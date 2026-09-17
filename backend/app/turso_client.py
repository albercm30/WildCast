"""
A minimal, stdlib-only client for Turso's HTTP API (https://docs.turso.tech/sdk/http/reference),
used as an optional persistence backend for app/db.py so accounts,
favorites, and saved-search alerts actually survive a redeploy on Render's
free tier -- which has no persistent disk at all (see app/db.py's
docstring for the real research behind that). Turso is a free,
SQLite-compatible hosted database (libSQL) reachable over plain HTTPS, so
it needs no new pip dependency (this sandbox's PyPI access is blocked, the
same reason this project has avoided every other third-party dependency --
see app/db.py's docstring) and no local disk at all.

Wire format verified against the real Hrana 3 protocol spec
(https://github.com/tursodatabase/libsql/blob/main/docs/HRANA_3_SPEC.md)
and Turso's HTTP API reference (both fetched 2026-09-17), not assumed:
POST {base_url}/v2/pipeline, `Authorization: Bearer <token>`, a JSON body
of `{"requests": [...]}` where each request is `{"type": "execute", "stmt":
{"sql": ..., "args": [...]}}` (or `{"type": "close"}` to end the
server-side session). Every Value in a request or response is tagged by
type -- notably **INTEGER values are JSON STRINGS** ("value": "42"), not
JSON numbers, specifically to avoid 64-bit precision loss; FLOAT values
are plain JSON numbers. Getting this backwards would silently corrupt
every integer parameter (user ids, favorite ids, ...), so it's worth
calling out explicitly here rather than leaving it as an implementation
detail.

IMPORTANT -- this has NOT been tested against a real Turso database. This
sandbox's outbound network is locked down to an allowlist that doesn't
include turso.io (confirmed: a plain `curl https://turso.tech` here gets a
407/403 from the proxy, the same restriction documented for GBIF/eBird/
iNaturalist/IUCN/Protected Planet elsewhere in this project), so this
module is written defensively against the real protocol spec and covered
by mocked unit tests (test_turso_client.py), the same "verify against real
docs, flag for live re-verification" standard already applied to
iucn_client.py and protected_planet_client.py. Re-verify against a real
Turso database (see README's "Accounts & saved-search alerts" section for
setup) before fully trusting this in production -- particularly the error
response shape (`_raise_if_error`), which the docs page didn't show a
worked example of.

Deliberately reuses sqlite3's own exception classes
(sqlite3.IntegrityError, sqlite3.OperationalError) for errors, purely as a
familiar DB-API-shaped error type -- NOT because this talks to sqlite3 at
all -- so app/services/auth_service.py's existing
`except sqlite3.IntegrityError` needs zero changes to work against either
backend.

Each Connection.execute() call is ONE HTTP round trip that auto-commits
immediately (bundled with an explicit `PRAGMA foreign_keys = ON` first, so
foreign-key enforcement -- e.g. a future "delete my account" cascading
delete -- works correctly even though nothing here holds a real
persistent connection open across calls the way sqlite3 does). This is a
deliberate simplification, not an oversight: every current caller in
app/services/{auth,favorites,saved_search,notification}_service.py issues
its statements sequentially within one `with db.get_connection() as conn:`
block and never depends on a later statement in the same block being able
to roll back an earlier one within that block (audited 2026-09-17 across
all four service files) -- so per-statement autocommit is safe here, not
just convenient. `commit()`/`rollback()`/`close()` on the Connection are
no-ops for exactly this reason.
"""
from __future__ import annotations

import base64
import json
import sqlite3
import urllib.error
import urllib.request
from typing import Any

_DEFAULT_TIMEOUT = 15.0


def _encode_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):  # must precede the int check -- bool is a subclass of int
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, int):
        return {"type": "integer", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "text", "value": value}
    if isinstance(value, (bytes, bytearray)):
        return {"type": "blob", "base64": base64.b64encode(bytes(value)).decode("ascii")}
    raise TypeError(f"Cannot encode {type(value).__name__!r} as a Turso SQL parameter")


def _decode_value(cell: dict[str, Any]) -> Any:
    kind = cell.get("type")
    if kind == "null":
        return None
    if kind == "integer":
        return int(cell["value"])
    if kind == "float":
        return float(cell["value"])
    if kind == "text":
        return cell["value"]
    if kind == "blob":
        return base64.b64decode(cell["base64"])
    raise ValueError(f"Unrecognized Turso value type {kind!r} in response: {cell!r}")


def _raise_if_error(result: dict[str, Any], sql: str) -> None:
    if result.get("type") == "ok":
        return
    err = result.get("error") or {}
    message = err.get("message") or json.dumps(result)
    # SQLite's own standard error text for a UNIQUE-constraint violation --
    # matched defensively (substring, case-sensitive to SQLite's own
    # wording) since the exact shape of Turso's error object wasn't
    # confirmed against a worked example in the docs (see module
    # docstring). Any other failure surfaces as OperationalError, the
    # usual DB-API catch-all for "the database rejected this statement."
    if "UNIQUE constraint failed" in message:
        raise sqlite3.IntegrityError(message)
    raise sqlite3.OperationalError(f"Turso error executing {sql!r}: {message}")


class Row:
    """A read-only, dict-convertible row -- mirrors the subset of
    sqlite3.Row's behavior this project's service-layer code actually uses
    (`row["col"]` and `dict(row)`), so a caller written against
    sqlite3.Row needs zero changes."""

    __slots__ = ("_cols", "_values", "_index")

    def __init__(self, cols: list[str], values: list[Any]):
        self._cols = cols
        self._values = values
        self._index = {name: i for i, name in enumerate(cols)}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return self._values[self._index[key]]

    def keys(self) -> list[str]:
        return list(self._cols)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"Row({dict(zip(self._cols, self._values))!r})"


class Cursor:
    """Just enough of sqlite3's cursor surface for this project's
    service-layer code: .lastrowid, .rowcount, .fetchone(), .fetchall()."""

    def __init__(self, rows: list[Row], affected_row_count: int | None, last_insert_rowid: int | None):
        self._rows = rows
        self._pos = 0
        self.rowcount = affected_row_count if affected_row_count is not None else -1
        self.lastrowid = last_insert_rowid

    def fetchone(self) -> Row | None:
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchall(self) -> list[Row]:
        remaining = self._rows[self._pos:]
        self._pos = len(self._rows)
        return remaining


def _post(url: str, token: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- fixed https host, not user input
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise sqlite3.OperationalError(f"Turso request failed (HTTP {exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise sqlite3.OperationalError(f"Turso request failed: {exc.reason}") from exc


class Connection:
    def __init__(self, base_url: str, token: str, timeout: float = _DEFAULT_TIMEOUT):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _pipeline(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        data = _post(f"{self._base_url}/v2/pipeline", self._token, {"requests": requests}, self._timeout)
        return data["results"]

    def execute(self, sql: str, params: tuple = ()) -> Cursor:
        args = [_encode_value(p) for p in params]
        results = self._pipeline(
            [
                {"type": "execute", "stmt": {"sql": "PRAGMA foreign_keys = ON"}},
                {"type": "execute", "stmt": {"sql": sql, "args": args}},
                {"type": "close"},
            ]
        )
        stmt_result = results[1]
        _raise_if_error(stmt_result, sql)
        payload = stmt_result["response"]["result"]
        cols = [c["name"] for c in payload.get("cols") or []]
        rows = [Row(cols, [_decode_value(cell) for cell in r]) for r in payload.get("rows") or []]
        last_id = payload.get("last_insert_rowid")
        if isinstance(last_id, str):
            last_id = int(last_id)
        return Cursor(rows, payload.get("affected_row_count"), last_id)

    def executescript(self, script: str) -> None:
        """Runs each ';'-separated statement in `script` in one pipeline
        call. Only ever used here with this project's own static schema
        DDL (app/db.py's SCHEMA / DROP TABLE strings), which contains no
        semicolons inside string literals, so a naive split is safe --
        this is NOT a general-purpose SQL statement splitter."""
        statements = [s.strip() for s in script.split(";") if s.strip()]
        if not statements:
            return
        requests: list[dict[str, Any]] = [{"type": "execute", "stmt": {"sql": "PRAGMA foreign_keys = ON"}}]
        requests.extend({"type": "execute", "stmt": {"sql": stmt}} for stmt in statements)
        requests.append({"type": "close"})
        results = self._pipeline(requests)
        for stmt, result in zip(statements, results[1:-1]):
            _raise_if_error(result, stmt)

    def commit(self) -> None:
        pass  # each execute() already committed individually -- see module docstring

    def rollback(self) -> None:
        pass  # no open cross-statement transaction to roll back -- see module docstring

    def close(self) -> None:
        pass  # stateless: no persistent connection is held open between calls


def connect(database_url: str, auth_token: str, timeout: float = _DEFAULT_TIMEOUT) -> Connection:
    """`database_url` may be the `libsql://...` form Turso's dashboard/CLI
    show by default (meant for their native SDKs) -- converted to
    `https://...` here since the HTTP API needs a real scheme, a real
    footgun this project has been bitten by before in a different form
    (see config.py's WILDCAST_DB_PATH blank-value fix)."""
    base = database_url.strip()
    if base.startswith("libsql://"):
        base = "https://" + base[len("libsql://"):]
    return Connection(base, auth_token, timeout)

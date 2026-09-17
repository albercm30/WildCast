"""
Unit tests for app.turso_client -- mocks urllib.request.urlopen throughout,
same "can't reach the real service from this sandbox" pattern as
test_iucn_client.py/test_protected_planet_client.py (this sandbox's egress
doesn't reach turso.io either, confirmed via a direct curl -- see the
module's own docstring). These pin the wire format against the real Hrana
3 spec's documented encoding, and should be re-checked against a real
Turso database once one exists (see README's "Accounts & saved-search
alerts" section).
"""
import json
import sqlite3
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from app import turso_client as tc


def _fake_response(body: dict) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _ok_execute_result(cols=None, rows=None, affected_row_count=None, last_insert_rowid=None):
    return {
        "type": "ok",
        "response": {
            "type": "execute",
            "result": {
                "cols": cols or [],
                "rows": rows or [],
                "affected_row_count": affected_row_count,
                "last_insert_rowid": last_insert_rowid,
            },
        },
    }


class EncodeValueTests(unittest.TestCase):
    def test_none(self):
        self.assertEqual(tc._encode_value(None), {"type": "null"})

    def test_bool_before_int(self):
        # bool is an int subclass -- must be checked first or True/False
        # would silently encode as integer 1/0 anyway (harmless here, but
        # the explicit ordering is what makes that intentional, not lucky).
        self.assertEqual(tc._encode_value(True), {"type": "integer", "value": "1"})
        self.assertEqual(tc._encode_value(False), {"type": "integer", "value": "0"})

    def test_int_is_stringified(self):
        # The one detail this whole client exists to get right -- see the
        # module docstring's explanation of why Hrana stringifies integers.
        self.assertEqual(tc._encode_value(42), {"type": "integer", "value": "42"})
        self.assertEqual(tc._encode_value(-7), {"type": "integer", "value": "-7"})

    def test_float_is_a_json_number(self):
        self.assertEqual(tc._encode_value(3.5), {"type": "float", "value": 3.5})

    def test_text(self):
        self.assertEqual(tc._encode_value("hello"), {"type": "text", "value": "hello"})

    def test_blob_is_base64(self):
        encoded = tc._encode_value(b"\x00\x01\xff")
        self.assertEqual(encoded["type"], "blob")
        self.assertIn("base64", encoded)

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            tc._encode_value(object())


class DecodeValueTests(unittest.TestCase):
    def test_null(self):
        self.assertIsNone(tc._decode_value({"type": "null"}))

    def test_integer_parsed_from_string(self):
        self.assertEqual(tc._decode_value({"type": "integer", "value": "123"}), 123)
        self.assertIsInstance(tc._decode_value({"type": "integer", "value": "123"}), int)

    def test_float(self):
        self.assertEqual(tc._decode_value({"type": "float", "value": 1.5}), 1.5)

    def test_text(self):
        self.assertEqual(tc._decode_value({"type": "text", "value": "hi"}), "hi")

    def test_blob(self):
        import base64

        raw = b"abc"
        cell = {"type": "blob", "base64": base64.b64encode(raw).decode("ascii")}
        self.assertEqual(tc._decode_value(cell), raw)

    def test_unrecognized_type_raises(self):
        with self.assertRaises(ValueError):
            tc._decode_value({"type": "weird"})


class RaiseIfErrorTests(unittest.TestCase):
    def test_ok_does_nothing(self):
        tc._raise_if_error({"type": "ok"}, "SELECT 1")  # no raise

    def test_unique_violation_raises_integrity_error(self):
        result = {"type": "error", "error": {"message": "UNIQUE constraint failed: users.email"}}
        with self.assertRaises(sqlite3.IntegrityError):
            tc._raise_if_error(result, "INSERT INTO users ...")

    def test_other_error_raises_operational_error(self):
        result = {"type": "error", "error": {"message": "no such table: bogus"}}
        with self.assertRaises(sqlite3.OperationalError):
            tc._raise_if_error(result, "SELECT * FROM bogus")

    def test_missing_message_falls_back_to_dumping_the_result(self):
        with self.assertRaises(sqlite3.OperationalError):
            tc._raise_if_error({"type": "error"}, "SELECT 1")


class RowTests(unittest.TestCase):
    def test_getitem_by_name_and_index(self):
        row = tc.Row(["id", "email"], [5, "a@example.com"])
        self.assertEqual(row["id"], 5)
        self.assertEqual(row["email"], "a@example.com")
        self.assertEqual(row[0], 5)

    def test_dict_conversion(self):
        row = tc.Row(["id", "email"], [5, "a@example.com"])
        self.assertEqual(dict(row), {"id": 5, "email": "a@example.com"})

    def test_keys(self):
        row = tc.Row(["id", "email"], [5, "a@example.com"])
        self.assertEqual(row.keys(), ["id", "email"])


class CursorTests(unittest.TestCase):
    def test_fetchone_then_fetchall(self):
        rows = [tc.Row(["id"], [1]), tc.Row(["id"], [2]), tc.Row(["id"], [3])]
        cur = tc.Cursor(rows, affected_row_count=3, last_insert_rowid=3)
        self.assertEqual(cur.fetchone()["id"], 1)
        remaining = cur.fetchall()
        self.assertEqual([r["id"] for r in remaining], [2, 3])
        self.assertIsNone(cur.fetchone())

    def test_rowcount_and_lastrowid(self):
        cur = tc.Cursor([], affected_row_count=1, last_insert_rowid=42)
        self.assertEqual(cur.rowcount, 1)
        self.assertEqual(cur.lastrowid, 42)

    def test_rowcount_defaults_to_minus_one_when_none(self):
        cur = tc.Cursor([], affected_row_count=None, last_insert_rowid=None)
        self.assertEqual(cur.rowcount, -1)


class ConnectTests(unittest.TestCase):
    def test_libsql_scheme_converted_to_https(self):
        conn = tc.connect("libsql://my-db-org.turso.io", "tok")
        self.assertEqual(conn._base_url, "https://my-db-org.turso.io")

    def test_https_scheme_left_as_is(self):
        conn = tc.connect("https://my-db-org.turso.io", "tok")
        self.assertEqual(conn._base_url, "https://my-db-org.turso.io")

    def test_trailing_slash_stripped(self):
        conn = tc.connect("https://my-db-org.turso.io/", "tok")
        self.assertEqual(conn._base_url, "https://my-db-org.turso.io")


class ConnectionExecuteTests(unittest.TestCase):
    def setUp(self):
        self.conn = tc.connect("https://db-org.turso.io", "tok")

    @patch("app.turso_client.urllib.request.urlopen")
    def test_sends_pragma_then_statement_then_close(self, mock_urlopen):
        body = {"results": [_ok_execute_result(), _ok_execute_result(), {"type": "ok"}]}
        mock_urlopen.return_value = _fake_response(body)
        self.conn.execute("SELECT 1")
        sent_request = mock_urlopen.call_args[0][0]
        sent_body = json.loads(sent_request.data.decode("utf-8"))
        types = [r["type"] for r in sent_body["requests"]]
        self.assertEqual(types, ["execute", "execute", "close"])
        self.assertEqual(sent_body["requests"][0]["stmt"]["sql"], "PRAGMA foreign_keys = ON")
        self.assertEqual(sent_body["requests"][1]["stmt"]["sql"], "SELECT 1")

    @patch("app.turso_client.urllib.request.urlopen")
    def test_sends_auth_header(self, mock_urlopen):
        body = {"results": [_ok_execute_result(), _ok_execute_result(), {"type": "ok"}]}
        mock_urlopen.return_value = _fake_response(body)
        self.conn.execute("SELECT 1")
        sent_request = mock_urlopen.call_args[0][0]
        self.assertEqual(sent_request.get_header("Authorization"), "Bearer tok")

    @patch("app.turso_client.urllib.request.urlopen")
    def test_params_encoded_as_tagged_args(self, mock_urlopen):
        body = {"results": [_ok_execute_result(), _ok_execute_result(), {"type": "ok"}]}
        mock_urlopen.return_value = _fake_response(body)
        self.conn.execute("SELECT * FROM users WHERE id = ? AND email = ?", (5, "a@example.com"))
        sent_body = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        args = sent_body["requests"][1]["stmt"]["args"]
        self.assertEqual(args, [{"type": "integer", "value": "5"}, {"type": "text", "value": "a@example.com"}])

    @patch("app.turso_client.urllib.request.urlopen")
    def test_parses_rows_and_lastrowid(self, mock_urlopen):
        result = _ok_execute_result(
            cols=[{"name": "id"}, {"name": "email"}],
            rows=[[{"type": "integer", "value": "1"}, {"type": "text", "value": "a@example.com"}]],
            affected_row_count=1,
            last_insert_rowid="1",
        )
        body = {"results": [_ok_execute_result(), result, {"type": "ok"}]}
        mock_urlopen.return_value = _fake_response(body)
        cur = self.conn.execute("INSERT INTO users ... RETURNING *")
        row = cur.fetchone()
        self.assertEqual(dict(row), {"id": 1, "email": "a@example.com"})
        self.assertEqual(cur.lastrowid, 1)
        self.assertIsInstance(cur.lastrowid, int)
        self.assertEqual(cur.rowcount, 1)

    @patch("app.turso_client.urllib.request.urlopen")
    def test_unique_violation_raises_integrity_error(self, mock_urlopen):
        error_result = {"type": "error", "error": {"message": "UNIQUE constraint failed: users.email"}}
        body = {"results": [_ok_execute_result(), error_result, {"type": "ok"}]}
        mock_urlopen.return_value = _fake_response(body)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("INSERT INTO users (email) VALUES (?)", ("dup@example.com",))

    @patch("app.turso_client.urllib.request.urlopen")
    def test_http_error_becomes_operational_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://db-org.turso.io/v2/pipeline", 401, "Unauthorized", {}, MagicMock(read=lambda: b"bad token")
        )
        with self.assertRaises(sqlite3.OperationalError):
            self.conn.execute("SELECT 1")

    @patch("app.turso_client.urllib.request.urlopen")
    def test_url_error_becomes_operational_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("network unreachable")
        with self.assertRaises(sqlite3.OperationalError):
            self.conn.execute("SELECT 1")

    def test_commit_rollback_close_are_harmless_noops(self):
        self.conn.commit()
        self.conn.rollback()
        self.conn.close()  # no exception is the assertion


class ConnectionExecuteScriptTests(unittest.TestCase):
    def setUp(self):
        self.conn = tc.connect("https://db-org.turso.io", "tok")

    @patch("app.turso_client.urllib.request.urlopen")
    def test_splits_and_sends_every_statement(self, mock_urlopen):
        body = {
            "results": [
                _ok_execute_result(),  # pragma
                _ok_execute_result(),  # statement 1
                _ok_execute_result(),  # statement 2
                {"type": "ok"},  # close
            ]
        }
        mock_urlopen.return_value = _fake_response(body)
        self.conn.executescript("CREATE TABLE a (id INTEGER); CREATE TABLE b (id INTEGER);")
        sent_body = json.loads(mock_urlopen.call_args[0][0].data.decode("utf-8"))
        sqls = [r["stmt"]["sql"] for r in sent_body["requests"] if r["type"] == "execute"]
        self.assertEqual(sqls, ["PRAGMA foreign_keys = ON", "CREATE TABLE a (id INTEGER)", "CREATE TABLE b (id INTEGER)"])

    @patch("app.turso_client.urllib.request.urlopen")
    def test_empty_script_sends_nothing(self, mock_urlopen):
        self.conn.executescript("   ;  ; ")
        mock_urlopen.assert_not_called()

    @patch("app.turso_client.urllib.request.urlopen")
    def test_error_in_one_statement_raises(self, mock_urlopen):
        body = {
            "results": [
                _ok_execute_result(),
                {"type": "error", "error": {"message": "syntax error"}},
                {"type": "ok"},
            ]
        }
        mock_urlopen.return_value = _fake_response(body)
        with self.assertRaises(sqlite3.OperationalError):
            self.conn.executescript("BOGUS SQL;")


if __name__ == "__main__":
    unittest.main()

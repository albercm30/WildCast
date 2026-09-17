"""
Lightweight persistence layer for user accounts, favorites, saved searches,
and saved-search notifications.

Built directly on Python's stdlib `sqlite3` rather than an ORM (SQLAlchemy
or similar) for the same reason this project avoids other unprovable
dependencies (see README's "Why Flask, not FastAPI" note): this sandbox's
egress is blocked to PyPI, so a new third-party dependency added here could
not actually be installed or tested end-to-end before delivery -- only
claimed to work. stdlib `sqlite3` needs no install step at all, so every
line of this module has actually been run against a real database file in
this sandbox, the same standard this project holds every other piece of
code to.

IMPORTANT -- read before deploying: Render's web services have an
EPHEMERAL filesystem by default (https://render.com/docs/disks, confirmed
2026-09-17) -- any local file, including this SQLite database, is wiped on
every redeploy or restart. Accounts, favorites and saved searches WILL be
lost on the next deploy unless you either:
  1. Attach a Render persistent disk (paid plans only -- not available on
     the free tier) and set WILDCAST_DB_PATH to a file inside its mount, or
  2. Migrate this module to a managed Postgres database instead (not done
     here -- would need a driver such as psycopg2, which this sandbox
     cannot install or test either, so it's left as a documented follow-up
     rather than shipped unverified. The service-layer functions in
     app/services/auth_service.py, favorites_service.py,
     saved_search_service.py and notification_service.py all go through
     this module alone, so that migration would only touch this one file).
See README.md's "Accounts & saved-search alerts" section for the full
deployment checklist.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from contextlib import contextmanager

from app.config import DB_PATH as _DEFAULT_DB_PATH

# Mutable on purpose (not a frozen import-time constant): tests point this
# at an isolated temp-file path per test case via `db.DB_PATH = ...` +
# `db.reset_for_tests()`, the same pattern app.ml.prediction_service's
# module-level caches use for test isolation.
DB_PATH = _DEFAULT_DB_PATH

# Guards the schema-creation step only (see _initialized below), not normal
# request traffic -- SQLite serializes writes at the file level on its own,
# and each caller here gets its own short-lived connection rather than a
# shared one, so plain concurrent requests need no extra locking.
_init_lock = threading.Lock()
_initialized = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('area', 'location')),
    area_id TEXT,
    lat REAL,
    lon REAL,
    radius_km REAL,
    species_key TEXT,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_id);

CREATE TABLE IF NOT EXISTS saved_searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('area', 'location')),
    area_id TEXT,
    lat REAL,
    lon REAL,
    radius_km REAL,
    species_key TEXT,
    min_probability REAL NOT NULL,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_checked_at TEXT,
    last_notified_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_saved_searches_user ON saved_searches(user_id);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    saved_search_id INTEGER REFERENCES saved_searches(id) ON DELETE SET NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    read_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Creates every table (idempotent -- IF NOT EXISTS) if this process
    hasn't already done so against the current DB_PATH. Called once eagerly
    at app startup (see app/main.py's create_app) and defensively by
    get_connection() too, so any code path that touches the database is
    guaranteed a ready schema regardless of import order."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        conn = _connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()
        _initialized = True


@contextmanager
def get_connection():
    """A short-lived connection for one unit of work: commits on success,
    rolls back on any exception, always closes. Every service-layer
    function in app/services/{auth,favorites,saved_search,notification}_service.py
    goes through this -- never a module-level shared connection -- so
    Flask's multi-threaded dev/gunicorn workers can't corrupt each other's
    in-flight transactions."""
    init_db()
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reset_for_tests() -> None:
    """Test-only: drops and recreates every table against the CURRENT
    DB_PATH. Tests set `db.DB_PATH` to an isolated temp-file path first
    (":memory:" doesn't work here -- this module opens a fresh connection
    per call, and SQLite's in-memory databases don't persist across
    connections) so no two tests share state."""
    global _initialized
    _initialized = False
    conn = _connect()
    try:
        conn.executescript(
            "DROP TABLE IF EXISTS notifications;"
            "DROP TABLE IF EXISTS saved_searches;"
            "DROP TABLE IF EXISTS favorites;"
            "DROP TABLE IF EXISTS users;"
        )
        conn.commit()
    finally:
        conn.close()
    init_db()


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

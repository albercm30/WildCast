"""
Shared test setup for anything touching app/db.py (accounts, favorites,
saved searches, notifications) -- not itself a test file (doesn't match
unittest discover's default `test*.py` pattern), just a helper the actual
test modules import.

Every test that touches the database gets its own fresh, isolated SQLite
file: app.db.DB_PATH is a mutable module attribute specifically so tests
can repoint it (the same pattern app.ml.prediction_service's module-level
caches use), and reset_for_tests() drops+recreates every table against
whatever DB_PATH currently points at. Without this, tests would either
share one real database file (state leaking between tests, and polluting
whatever's at backend/data/wildcast.db) or need a real running server.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app import db


class TempDbTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._original_db_path = db.DB_PATH
        fd, path = tempfile.mkstemp(suffix=".db", prefix="wildcast_test_")
        import os

        os.close(fd)
        db.DB_PATH = Path(path)
        db.reset_for_tests()

    def tearDown(self):
        super().tearDown()
        stale_path = db.DB_PATH
        db.DB_PATH = self._original_db_path
        db._initialized = False  # noqa: SLF001 -- test-only reset of the other module's cache flag
        try:
            stale_path.unlink(missing_ok=True)
        except OSError:
            pass

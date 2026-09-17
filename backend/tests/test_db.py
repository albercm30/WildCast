import sqlite3
import unittest

from app import db
from tests._db_test_utils import TempDbTestCase


class SchemaTests(TempDbTestCase):
    def test_all_four_tables_exist_after_init(self):
        with db.get_connection() as conn:
            names = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
        self.assertTrue({"users", "favorites", "saved_searches", "notifications"}.issubset(names))

    def test_init_db_is_idempotent(self):
        db.init_db()
        db.init_db()  # must not raise (CREATE TABLE IF NOT EXISTS, re-running the whole script)
        with db.get_connection() as conn:
            conn.execute("SELECT 1 FROM users LIMIT 1")  # table still usable, not dropped/corrupted

    def test_foreign_keys_are_enforced(self):
        # A favorite/saved_search/notification pointing at a nonexistent
        # user_id must be rejected -- without `PRAGMA foreign_keys = ON`,
        # SQLite silently allows orphaned rows by default.
        with self.assertRaises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                conn.execute(
                    "INSERT INTO favorites (user_id, kind, area_id, label, created_at) VALUES (?, ?, ?, ?, ?)",
                    (999999, "area", "yellowstone", "test", db.now_iso()),
                )

    def test_deleting_a_user_cascades_to_their_favorites(self):
        with db.get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                ("cascade@example.com", "x", "", db.now_iso()),
            )
            user_id = cur.lastrowid
            conn.execute(
                "INSERT INTO favorites (user_id, kind, area_id, label, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, "area", "yellowstone", "test", db.now_iso()),
            )
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            remaining = conn.execute("SELECT COUNT(*) FROM favorites WHERE user_id = ?", (user_id,)).fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_now_iso_returns_a_real_iso_timestamp(self):
        self.assertRegex(db.now_iso(), r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def test_get_connection_rolls_back_on_exception(self):
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                ("rollback@example.com", "x", "", db.now_iso()),
            )
        try:
            with db.get_connection() as conn:
                conn.execute(
                    "INSERT INTO users (email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                    ("second@example.com", "x", "", db.now_iso()),
                )
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        with db.get_connection() as conn:
            emails = {r[0] for r in conn.execute("SELECT email FROM users").fetchall()}
        self.assertIn("rollback@example.com", emails)
        self.assertNotIn("second@example.com", emails)  # rolled back, never committed


if __name__ == "__main__":
    unittest.main()

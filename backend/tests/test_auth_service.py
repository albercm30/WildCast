import time
import unittest
from unittest.mock import patch

from app.services import auth_service
from tests._db_test_utils import TempDbTestCase


class RegisterTests(TempDbTestCase):
    def test_registering_returns_a_user_and_a_token(self):
        user, token = auth_service.register("Alber@Example.com", "correcthorse123", "Alber")
        self.assertEqual(user["email"], "alber@example.com")  # normalized to lowercase
        self.assertEqual(user["display_name"], "Alber")
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 10)

    def test_password_hash_is_never_returned(self):
        user, _token = auth_service.register("hidden@example.com", "correcthorse123")
        self.assertNotIn("password_hash", user)

    def test_duplicate_email_is_rejected(self):
        auth_service.register("dupe@example.com", "correcthorse123")
        with self.assertRaises(auth_service.AuthError):
            auth_service.register("dupe@example.com", "anotherpassword")

    def test_duplicate_email_is_rejected_case_insensitively(self):
        auth_service.register("Case@Example.com", "correcthorse123")
        with self.assertRaises(auth_service.AuthError):
            auth_service.register("case@example.com", "anotherpassword")

    def test_invalid_email_is_rejected(self):
        with self.assertRaises(auth_service.AuthError):
            auth_service.register("not-an-email", "correcthorse123")

    def test_short_password_is_rejected(self):
        with self.assertRaises(auth_service.AuthError):
            auth_service.register("short@example.com", "1234567")  # 7 chars, under the 8-char floor


class LoginTests(TempDbTestCase):
    def setUp(self):
        super().setUp()
        auth_service.register("login@example.com", "correcthorse123", "Login Test")

    def test_correct_credentials_succeed(self):
        user, token = auth_service.login("login@example.com", "correcthorse123")
        self.assertEqual(user["email"], "login@example.com")
        self.assertIsInstance(token, str)

    def test_login_is_case_insensitive_on_email(self):
        user, _token = auth_service.login("Login@Example.com", "correcthorse123")
        self.assertEqual(user["email"], "login@example.com")

    def test_wrong_password_fails(self):
        with self.assertRaises(auth_service.AuthError):
            auth_service.login("login@example.com", "wrongpassword")

    def test_unknown_email_fails(self):
        with self.assertRaises(auth_service.AuthError):
            auth_service.login("nobody@example.com", "correcthorse123")

    def test_wrong_password_and_unknown_email_give_the_same_error_message(self):
        # Distinguishing them would let an attacker enumerate which emails
        # have real accounts -- see auth_service.login's comment.
        try:
            auth_service.login("nobody@example.com", "whatever123")
        except auth_service.AuthError as exc:
            unknown_email_message = str(exc)
        try:
            auth_service.login("login@example.com", "wrongpassword")
        except auth_service.AuthError as exc:
            wrong_password_message = str(exc)
        self.assertEqual(unknown_email_message, wrong_password_message)


class TokenTests(TempDbTestCase):
    def test_a_freshly_issued_token_resolves_to_the_right_user(self):
        user, token = auth_service.register("tok@example.com", "correcthorse123")
        resolved = auth_service.user_from_token(token)
        self.assertEqual(resolved["id"], user["id"])
        self.assertEqual(resolved["email"], "tok@example.com")

    def test_a_garbage_token_is_rejected(self):
        with self.assertRaises(auth_service.AuthError):
            auth_service.user_from_token("not-a-real-token")

    def test_a_token_for_a_since_deleted_user_is_rejected(self):
        user, token = auth_service.register("deleteme@example.com", "correcthorse123")
        from app import db

        with db.get_connection() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user["id"],))
        with self.assertRaises(auth_service.AuthError):
            auth_service.user_from_token(token)

    def test_an_expired_token_is_rejected(self):
        _user, token = auth_service.register("expired@example.com", "correcthorse123")
        with patch.object(auth_service, "_TOKEN_MAX_AGE_SECONDS", 0):
            time.sleep(1.1)
            with self.assertRaises(auth_service.AuthError):
                auth_service.user_from_token(token)


class UpdateDisplayNameTests(TempDbTestCase):
    def test_display_name_can_be_updated(self):
        user, _token = auth_service.register("rename@example.com", "correcthorse123", "Old Name")
        updated = auth_service.update_display_name(user["id"], "New Name")
        self.assertEqual(updated["email"], "rename@example.com")
        self.assertEqual(updated["display_name"], "New Name")


if __name__ == "__main__":
    unittest.main()

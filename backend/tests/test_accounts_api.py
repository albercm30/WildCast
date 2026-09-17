"""
Flask test-client-level tests for the accounts/favorites/saved-searches/
notifications/internal-cron endpoints added in app/routers/{auth,favorites,
saved_searches,notifications,internal}.py -- same style as test_api.py's
existing endpoint tests, but with an isolated temp SQLite database per test
(see tests/_db_test_utils.py) instead of the shared prediction-model cache.
"""
import unittest
from unittest.mock import patch

from app.main import create_app
from tests._db_test_utils import TempDbTestCase


class AccountsApiTestCase(TempDbTestCase):
    def setUp(self):
        super().setUp()
        self.client = create_app().test_client()

    def _register(self, email="user@example.com", password="correcthorse123", display_name="Test User"):
        resp = self.client.post(
            "/api/auth/register", json={"email": email, "password": password, "display_name": display_name}
        )
        body = resp.get_json()
        return resp, body

    def _auth_header(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}


class RegisterLoginMeTests(AccountsApiTestCase):
    def test_register_returns_201_with_user_and_token(self):
        resp, body = self._register()
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(body["user"]["email"], "user@example.com")
        self.assertIn("token", body)
        self.assertNotIn("password_hash", body["user"])

    def test_register_duplicate_email_is_400(self):
        self._register()
        resp, body = self._register()
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", body)

    def test_register_missing_fields_is_400(self):
        resp = self.client.post("/api/auth/register", json={})
        self.assertEqual(resp.status_code, 400)

    def test_login_with_correct_credentials(self):
        self._register(email="login@example.com", password="correcthorse123")
        resp = self.client.post("/api/auth/login", json={"email": "login@example.com", "password": "correcthorse123"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("token", resp.get_json())

    def test_login_with_wrong_password_is_401(self):
        self._register(email="login2@example.com", password="correcthorse123")
        resp = self.client.post("/api/auth/login", json={"email": "login2@example.com", "password": "nope"})
        self.assertEqual(resp.status_code, 401)

    def test_me_without_token_is_401(self):
        resp = self.client.get("/api/auth/me")
        self.assertEqual(resp.status_code, 401)

    def test_me_with_bad_token_is_401(self):
        resp = self.client.get("/api/auth/me", headers=self._auth_header("garbage"))
        self.assertEqual(resp.status_code, 401)

    def test_me_with_valid_token_returns_the_right_user(self):
        _resp, body = self._register(email="whoami@example.com")
        resp = self.client.get("/api/auth/me", headers=self._auth_header(body["token"]))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["user"]["email"], "whoami@example.com")

    def test_patch_me_updates_display_name(self):
        _resp, body = self._register(email="patchme@example.com", display_name="Old")
        resp = self.client.patch("/api/auth/me", json={"display_name": "New"}, headers=self._auth_header(body["token"]))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["user"]["display_name"], "New")


class FavoritesApiTests(AccountsApiTestCase):
    def setUp(self):
        super().setUp()
        _resp, body = self._register()
        self.token = body["token"]

    def test_favorites_requires_auth(self):
        resp = self.client.get("/api/favorites")
        self.assertEqual(resp.status_code, 401)

    def test_add_list_and_remove_a_favorite(self):
        resp = self.client.post(
            "/api/favorites",
            json={"kind": "area", "area_id": "yellowstone", "label": "Yellowstone"},
            headers=self._auth_header(self.token),
        )
        self.assertEqual(resp.status_code, 201)
        favorite_id = resp.get_json()["id"]

        resp = self.client.get("/api/favorites", headers=self._auth_header(self.token))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.get_json()), 1)

        resp = self.client.delete(f"/api/favorites/{favorite_id}", headers=self._auth_header(self.token))
        self.assertEqual(resp.status_code, 204)
        resp = self.client.get("/api/favorites", headers=self._auth_header(self.token))
        self.assertEqual(resp.get_json(), [])

    def test_add_invalid_favorite_is_400(self):
        resp = self.client.post("/api/favorites", json={"kind": "area"}, headers=self._auth_header(self.token))
        self.assertEqual(resp.status_code, 400)

    def test_deleting_someone_elses_favorite_is_404(self):
        resp = self.client.post(
            "/api/favorites",
            json={"kind": "area", "area_id": "yellowstone", "label": "mine"},
            headers=self._auth_header(self.token),
        )
        favorite_id = resp.get_json()["id"]

        _resp, other_body = self._register(email="other@example.com")
        resp = self.client.delete(f"/api/favorites/{favorite_id}", headers=self._auth_header(other_body["token"]))
        self.assertEqual(resp.status_code, 404)


class SavedSearchesApiTests(AccountsApiTestCase):
    def setUp(self):
        super().setUp()
        _resp, body = self._register()
        self.token = body["token"]

    def test_add_and_list_a_saved_search(self):
        resp = self.client.post(
            "/api/saved-searches",
            json={"kind": "area", "area_id": "yellowstone", "label": "Wolves", "species_key": "Canis lupus", "min_probability": 0.6},
            headers=self._auth_header(self.token),
        )
        self.assertEqual(resp.status_code, 201)
        resp = self.client.get("/api/saved-searches", headers=self._auth_header(self.token))
        self.assertEqual(len(resp.get_json()), 1)

    def test_invalid_probability_is_400(self):
        resp = self.client.post(
            "/api/saved-searches",
            json={"kind": "area", "area_id": "yellowstone", "label": "x", "min_probability": 5},
            headers=self._auth_header(self.token),
        )
        self.assertEqual(resp.status_code, 400)

    def test_remove_a_saved_search(self):
        resp = self.client.post(
            "/api/saved-searches",
            json={"kind": "area", "area_id": "yellowstone", "label": "x", "min_probability": 0.5},
            headers=self._auth_header(self.token),
        )
        search_id = resp.get_json()["id"]
        resp = self.client.delete(f"/api/saved-searches/{search_id}", headers=self._auth_header(self.token))
        self.assertEqual(resp.status_code, 204)


class NotificationsApiTests(AccountsApiTestCase):
    def setUp(self):
        super().setUp()
        _resp, body = self._register()
        self.token = body["token"]
        self.user_id = body["user"]["id"]

    def test_empty_notifications_list(self):
        resp = self.client.get("/api/notifications", headers=self._auth_header(self.token))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), [])

    def test_mark_read_on_a_real_notification(self):
        from app import db

        with db.get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO notifications (user_id, message, created_at) VALUES (?, ?, ?)",
                (self.user_id, "test alert", db.now_iso()),
            )
            notification_id = cur.lastrowid
        resp = self.client.post(f"/api/notifications/{notification_id}/read", headers=self._auth_header(self.token))
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get("/api/notifications?unread_only=1", headers=self._auth_header(self.token))
        self.assertEqual(resp.get_json(), [])

    def test_mark_read_on_a_nonexistent_notification_is_404(self):
        resp = self.client.post("/api/notifications/999999/read", headers=self._auth_header(self.token))
        self.assertEqual(resp.status_code, 404)


class InternalCronEndpointTests(AccountsApiTestCase):
    def test_missing_secret_key_config_is_503(self):
        with patch("app.routers.internal.CRON_SECRET_KEY", ""):
            resp = self.client.post("/api/internal/check-saved-searches")
        self.assertEqual(resp.status_code, 503)

    def test_missing_header_is_401(self):
        with patch("app.routers.internal.CRON_SECRET_KEY", "s3cret"):
            resp = self.client.post("/api/internal/check-saved-searches")
        self.assertEqual(resp.status_code, 401)

    def test_wrong_header_value_is_401(self):
        with patch("app.routers.internal.CRON_SECRET_KEY", "s3cret"):
            resp = self.client.post("/api/internal/check-saved-searches", headers={"X-Cron-Key": "wrong"})
        self.assertEqual(resp.status_code, 401)

    def test_correct_header_runs_the_check(self):
        with (
            patch("app.routers.internal.CRON_SECRET_KEY", "s3cret"),
            patch("app.services.saved_search_service.check_due_saved_searches", return_value=3) as mocked,
        ):
            resp = self.client.post("/api/internal/check-saved-searches", headers={"X-Cron-Key": "s3cret"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["notifications_created"], 3)
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()

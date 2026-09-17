import unittest
from unittest.mock import patch

from app import db
from app.services import auth_service, notification_service, saved_search_service
from tests._db_test_utils import TempDbTestCase


class ValidationTests(TempDbTestCase):
    def setUp(self):
        super().setUp()
        user, _ = auth_service.register("saved-a@example.com", "correcthorse123")
        self.user_id = user["id"]

    def test_add_an_area_saved_search(self):
        s = saved_search_service.add_saved_search(
            self.user_id, kind="area", area_id="yellowstone", label="Wolves", species_key="Canis lupus",
            min_probability=0.6,
        )
        self.assertEqual(s["area_id"], "yellowstone")
        self.assertAlmostEqual(s["min_probability"], 0.6)

    def test_area_search_without_area_id_is_rejected(self):
        with self.assertRaises(saved_search_service.SavedSearchError):
            saved_search_service.add_saved_search(self.user_id, kind="area", label="x", min_probability=0.5)

    def test_location_search_without_coordinates_is_rejected(self):
        with self.assertRaises(saved_search_service.SavedSearchError):
            saved_search_service.add_saved_search(self.user_id, kind="location", label="x", min_probability=0.5)

    def test_probability_out_of_range_is_rejected(self):
        with self.assertRaises(saved_search_service.SavedSearchError):
            saved_search_service.add_saved_search(
                self.user_id, kind="area", area_id="yellowstone", label="x", min_probability=1.5
            )

    def test_non_numeric_probability_is_rejected(self):
        with self.assertRaises(saved_search_service.SavedSearchError):
            saved_search_service.add_saved_search(
                self.user_id, kind="area", area_id="yellowstone", label="x", min_probability="high"
            )

    def test_remove_enforces_ownership(self):
        other, _ = auth_service.register("saved-b@example.com", "correcthorse123")
        s = saved_search_service.add_saved_search(
            self.user_id, kind="area", area_id="yellowstone", label="x", min_probability=0.5
        )
        with self.assertRaises(saved_search_service.SavedSearchError):
            saved_search_service.remove_saved_search(other["id"], s["id"])


def _fake_predict_area(_area_id, _date, species_key=None):
    return [
        {"scientific_name": "Ursus arctos", "common_name": "Grizzly Bear", "probability": 0.82},
        {"scientific_name": "Canis lupus", "common_name": "Gray Wolf", "probability": 0.3},
    ]


class CheckDueSavedSearchesTests(TempDbTestCase):
    def setUp(self):
        super().setUp()
        user, _ = auth_service.register("alerts@example.com", "correcthorse123")
        self.user_id = user["id"]

    def test_a_search_above_threshold_creates_a_notification(self):
        saved_search_service.add_saved_search(
            self.user_id, kind="area", area_id="yellowstone", label="Bears", min_probability=0.5,
        )
        with patch("app.ml.prediction_service.predict", side_effect=_fake_predict_area):
            created = saved_search_service.check_due_saved_searches()
        self.assertEqual(created, 1)
        notes = notification_service.list_notifications(self.user_id)
        self.assertEqual(len(notes), 1)
        self.assertIn("Grizzly Bear", notes[0]["message"])
        self.assertIn("82%", notes[0]["message"])

    def test_a_search_below_threshold_creates_nothing(self):
        saved_search_service.add_saved_search(
            self.user_id, kind="area", area_id="yellowstone", label="Bears", min_probability=0.95,
        )
        with patch("app.ml.prediction_service.predict", side_effect=_fake_predict_area):
            created = saved_search_service.check_due_saved_searches()
        self.assertEqual(created, 0)
        self.assertEqual(notification_service.list_notifications(self.user_id), [])

    def test_a_failing_prediction_is_treated_as_nothing_to_report_not_an_error(self):
        saved_search_service.add_saved_search(
            self.user_id, kind="area", area_id="yellowstone", label="Bears", min_probability=0.1,
        )
        with patch("app.ml.prediction_service.predict", side_effect=RuntimeError("model not trained")):
            created = saved_search_service.check_due_saved_searches()  # must not raise
        self.assertEqual(created, 0)

    def test_recheck_cooldown_skips_a_recently_checked_search(self):
        search = saved_search_service.add_saved_search(
            self.user_id, kind="area", area_id="yellowstone", label="Bears", min_probability=0.5,
        )
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE saved_searches SET last_checked_at = ? WHERE id = ?", (db.now_iso(), search["id"])
            )
        with patch("app.ml.prediction_service.predict", side_effect=_fake_predict_area) as mocked:
            created = saved_search_service.check_due_saved_searches()
        self.assertEqual(created, 0)
        mocked.assert_not_called()

    def test_notify_cooldown_prevents_re_notifying_for_a_still_true_condition(self):
        search = saved_search_service.add_saved_search(
            self.user_id, kind="area", area_id="yellowstone", label="Bears", min_probability=0.5,
        )
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE saved_searches SET last_notified_at = ? WHERE id = ?", (db.now_iso(), search["id"])
            )
        # last_checked_at is untouched (still NULL), so the recheck cooldown
        # doesn't block this -- only the *notify* cooldown should.
        with patch("app.ml.prediction_service.predict", side_effect=_fake_predict_area):
            created = saved_search_service.check_due_saved_searches()
        self.assertEqual(created, 0)
        self.assertEqual(notification_service.list_notifications(self.user_id), [])

    def test_location_search_uses_predict_at_location(self):
        saved_search_service.add_saved_search(
            self.user_id, kind="location", lat=44.65, lon=-110.45, radius_km=150, label="My spot",
            min_probability=0.5,
        )
        fake_result = {"predictions": [{"scientific_name": "Ursus arctos", "common_name": "Grizzly Bear", "probability": 0.9}]}
        with patch("app.ml.prediction_service.predict_at_location", return_value=fake_result) as mocked:
            created = saved_search_service.check_due_saved_searches()
        self.assertEqual(created, 1)
        mocked.assert_called_once()

    def test_one_failing_search_does_not_block_another_users_search(self):
        other, _ = auth_service.register("alerts-b@example.com", "correcthorse123")
        saved_search_service.add_saved_search(
            self.user_id, kind="area", area_id="broken-area", label="Broken", min_probability=0.1,
        )
        saved_search_service.add_saved_search(
            other["id"], kind="area", area_id="yellowstone", label="Bears", min_probability=0.5,
        )

        def fake_predict(area_id, _date, species_key=None):
            if area_id == "broken-area":
                raise KeyError("no such area")
            return _fake_predict_area(area_id, _date, species_key)

        with patch("app.ml.prediction_service.predict", side_effect=fake_predict):
            created = saved_search_service.check_due_saved_searches()
        self.assertEqual(created, 1)
        self.assertEqual(notification_service.list_notifications(other["id"])[0]["message"].count("Grizzly"), 1)


if __name__ == "__main__":
    unittest.main()

import unittest

from app.services import auth_service, favorites_service
from tests._db_test_utils import TempDbTestCase


class FavoritesServiceTests(TempDbTestCase):
    def setUp(self):
        super().setUp()
        user_a, _ = auth_service.register("fav-a@example.com", "correcthorse123")
        user_b, _ = auth_service.register("fav-b@example.com", "correcthorse123")
        self.user_a_id = user_a["id"]
        self.user_b_id = user_b["id"]

    def test_add_an_area_favorite(self):
        fav = favorites_service.add_favorite(self.user_a_id, kind="area", area_id="yellowstone", label="Yellowstone")
        self.assertEqual(fav["kind"], "area")
        self.assertEqual(fav["area_id"], "yellowstone")
        self.assertIsNone(fav["lat"])

    def test_add_a_location_favorite(self):
        fav = favorites_service.add_favorite(
            self.user_a_id, kind="location", lat=44.65, lon=-110.45, radius_km=150, label="My spot",
            species_key="Ursus arctos",
        )
        self.assertEqual(fav["kind"], "location")
        self.assertAlmostEqual(fav["lat"], 44.65)
        self.assertEqual(fav["species_key"], "Ursus arctos")

    def test_area_favorite_without_area_id_is_rejected(self):
        with self.assertRaises(favorites_service.FavoritesError):
            favorites_service.add_favorite(self.user_a_id, kind="area", label="Nowhere")

    def test_location_favorite_without_coordinates_is_rejected(self):
        with self.assertRaises(favorites_service.FavoritesError):
            favorites_service.add_favorite(self.user_a_id, kind="location", label="Nowhere")

    def test_out_of_range_coordinates_are_rejected(self):
        with self.assertRaises(favorites_service.FavoritesError):
            favorites_service.add_favorite(self.user_a_id, kind="location", lat=999, lon=0, label="Bad")

    def test_invalid_kind_is_rejected(self):
        with self.assertRaises(favorites_service.FavoritesError):
            favorites_service.add_favorite(self.user_a_id, kind="planet", label="Nowhere")

    def test_list_returns_only_that_users_favorites(self):
        favorites_service.add_favorite(self.user_a_id, kind="area", area_id="yellowstone", label="A's")
        favorites_service.add_favorite(self.user_b_id, kind="area", area_id="kruger", label="B's")
        a_favorites = favorites_service.list_favorites(self.user_a_id)
        self.assertEqual(len(a_favorites), 1)
        self.assertEqual(a_favorites[0]["area_id"], "yellowstone")

    def test_remove_a_favorite(self):
        fav = favorites_service.add_favorite(self.user_a_id, kind="area", area_id="yellowstone", label="A's")
        favorites_service.remove_favorite(self.user_a_id, fav["id"])
        self.assertEqual(favorites_service.list_favorites(self.user_a_id), [])

    def test_removing_someone_elses_favorite_fails(self):
        # user_b must not be able to delete user_a's favorite just by
        # guessing/knowing its id -- ownership is enforced in the DELETE's
        # WHERE clause (user_id = ?), not just at the API layer.
        fav = favorites_service.add_favorite(self.user_a_id, kind="area", area_id="yellowstone", label="A's")
        with self.assertRaises(favorites_service.FavoritesError):
            favorites_service.remove_favorite(self.user_b_id, fav["id"])
        self.assertEqual(len(favorites_service.list_favorites(self.user_a_id)), 1)  # still there

    def test_removing_a_nonexistent_favorite_fails(self):
        with self.assertRaises(favorites_service.FavoritesError):
            favorites_service.remove_favorite(self.user_a_id, 999999)


if __name__ == "__main__":
    unittest.main()

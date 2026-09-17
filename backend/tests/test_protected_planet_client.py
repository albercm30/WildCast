"""
Unit tests for app.services.protected_planet_client using mocked HTTP
responses -- same pattern as test_iucn_client.py/test_ebird_client.py, and
for the same reason: this sandbox's egress can't reach
api.protectedplanet.net to verify a real response (see that module's
docstring). These pin the CODE's behavior against the best-read of the
live v4 docs, and should be re-checked against a real response once a real
PROTECTED_PLANET_API_KEY is available.
"""
import unittest
from unittest.mock import MagicMock, patch

from app.services import protected_planet_client as ppc


class IsConfiguredTests(unittest.TestCase):
    def test_false_when_no_key(self):
        with patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", ""):
            self.assertFalse(ppc.is_configured())

    def test_true_when_key_set(self):
        with patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", "fake-key"):
            self.assertTrue(ppc.is_configured())


class SearchProtectedAreasTests(unittest.TestCase):
    @patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", "")
    def test_raises_when_not_configured(self):
        with self.assertRaises(ppc.ProtectedPlanetUnavailable):
            ppc.search_protected_areas(country_iso3="USA")

    @patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", "fake-key")
    @patch("app.services.protected_planet_client.requests.get")
    def test_sends_token_and_filters(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"protected_areas": []})
        mock_get.return_value.raise_for_status = lambda: None
        ppc.search_protected_areas(country_iso3="usa", marine=False, page=2, per_page=10)
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["token"], "fake-key")
        self.assertEqual(params["country"], "USA")  # uppercased
        self.assertEqual(params["marine"], "false")
        self.assertEqual(params["page"], 2)
        self.assertEqual(params["per_page"], 10)

    @patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", "fake-key")
    @patch("app.services.protected_planet_client.requests.get")
    def test_per_page_clamped_to_api_max(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"protected_areas": []})
        mock_get.return_value.raise_for_status = lambda: None
        ppc.search_protected_areas(per_page=500)
        self.assertEqual(mock_get.call_args[1]["params"]["per_page"], 50)

    @patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", "fake-key")
    @patch("app.services.protected_planet_client.requests.get")
    def test_parses_and_normalizes_results(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "protected_areas": [
                    {
                        "site_id": "555",
                        "name": "Yellowstone National Park",
                        "designation": {"name": "National Park"},
                        "iucn_category": {"name": "II"},
                        "marine": False,
                        "countries": [{"iso_3": "USA", "name": "United States"}],
                        "geojson": {
                            "type": "Polygon",
                            "coordinates": [[[-110.0, 44.0], [-110.0, 45.0], [-109.0, 45.0], [-109.0, 44.0]]],
                        },
                    }
                ]
            },
        )
        mock_get.return_value.raise_for_status = lambda: None
        result = ppc.search_protected_areas(country_iso3="USA")
        self.assertEqual(len(result["protected_areas"]), 1)
        area = result["protected_areas"][0]
        self.assertEqual(area["site_id"], "555")
        self.assertEqual(area["name"], "Yellowstone National Park")
        self.assertEqual(area["designation"], "National Park")
        self.assertEqual(area["iucn_category"], "II")
        self.assertFalse(area["marine"])
        self.assertEqual(area["countries"], [{"iso3": "USA", "name": "United States"}])
        self.assertIsNotNone(area["centroid"])
        # Average of the 4 corners above.
        self.assertAlmostEqual(area["centroid"]["lon"], -109.5)
        self.assertAlmostEqual(area["centroid"]["lat"], 44.5)

    @patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", "fake-key")
    @patch("app.services.protected_planet_client.requests.get")
    def test_has_more_true_when_full_page_returned(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"protected_areas": [{"site_id": str(i)} for i in range(25)]},
        )
        mock_get.return_value.raise_for_status = lambda: None
        result = ppc.search_protected_areas(per_page=25)
        self.assertTrue(result["has_more"])

    @patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", "fake-key")
    @patch("app.services.protected_planet_client.requests.get")
    def test_has_more_false_when_partial_page_returned(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"protected_areas": [{"site_id": "1"}]},
        )
        mock_get.return_value.raise_for_status = lambda: None
        result = ppc.search_protected_areas(per_page=25)
        self.assertFalse(result["has_more"])


class GetProtectedAreaTests(unittest.TestCase):
    @patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", "")
    def test_raises_when_not_configured(self):
        with self.assertRaises(ppc.ProtectedPlanetUnavailable):
            ppc.get_protected_area("555")

    @patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", "fake-key")
    @patch("app.services.protected_planet_client.requests.get")
    def test_returns_none_on_404(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        result = ppc.get_protected_area("does-not-exist")
        self.assertIsNone(result)

    @patch("app.services.protected_planet_client.PROTECTED_PLANET_API_KEY", "fake-key")
    @patch("app.services.protected_planet_client.requests.get")
    def test_returns_parsed_area(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"protected_area": {"site_id": "555", "name": "Kruger National Park"}},
        )
        mock_get.return_value.raise_for_status = lambda: None
        result = ppc.get_protected_area("555")
        self.assertEqual(result["name"], "Kruger National Park")


class CentroidFromGeojsonTests(unittest.TestCase):
    def test_none_input_returns_none(self):
        self.assertIsNone(ppc._centroid_from_geojson(None))

    def test_multipolygon_averages_all_vertices(self):
        geojson = {
            "type": "MultiPolygon",
            "coordinates": [
                [[[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]]],
                [[[10.0, 10.0], [10.0, 12.0], [12.0, 12.0], [12.0, 10.0]]],
            ],
        }
        centroid = ppc._centroid_from_geojson(geojson)
        self.assertIsNotNone(centroid)
        # 8 points total, 4 near (1,1) and 4 near (11,11) -> average (6, 6).
        self.assertAlmostEqual(centroid["lon"], 6.0)
        self.assertAlmostEqual(centroid["lat"], 6.0)

    def test_feature_wrapper_is_tolerated(self):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]]],
            },
        }
        centroid = ppc._centroid_from_geojson(feature)
        self.assertAlmostEqual(centroid["lon"], 0.0)
        self.assertAlmostEqual(centroid["lat"], 0.0)


if __name__ == "__main__":
    unittest.main()

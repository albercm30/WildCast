"""
Tests for the /api/parks endpoints (app.routers.parks), via Flask's own
test client -- same "no running server, no network" pattern as test_api.py.
Protected Planet itself is always mocked (see test_protected_planet_client.py
for the client-level tests); these exercise the router's query-param
handling, the "not configured" 503, and the countries list.
"""
import unittest
from unittest.mock import patch

from app.main import create_app


class ParksApiTestCase(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()


class CountriesEndpointTests(ParksApiTestCase):
    def test_works_without_a_protected_planet_key(self):
        # The country list is a static local file, not a Protected Planet
        # API call -- must work even with no key configured at all.
        with patch("app.routers.parks.ppc.PROTECTED_PLANET_API_KEY", ""):
            resp = self.client.get("/api/parks/countries")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertGreater(len(body), 100)
        self.assertIn({"iso3": "USA", "name": "United States"}, body)


class SearchParksEndpointTests(ParksApiTestCase):
    def test_503_when_not_configured(self):
        with patch("app.routers.parks.ppc.is_configured", return_value=False):
            resp = self.client.get("/api/parks?country=USA")
        self.assertEqual(resp.status_code, 503)
        self.assertIn("error", resp.get_json())

    def test_bad_page_is_400(self):
        with patch("app.routers.parks.ppc.is_configured", return_value=True):
            resp = self.client.get("/api/parks?page=not-a-number")
        self.assertEqual(resp.status_code, 400)

    def test_bad_per_page_is_400(self):
        with patch("app.routers.parks.ppc.is_configured", return_value=True):
            resp = self.client.get("/api/parks?per_page=not-a-number")
        self.assertEqual(resp.status_code, 400)

    def test_valid_request_passes_filters_through(self):
        fake_result = {"protected_areas": [{"site_id": "1", "name": "Test Park"}], "page": 1, "per_page": 25, "has_more": False}
        with (
            patch("app.routers.parks.ppc.is_configured", return_value=True),
            patch("app.routers.parks.ppc.search_protected_areas", return_value=fake_result) as mock_search,
        ):
            resp = self.client.get("/api/parks?country=usa&marine=true&page=1&per_page=25")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), fake_result)
        _, kwargs = mock_search.call_args
        self.assertEqual(kwargs["country_iso3"], "usa")
        self.assertTrue(kwargs["marine"])
        self.assertEqual(kwargs["page"], 1)
        self.assertEqual(kwargs["per_page"], 25)

    def test_marine_param_absent_when_not_given(self):
        fake_result = {"protected_areas": [], "page": 1, "per_page": 25, "has_more": False}
        with (
            patch("app.routers.parks.ppc.is_configured", return_value=True),
            patch("app.routers.parks.ppc.search_protected_areas", return_value=fake_result) as mock_search,
        ):
            self.client.get("/api/parks")
        self.assertIsNone(mock_search.call_args[1]["marine"])


class GetParkEndpointTests(ParksApiTestCase):
    def test_503_when_not_configured(self):
        with patch("app.routers.parks.ppc.is_configured", return_value=False):
            resp = self.client.get("/api/parks/555")
        self.assertEqual(resp.status_code, 503)

    def test_404_when_not_found(self):
        with (
            patch("app.routers.parks.ppc.is_configured", return_value=True),
            patch("app.routers.parks.ppc.get_protected_area", return_value=None),
        ):
            resp = self.client.get("/api/parks/does-not-exist")
        self.assertEqual(resp.status_code, 404)

    def test_200_when_found(self):
        fake_area = {"site_id": "555", "name": "Kruger National Park"}
        with (
            patch("app.routers.parks.ppc.is_configured", return_value=True),
            patch("app.routers.parks.ppc.get_protected_area", return_value=fake_area),
        ):
            resp = self.client.get("/api/parks/555")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), fake_area)


if __name__ == "__main__":
    unittest.main()

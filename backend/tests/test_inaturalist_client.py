"""
Unit tests for app.services.inaturalist_client using mocked HTTP responses --
same pattern as test_gbif_client.py: verify the CODE (URL/param construction,
response parsing), not that iNaturalist's live API still matches this exact
shape (this sandbox's egress can't reach api.inaturalist.org to check --
see the module docstring).
"""
import unittest
from unittest.mock import MagicMock, patch

from app.services import inaturalist_client


class SearchObservationsTests(unittest.TestCase):
    @patch("app.services.inaturalist_client._SESSION.get")
    def test_sends_expected_params_and_wild_only_filter(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"total_results": 0, "results": []})
        inaturalist_client.search_observations(scientific_name="Ursus arctos", lat=44.6, lon=-110.5, radius_km=60)
        called_url, called_kwargs = mock_get.call_args[0][0], mock_get.call_args[1]
        self.assertIn("/observations", called_url)
        params = called_kwargs["params"]
        self.assertEqual(params["taxon_name"], "Ursus arctos")
        self.assertEqual(params["lat"], 44.6)
        self.assertEqual(params["lng"], -110.5)
        self.assertEqual(params["radius"], 60)
        self.assertEqual(params["captive"], "false")
        self.assertEqual(params["quality_grade"], "research")

    @patch("app.services.inaturalist_client._SESSION.get")
    def test_wild_only_false_omits_captive_filter(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"total_results": 0, "results": []})
        inaturalist_client.search_observations(
            scientific_name="Ursus arctos", lat=44.6, lon=-110.5, radius_km=60, wild_only=False,
        )
        params = mock_get.call_args[1]["params"]
        self.assertNotIn("captive", params)


class PresenceCountTests(unittest.TestCase):
    @patch("app.services.inaturalist_client._SESSION.get")
    def test_returns_total_results_as_int(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"total_results": 42, "results": []})
        count = inaturalist_client.presence_count("Ursus arctos", 44.6, -110.5, 60)
        self.assertEqual(count, 42)

    @patch("app.services.inaturalist_client._SESSION.get")
    def test_missing_total_results_defaults_to_zero(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"results": []})
        count = inaturalist_client.presence_count("Ursus arctos", 44.6, -110.5, 60)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()

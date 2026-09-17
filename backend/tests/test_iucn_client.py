"""
Unit tests for app.services.iucn_client using mocked HTTP responses -- same
pattern as the other *_client tests. This sandbox's egress can't reach
api.iucnredlist.org to verify the real response shape (see the module
docstring); these tests pin the CODE's behavior against the best-guess
shape documented there, and must be re-checked against a live response
once a real IUCN_API_KEY is available.
"""
import unittest
from unittest.mock import MagicMock, patch

from app.services import iucn_client


class IsConfiguredTests(unittest.TestCase):
    def test_false_when_no_key(self):
        with patch("app.services.iucn_client.IUCN_API_KEY", None):
            self.assertFalse(iucn_client.is_configured())

    def test_true_when_key_set(self):
        with patch("app.services.iucn_client.IUCN_API_KEY", "fake-key"):
            self.assertTrue(iucn_client.is_configured())


class SpeciesAssessmentTests(unittest.TestCase):
    @patch("app.services.iucn_client.IUCN_API_KEY", None)
    def test_raises_when_not_configured(self):
        with self.assertRaises(iucn_client.IUCNUnavailable):
            iucn_client.species_assessment("Ursus arctos")

    @patch("app.services.iucn_client.IUCN_API_KEY", "fake-key")
    @patch("app.services.iucn_client.requests.get")
    def test_splits_scientific_name_into_genus_and_species_params(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"assessments": []})
        mock_get.return_value.raise_for_status = lambda: None
        iucn_client.species_assessment("Ursus arctos")
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["genus_name"], "Ursus")
        self.assertEqual(params["species_name"], "arctos")

    @patch("app.services.iucn_client.IUCN_API_KEY", "fake-key")
    @patch("app.services.iucn_client.requests.get")
    def test_returns_none_on_404(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        result = iucn_client.species_assessment("Nonexistens fakeus")
        self.assertIsNone(result)

    @patch("app.services.iucn_client.IUCN_API_KEY", "fake-key")
    @patch("app.services.iucn_client.requests.get")
    def test_returns_none_when_no_assessments(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"assessments": []})
        mock_get.return_value.raise_for_status = lambda: None
        result = iucn_client.species_assessment("Ursus arctos")
        self.assertIsNone(result)

    @patch("app.services.iucn_client.IUCN_API_KEY", "fake-key")
    @patch("app.services.iucn_client.requests.get")
    def test_picks_the_latest_published_assessment(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "assessments": [
                    {"year_published": 2010, "red_list_category": {"code": "VU"}, "population_trend": "Decreasing"},
                    {"year_published": 2022, "red_list_category": {"code": "EN"}, "population_trend": "Decreasing"},
                    {"year_published": 2016, "red_list_category": {"code": "VU"}, "population_trend": "Stable"},
                ]
            },
        )
        mock_get.return_value.raise_for_status = lambda: None
        result = iucn_client.species_assessment("Ursus arctos")
        self.assertEqual(result["category"], "EN")
        self.assertEqual(result["year_published"], 2022)
        self.assertEqual(result["population_trend"], "Decreasing")


if __name__ == "__main__":
    unittest.main()

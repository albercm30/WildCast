"""
Unit tests for app.services.ebird_client -- focused on `presence_count`
(WildCast's actual per-species usage of this client; see
app.ml.prediction_service._local_evidence_cached) and the is_configured/
_headers key-gating every eBird call goes through. Uses mocked HTTP
responses, same pattern as test_gbif_client.py / test_inaturalist_client.py.
"""
import unittest
from unittest.mock import MagicMock, patch

from app.services import ebird_client


class IsConfiguredTests(unittest.TestCase):
    def test_false_when_no_key(self):
        with patch("app.services.ebird_client.EBIRD_API_KEY", None):
            self.assertFalse(ebird_client.is_configured())

    def test_true_when_key_set(self):
        with patch("app.services.ebird_client.EBIRD_API_KEY", "fake-key"):
            self.assertTrue(ebird_client.is_configured())


class PresenceCountTests(unittest.TestCase):
    @patch("app.services.ebird_client.EBIRD_API_KEY", "fake-key")
    @patch("app.services.ebird_client.requests.get")
    def test_counts_only_matching_scientific_name(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: [
                {"sciName": "Haliaeetus leucocephalus", "comName": "Bald Eagle"},
                {"sciName": "Corvus corax", "comName": "Common Raven"},
                {"sciName": "Haliaeetus leucocephalus", "comName": "Bald Eagle"},
            ],
        )
        mock_get.return_value.raise_for_status = lambda: None
        count = ebird_client.presence_count("Haliaeetus leucocephalus", 44.6, -110.5, 60)
        self.assertEqual(count, 2)

    @patch("app.services.ebird_client.EBIRD_API_KEY", "fake-key")
    @patch("app.services.ebird_client.requests.get")
    def test_caps_distance_at_ebirds_50km_max(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        mock_get.return_value.raise_for_status = lambda: None
        ebird_client.presence_count("Corvus corax", 44.6, -110.5, 150)
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["dist"], 50)

    @patch("app.services.ebird_client.EBIRD_API_KEY", None)
    def test_raises_when_not_configured(self):
        with self.assertRaises(ebird_client.EBirdUnavailable):
            ebird_client.presence_count("Corvus corax", 44.6, -110.5, 60)


if __name__ == "__main__":
    unittest.main()

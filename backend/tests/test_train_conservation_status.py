"""
Unit tests for app.ml.train._compute_conservation_status -- specifically
that it prefers a committed data/cache/conservation_status.json (the real
production path, written by scripts/ingest_iucn.py) over a live IUCN call,
and only falls back to a live call as a local-dev convenience. This is a
regression test for a real bug: this function used to call iucn_client
directly and unconditionally, which meant conservation status could never
actually reach a model trained inside docker/Dockerfile.backend's build
step (no secrets available there) no matter how IUCN_API_KEY was
configured as a repo secret. See scripts/ingest_iucn.py's docstring.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ml import train


class ComputeConservationStatusTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cache_path = Path(self._tmpdir.name) / "conservation_status.json"
        self._patcher = patch("app.ml.train._CONSERVATION_STATUS_CACHE_PATH", self.cache_path)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_uses_committed_cache_without_touching_iucn_client(self):
        self.cache_path.write_text(json.dumps({"Ursus arctos": {"category": "EN", "category_label": "Endangered"}}))
        with patch("app.ml.train.iucn_client.species_assessment") as mock_assess:
            result = train._compute_conservation_status()
        mock_assess.assert_not_called()
        self.assertEqual(result["Ursus arctos"]["category"], "EN")

    def test_falls_back_to_live_call_when_no_cache_and_key_configured(self):
        with (
            patch("app.ml.train.iucn_client.is_configured", return_value=True),
            patch("app.ml.train.iucn_client.species_assessment", return_value={"category": "VU", "population_trend": "Stable"}),
        ):
            result = train._compute_conservation_status()
        self.assertGreater(len(result), 0)
        self.assertTrue(all(v["category"] == "VU" for v in result.values()))

    def test_returns_empty_when_no_cache_and_no_key(self):
        with patch("app.ml.train.iucn_client.is_configured", return_value=False):
            result = train._compute_conservation_status()
        self.assertEqual(result, {})

    def test_corrupt_cache_falls_back_instead_of_crashing(self):
        self.cache_path.write_text("{not valid json")
        with patch("app.ml.train.iucn_client.is_configured", return_value=False):
            result = train._compute_conservation_status()
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()

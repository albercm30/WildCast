"""
Unit tests for scripts.ingest_iucn -- the real production path for IUCN
conservation-status enrichment (see that script's docstring for why this
exists as a separate script rather than a live call inside app.ml.train:
the Docker build step that actually trains the deployed model has no
access to secrets, so this must run where IUCN_API_KEY genuinely is
available -- .github/workflows/retrain.yml -- and commit its result as a
cache file app.ml.train can just read).
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import iucn_client
from scripts import ingest_iucn


class IngestIucnTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cache_path = Path(self._tmpdir.name) / "conservation_status.json"
        self._patcher = patch("scripts.ingest_iucn.CONSERVATION_STATUS_PATH", self.cache_path)
        self._patcher.start()
        # Keep the species list small and deterministic for these tests,
        # independent of however many are in the real seed_species.json.
        self._species_patcher = patch(
            "scripts.ingest_iucn.SEED_SPECIES",
            [{"scientific_name": "Ursus arctos"}, {"scientific_name": "Canis lupus"}],
        )
        self._species_patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._species_patcher.stop()
        self._tmpdir.cleanup()

    @patch("scripts.ingest_iucn.iucn_client.is_configured", return_value=False)
    def test_skips_and_writes_nothing_when_not_configured(self, _mock_configured):
        ingest_iucn.main()
        self.assertFalse(self.cache_path.exists())

    @patch("scripts.ingest_iucn.iucn_client.is_configured", return_value=True)
    @patch("scripts.ingest_iucn.iucn_client.species_assessment")
    def test_writes_cache_with_category_labels_when_configured(self, mock_assess, _mock_configured):
        mock_assess.side_effect = lambda name: (
            {"category": "EN", "population_trend": "Decreasing", "year_published": 2022}
            if name == "Ursus arctos"
            else None
        )
        ingest_iucn.main()
        self.assertTrue(self.cache_path.exists())
        result = json.loads(self.cache_path.read_text())
        self.assertEqual(set(result.keys()), {"Ursus arctos"})
        self.assertEqual(result["Ursus arctos"]["category"], "EN")
        self.assertEqual(result["Ursus arctos"]["category_label"], "Endangered")

    @patch("scripts.ingest_iucn.iucn_client.is_configured", return_value=True)
    @patch("scripts.ingest_iucn.iucn_client.species_assessment", side_effect=RuntimeError("outage"))
    def test_a_bad_run_never_overwrites_an_existing_good_cache(self, _mock_assess, _mock_configured):
        self.cache_path.write_text(json.dumps({"Ursus arctos": {"category": "VU", "category_label": "Vulnerable"}}))
        ingest_iucn.main()
        # The transient failure above must not have wiped the previously
        # committed real data -- see the script's docstring on this exact
        # failure mode.
        result = json.loads(self.cache_path.read_text())
        self.assertEqual(result["Ursus arctos"]["category"], "VU")

    @patch("scripts.ingest_iucn.iucn_client.is_configured", return_value=True)
    @patch("scripts.ingest_iucn.iucn_client.species_assessment", return_value=None)
    def test_nothing_written_when_no_cache_exists_and_nothing_found(self, _mock_assess, _mock_configured):
        ingest_iucn.main()
        self.assertFalse(self.cache_path.exists())


class CategoryLabelsTests(unittest.TestCase):
    def test_every_standard_category_has_a_label(self):
        for code in ("EX", "EW", "CR", "EN", "VU", "NT", "LC", "DD", "NE"):
            self.assertIn(code, iucn_client.CATEGORY_LABELS)


if __name__ == "__main__":
    unittest.main()

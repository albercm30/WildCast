"""
Unit tests for app.services.data_source_marker, using a temp file (via
tempfile) swapped in for DATA_SOURCE_MARKER_PATH so these never touch the
repo's real cache/data_source.json.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import data_source_marker


class DataSourceMarkerTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.marker_path = Path(self._tmpdir.name) / "nested" / "data_source.json"
        self._patcher = patch("app.services.data_source_marker.DATA_SOURCE_MARKER_PATH", self.marker_path)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_read_with_no_marker_file_returns_unknown(self):
        self.assertEqual(data_source_marker.read(), {"source": "unknown"})

    def test_mark_real_then_read_round_trips(self):
        data_source_marker.mark("real", {"ingested_by": "scripts.ingest_gbif"})
        result = data_source_marker.read()
        self.assertEqual(result["source"], "real")
        self.assertEqual(result["ingested_by"], "scripts.ingest_gbif")
        self.assertIn("written_at", result)

    def test_mark_synthetic_demo_then_read_round_trips(self):
        data_source_marker.mark("synthetic_demo")
        result = data_source_marker.read()
        self.assertEqual(result["source"], "synthetic_demo")

    def test_mark_creates_parent_directories(self):
        self.assertFalse(self.marker_path.parent.exists())
        data_source_marker.mark("real")
        self.assertTrue(self.marker_path.exists())

    def test_mark_rejects_unknown_source(self):
        with self.assertRaises(ValueError):
            data_source_marker.mark("fabricated")

    def test_a_later_mark_overwrites_an_earlier_one(self):
        data_source_marker.mark("synthetic_demo")
        data_source_marker.mark("real")
        self.assertEqual(data_source_marker.read()["source"], "real")

    def test_read_on_corrupt_json_returns_unknown_not_a_crash(self):
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text("{not valid json")
        self.assertEqual(data_source_marker.read(), {"source": "unknown"})

    def test_read_on_unrecognized_source_value_returns_unknown(self):
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text(json.dumps({"source": "made_up"}))
        self.assertEqual(data_source_marker.read(), {"source": "unknown"})


if __name__ == "__main__":
    unittest.main()

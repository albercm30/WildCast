"""
Unit tests for scripts.check_real_data_complete -- the check
docker/Dockerfile.backend's build step uses to decide whether the
committed data/cache/ is trustworthy real data or needs the synthetic
fixtures fallback. Regression coverage for a real failure: the old check
only tested for one hardcoded file (occurrences_yellowstone.csv), which
kept "passing" even after 5 new pilot areas were added with no real data
cache for them yet, so the Docker build skipped the fallback and crashed
inside app.ml.train instead of generating a consistent synthetic dataset.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_real_data_complete as check


_TWO_AREAS = [{"id": "yellowstone"}, {"id": "serengeti"}]


def _write_area_cache(cache_dir: Path, area_id: str) -> None:
    for prefix in ("occurrences", "weather_daily", "climate_normals"):
        (cache_dir / f"{prefix}_{area_id}.csv").write_text("placeholder\n")


class CheckRealDataCompleteTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmpdir.name)
        self._cache_patcher = patch("scripts.check_real_data_complete.CACHE_DIR", self.cache_dir)
        self._areas_patcher = patch("scripts.check_real_data_complete.PILOT_AREAS", _TWO_AREAS)
        self._cache_patcher.start()
        self._areas_patcher.start()

    def tearDown(self):
        self._cache_patcher.stop()
        self._areas_patcher.stop()
        self._tmpdir.cleanup()

    def test_missing_areas_empty_when_every_area_fully_cached(self):
        _write_area_cache(self.cache_dir, "yellowstone")
        _write_area_cache(self.cache_dir, "serengeti")
        self.assertEqual(check.missing_areas(), [])
        self.assertEqual(check.main(), 0)

    def test_flags_an_area_with_no_cache_at_all(self):
        # Regression case: this is exactly the real-world scenario found in
        # production -- one area (here, the stand-in for a newly added one
        # like Serengeti) has never been ingested at all.
        _write_area_cache(self.cache_dir, "yellowstone")
        self.assertEqual(check.missing_areas(), ["serengeti"])
        self.assertEqual(check.main(), 1)

    def test_flags_an_area_with_a_partial_cache(self):
        _write_area_cache(self.cache_dir, "yellowstone")
        _write_area_cache(self.cache_dir, "serengeti")
        (self.cache_dir / "weather_daily_serengeti.csv").unlink()  # one of the three files goes missing
        self.assertEqual(check.missing_areas(), ["serengeti"])

    def test_empty_cache_dir_flags_every_area(self):
        self.assertEqual(sorted(check.missing_areas()), ["serengeti", "yellowstone"])
        self.assertEqual(check.main(), 1)


if __name__ == "__main__":
    unittest.main()

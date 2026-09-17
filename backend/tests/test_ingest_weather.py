"""
Unit tests for scripts.ingest_weather.main()'s date-parsing defense. This is
the second, defensive layer against a real 2026-09-17 crash: a bare-year
GBIF eventDate ("2008") made it into occurrences_scottish_highlands.csv (the
real fix is scripts.ingest_gbif._record_date -- see test_ingest_gbif.py),
and pd.to_datetime(occ["date"]) with no error handling took down the whole
ingestion run on that one bad row. main() now coerces unparseable dates to
NaT, drops them, and only skips the area entirely if nothing is left --
so a stale or unexpected bad date can't crash the run, before or after the
real fix lands.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import ingest_weather


_ONE_AREA = [{"id": "testarea", "name": "Test Area", "lat": 1.0, "lon": 2.0}]


def _daily_frame(n=1):
    return pd.DataFrame({
        "date": ["2026-06-01"] * n,
        "temperature_2m_max": [25.0] * n,
        "temperature_2m_min": [15.0] * n,
        "precipitation_sum": [0.0] * n,
        "windspeed_10m_max": [10.0] * n,
        "cloudcover_mean": [30.0] * n,
    })


class IngestWeatherDateParsingTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmpdir.name)
        self._cache_patcher = patch("scripts.ingest_weather.CACHE_DIR", self.cache_dir)
        self._areas_patcher = patch("scripts.ingest_weather.PILOT_AREAS", _ONE_AREA)
        self._sleep_patcher = patch("scripts.ingest_weather.time.sleep")
        self._marker_patcher = patch("scripts.ingest_weather.data_source_marker.mark")
        self._cache_patcher.start()
        self._areas_patcher.start()
        self._sleep_patcher.start()
        self._marker_patcher.start()

    def tearDown(self):
        self._cache_patcher.stop()
        self._areas_patcher.stop()
        self._sleep_patcher.stop()
        self._marker_patcher.stop()
        self._tmpdir.cleanup()

    def _write_occurrences(self, dates):
        pd.DataFrame({
            "species": ["Ursus arctos"] * len(dates),
            "common_name": ["Grizzly bear"] * len(dates),
            "taxon_class": ["Mammalia"] * len(dates),
            "area_id": ["testarea"] * len(dates),
            "lat": [1.0] * len(dates),
            "lon": [2.0] * len(dates),
            "date": dates,
        }).to_csv(self.cache_dir / "occurrences_testarea.csv", index=False)

    @patch("scripts.ingest_weather.weather_client.climate_normals")
    @patch("scripts.ingest_weather.weather_client.historical_daily")
    def test_a_mix_of_good_and_bad_dates_skips_only_the_bad_ones(self, mock_hist, mock_normals):
        # Regression case: one bad row ("2008", the exact real value) mixed
        # in with otherwise-good dates must not crash the whole area.
        self._write_occurrences(["2021-01-01", "2008", "2021-06-15"])
        mock_hist.return_value = _daily_frame()
        mock_normals.return_value = pd.DataFrame({"day_of_year": [1]})

        ingest_weather.main()  # must not raise

        self.assertTrue((self.cache_dir / "weather_daily_testarea.csv").exists())
        # The good dates (2021-01-01 and 2021-06-15) should have driven the
        # fetched range, with the unparseable "2008" row simply dropped.
        called_start, called_end = mock_hist.call_args[0][2], mock_hist.call_args[0][3]
        self.assertEqual(called_start, "2021-01-01")
        self.assertEqual(called_end, "2021-06-15")

    @patch("scripts.ingest_weather.weather_client.climate_normals")
    @patch("scripts.ingest_weather.weather_client.historical_daily")
    def test_area_is_skipped_cleanly_when_every_date_is_unparseable(self, mock_hist, mock_normals):
        self._write_occurrences(["2008", "not a date", "1999"])

        ingest_weather.main()  # must not raise

        mock_hist.assert_not_called()
        self.assertFalse((self.cache_dir / "weather_daily_testarea.csv").exists())

    @patch("scripts.ingest_weather.weather_client.climate_normals")
    @patch("scripts.ingest_weather.weather_client.historical_daily")
    def test_all_good_dates_behaves_exactly_as_before(self, mock_hist, mock_normals):
        self._write_occurrences(["2021-01-01", "2021-06-15"])
        mock_hist.return_value = _daily_frame()
        mock_normals.return_value = pd.DataFrame({"day_of_year": [1]})

        ingest_weather.main()

        called_start, called_end = mock_hist.call_args[0][2], mock_hist.call_args[0][3]
        self.assertEqual(called_start, "2021-01-01")
        self.assertEqual(called_end, "2021-06-15")


if __name__ == "__main__":
    unittest.main()

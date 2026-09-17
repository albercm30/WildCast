"""
Unit tests for app.services.weather_client using mocked HTTP responses (see
test_gbif_client.py's docstring for why: no live network in this
environment). climate_normals() is memoized with lru_cache, so tests clear
that cache first to avoid one test's mock leaking into another's result.
"""
import datetime as dt
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from app.services import weather_client


def _archive_response(dates, temp_max, temp_min=None, precip=None):
    temp_min = temp_min or [t - 10 for t in temp_max]
    precip = precip or [0.0] * len(dates)
    return {
        "daily": {
            "time": dates,
            "temperature_2m_max": temp_max,
            "temperature_2m_min": temp_min,
            "precipitation_sum": precip,
            "windspeed_10m_max": [10.0] * len(dates),
            "cloudcover_mean": [30.0] * len(dates),
        }
    }


class HistoricalDailyTests(unittest.TestCase):
    @patch("app.services.weather_client._SESSION.get")
    def test_parses_daily_arrays_into_a_dataframe(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: _archive_response(["2026-06-01", "2026-06-02"], [25.0, 27.0]),
        )
        df = weather_client.historical_daily(44.6, -110.5, "2026-06-01", "2026-06-02")
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["temperature_2m_max"]), [25.0, 27.0])

    @patch("app.services.weather_client._SESSION.get")
    def test_rounds_coordinates_for_a_stable_cache_key(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: _archive_response(["2026-06-01"], [20.0]))
        weather_client.historical_daily(44.60001234, -110.50009876, "2026-06-01", "2026-06-01")
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["latitude"], 44.6)
        self.assertEqual(params["longitude"], -110.5)

    @patch("app.services.weather_client._SESSION.get")
    def test_empty_daily_payload_returns_empty_dataframe_not_a_crash(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"daily": {}})
        df = weather_client.historical_daily(44.6, -110.5, "2026-06-01", "2026-06-01")
        self.assertTrue(df.empty)
        self.assertIn("date", df.columns)


def _response(status_code, payload=None, headers=None):
    resp = MagicMock(status_code=status_code, headers=headers or {})
    resp.json.return_value = payload or {}
    if status_code == 200:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code} error", response=resp)
    return resp


class RetryOnRateLimitTests(unittest.TestCase):
    """
    Regression coverage for the retry-with-backoff logic added after a real
    ingestion run failed outright on a 429 from Open-Meteo (2026-09-17) --
    a shared CI-runner IP tripped its rate limit at a low request volume.
    time.sleep is patched throughout so these tests run instantly rather
    than actually waiting out the backoff.
    """

    @patch("app.services.weather_client.time.sleep")
    @patch("app.services.weather_client._SESSION.get")
    def test_succeeds_after_one_429_then_a_200(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _response(429),
            _response(200, _archive_response(["2026-06-01"], [25.0])),
        ]
        df = weather_client.historical_daily(44.6, -110.5, "2026-06-01", "2026-06-01")
        self.assertEqual(len(df), 1)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("app.services.weather_client.time.sleep")
    @patch("app.services.weather_client._SESSION.get")
    def test_honors_retry_after_header_when_present(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _response(429, headers={"Retry-After": "7"}),
            _response(200, _archive_response(["2026-06-01"], [25.0])),
        ]
        weather_client.historical_daily(44.6, -110.5, "2026-06-01", "2026-06-01")
        mock_sleep.assert_called_once_with(7.0)

    @patch("app.services.weather_client.time.sleep")
    @patch("app.services.weather_client._SESSION.get")
    def test_gives_up_and_raises_after_exhausting_retries(self, mock_get, mock_sleep):
        mock_get.side_effect = [_response(429) for _ in range(weather_client._MAX_RETRIES)]
        with self.assertRaises(requests.exceptions.HTTPError):
            weather_client.historical_daily(44.6, -110.5, "2026-06-01", "2026-06-01")
        self.assertEqual(mock_get.call_count, weather_client._MAX_RETRIES)

    @patch("app.services.weather_client.time.sleep")
    @patch("app.services.weather_client._SESSION.get")
    def test_a_real_error_like_404_is_not_retried(self, mock_get, mock_sleep):
        # Only 429 and 5xx are worth retrying -- a 404 (e.g. a bad URL) is
        # never going to succeed on retry and should fail immediately.
        mock_get.return_value = _response(404)
        with self.assertRaises(requests.exceptions.HTTPError):
            weather_client.historical_daily(44.6, -110.5, "2026-06-01", "2026-06-01")
        self.assertEqual(mock_get.call_count, 1)
        mock_sleep.assert_not_called()


class RetryOnTransportFailureTests(unittest.TestCase):
    """
    Regression coverage for a real 2026-09-17 incident: fetching ~20 years
    of Kruger's historical archive hit a raw requests.exceptions.ReadTimeout
    -- a transport-level failure with no HTTP response at all -- which used
    to propagate straight out of _get_with_retry uncaught, since the retry
    loop only ever inspected resp.status_code. That killed the whole
    scripts/ingest_weather.py run even though Yellowstone (a shorter date
    range, fetched just before) had already succeeded in the same run.
    """

    @patch("app.services.weather_client.time.sleep")
    @patch("app.services.weather_client._SESSION.get")
    def test_succeeds_after_a_read_timeout_then_a_200(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            requests.exceptions.ReadTimeout("read timed out"),
            _response(200, _archive_response(["2026-06-01"], [25.0])),
        ]
        df = weather_client.historical_daily(44.6, -110.5, "2026-06-01", "2026-06-01")
        self.assertEqual(len(df), 1)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("app.services.weather_client.time.sleep")
    @patch("app.services.weather_client._SESSION.get")
    def test_also_retries_a_connection_error(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("connection reset"),
            _response(200, _archive_response(["2026-06-01"], [25.0])),
        ]
        df = weather_client.historical_daily(44.6, -110.5, "2026-06-01", "2026-06-01")
        self.assertEqual(len(df), 1)
        self.assertEqual(mock_get.call_count, 2)

    @patch("app.services.weather_client.time.sleep")
    @patch("app.services.weather_client._SESSION.get")
    def test_gives_up_and_raises_the_transport_error_after_exhausting_retries(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            requests.exceptions.ReadTimeout("read timed out") for _ in range(weather_client._MAX_RETRIES)
        ]
        with self.assertRaises(requests.exceptions.ReadTimeout):
            weather_client.historical_daily(44.6, -110.5, "2026-06-01", "2026-06-01")
        self.assertEqual(mock_get.call_count, weather_client._MAX_RETRIES)

    @patch("app.services.weather_client.time.sleep")
    @patch("app.services.weather_client._SESSION.get")
    def test_a_timeout_then_a_429_then_success_rides_out_both_failure_kinds(self, mock_get, mock_sleep):
        # The two retry paths (transport exceptions and 429/5xx responses)
        # share one delay/attempt counter, so a run that hits one of each
        # kind of transient failure before succeeding must still work.
        mock_get.side_effect = [
            requests.exceptions.ReadTimeout("read timed out"),
            _response(429),
            _response(200, _archive_response(["2026-06-01"], [25.0])),
        ]
        df = weather_client.historical_daily(44.6, -110.5, "2026-06-01", "2026-06-01")
        self.assertEqual(len(df), 1)
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)


class ClimateNormalsTests(unittest.TestCase):
    def setUp(self):
        weather_client.climate_normals.cache_clear()

    @patch("app.services.weather_client.historical_daily")
    def test_produces_mean_and_std_per_day_of_year(self, mock_hist):
        # Two years of the same June 1st value (25, 27) and nothing else.
        # Both 2023 and 2025 are common (non-leap) years, so June 1 lands on
        # the same day-of-year (152) in both -- using a leap and a non-leap
        # year here would put them one day apart and silently average the
        # wrong pair (caught while first writing this test). The column
        # names must be *_normal_mean / *_normal_std (this exact mismatch
        # was a real bug caught while writing this test -- see
        # weather_anomaly_features's use of these columns).
        mock_hist.return_value = pd.DataFrame({
            "date": ["2023-06-01", "2025-06-01"],
            "temperature_2m_max": [25.0, 27.0],
            "temperature_2m_min": [10.0, 12.0],
            "precipitation_sum": [0.0, 1.0],
            "windspeed_10m_max": [10.0, 10.0],
            "cloudcover_mean": [30.0, 30.0],
        })
        normals = weather_client.climate_normals(44.6, -110.5, years=2)
        self.assertIn("temperature_2m_max_normal_mean", normals.columns)
        self.assertIn("temperature_2m_max_normal_std", normals.columns)
        row = normals[normals["day_of_year"] == 152]  # June 1 in a non-leap year
        self.assertFalse(row.empty)
        self.assertAlmostEqual(float(row["temperature_2m_max_normal_mean"].iloc[0]), 26.0)

    @patch("app.services.weather_client.historical_daily")
    def test_empty_archive_returns_empty_frame(self, mock_hist):
        mock_hist.return_value = pd.DataFrame(columns=["date", *weather_client.WEATHER_DAILY_VARS])
        normals = weather_client.climate_normals(1.0, 2.0, years=1)
        self.assertTrue(normals.empty)


class WeatherAnomalyFeaturesTests(unittest.TestCase):
    """
    Regression test for the *_mean/*_std vs *_normal_mean/*_normal_std
    column-naming bug: this function reads the normals frame that
    climate_normals() actually produces, so a naming mismatch between the
    two must fail loudly here rather than only in production.
    """

    def setUp(self):
        weather_client.climate_normals.cache_clear()

    @patch("app.services.weather_client.historical_daily")
    def test_uses_the_correct_normals_column_names_end_to_end(self, mock_hist):
        # A pre-built normals frame in the exact shape climate_normals()
        # produces (temperature_2m_max_normal_mean / _normal_std, ...): a
        # flat 20-degree normal for every day of the year. Passing `normals`
        # explicitly skips the (separately-tested) climate_normals() call
        # and its own live archive fetch, isolating this test to the one
        # thing it's checking: the column-name lookup.
        day_of_year = list(range(1, 367))
        precomputed_normals = pd.DataFrame({
            "day_of_year": day_of_year,
            "temperature_2m_max_normal_mean": [20.0] * len(day_of_year),
            "temperature_2m_max_normal_std": [5.0] * len(day_of_year),
            "temperature_2m_min_normal_mean": [10.0] * len(day_of_year),
            "temperature_2m_min_normal_std": [4.0] * len(day_of_year),
            "precipitation_sum_normal_mean": [1.0] * len(day_of_year),
            "precipitation_sum_normal_std": [1.0] * len(day_of_year),
            "windspeed_10m_max_normal_mean": [10.0] * len(day_of_year),
            "windspeed_10m_max_normal_std": [3.0] * len(day_of_year),
            "cloudcover_mean_normal_mean": [30.0] * len(day_of_year),
            "cloudcover_mean_normal_std": [10.0] * len(day_of_year),
        })

        # A real date 5 days in the past, so the function's own horizon
        # check picks the live-historical branch without needing to mock
        # datetime.date.today() at all.
        target = dt.date.today() - dt.timedelta(days=5)
        mock_hist.return_value = pd.DataFrame({
            "date": [target.isoformat()], "temperature_2m_max": [30.0], "temperature_2m_min": [15.0],
            "precipitation_sum": [1.0], "windspeed_10m_max": [10.0], "cloudcover_mean": [30.0],
        })

        row = weather_client.weather_anomaly_features(44.6, -110.5, target.isoformat(), normals=precomputed_normals)

        # This is the assertion that would have failed before the fix: the
        # function used to KeyError looking up "temperature_2m_max_mean"
        # against a frame that only had "temperature_2m_max_normal_mean".
        self.assertAlmostEqual(row["temperature_2m_max_normal_mean"], 20.0)
        self.assertAlmostEqual(row["temperature_2m_max"], 30.0)
        self.assertGreater(row["temperature_2m_max_anomaly"], 0)  # 30 is warmer than the 20-degree normal


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for app.services.gbif_client using mocked HTTP responses -- these
verify the CODE (URL/param construction, response parsing, pagination,
bbox math) without needing the live network access this environment does
not have. They do not prove the live GBIF API still matches this shape;
that was checked manually (see the module docstring and the design doc) and
should be re-verified with a real call before trusting this in production.
"""
import unittest
from unittest.mock import MagicMock, patch

import requests

from app.services import gbif_client


def _response(status_code, payload=None, headers=None):
    resp = MagicMock(status_code=status_code, headers=headers or {})
    resp.json.return_value = payload or {}
    if status_code == 200:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code} error", response=resp)
    return resp


class BboxFromPointTests(unittest.TestCase):
    def test_bbox_is_centered_on_the_point(self):
        min_lat, max_lat, min_lon, max_lon = gbif_client._bbox_from_point(44.6, -110.5, 60)
        self.assertAlmostEqual((min_lat + max_lat) / 2, 44.6, places=3)
        self.assertAlmostEqual((min_lon + max_lon) / 2, -110.5, places=3)

    def test_bbox_widens_in_longitude_near_the_poles(self):
        # A degree of longitude covers less ground near the poles, so the
        # box must be WIDER in degrees there to cover the same radius_km.
        _, _, lo_min, lo_max = gbif_client._bbox_from_point(10, 0, 100)   # near the equator
        _, _, hi_min, hi_max = gbif_client._bbox_from_point(70, 0, 100)   # near the pole
        self.assertGreater(hi_max - hi_min, lo_max - lo_min)


class MatchSpeciesTests(unittest.TestCase):
    @patch("app.services.gbif_client._SESSION.get")
    def test_sends_the_name_param_and_returns_parsed_json(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"usageKey": 2433433, "scientificName": "Ursus arctos Linnaeus, 1758",
                           "canonicalName": "Ursus arctos", "matchType": "EXACT", "confidence": 99},
        )
        result = gbif_client.match_species("Ursus arctos")
        called_url, called_kwargs = mock_get.call_args[0][0], mock_get.call_args[1]
        self.assertIn("/species/match", called_url)
        self.assertEqual(called_kwargs["params"]["name"], "Ursus arctos")
        self.assertEqual(result["usageKey"], 2433433)
        self.assertEqual(result["matchType"], "EXACT")


class OccurrenceSearchTests(unittest.TestCase):
    @patch("app.services.gbif_client._SESSION.get")
    def test_builds_expected_query_params(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"results": [], "endOfRecords": True, "count": 0})
        gbif_client.occurrence_search(
            scientific_name="Ursus arctos", lat=44.6, lon=-110.5, radius_km=60,
            year_range=(2006, 2026), limit=300, offset=0,
        )
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["scientificName"], "Ursus arctos")
        self.assertEqual(params["hasCoordinate"], "true")
        self.assertEqual(params["year"], "2006,2026")
        self.assertIn("decimalLatitude", params)
        self.assertIn("decimalLongitude", params)

    @patch("app.services.gbif_client._SESSION.get")
    def test_limit_is_capped_at_gbifs_max_page_size(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"results": [], "endOfRecords": True, "count": 0})
        gbif_client.occurrence_search(scientific_name="Ursus arctos", limit=10_000)
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["limit"], gbif_client._MAX_PAGE_SIZE)

    @patch("app.services.gbif_client._SESSION.get")
    def test_wild_only_excludes_captive_and_fossil_records(self, mock_get):
        # Regression test for a real report: a lion "verified live against
        # GBIF" near Bangkok, Thailand -- almost certainly a zoo/safari-park
        # record (LIVING_SPECIMEN), not a wild population. wild_only=True
        # must ask GBIF to exclude that basisOfRecord (and FOSSIL_SPECIMEN).
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"results": [], "endOfRecords": True, "count": 0})
        gbif_client.occurrence_search(scientific_name="Panthera leo", wild_only=True)
        params = mock_get.call_args[1]["params"]
        self.assertIn("basisOfRecord", params)
        self.assertNotIn("LIVING_SPECIMEN", params["basisOfRecord"])
        self.assertNotIn("FOSSIL_SPECIMEN", params["basisOfRecord"])

    @patch("app.services.gbif_client._SESSION.get")
    def test_wild_only_defaults_off_for_general_callers(self, mock_get):
        # occurrences_near (used to build real training examples from dated
        # sighting records) must NOT filter by basisOfRecord by default --
        # only the plausibility check (has_any_presence) opts into that.
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"results": [], "endOfRecords": True, "count": 0})
        gbif_client.occurrence_search(scientific_name="Ursus arctos")
        params = mock_get.call_args[1]["params"]
        self.assertNotIn("basisOfRecord", params)


class GetRetryTests(unittest.TestCase):
    """
    `_get` used to be a bare `_SESSION.get(...).raise_for_status()` with no
    retry logic of any kind -- hardened proactively after the same class of
    bug (an unretried transport timeout) took down
    scripts/ingest_weather.py on 2026-09-17 (see test_weather_client.py's
    RetryOnTransportFailureTests); GBIF ingestion runs earlier in the same
    retrain.yml pipeline and was structurally even more exposed.
    """

    @patch("app.services.gbif_client.time.sleep")
    @patch("app.services.gbif_client._SESSION.get")
    def test_succeeds_after_a_read_timeout_then_a_200(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            requests.exceptions.ReadTimeout("read timed out"),
            _response(200, {"usageKey": 1, "scientificName": "Ursus arctos"}),
        ]
        result = gbif_client.match_species("Ursus arctos")
        self.assertEqual(result["usageKey"], 1)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("app.services.gbif_client.time.sleep")
    @patch("app.services.gbif_client._SESSION.get")
    def test_succeeds_after_one_429_then_a_200(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _response(429),
            _response(200, {"results": [], "endOfRecords": True, "count": 0}),
        ]
        gbif_client.occurrence_search(scientific_name="Ursus arctos")
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("app.services.gbif_client.time.sleep")
    @patch("app.services.gbif_client._SESSION.get")
    def test_gives_up_and_raises_the_transport_error_after_exhausting_retries(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("connection reset") for _ in range(gbif_client._MAX_RETRIES)
        ]
        with self.assertRaises(requests.exceptions.ConnectionError):
            gbif_client.match_species("Ursus arctos")
        self.assertEqual(mock_get.call_count, gbif_client._MAX_RETRIES)

    @patch("app.services.gbif_client.time.sleep")
    @patch("app.services.gbif_client._SESSION.get")
    def test_a_real_error_like_404_is_not_retried(self, mock_get, mock_sleep):
        mock_get.return_value = _response(404)
        with self.assertRaises(requests.exceptions.HTTPError):
            gbif_client.match_species("Ursus arctos")
        self.assertEqual(mock_get.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("app.services.gbif_client.time.sleep")
    @patch("app.services.gbif_client._SESSION.get")
    def test_presence_count_passes_the_interactive_retry_schedule_through(self, mock_get, mock_sleep):
        # species_for_location's live "Explore Anywhere" calls must fail
        # fast (INTERACTIVE_*), not ride out the patient batch schedule and
        # leave a real user's click hanging.
        mock_get.side_effect = [
            requests.exceptions.ReadTimeout("read timed out")
            for _ in range(gbif_client.INTERACTIVE_MAX_RETRIES)
        ]
        with self.assertRaises(requests.exceptions.ReadTimeout):
            gbif_client.presence_count(
                "Panthera pardus", 14.24, 101.87, 150,
                max_retries=gbif_client.INTERACTIVE_MAX_RETRIES,
                base_backoff=gbif_client.INTERACTIVE_BASE_BACKOFF,
                max_backoff=gbif_client.INTERACTIVE_MAX_BACKOFF,
            )
        self.assertEqual(mock_get.call_count, gbif_client.INTERACTIVE_MAX_RETRIES)


class OccurrencesNearPaginationTests(unittest.TestCase):
    @patch("app.services.gbif_client.time.sleep", lambda *_: None)  # skip the polite delay in tests
    @patch("app.services.gbif_client.occurrence_search")
    def test_pages_until_end_of_records(self, mock_search):
        page1 = {"results": [{"key": i} for i in range(300)], "endOfRecords": False, "count": 500}
        page2 = {"results": [{"key": i} for i in range(300, 500)], "endOfRecords": True, "count": 500}
        mock_search.side_effect = [page1, page2]

        results = gbif_client.occurrences_near("Ursus arctos", 44.6, -110.5, 60, max_records=1000)
        self.assertEqual(len(results), 500)
        self.assertEqual(mock_search.call_count, 2)

    @patch("app.services.gbif_client.occurrence_search")
    def test_stops_at_max_records_even_if_more_are_available(self, mock_search):
        page = {"results": [{"key": i} for i in range(300)], "endOfRecords": False, "count": 10_000}
        mock_search.return_value = page
        with patch("app.services.gbif_client.time.sleep", lambda *_: None):
            results = gbif_client.occurrences_near("Ursus arctos", 44.6, -110.5, 60, max_records=450)
        self.assertEqual(len(results), 450)


class PresenceCountTests(unittest.TestCase):
    @patch("app.services.gbif_client.occurrence_search")
    def test_returns_gbifs_real_count_not_just_a_boolean(self, mock_search):
        # Regression test for a real report: a Leopard showed "91% high
        # confidence" near a Thailand click point off the strength of its
        # curated home area's training count, because the live GBIF check
        # only ever asked yes/no ("has this species EVER been seen here").
        # presence_count must surface the actual number so calibration can
        # be based on real local evidence instead.
        mock_search.return_value = {"count": 37, "results": []}
        self.assertEqual(gbif_client.presence_count("Panthera pardus", 14.24, 101.87, 150), 37)

    @patch("app.services.gbif_client.occurrence_search")
    def test_zero_when_gbif_reports_zero_records(self, mock_search):
        mock_search.return_value = {"count": 0, "results": []}
        self.assertEqual(gbif_client.presence_count("Ursus maritimus", -1.0, -60.0, 60), 0)

    @patch("app.services.gbif_client.occurrence_search")
    def test_asks_for_wild_records_only(self, mock_search):
        mock_search.return_value = {"count": 0, "results": []}
        gbif_client.presence_count("Panthera leo", 14.24, 101.87, 150)
        self.assertTrue(mock_search.call_args[1]["wild_only"])

    @patch("app.services.gbif_client.occurrence_search")
    def test_only_asks_gbif_for_one_record_since_only_the_total_count_is_needed(self, mock_search):
        mock_search.return_value = {"count": 500, "results": []}
        gbif_client.presence_count("Panthera pardus", 14.24, 101.87, 150)
        self.assertEqual(mock_search.call_args[1]["limit"], 1)


class HasAnyPresenceTests(unittest.TestCase):
    @patch("app.services.gbif_client.occurrence_search")
    def test_true_when_gbif_reports_a_nonzero_count(self, mock_search):
        mock_search.return_value = {"count": 5, "results": []}
        self.assertTrue(gbif_client.has_any_presence("Ursus arctos", 44.6, -110.5, 60))

    @patch("app.services.gbif_client.occurrence_search")
    def test_false_when_gbif_reports_zero_records(self, mock_search):
        mock_search.return_value = {"count": 0, "results": []}
        self.assertFalse(gbif_client.has_any_presence("Ursus maritimus", -1.0, -60.0, 60))  # polar bear near the equator

    @patch("app.services.gbif_client.occurrence_search")
    def test_asks_for_wild_records_only(self, mock_search):
        mock_search.return_value = {"count": 0, "results": []}
        gbif_client.has_any_presence("Panthera leo", 14.24, 101.87, 150)
        self.assertTrue(mock_search.call_args[1]["wild_only"])

    @patch("app.services.gbif_client.occurrence_search")
    def test_passes_plausible_countries_through_as_the_country_filter(self, mock_search):
        # Regression test for a real report: wild_only alone (basisOfRecord
        # filtering) was not enough -- a lion and a gray wolf both still
        # showed up as "verified" near Bangkok, Thailand, because a casual
        # zoo visitor's photo typically carries the SAME basisOfRecord
        # (HUMAN_OBSERVATION) as a genuine wild sighting. Restricting the
        # query to the species' real native-range countries is what
        # actually closes that gap.
        mock_search.return_value = {"count": 0, "results": []}
        gbif_client.has_any_presence(
            "Panthera leo", 14.24, 101.87, 150, plausible_countries=["ZA", "KE", "TZ"]
        )
        self.assertEqual(mock_search.call_args[1]["country"], ["ZA", "KE", "TZ"])

    @patch("app.services.gbif_client._SESSION.get")
    def test_country_list_is_sent_as_the_gbif_country_param(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"results": [], "endOfRecords": True, "count": 0})
        gbif_client.occurrence_search(scientific_name="Panthera leo", country=["ZA", "KE"])
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["country"], ["ZA", "KE"])


if __name__ == "__main__":
    unittest.main()

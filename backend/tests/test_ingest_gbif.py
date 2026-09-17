"""
Unit tests for scripts.ingest_gbif._record_date -- the day-precision date
extractor for a GBIF occurrence record. Regression coverage for a real
2026-09-17 retrain.yml crash: a Scottish Highlands occurrence record's
`eventDate` was a bare year ("2008", no month/day at all), which the old
code's `str(date)[:10]` let straight through into
data/cache/occurrences_scottish_highlands.csv unchanged (a 4-character
string sliced to 10 chars is still 4 characters) -- and later crashed
scripts/ingest_weather.py's `pd.to_datetime(occ["date"])` with
`ValueError: time data "2008" doesn't match format "%Y-%m-%d"`.
"""
import unittest

from scripts.ingest_gbif import _record_date


class RecordDateTests(unittest.TestCase):
    def test_full_iso_event_date_is_used_as_is(self):
        self.assertEqual(_record_date({"eventDate": "2021-06-15"}), "2021-06-15")

    def test_full_timestamp_event_date_is_truncated_to_the_date(self):
        self.assertEqual(_record_date({"eventDate": "2021-06-15T14:30:00"}), "2021-06-15")

    def test_interval_event_date_uses_the_start(self):
        self.assertEqual(_record_date({"eventDate": "2008-05-01/2008-05-03"}), "2008-05-01")

    def test_bare_year_event_date_is_rejected_not_truncated(self):
        # This is the exact real-world regression: GBIF returned
        # eventDate="2008" for a real record (year-only precision -- GBIF
        # allows this), and the old code's str(date)[:10] passed "2008"
        # straight through as if it were a full date.
        self.assertIsNone(_record_date({"eventDate": "2008"}))

    def test_year_month_event_date_is_rejected(self):
        self.assertIsNone(_record_date({"eventDate": "2008-05"}))

    def test_falls_back_to_year_month_day_fields_when_event_date_is_missing(self):
        self.assertEqual(_record_date({"year": 2019, "month": 3, "day": 7}), "2019-03-07")

    def test_year_month_day_fields_pad_single_digit_month_and_day(self):
        self.assertEqual(_record_date({"year": 2019, "month": 3, "day": 7}), "2019-03-07")

    def test_no_date_at_all_when_only_year_is_present(self):
        # The old fallback defaulted a missing month/day to 1, fabricating
        # a plausible-looking "January 1st" date for a record that might
        # only ever have had year-level precision. A record without real
        # day precision is worse than useless for joining to one day's
        # weather, so it must be dropped, not guessed at.
        self.assertIsNone(_record_date({"year": 2019}))

    def test_no_date_at_all_when_year_and_month_present_but_no_day(self):
        self.assertIsNone(_record_date({"year": 2019, "month": 3}))

    def test_no_date_fields_at_all_returns_none(self):
        self.assertIsNone(_record_date({}))

    def test_malformed_event_date_falls_back_to_year_month_day_fields(self):
        # An unparseable eventDate string should not stop the year/month/day
        # fallback from being tried.
        self.assertEqual(
            _record_date({"eventDate": "not a date", "year": 2019, "month": 3, "day": 7}),
            "2019-03-07",
        )


if __name__ == "__main__":
    unittest.main()

"""
Tests for app.ml.prediction_service. These are integration-style: they use
the trained model + cached data that ship in the repo (backend/app/ml/artifacts
and backend/data/cache), rather than mocking, because the whole point is to
catch problems in how the pieces fit together -- exactly the class of bug
(a collapsed/inverted calibration) that unit tests of the individual pieces
would not have caught. If these fail, retrain first: `python -m app.ml.train`.
"""
import datetime as dt
import threading
import time
import unittest
from unittest.mock import patch

import pandas as pd

from app.ml import prediction_service as ps

ALL_PILOT_AREA_IDS = {
    "yellowstone", "kruger", "brisbane_urban", "svalbard_arctic", "amazon_rainforest",
    "serengeti", "borneo_rainforest", "patagonia", "scottish_highlands", "costa_rica_cloud_forest",
}


def _synthetic_normals() -> pd.DataFrame:
    """A minimal but schema-correct climate-normals DataFrame, standing in
    for a live `weather_client.climate_normals()` call in tests that must
    not hit the real network."""
    from app.config import WEATHER_DAILY_VARS

    rows = []
    for doy in range(1, 367):
        row = {"day_of_year": doy}
        for var in WEATHER_DAILY_VARS:
            row[f"{var}_normal_mean"] = 15.0
            row[f"{var}_normal_std"] = 3.0
        rows.append(row)
    return pd.DataFrame(rows)


class ListAndGetAreaTests(unittest.TestCase):
    def test_all_five_pilot_areas_are_present(self):
        ids = {a["id"] for a in ps.list_areas()}
        self.assertEqual(ids, ALL_PILOT_AREA_IDS)

    def test_unknown_area_raises_key_error(self):
        with self.assertRaises(KeyError):
            ps.get_area("atlantis")


class SpeciesForAreaTests(unittest.TestCase):
    def test_yellowstone_species_are_yellowstone_only(self):
        species = ps.species_for_area("yellowstone")
        self.assertEqual({s["area_id"] for s in species}, {"yellowstone"})
        names = {s["scientific_name"] for s in species}
        self.assertIn("Ursus arctos", names)
        self.assertNotIn("Loxodonta africana", names)  # Kruger-only: the plausibility gate must exclude it

    def test_every_seed_species_has_training_support(self):
        for area_id in ALL_PILOT_AREA_IDS:
            for s in ps.species_for_area(area_id):
                self.assertGreater(s["training_support"], 0, f"{s['scientific_name']} has no training support")

    def test_no_species_is_offered_outside_its_own_area(self):
        """
        Direct operationalization of the project's own founding example:
        "there's no point predicting a polar bear in a tropical rainforest."
        Every species must appear in exactly the one area's candidate list
        it was seeded for, and nowhere else -- the plausibility gate must
        never suggest an animal where it can't live.
        """
        by_area = {area_id: {s["scientific_name"] for s in ps.species_for_area(area_id)} for area_id in ALL_PILOT_AREA_IDS}
        for area_id, names in by_area.items():
            for other_area_id, other_names in by_area.items():
                if other_area_id == area_id:
                    continue
                overlap = names & other_names
                self.assertEqual(overlap, set(), f"{overlap} appears in both '{area_id}' and '{other_area_id}'")

    def test_polar_bear_never_offered_for_the_amazon_rainforest(self):
        # The project's own headline example, made concrete and literal.
        amazon_species = {s["scientific_name"] for s in ps.species_for_area("amazon_rainforest")}
        self.assertNotIn("Ursus maritimus", amazon_species)

    def test_jaguar_never_offered_for_the_arctic_tundra(self):
        arctic_species = {s["scientific_name"] for s in ps.species_for_area("svalbard_arctic")}
        self.assertNotIn("Panthera onca", arctic_species)


class PredictTests(unittest.TestCase):
    def test_predict_returns_one_row_per_area_species_sorted_descending(self):
        results = ps.predict("yellowstone", "2027-06-15")
        self.assertEqual(len(results), len(ps.species_for_area("yellowstone")))
        probs = [r["probability"] for r in results]
        self.assertEqual(probs, sorted(probs, reverse=True))
        for r in results:
            self.assertGreaterEqual(r["probability"], 0.0)
            self.assertLessEqual(r["probability"], 1.0)
            self.assertIn(r["confidence"], ("high", "medium", "low"))

    def test_single_species_filter(self):
        results = ps.predict("yellowstone", "2027-06-15", species_key="Ursus arctos")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["scientific_name"], "Ursus arctos")

    def test_species_not_in_area_raises_key_error(self):
        with self.assertRaises(KeyError):
            ps.predict("yellowstone", "2027-06-15", species_key="Loxodonta africana")

    def test_unknown_area_raises_key_error(self):
        with self.assertRaises(KeyError):
            ps.predict("atlantis", "2027-06-15")

    def test_grizzly_bear_scores_higher_in_summer_than_deep_winter(self):
        """
        Regression test for the calibration-collapse bug found while
        building this project: an earlier version of the trained model
        scored a hibernating-season date HIGHER than a summer date for
        grizzly bears -- the opposite of both real bear ecology and this
        project's own training data. This test pins the correct direction
        so a future retrain that reintroduces that bug fails loudly here
        instead of silently shipping backwards predictions.
        """
        summer = ps.predict("yellowstone", "2027-07-15", species_key="Ursus arctos")[0]["probability"]
        deep_winter = ps.predict("yellowstone", "2027-01-15", species_key="Ursus arctos")[0]["probability"]
        self.assertGreater(summer, deep_winter)

    def test_kangaroo_scores_higher_in_cooler_months_than_summer_heat(self):
        # Eastern grey kangaroos avoid extreme heat (see the sensitivity
        # profile in scripts/generate_sample_fixtures.py) -- another
        # concrete, checkable seasonal claim, this time on a Southern
        # Hemisphere / urban species.
        cool = ps.predict("brisbane_urban", "2027-07-01", species_key="Macropus giganteus")[0]["probability"]
        hot = ps.predict("brisbane_urban", "2027-01-15", species_key="Macropus giganteus")[0]["probability"]
        self.assertGreater(cool, hot)

    def test_polar_bear_scores_higher_in_ice_free_season_than_deep_winter(self):
        # Polar bears here are modeled as most visible ashore in the
        # ice-free season while waiting for the sea to refreeze (Aug), and
        # least visible in deep winter/spring while out hunting on distant
        # sea ice (Feb) -- see the profile comment in generate_sample_fixtures.py.
        ice_free = ps.predict("svalbard_arctic", "2027-08-15", species_key="Ursus maritimus")[0]["probability"]
        deep_winter = ps.predict("svalbard_arctic", "2027-02-15", species_key="Ursus maritimus")[0]["probability"]
        self.assertGreater(ice_free, deep_winter)

    def test_jaguar_scores_higher_in_dry_season_than_wet_season(self):
        # Jaguars here are modeled as more visible in the drier months near
        # shrinking water sources (Jul), and less visible in the wet season
        # when floods disperse them (Feb).
        dry = ps.predict("amazon_rainforest", "2027-07-15", species_key="Panthera onca")[0]["probability"]
        wet = ps.predict("amazon_rainforest", "2027-02-15", species_key="Panthera onca")[0]["probability"]
        self.assertGreater(dry, wet)

    def test_predictions_include_plain_language_factors(self):
        # The design doc's Product & UX "why" explanation text: every
        # prediction should carry 2-3 short, human-readable reasons, not
        # just a bare number.
        results = ps.predict("yellowstone", "2027-06-15", species_key="Ursus arctos")
        factors = results[0]["factors"]
        self.assertIsInstance(factors, list)
        self.assertGreaterEqual(len(factors), 1)
        for f in factors:
            self.assertIsInstance(f, str)
            self.assertGreater(len(f), 0)
        # The first factor is always the data-support explanation.
        self.assertIn("historical sighting records", factors[0])


class EvidenceFactorTests(unittest.TestCase):
    def test_zero_primary_count_is_zero(self):
        self.assertEqual(ps._evidence_factor(0), 0.0)
        self.assertEqual(ps._evidence_factor(0, corroborating_count=2), 0.0)

    def test_factor_increases_with_primary_count(self):
        low = ps._evidence_factor(1)
        mid = ps._evidence_factor(40)
        high = ps._evidence_factor(150)
        self.assertLess(low, mid)
        self.assertLess(mid, high)
        self.assertAlmostEqual(high, 1.0, places=6)

    def test_corroboration_gives_a_modest_bonus_but_never_exceeds_one(self):
        base = ps._evidence_factor(40, corroborating_count=0)
        one_source = ps._evidence_factor(40, corroborating_count=1)
        two_sources = ps._evidence_factor(40, corroborating_count=2)
        self.assertLess(base, one_source)
        self.assertLess(one_source, two_sources)
        # Extra corroborating sources beyond the cap add nothing further.
        self.assertEqual(two_sources, ps._evidence_factor(40, corroborating_count=5))
        # A saturated primary count plus corroboration still caps at 1.0.
        self.assertEqual(ps._evidence_factor(150, corroborating_count=2), 1.0)


class EffortAdjustedEvidenceFactorTests(unittest.TestCase):
    """
    Regression coverage for a real data-reliability problem raised
    directly by a user: "5000 polar bear observations doesn't mean they're
    easy to find, and 5 kangaroo observations doesn't mean it's hard."
    Raw local counts alone conflate true abundance with how much anyone
    happens to be looking/reporting nearby. `effort_index` (total wild
    records of ANY species nearby) corrects for that.
    """

    def test_missing_effort_index_falls_back_to_original_absolute_only_behavior(self):
        # None (the live check failed or was never run) must reproduce the
        # exact pre-existing behavior -- no silent change for every caller
        # that doesn't pass this new argument at all.
        with_none = ps._evidence_factor(40, corroborating_count=1, effort_index=None)
        without_arg = ps._evidence_factor(40, corroborating_count=1)
        self.assertEqual(with_none, without_arg)

    def test_zero_or_negative_effort_index_also_falls_back(self):
        # A genuinely-zero or invalid baseline is not a usable denominator
        # -- must not raise (division by zero) or silently maximize
        # confidence, just fall back like None does.
        self.assertEqual(
            ps._evidence_factor(40, effort_index=0), ps._evidence_factor(40, effort_index=None),
        )

    def test_a_large_count_in_a_high_effort_area_is_dampened_more_than_without_effort_data(self):
        # The "5000 polar bear observations" case: a big raw count, but
        # this species is a small share of everything being reported near
        # this very busy (high-effort) point -- the effort-adjusted factor
        # must come out LOWER than the same count would score with no
        # effort data at all.
        unadjusted = ps._evidence_factor(500, effort_index=None)
        # 500 out of 100,000 total local wild records is a tiny (0.5%) share.
        effort_adjusted = ps._evidence_factor(500, effort_index=100_000)
        self.assertLess(effort_adjusted, unadjusted)

    def test_a_small_count_in_a_low_effort_area_is_boosted_relative_to_no_effort_data(self):
        # The "5 kangaroo observations" case: a small raw count, but it's
        # a LARGE share of the (small) total local activity -- the
        # effort-adjusted factor must come out HIGHER than the same small
        # count would score with no effort data at all.
        unadjusted = ps._evidence_factor(5, effort_index=None)
        # 5 out of 8 total local wild records is the large majority of all
        # local activity.
        effort_adjusted = ps._evidence_factor(5, effort_index=8)
        self.assertGreater(effort_adjusted, unadjusted)

    def test_effort_index_below_the_noise_floor_is_ignored(self):
        # A baseline of 1-2 total records is too thin to divide by without
        # producing a wildly noisy ratio -- must fall back to
        # absolute-count-only rather than swing on one extra record.
        self.assertLess(ps._MIN_EFFORT_INDEX_FOR_SHARE, 5 + 1)  # sanity: test below actually exercises the guard
        ignored = ps._evidence_factor(3, effort_index=2)
        fallback = ps._evidence_factor(3, effort_index=None)
        self.assertEqual(ignored, fallback)

    def test_effort_adjusted_factor_never_exceeds_one(self):
        self.assertLessEqual(ps._evidence_factor(1000, corroborating_count=2, effort_index=1001), 1.0)


class ExplainFactorsTests(unittest.TestCase):
    """Direct tests of _explain's new activity-pattern and conservation-status
    factor lines -- the user's explicit "population" and "species ability to
    evade" asks, kept as real qualitative/authoritative context rather than
    an invented numeric statistic (see the module's _ACTIVITY_PATTERN_NOTES
    and _explain docstrings)."""

    def _factors(self, **kwargs):
        return ps._explain(
            "Ursus arctos", dt.date(2027, 6, 15), {}, {}, support=100, confidence="high", **kwargs
        )

    def test_nocturnal_gets_an_evasion_note(self):
        factors = self._factors(activity_pattern="nocturnal")
        self.assertTrue(any("nocturnal" in f for f in factors))

    def test_diurnal_gets_no_extra_note(self):
        with_note = self._factors(activity_pattern="diurnal")
        without = self._factors(activity_pattern=None)
        self.assertEqual(len(with_note), len(without))

    def test_unknown_activity_pattern_is_silently_skipped(self):
        factors = self._factors(activity_pattern="made_up_value")
        without = self._factors(activity_pattern=None)
        self.assertEqual(len(factors), len(without))

    def test_conservation_status_adds_a_real_data_labeled_line(self):
        factors = self._factors(
            conservation_status={"category": "EN", "category_label": "Endangered", "population_trend": "Decreasing"}
        )
        self.assertTrue(any("IUCN Red List status: Endangered" in f for f in factors))
        self.assertTrue(any("decreasing" in f for f in factors))
        self.assertTrue(any("real conservation data, not sighting data" in f for f in factors))

    def test_missing_conservation_status_adds_no_line(self):
        with_none = self._factors(conservation_status=None)
        with_empty = self._factors(conservation_status={})
        without = self._factors()
        self.assertEqual(len(with_none), len(without))
        self.assertEqual(len(with_empty), len(without))


class DataSourceTests(unittest.TestCase):
    def test_get_data_source_returns_a_source_key(self):
        result = ps.get_data_source()
        self.assertIn("source", result)
        self.assertIn(result["source"], ("real", "synthetic_demo", "unknown"))


class BestWindowTests(unittest.TestCase):
    def test_returns_a_point_roughly_every_five_days_across_the_year(self):
        points = ps.best_window("yellowstone", "Ursus arctos")
        self.assertGreaterEqual(len(points), 70)
        dates = [dt.date.fromisoformat(p["date"]) for p in points]
        self.assertEqual(dates, sorted(dates))
        for p in points:
            self.assertGreaterEqual(p["probability"], 0.0)
            self.assertLessEqual(p["probability"], 1.0)

    def test_unknown_species_raises_key_error(self):
        with self.assertRaises(KeyError):
            ps.best_window("yellowstone", "Loxodonta africana")


def _no_inaturalist(**overrides):
    """Patches iNaturalist to report nothing, by default, in every Anywhere
    test below -- it's queried unconditionally (see
    `_local_evidence_cached`), so leaving it unmocked would make these tests
    hit the real network (which this sandbox's egress blocks anyway) and
    make results depend on live iNaturalist data instead of the fixed
    values each test is actually pinning. `overrides` lets a specific test
    give iNaturalist a real mocked value to test corroboration."""
    return patch("app.services.inaturalist_client.presence_count", **(overrides or {"return_value": 0}))


class AnywhereModeTests(unittest.TestCase):
    """
    'Explore Anywhere' mode (species_for_location / predict_at_location /
    best_window_at_location) is tested with GBIF, iNaturalist, and live
    weather mocked out -- the point of these tests is the wiring (does a
    live presence result correctly gate the candidate list? does an
    arbitrary point still produce valid, correctly-labeled predictions?),
    not GBIF's/iNaturalist's/Open-Meteo's actual live data, which the
    *_client modules' own tests and the manual Render smoke test already
    cover. eBird is skipped automatically in tests (no EBIRD_API_KEY set),
    so it needs no mocking here.
    """

    def setUp(self):
        # The live-evidence and effort-index caches are both process-global
        # (by design -- see their docstrings) so neither must leak results
        # between tests that reuse the same coordinates with different
        # mocked source answers.
        ps._presence_cache.clear()
        ps._effort_cache.clear()

    def test_haversine_zero_for_same_point(self):
        self.assertAlmostEqual(ps._haversine_km(44.6, -110.5, 44.6, -110.5), 0.0, places=6)

    def test_haversine_roughly_correct_known_distance(self):
        # New York City to London is ~5570km.
        d = ps._haversine_km(40.7128, -74.0060, 51.5074, -0.1278)
        self.assertGreater(d, 5400)
        self.assertLess(d, 5700)

    def test_nearest_area_finds_the_close_one(self):
        # A point a few km from Yellowstone's center should resolve to
        # Yellowstone, not one of the other 4 (much farther) areas.
        nearest = ps._nearest_area(44.65, -110.45)
        self.assertEqual(nearest["id"], "yellowstone")

    def test_species_for_location_only_returns_gbif_confirmed_species(self):
        confirmed = {"Ursus arctos", "Canis lupus"}
        with (
            patch(
                "app.services.gbif_client.presence_count",
                side_effect=lambda name, *a, **k: 200 if name in confirmed else 0,
            ),
            _no_inaturalist(),
        ):
            results = ps.species_for_location(44.6, -110.5)
        names = {s["scientific_name"] for s in results}
        self.assertEqual(names, confirmed)
        for s in results:
            self.assertGreater(s["training_support"], 0)
            self.assertEqual(s["local_evidence_count"], 200)
            self.assertEqual(s["local_corroborating_count"], 0)
            self.assertEqual(s["local_evidence_sources"]["gbif"], 200)

    def test_species_for_location_passes_each_species_curated_country_list(self):
        # Regression test for a real report: a lion and a gray wolf both
        # showed up as "verified" near Bangkok, Thailand, even after
        # excluding captive/fossil GBIF records -- because a casual zoo
        # visitor's photo typically carries the same basisOfRecord as a
        # genuine wild sighting. The fix restricts each species' live GBIF
        # query to its own curated native-range countries from
        # seed_species.json. This test pins that Lion's country list does
        # NOT include Thailand (so a Bangkok click can never confirm it)
        # while Leopard's DOES (Indochinese leopards are a real native
        # subspecies there, and the plausibility check must not falsely
        # exclude a genuinely plausible species).
        seen_countries = {}

        def fake_presence(name, *_args, **kwargs):
            seen_countries[name] = kwargs.get("plausible_countries")
            return 50

        with patch("app.services.gbif_client.presence_count", side_effect=fake_presence), _no_inaturalist():
            ps.species_for_location(14.24, 101.87)

        self.assertNotIn("TH", seen_countries["Panthera leo"] or [])
        self.assertIn("TH", seen_countries["Panthera pardus"] or [])
        self.assertNotIn("TH", seen_countries["Canis lupus"] or [])

    def test_species_for_location_excludes_everything_on_gbif_outage(self):
        # A live API failure must fail closed (nothing offered), never open
        # (everything offered) -- an empty/erroring response from every
        # source is not evidence a species is plausible somewhere.
        with (
            patch("app.services.gbif_client.presence_count", side_effect=RuntimeError("network down")),
            _no_inaturalist(),
        ):
            results = ps.species_for_location(44.6, -110.5)
        self.assertEqual(results, [])

    def test_species_for_location_falls_back_to_inaturalist_when_gbif_finds_nothing(self):
        # primary_count should use whichever source actually found
        # something when GBIF itself comes up empty -- corroboration from
        # an independent source is real evidence too, not just a GBIF
        # monopoly.
        with (
            patch("app.services.gbif_client.presence_count", return_value=0),
            _no_inaturalist(return_value=30),
        ):
            results = ps.species_for_location(44.6, -110.5)
        names = {s["scientific_name"] for s in results}
        self.assertGreater(len(names), 0)
        for s in results:
            self.assertEqual(s["local_evidence_count"], 30)
            self.assertEqual(s["local_corroborating_count"], 0)  # only one source found anything

    def test_species_for_location_counts_corroboration_from_a_second_source(self):
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            _no_inaturalist(return_value=15),
        ):
            results = ps.species_for_location(44.6, -110.5)
        for s in results:
            self.assertEqual(s["local_evidence_count"], 200)  # GBIF is primary when it finds anything
            self.assertEqual(s["local_corroborating_count"], 1)  # iNaturalist independently confirmed too

    def test_species_for_location_attaches_the_local_effort_index(self):
        # Wiring test for the effort-correction feature (see
        # _local_effort_index_cached / _evidence_factor): the area-level
        # "how much wildlife-reporting activity happens near this point at
        # all" number must actually reach each matched species, not just
        # exist as a standalone helper.
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            patch("app.services.gbif_client.total_wild_occurrence_count", return_value=4000),
            _no_inaturalist(),
        ):
            results = ps.species_for_location(44.6, -110.5)
        self.assertGreater(len(results), 0)
        for s in results:
            self.assertEqual(s["local_effort_index"], 4000)

    def test_species_for_location_effort_index_is_none_when_the_live_check_fails(self):
        # Fail-closed, not fail-zero: a real 0 would make _evidence_factor
        # treat the area as having no reporting activity at all (an
        # incorrect share-based boost), whereas None correctly falls back
        # to the original absolute-count-only behavior.
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            patch("app.services.gbif_client.total_wild_occurrence_count", side_effect=RuntimeError("network down")),
            _no_inaturalist(),
        ):
            results = ps.species_for_location(44.6, -110.5)
        self.assertGreater(len(results), 0)
        for s in results:
            self.assertIsNone(s["local_effort_index"])

    def test_predict_at_location_labels_the_nearest_area_and_distance(self):
        # A generous local count (well above the "high confidence" floor)
        # so this test's focus -- area/distance labeling and factor text --
        # isn't entangled with the evidence-dampening behavior covered
        # separately below.
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            result = ps.predict_at_location(44.65, -110.45, "2027-06-15")
        self.assertEqual(result["nearest_area"]["id"], "yellowstone")
        self.assertGreaterEqual(result["nearest_area"]["distance_km"], 0)
        self.assertGreater(len(result["predictions"]), 0)
        for p in result["predictions"]:
            self.assertGreaterEqual(p["probability"], 0.0)
            self.assertLessEqual(p["probability"], 1.0)
            self.assertTrue(any("Verified live against real wildlife-sighting databases" in f for f in p["factors"]))
            # The confidence/record-count factor must be about the real
            # local evidence count near the click, not the curated home
            # area's training count -- regression test for a real report
            # where a Leopard showed "91% high confidence" and "500
            # historical sighting records" near a Thailand click point off
            # the strength of Kruger, South Africa's training data alone.
            self.assertFalse(any("in this area" in f for f in p["factors"]))
            self.assertFalse(any("closest trained analog" in f for f in p["factors"]))
            self.assertTrue(any("200 confirmed wild sighting records" in f for f in p["factors"]))
            self.assertTrue(any("Yellowstone" in f for f in p["factors"]))
            self.assertEqual(p["local_evidence_count"], 200)
            self.assertEqual(p["training_records"], 200)

    def test_predict_at_location_mentions_corroborating_sources_when_present(self):
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            _no_inaturalist(return_value=25),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            result = ps.predict_at_location(44.65, -110.45, "2027-06-15", species_key="Ursus arctos")
        p = result["predictions"][0]
        self.assertEqual(p["local_corroborating_count"], 1)
        self.assertTrue(any("independently confirmed by iNaturalist" in f for f in p["factors"]))

    def test_predict_at_location_dampens_probability_for_thin_local_evidence(self):
        # The user's explicit calibration requirement: a species that is
        # technically plausible (passes the presence gate) but has barely
        # any real local sightings must NOT show a high probability. A
        # single confirmed record should score far below the same species'
        # raw model probability -- proof the dampening is actually wired
        # in, not just documented.
        with (
            patch("app.services.gbif_client.presence_count", return_value=1),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            thin = ps.predict_at_location(44.65, -110.45, "2027-06-15", species_key="Ursus arctos")
        ps._presence_cache.clear()  # otherwise the 24h TTL cache would just replay the "thin" count above
        with (
            patch("app.services.gbif_client.presence_count", return_value=1000),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            strong = ps.predict_at_location(44.65, -110.45, "2027-06-15", species_key="Ursus arctos")

        thin_pred, strong_pred = thin["predictions"][0], strong["predictions"][0]
        self.assertEqual(thin_pred["confidence"], "low")
        self.assertEqual(strong_pred["confidence"], "high")
        # Both draw the exact same raw model probability (same inputs);
        # only the displayed, evidence-scaled probability should differ.
        self.assertAlmostEqual(thin_pred["raw_model_probability"], strong_pred["raw_model_probability"], places=6)
        self.assertLess(thin_pred["probability"], thin_pred["raw_model_probability"])
        self.assertLess(thin_pred["probability"], strong_pred["probability"])
        self.assertAlmostEqual(strong_pred["probability"], strong_pred["raw_model_probability"], places=6)

    def test_predict_at_location_includes_effort_index_and_downward_adjustment_note(self):
        # A busy area (high effort_index) relative to this species' own
        # count should dampen the factor below the raw-count-only value and
        # say so explicitly -- the user's "5000 polar bear observations
        # doesn't mean they're easy to find" scenario.
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            patch("app.services.gbif_client.total_wild_occurrence_count", return_value=40000),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            result = ps.predict_at_location(44.65, -110.45, "2027-06-15", species_key="Ursus arctos")
        p = result["predictions"][0]
        self.assertEqual(p["local_effort_index"], 40000)
        self.assertLess(p["probability"], p["raw_model_probability"])
        self.assertTrue(any("a lot of overall wildlife-reporting activity" in f for f in p["factors"]))

    def test_predict_at_location_includes_effort_index_and_upward_adjustment_note(self):
        # A quiet area (thin effort_index) makes even a small count more
        # meaningful than the raw-count-only value would suggest -- the
        # user's "5 kangaroo observations doesn't mean it's hard" scenario.
        with (
            patch("app.services.gbif_client.presence_count", return_value=5),
            patch("app.services.gbif_client.total_wild_occurrence_count", return_value=6),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            result = ps.predict_at_location(44.65, -110.45, "2027-06-15", species_key="Ursus arctos")
        p = result["predictions"][0]
        self.assertEqual(p["local_effort_index"], 6)
        self.assertTrue(any("relatively little overall wildlife-reporting activity" in f for f in p["factors"]))

    def test_predict_at_location_with_no_gbif_matches_returns_empty_not_error(self):
        # A real point with genuinely no curated species nearby (e.g. open
        # ocean) is a valid, non-error result -- just an empty list.
        with (
            patch("app.services.gbif_client.presence_count", return_value=0),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            result = ps.predict_at_location(0.0, -140.0, "2027-06-15")
        self.assertEqual(result["predictions"], [])

    def test_predict_at_location_species_filter_raises_if_not_confirmed_nearby(self):
        with (
            patch("app.services.gbif_client.presence_count", return_value=0),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            with self.assertRaises(KeyError):
                ps.predict_at_location(44.65, -110.45, "2027-06-15", species_key="Ursus arctos")

    def test_predict_at_location_raises_live_data_unavailable_on_weather_outage(self):
        # Unlike curated areas (pre-cached normals, zero network dependency),
        # anywhere-mode's climate normals are a live call with no offline
        # fallback. A failure there must surface as a clear, catchable
        # error -- not an unhandled crash (this was a real bug found via a
        # local end-to-end smoke test: a live Open-Meteo outage took the
        # whole request down with an unhandled exception instead of a
        # clean 503).
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", side_effect=RuntimeError("network down")),
        ):
            with self.assertRaises(ps.LiveDataUnavailable):
                ps.predict_at_location(44.65, -110.45, "2027-06-15")

    def test_best_window_at_location_returns_points_across_the_year(self):
        with (
            patch("app.services.gbif_client.presence_count", return_value=200),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            points = ps.best_window_at_location(44.65, -110.45, "Ursus arctos")
        self.assertGreaterEqual(len(points), 70)
        for p in points:
            self.assertGreaterEqual(p["probability"], 0.0)
            self.assertLessEqual(p["probability"], 1.0)

    def test_best_window_at_location_dampens_for_thin_local_evidence(self):
        with (
            patch("app.services.gbif_client.presence_count", return_value=1),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            thin_points = ps.best_window_at_location(44.65, -110.45, "Ursus arctos")
        ps._presence_cache.clear()  # otherwise the 24h TTL cache would just replay the "thin" count above
        with (
            patch("app.services.gbif_client.presence_count", return_value=1000),
            _no_inaturalist(),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            strong_points = ps.best_window_at_location(44.65, -110.45, "Ursus arctos")
        self.assertLess(max(p["probability"] for p in thin_points), max(p["probability"] for p in strong_points))


class LiveEvidenceConcurrencyTests(unittest.TestCase):
    """
    Regression coverage for a real 2026-09-17 production incident:
    `_local_evidence_cached` used to fetch GBIF, iNaturalist, and (for Aves
    species) eBird one after another, so their timeouts SUMMED. That was
    fine with just GBIF + iNaturalist, but the moment EBIRD_API_KEY was
    first configured in production, a single "Explore Anywhere" request for
    one bird species could exceed gunicorn's default 30s worker timeout and
    Render's proxy returned a bare 502 -- confirmed live against
    wildcast.onrender.com. The fix fetches all sources concurrently, so one
    species' worst-case wait is bounded by its single slowest source, not
    the sum of all of them.
    """

    def setUp(self):
        ps._presence_cache.clear()

    def test_sources_are_fetched_concurrently_not_sequentially(self):
        start_times = {}
        lock = threading.Lock()

        def _make_source(name, value):
            def _fn(*args, **kwargs):
                with lock:
                    start_times[name] = time.monotonic()
                time.sleep(0.2)
                return value
            return _fn

        with (
            patch("app.services.gbif_client.presence_count", side_effect=_make_source("gbif", 10)),
            patch("app.services.inaturalist_client.presence_count", side_effect=_make_source("inaturalist", 5)),
            patch("app.services.ebird_client.presence_count", side_effect=_make_source("ebird", 3)),
            patch("app.services.ebird_client.is_configured", return_value=True),
        ):
            t0 = time.monotonic()
            evidence = ps._local_evidence_cached("Corvus corax", 44.6, -110.5, 60, "Aves")
            elapsed = time.monotonic() - t0

        # If these ran sequentially, the three start times would be spread
        # ~0.2s apart (one source's whole sleep duration) and the call
        # would take ~0.6s total. Run concurrently, all three should start
        # within a few ms of each other and the whole call should take
        # roughly one sleep's worth of time, not the sum of three.
        self.assertEqual(set(start_times), {"gbif", "inaturalist", "ebird"})
        spread = max(start_times.values()) - min(start_times.values())
        self.assertLess(spread, 0.1)
        self.assertLess(elapsed, 0.4)
        self.assertEqual(evidence["sources"], {"gbif": 10, "inaturalist": 5, "ebird": 3})

    def test_ebird_receives_the_short_interactive_timeout_not_ebirds_own_default(self):
        seen_kwargs = {}

        def _fake_ebird(*args, **kwargs):
            seen_kwargs.update(kwargs)
            return 0

        with (
            patch("app.services.gbif_client.presence_count", return_value=0),
            patch("app.services.inaturalist_client.presence_count", return_value=0),
            patch("app.services.ebird_client.presence_count", side_effect=_fake_ebird),
            patch("app.services.ebird_client.is_configured", return_value=True),
        ):
            ps._local_evidence_cached("Corvus corax", 44.6, -110.5, 60, "Aves")

        self.assertEqual(seen_kwargs.get("timeout"), ps._ANYWHERE_PRESENCE_TIMEOUT_S)

    def test_ebird_is_skipped_entirely_for_non_aves_species(self):
        with (
            patch("app.services.gbif_client.presence_count", return_value=5),
            patch("app.services.inaturalist_client.presence_count", return_value=0),
            patch("app.services.ebird_client.presence_count") as mock_ebird,
            patch("app.services.ebird_client.is_configured", return_value=True),
        ):
            evidence = ps._local_evidence_cached("Ursus arctos", 44.6, -110.5, 60, "Mammalia")

        mock_ebird.assert_not_called()
        self.assertNotIn("ebird", evidence["sources"])


if __name__ == "__main__":
    unittest.main()

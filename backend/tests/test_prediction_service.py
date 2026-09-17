"""
Tests for app.ml.prediction_service. These are integration-style: they use
the trained model + cached data that ship in the repo (backend/app/ml/artifacts
and backend/data/cache), rather than mocking, because the whole point is to
catch problems in how the pieces fit together -- exactly the class of bug
(a collapsed/inverted calibration) that unit tests of the individual pieces
would not have caught. If these fail, retrain first: `python -m app.ml.train`.
"""
import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

from app.ml import prediction_service as ps

ALL_PILOT_AREA_IDS = {"yellowstone", "kruger", "brisbane_urban", "svalbard_arctic", "amazon_rainforest"}


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


class AnywhereModeTests(unittest.TestCase):
    """
    'Explore Anywhere' mode (species_for_location / predict_at_location /
    best_window_at_location) is tested with GBIF and live weather mocked
    out -- the point of these tests is the wiring (does a live presence
    result correctly gate the candidate list? does an arbitrary point still
    produce valid, correctly-labeled predictions?), not GBIF's or
    Open-Meteo's actual live data, which the *_client modules' own tests
    and the manual Render smoke test already cover.
    """

    def setUp(self):
        # The live-presence cache is process-global (by design -- see its
        # docstring) so it must not leak results between tests that reuse
        # the same coordinates with different mocked GBIF answers.
        ps._presence_cache.clear()

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
        with patch("app.services.gbif_client.has_any_presence", side_effect=lambda name, *a, **k: name in confirmed):
            results = ps.species_for_location(44.6, -110.5)
        names = {s["scientific_name"] for s in results}
        self.assertEqual(names, confirmed)
        for s in results:
            self.assertGreater(s["training_support"], 0)

    def test_species_for_location_excludes_everything_on_gbif_outage(self):
        # A live API failure must fail closed (nothing offered), never open
        # (everything offered) -- an empty/erroring GBIF response is not
        # evidence a species is plausible somewhere.
        with patch("app.services.gbif_client.has_any_presence", side_effect=RuntimeError("network down")):
            results = ps.species_for_location(44.6, -110.5)
        self.assertEqual(results, [])

    def test_predict_at_location_labels_the_nearest_area_and_distance(self):
        with (
            patch("app.services.gbif_client.has_any_presence", return_value=True),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            result = ps.predict_at_location(44.65, -110.45, "2027-06-15")
        self.assertEqual(result["nearest_area"]["id"], "yellowstone")
        self.assertGreaterEqual(result["nearest_area"]["distance_km"], 0)
        self.assertGreater(len(result["predictions"]), 0)
        for p in result["predictions"]:
            self.assertGreaterEqual(p["probability"], 0.0)
            self.assertLessEqual(p["probability"], 1.0)
            self.assertTrue(any("Verified live against GBIF" in f for f in p["factors"]))

    def test_predict_at_location_with_no_gbif_matches_returns_empty_not_error(self):
        # A real point with genuinely no curated species nearby (e.g. open
        # ocean) is a valid, non-error result -- just an empty list.
        with (
            patch("app.services.gbif_client.has_any_presence", return_value=False),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            result = ps.predict_at_location(0.0, -140.0, "2027-06-15")
        self.assertEqual(result["predictions"], [])

    def test_predict_at_location_species_filter_raises_if_not_confirmed_nearby(self):
        with (
            patch("app.services.gbif_client.has_any_presence", return_value=False),
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
            patch("app.services.gbif_client.has_any_presence", return_value=True),
            patch("app.services.weather_client.climate_normals", side_effect=RuntimeError("network down")),
        ):
            with self.assertRaises(ps.LiveDataUnavailable):
                ps.predict_at_location(44.65, -110.45, "2027-06-15")

    def test_best_window_at_location_returns_points_across_the_year(self):
        with (
            patch("app.services.gbif_client.has_any_presence", return_value=True),
            patch("app.services.weather_client.climate_normals", return_value=_synthetic_normals()),
        ):
            points = ps.best_window_at_location(44.65, -110.45, "Ursus arctos")
        self.assertGreaterEqual(len(points), 70)
        for p in points:
            self.assertGreaterEqual(p["probability"], 0.0)
            self.assertLessEqual(p["probability"], 1.0)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for app.ml.features. Pure functions, no network, no trained
model needed -- these should run anywhere in well under a second.
"""
import math
import unittest

import pandas as pd

from app.ml.features import ALL_FEATURE_COLUMNS, build_feature_frame, categorical_feature_mask


def _sample_row(**overrides):
    row = {
        "species": "Ursus arctos",
        "taxon_class": "Mammalia",
        "area_id": "yellowstone",
        "day_of_year": 166,
        "temperature_2m_max": 25.0,
        "temperature_2m_min": 10.0,
        "precipitation_sum": 0.0,
        "windspeed_10m_max": 10.0,
        "cloudcover_mean": 20.0,
        "temperature_2m_max_normal_mean": 25.0,
        "temperature_2m_max_normal_std": 5.0,
        "temperature_2m_min_normal_mean": 10.0,
        "temperature_2m_min_normal_std": 4.0,
        "precipitation_sum_normal_mean": 2.0,
        "precipitation_sum_normal_std": 1.0,
        "windspeed_10m_max_normal_mean": 12.0,
        "windspeed_10m_max_normal_std": 3.0,
        "cloudcover_mean_normal_mean": 40.0,
        "cloudcover_mean_normal_std": 10.0,
    }
    row.update(overrides)
    return row


class CyclicalDayOfYearTests(unittest.TestCase):
    def test_day_1_and_day_365_are_close_on_the_circle(self):
        df = pd.DataFrame([_sample_row(day_of_year=1), _sample_row(day_of_year=365)])
        feat = build_feature_frame(df)
        d1 = (feat.iloc[0]["doy_sin"], feat.iloc[0]["doy_cos"])
        d365 = (feat.iloc[1]["doy_sin"], feat.iloc[1]["doy_cos"])
        euclidean = math.dist(d1, d365)
        # Adjacent calendar days must land close together on the unit circle,
        # nowhere near as far apart as the raw integers (1 vs 365) suggest.
        self.assertLess(euclidean, 0.05)

    def test_day_1_and_day_183_are_far_apart(self):
        df = pd.DataFrame([_sample_row(day_of_year=1), _sample_row(day_of_year=183)])
        feat = build_feature_frame(df)
        d1 = (feat.iloc[0]["doy_sin"], feat.iloc[0]["doy_cos"])
        d183 = (feat.iloc[1]["doy_sin"], feat.iloc[1]["doy_cos"])
        self.assertGreater(math.dist(d1, d183), 1.8)  # opposite sides of the year


class BuildFeatureFrameTests(unittest.TestCase):
    def test_output_has_exactly_the_declared_columns(self):
        df = pd.DataFrame([_sample_row()])
        feat = build_feature_frame(df)
        self.assertEqual(list(feat.columns), ALL_FEATURE_COLUMNS)

    def test_categorical_columns_get_category_dtype(self):
        df = pd.DataFrame([_sample_row()])
        feat = build_feature_frame(df)
        for col in ("species", "taxon_class", "area_id"):
            self.assertEqual(str(feat[col].dtype), "category")

    def test_lat_lon_are_not_model_features(self):
        # Deliberate design choice (see features.py's module comment) --
        # guard against it being silently reintroduced.
        self.assertNotIn("lat", ALL_FEATURE_COLUMNS)
        self.assertNotIn("lon", ALL_FEATURE_COLUMNS)

    def test_anomaly_is_zero_when_value_equals_normal_mean(self):
        df = pd.DataFrame([_sample_row(temperature_2m_max=25.0, temperature_2m_max_normal_mean=25.0)])
        feat = build_feature_frame(df)
        self.assertAlmostEqual(feat.iloc[0]["temperature_2m_max_anomaly"], 0.0)

    def test_anomaly_sign_matches_direction_of_deviation(self):
        hot = pd.DataFrame([_sample_row(temperature_2m_max=35.0, temperature_2m_max_normal_mean=25.0,
                                         temperature_2m_max_normal_std=5.0)])
        cold = pd.DataFrame([_sample_row(temperature_2m_max=15.0, temperature_2m_max_normal_mean=25.0,
                                          temperature_2m_max_normal_std=5.0)])
        hot_feat = build_feature_frame(hot)
        cold_feat = build_feature_frame(cold)
        self.assertGreater(hot_feat.iloc[0]["temperature_2m_max_anomaly"], 0)
        self.assertLess(cold_feat.iloc[0]["temperature_2m_max_anomaly"], 0)
        self.assertAlmostEqual(hot_feat.iloc[0]["temperature_2m_max_anomaly"], 2.0)  # (35-25)/5
        self.assertAlmostEqual(cold_feat.iloc[0]["temperature_2m_max_anomaly"], -2.0)

    def test_zero_std_does_not_raise_divide_by_zero(self):
        df = pd.DataFrame([_sample_row(precipitation_sum_normal_std=0.0)])
        feat = build_feature_frame(df)  # must not raise / produce inf or NaN
        self.assertTrue(math.isfinite(feat.iloc[0]["precipitation_sum_anomaly"]))


class CategoricalFeatureMaskTests(unittest.TestCase):
    def test_mask_matches_column_order(self):
        df = pd.DataFrame([_sample_row()])
        feat = build_feature_frame(df)
        mask = categorical_feature_mask(feat)
        expected = [str(feat[c].dtype) == "category" for c in feat.columns]
        self.assertEqual(mask, expected)
        # species, taxon_class, area_id are the first three declared columns.
        self.assertEqual(mask[:3], [True, True, True])
        self.assertNotIn(True, mask[3:])


if __name__ == "__main__":
    unittest.main()

"""
Shared feature engineering -- used at BOTH training time and inference time
so the two can never drift apart (a classic source of silent accuracy loss
in ML systems: train/serve skew).
"""
from __future__ import annotations

import math

import pandas as pd

from app.config import WEATHER_DAILY_VARS

CATEGORICAL_COLUMNS = ["species", "taxon_class", "area_id"]

NUMERIC_WEATHER_COLUMNS = WEATHER_DAILY_VARS + [f"{v}_anomaly" for v in WEATHER_DAILY_VARS]

ALL_FEATURE_COLUMNS = (
    CATEGORICAL_COLUMNS
    + ["doy_sin", "doy_cos"]
    + NUMERIC_WEATHER_COLUMNS
)
# Note: raw continuous lat/lon are deliberately NOT model features at Phase 0
# scale. With only 3 pilot areas, a record's lat/lon is a near-constant
# (area center +/- a few tens of km of jitter) -- a high-cardinality
# continuous column that mostly gives a tree model something to overfit to
# rather than anything to generalize from; `area_id` already captures
# location at the resolution this dataset supports. Reintroduce lat/lon as
# a real feature once the catalog spans many distinct locations (Phase 2),
# where it starts carrying actual cross-area climate/geography signal.


def add_cyclical_day_of_year(df: pd.DataFrame, day_of_year_col: str = "day_of_year") -> pd.DataFrame:
    """
    Day-of-year as raw integer (1-365) is a terrible ML feature: Dec 31 and
    Jan 1 are a day apart in reality but 364 apart numerically. Encoding it
    on a circle fixes that.
    """
    df = df.copy()
    radians = 2 * math.pi * df[day_of_year_col] / 365.0
    df["doy_sin"] = radians.apply(math.sin)
    df["doy_cos"] = radians.apply(math.cos)
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a dataframe that already has species/taxon_class/area_id/lat/lon,
    a day_of_year column, raw weather columns, and <var>_normal_mean /
    <var>_normal_std columns, and return the final model-ready feature
    frame (categorical dtypes set so sklearn's native categorical support
    picks them up without one-hot encoding).
    """
    df = add_cyclical_day_of_year(df)
    for var in WEATHER_DAILY_VARS:
        mean_col, std_col, anomaly_col = f"{var}_normal_mean", f"{var}_normal_std", f"{var}_anomaly"
        if anomaly_col not in df.columns:
            std = df[std_col].replace(0, 1e-6)
            df[anomaly_col] = (df[var] - df[mean_col]) / std
    out = df[ALL_FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype("category")
    return out


def categorical_feature_mask(frame: pd.DataFrame) -> list[bool]:
    """Boolean mask (column order of `frame`) for sklearn's `categorical_features=`."""
    return [str(frame[c].dtype) == "category" for c in frame.columns]

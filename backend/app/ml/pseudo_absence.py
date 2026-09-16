"""
Target-group background sampling for presence-only data.

GBIF/iNaturalist records tell us where a species WAS seen, never where
someone looked and did NOT see it -- so we can't train a classifier on
presence records alone (it would just learn "where do people submit
records", which correlates with population density, not wildlife).

The standard fix (Phillips et al. 2009; this is what most modern species
distribution models, including eBird's, use): for a target species, treat
observations of OTHER species from the SAME taxonomic group, area and
season as implicit "someone was out looking here" background points. This
cancels out most of the observer-effort bias that a species' own presence
records alone would carry.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from app.config import RANDOM_SEED


def _to_date(x) -> dt.date:
    return pd.to_datetime(x).date()


def generate_pseudo_absences(
    occurrences: pd.DataFrame,
    *,
    ratio: float = 1.5,
    min_gap_days: int = 5,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    `occurrences` must have columns: species, taxon_class, area_id, lat, lon, date
    (one row per real occurrence record, across all species/areas).

    Returns a dataframe of the same shape representing pseudo-absences: for
    each species, dates/locations drawn from OTHER species in the same
    area + taxon_class, excluded if within `min_gap_days` of a real record
    of the target species (to avoid mislabeling a true presence as absent).
    """
    rng = np.random.default_rng(seed)
    occurrences = occurrences.copy()
    occurrences["date"] = pd.to_datetime(occurrences["date"])

    rows = []
    for (area_id, taxon_class), group in occurrences.groupby(["area_id", "taxon_class"]):
        species_list = group["species"].unique()
        for species in species_list:
            own = group[group["species"] == species]
            own_dates = set(own["date"].dt.date)
            background_pool = group[group["species"] != species]
            if background_pool.empty:
                continue
            n_positive = len(own)
            n_needed = max(1, int(round(n_positive * ratio)))
            candidates = background_pool.sample(
                n=min(n_needed * 3, len(background_pool)), random_state=int(rng.integers(0, 1_000_000))
            )
            picked = 0
            for _, cand in candidates.iterrows():
                cand_date = cand["date"].date()
                if any(abs((cand_date - d).days) < min_gap_days for d in own_dates):
                    continue
                rows.append(
                    {
                        "species": species,
                        "taxon_class": taxon_class,
                        "area_id": area_id,
                        "lat": cand["lat"],
                        "lon": cand["lon"],
                        "date": cand_date,
                        "label": 0,
                    }
                )
                picked += 1
                if picked >= n_needed:
                    break

    return pd.DataFrame(rows)


def build_labeled_dataset(occurrences: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Positives (label=1, real records) + generated pseudo-absences (label=0), concatenated."""
    positives = occurrences[["species", "taxon_class", "area_id", "lat", "lon", "date"]].copy()
    positives["date"] = pd.to_datetime(positives["date"]).dt.date
    positives["label"] = 1
    negatives = generate_pseudo_absences(occurrences, **kwargs)
    return pd.concat([positives, negatives], ignore_index=True)

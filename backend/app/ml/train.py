"""
Train the WildCast encounter-probability model from cached occurrence +
weather data (populated either by the real scripts/ingest_*.py against
live GBIF/Open-Meteo, or by scripts/generate_sample_fixtures.py for an
offline demo -- both write the exact same cache file schema, so this
script does not know or care which one produced its inputs).

Run as: python -m app.ml.train
"""
from __future__ import annotations

import json
import warnings

# app.config must be imported before joblib/sklearn: it sets LOKY_MAX_CPU_COUNT,
# which must be set before loky's cpu-count detection runs (see the comment
# in app/config.py for why).
from app.config import CACHE_DIR, DATA_DIR, MODEL_DIR, RANDOM_SEED, WEATHER_DAILY_VARS

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.ml.features import ALL_FEATURE_COLUMNS, build_feature_frame, categorical_feature_mask
from app.ml.pseudo_absence import build_labeled_dataset
from app.services import data_source_marker, iucn_client

warnings.filterwarnings("ignore", category=UserWarning)

PILOT_AREAS = json.loads((DATA_DIR / "pilot_areas.json").read_text())
SEED_SPECIES = json.loads((DATA_DIR / "seed_species.json").read_text())


def _make_base_model(categorical_features: list[bool]) -> HistGradientBoostingClassifier:
    """
    Deliberately modest capacity for Phase 0's data volume (thousands, not
    millions, of rows). An earlier version of this script used max_depth=6
    with no regularization: each of the calibration wrapper's CV folds
    overfit so hard (raw probabilities collapsing to ~0 or ~1 per fold) that
    averaging them produced an almost CONSTANT ~0.49 for every input --
    predictions that looked plausible (a real number between 0 and 1) but
    carried no information at all. `early_stopping` + `l2_regularization` +
    a shallower `max_depth` fix that; scale these back up as the real
    dataset (not this Phase 0 seed set) grows into the millions of records.
    """
    return HistGradientBoostingClassifier(
        categorical_features=categorical_features,
        random_state=RANDOM_SEED,
        max_depth=4,
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=2.0,
        min_samples_leaf=40,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.15,
    )


def _load_area_caches() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Concatenate every pilot area's occurrence / daily-weather / climate-normal caches."""
    occ_frames, weather_frames, normal_frames = [], [], []
    for area in PILOT_AREAS:
        area_id = area["id"]
        occ_path = CACHE_DIR / f"occurrences_{area_id}.csv"
        weather_path = CACHE_DIR / f"weather_daily_{area_id}.csv"
        normals_path = CACHE_DIR / f"climate_normals_{area_id}.csv"
        if not (occ_path.exists() and weather_path.exists() and normals_path.exists()):
            raise FileNotFoundError(
                f"Missing cache for '{area_id}'. Run scripts/ingest_gbif.py + "
                f"scripts/ingest_weather.py (needs internet), or "
                f"scripts/generate_sample_fixtures.py for an instant offline demo."
            )
        occ_frames.append(pd.read_csv(occ_path))
        weather_frames.append(pd.read_csv(weather_path))
        normal_frames.append(pd.read_csv(normals_path))
    return pd.concat(occ_frames, ignore_index=True), pd.concat(weather_frames, ignore_index=True), pd.concat(
        normal_frames, ignore_index=True
    )


def _join_weather(dataset: pd.DataFrame, weather: pd.DataFrame, normals: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["day_of_year"] = dataset["date"].dt.dayofyear
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"])

    merged = dataset.merge(weather, on=["area_id", "date"], how="left")
    merged = merged.merge(normals, on=["area_id", "day_of_year"], how="left")
    # A handful of rows can miss a weather join (gap in the archive) -- drop rather than impute silently.
    before = len(merged)
    raw_weather_cols = (
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max", "cloudcover_mean",
    )
    required_cols = [c for c in merged.columns if c.endswith("_mean") or c in raw_weather_cols]
    merged = merged.dropna(subset=required_cols)
    dropped = before - len(merged)
    if dropped:
        print(f"  (dropped {dropped}/{before} rows with no matching weather record)")
    return merged


def _temporal_block_cv(features: pd.DataFrame, labels: pd.Series, years: pd.Series) -> dict:
    """
    Leave-one-year-out AUC/Brier, PLUS the same for a climatology-only
    baseline (season + location, no weather) so we can see how much the
    daily weather signal actually buys us.

    Blocking by YEAR rather than a random row split is what "spatially and
    temporally blocked cross-validation" (design doc, Prediction methodology
    #5) means in practice here: it tests "can the model predict a year of
    dates it has never seen," which is the real deployment scenario, not
    "can it memorize a specific date within a year it has already partly
    seen." (Leave-one-AREA-out was tried first and is a bad fit for this
    seed dataset: each pilot species lives in only one area, so holding out
    an area also holds out every example of that area's species -- there is
    nothing left to generalize from. Blocking by year avoids that trap.)
    """
    raw_weather_prefixes = [
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max", "cloudcover_mean",
    ]
    weather_cols = [c for c in features.columns if any(c.startswith(v) for v in raw_weather_prefixes)]
    baseline_cols = [c for c in features.columns if c not in weather_cols]

    results = {"full_model": [], "climatology_baseline": []}
    for held_out in sorted(years.unique()):
        train_idx = years != held_out
        test_idx = years == held_out
        if labels[test_idx].nunique() < 2 or labels[train_idx].nunique() < 2:
            continue  # can't score AUC on a fold with only one class present

        for name, cols in (("full_model", features.columns), ("climatology_baseline", baseline_cols)):
            cat_mask = categorical_feature_mask(features[cols])
            clf = _make_base_model(cat_mask)
            clf.fit(features.loc[train_idx, cols], labels[train_idx])
            proba = clf.predict_proba(features.loc[test_idx, cols])[:, 1]
            auc = roc_auc_score(labels[test_idx], proba)
            brier = brier_score_loss(labels[test_idx], proba)
            results[name].append({"held_out_year": int(held_out), "auc": auc, "brier": brier})
    return results


def _compute_species_seasonality(calibrated: CalibratedClassifierCV, normals: pd.DataFrame) -> dict:
    """
    For every seed species, run the trained model over its home area's full
    366-day climate-normal curve (weather = that day's normal, so anomaly=0
    -- i.e. "an average day"), and record which day-of-year scores highest
    and lowest. This is what powers the plain-language "why" explanations
    in prediction_service.predict() (e.g. "September is this species' peak
    season" / "near its lowest-activity time of year") without a live call:
    it's computed once here and shipped in the model bundle.
    """
    rows = []
    for sp in SEED_SPECIES:
        sub = normals[normals["area_id"] == sp["area_id"]].copy()
        if sub.empty:
            continue
        sub["species"] = sp["scientific_name"]
        sub["taxon_class"] = sp["taxon_class"]
        for var in WEATHER_DAILY_VARS:
            sub[var] = sub[f"{var}_normal_mean"]  # anomaly=0: an "average" day
        rows.append(sub)
    if not rows:
        return {}
    combined = pd.concat(rows, ignore_index=True)
    features = build_feature_frame(combined)
    combined = combined.assign(probability=calibrated.predict_proba(features)[:, 1])

    seasonality = {}
    for species, grp in combined.groupby("species"):
        peak = grp.loc[grp["probability"].idxmax()]
        trough = grp.loc[grp["probability"].idxmin()]
        seasonality[species] = {
            "peak_doy": int(peak["day_of_year"]),
            "peak_probability": round(float(peak["probability"]), 4),
            "trough_doy": int(trough["day_of_year"]),
            "trough_probability": round(float(trough["probability"]), 4),
        }
    return seasonality


# IUCN Red List category codes -> plain-language labels, for the factor text
# in prediction_service._explain(). Codes per IUCN's own standard categories.
_IUCN_CATEGORY_LABELS = {
    "EX": "Extinct",
    "EW": "Extinct in the Wild",
    "CR": "Critically Endangered",
    "EN": "Endangered",
    "VU": "Vulnerable",
    "NT": "Near Threatened",
    "LC": "Least Concern",
    "DD": "Data Deficient",
    "NE": "Not Evaluated",
}


def _compute_conservation_status() -> dict:
    """
    Real IUCN Red List category + population trend per seed species, fetched
    once here (not per-request -- see iucn_client.py's docstring for why)
    and shipped in the model bundle. Skipped entirely, returning {}, when
    IUCN_API_KEY is unset -- this is optional enrichment, the same pattern
    eBird already uses (app.services.ebird_client.is_configured()). A
    species IUCN has no findable assessment for is simply left out of the
    dict rather than guessed at.
    """
    if not iucn_client.is_configured():
        print("  IUCN_API_KEY not set -- skipping conservation-status enrichment (optional; see .env.example).")
        return {}
    status = {}
    for sp in SEED_SPECIES:
        name = sp["scientific_name"]
        try:
            assessment = iucn_client.species_assessment(name)
        except Exception as exc:
            print(f"  IUCN lookup failed for {name}, skipping: {exc}")
            continue
        if not assessment or not assessment.get("category"):
            continue
        status[name] = {
            **assessment,
            "category_label": _IUCN_CATEGORY_LABELS.get(assessment["category"], assessment["category"]),
        }
    return status


def train() -> dict:
    print("Loading cached occurrence + weather data...")
    occ, weather, normals = _load_area_caches()
    occ = occ.rename(columns={"eventDate": "date"}) if "eventDate" in occ.columns else occ
    occ["date"] = pd.to_datetime(occ["date"]).dt.date.astype(str)

    print(f"  {len(occ)} occurrence records across {occ['area_id'].nunique()} areas, "
          f"{occ['species'].nunique()} species")

    print("Generating target-group pseudo-absences...")
    dataset = build_labeled_dataset(occ)
    print(f"  {int((dataset.label == 1).sum())} positive, {int((dataset.label == 0).sum())} pseudo-absence rows")

    print("Joining daily weather + climate normals...")
    dataset = _join_weather(dataset, weather, normals)

    print("Building model features...")
    features = build_feature_frame(dataset)
    labels = dataset["label"].astype(int).reset_index(drop=True)
    features = features.reset_index(drop=True)
    years = pd.to_datetime(dataset["date"]).dt.year.reset_index(drop=True)

    print("Temporally-blocked cross-validation (leave-one-year-out)...")
    cv_results = _temporal_block_cv(features, labels, years)
    for name, folds in cv_results.items():
        if folds:
            mean_auc = float(np.mean([f["auc"] for f in folds]))
            mean_brier = float(np.mean([f["brier"] for f in folds]))
            print(f"  {name}: mean AUC={mean_auc:.3f}, mean Brier={mean_brier:.3f} over {len(folds)} folds")

    print("Training final calibrated model on all data...")
    cat_mask = categorical_feature_mask(features)
    base_clf = _make_base_model(cat_mask)
    # Sigmoid (Platt) calibration rather than isotonic: isotonic regression
    # needs more held-out points per fold than this Phase 0 dataset gives it
    # per species to fit a stable monotonic curve, and degrades toward the
    # same collapse described in _make_base_model's docstring. Revisit
    # isotonic once training data is much larger (Phase 1/2).
    #
    # shuffle=True matters here: `dataset` is built area-by-area, species-by-
    # species (see _load_area_caches / build_labeled_dataset), so it is
    # nowhere close to IID-ordered. An unshuffled KFold split (the default)
    # hands some folds a training slice that under-represents whole species
    # or areas, occasionally producing one wildly miscalibrated fold that
    # then skews the whole ensemble's average -- caught by manually probing
    # a single species' predicted probability across months and finding it
    # came out with summer LOWER than winter for a species that should
    # clearly peak in summer (see project notes / commit history).
    cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv=cv_splitter)
    calibrated.fit(features, labels)

    # Data-density indicator per species: count real/positive occurrence
    # records only (not pseudo-absences) -- this is what should drive a
    # user-facing "low data" confidence flag.
    support = dataset[dataset["label"] == 1].groupby("species").size().to_dict()

    print("Computing per-species seasonal peak/trough (for the 'why' explanations)...")
    seasonality = _compute_species_seasonality(calibrated, normals)

    print("Fetching real IUCN Red List conservation status (optional enrichment)...")
    conservation_status = _compute_conservation_status()
    if conservation_status:
        print(f"  got real conservation status for {len(conservation_status)}/{len(SEED_SPECIES)} species")

    data_source = data_source_marker.read()
    print(f"Data source for this training run: {data_source['source']} "
          f"(see backend/data/cache/data_source.json)")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": calibrated,
            "feature_columns": ALL_FEATURE_COLUMNS,
            "categorical_columns": [c for c in features.columns if str(features[c].dtype) == "category"],
            "species_support": support,
            "species_seasonality": seasonality,
            "species_conservation_status": conservation_status,
            "areas": {a["id"]: a for a in PILOT_AREAS},
            "data_source": data_source,
        },
        MODEL_DIR / "model.joblib",
    )

    metrics = {
        "n_rows": len(dataset),
        "n_positive": int((labels == 1).sum()),
        "n_pseudo_absence": int((labels == 0).sum()),
        "cv": cv_results,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Saved model to {MODEL_DIR / 'model.joblib'} and metrics to {MODEL_DIR / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    train()

"""
Loads the trained model once and answers prediction requests. This is the
one place request-time code and training-time code meet, so it reuses
`app.ml.features.build_feature_frame` for both -- the train/serve parity
guarantee described in that module's docstring.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from functools import lru_cache

import joblib
import pandas as pd

from app.config import CACHE_DIR, DATA_DIR, MODEL_DIR, WEATHER_DAILY_VARS
from app.ml.features import build_feature_frame
from app.services import weather_client

log = logging.getLogger("wildcast.prediction")

SEED_SPECIES = json.loads((DATA_DIR / "seed_species.json").read_text())
SPECIES_BY_KEY = {s["scientific_name"]: s for s in SEED_SPECIES}


class ModelNotTrained(RuntimeError):
    pass


# Human-readable labels for the "why" explanations: which direction is
# "unusual" for each weather variable, in plain language.
_WEATHER_VAR_META = {
    "temperature_2m_max": {"name": "temperature", "hi": "warmer", "lo": "cooler"},
    "precipitation_sum": {"name": "rainfall", "hi": "wetter", "lo": "drier"},
    "windspeed_10m_max": {"name": "wind", "hi": "windier", "lo": "calmer"},
    "cloudcover_mean": {"name": "cloud cover", "hi": "cloudier", "lo": "clearer"},
}


def _month_label(day_of_year: int) -> str:
    """'around <Month>' label for a day-of-year. A non-leap reference year is
    fine here -- this is a coarse, human-facing label, not a date computation."""
    date = dt.date(2023, 1, 1) + dt.timedelta(days=(day_of_year - 1) % 365)
    return date.strftime("%B")


def _circular_day_distance(a: int, b: int, period: int = 365) -> int:
    diff = abs(a - b) % period
    return min(diff, period - diff)


def _explain(
    species_key: str,
    target_date: dt.date,
    weather_row: dict,
    seasonality: dict,
    support: int,
    confidence: str,
) -> list[str]:
    """
    2-3 short, plain-language reasons behind a prediction: data support,
    seasonal timing, and the day's most unusual weather factor (if any).
    This is Phase 0's version of the design doc's Product & UX "why"
    explanation text -- deliberately simple (no free-text LLM generation,
    so it's fast, deterministic, and always traceable to a concrete number).
    """
    factors = []

    if support >= 150:
        factors.append(f"High confidence: {support} historical sighting records for this species in this area.")
    elif support >= 40:
        factors.append(f"Medium confidence: {support} historical sighting records for this species in this area.")
    else:
        factors.append(f"Low confidence: only {support} historical sighting records for this species in this area.")

    season = seasonality.get(species_key)
    if season:
        doy = target_date.timetuple().tm_yday
        dist_peak = _circular_day_distance(doy, season["peak_doy"])
        dist_trough = _circular_day_distance(doy, season["trough_doy"])
        peak_month = _month_label(season["peak_doy"])
        trough_month = _month_label(season["trough_doy"])
        if dist_peak <= 35 and dist_peak <= dist_trough:
            factors.append(f"Near this species' peak season here (historically strongest around {peak_month}).")
        elif dist_trough <= 35 and dist_trough < dist_peak:
            factors.append(f"Near this species' quietest time of year here (weakest around {trough_month}).")
        else:
            factors.append(f"Shoulder season here -- peak is around {peak_month}, quietest around {trough_month}.")

    best_var, best_z = None, 0.0
    for var in _WEATHER_VAR_META:
        z = weather_row.get(f"{var}_anomaly") or 0.0
        if abs(z) > abs(best_z):
            best_var, best_z = var, z
    if best_var and abs(best_z) >= 0.75:
        meta = _WEATHER_VAR_META[best_var]
        word = meta["hi"] if best_z > 0 else meta["lo"]
        factors.append(f"Today looks {word} than typical for this date (unusual {meta['name']}).")

    return factors


@lru_cache(maxsize=1)
def _load_model() -> dict:
    path = MODEL_DIR / "model.joblib"
    if not path.exists():
        raise ModelNotTrained(
            "No trained model found. Run `python -m scripts.generate_sample_fixtures` (offline demo data) "
            "or the real scripts/ingest_gbif.py + scripts/ingest_weather.py, then `python -m app.ml.train`."
        )
    return joblib.load(path)


@lru_cache(maxsize=8)
def _load_climate_normals(area_id: str) -> pd.DataFrame:
    path = CACHE_DIR / f"climate_normals_{area_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No cached climate normals for '{area_id}' -- run the ingestion/fixture scripts.")
    return pd.read_csv(path)


def list_areas() -> list[dict]:
    return list(_load_model()["areas"].values())


def get_area(area_id: str) -> dict:
    areas = _load_model()["areas"]
    if area_id not in areas:
        raise KeyError(f"Unknown area_id '{area_id}'. Known areas: {list(areas)}")
    return areas[area_id]


def species_for_area(area_id: str) -> list[dict]:
    """
    The plausibility-gated candidate list for an area: only species the
    model actually has training support for there. This is Phase 0's
    concrete version of the design doc's plausibility gate -- every species
    ever offered for an area is one WildCast has real (or, in offline demo
    mode, clearly-labeled synthetic) evidence for in that area; nothing is
    ever guessed into the list. Phase 1 upgrades this to a live per-request
    GBIF presence check (see app.services.gbif_client.has_any_presence) so
    new areas don't need a curated seed list.
    """
    get_area(area_id)  # raises if unknown
    support = _load_model()["species_support"]
    return [
        {**s, "training_support": support.get(s["scientific_name"], 0)}
        for s in SEED_SPECIES
        if s["area_id"] == area_id
    ]


def _normal_row_for_date(area_id: str, date: dt.date) -> dict:
    """Climate-normal baseline only (no live call) -- the always-available fallback."""
    normals = _load_climate_normals(area_id)
    day_of_year = date.timetuple().tm_yday
    normal_row = normals[normals["day_of_year"] == day_of_year]
    row: dict = {"day_of_year": day_of_year}
    for var in WEATHER_DAILY_VARS:
        mean = float(normal_row[f"{var}_normal_mean"].iloc[0]) if len(normal_row) else 0.0
        std = float(normal_row[f"{var}_normal_std"].iloc[0]) if len(normal_row) else 1.0
        row[f"{var}_normal_mean"] = mean
        row[f"{var}_normal_std"] = std
        row[var] = mean  # no anomaly unless a live observation overrides it
        row[f"{var}_anomaly"] = 0.0
    return row


def _apply_observation(row: dict, obs_row: "pd.Series | None") -> dict:
    if obs_row is None:
        return row
    for var in WEATHER_DAILY_VARS:
        if var in obs_row and pd.notna(obs_row[var]):
            value = float(obs_row[var])
            std = max(row[f"{var}_normal_std"], 1e-6)
            row[var] = value
            row[f"{var}_anomaly"] = (value - row[f"{var}_normal_mean"]) / std
    return row


def _weather_features_for_date(area: dict, date: dt.date) -> dict:
    """
    Cached climate-normal baseline + a best-effort live observation/forecast
    for the anomaly term, falling back gracefully (anomaly=0) when the date
    is outside the live archive/forecast window or the network call fails --
    which also covers this being run somewhere with restricted egress. For a
    single date; see `_weather_features_for_dates` for the batched version
    best_window() uses (one live call for the whole date range instead of
    one per date).
    """
    area_id = area.get("area_id", area.get("id"))
    row = _normal_row_for_date(area_id, date)
    today = dt.date.today()
    horizon = (date - today).days
    try:
        if -30 <= horizon <= 0:
            obs = weather_client.historical_daily(area["lat"], area["lon"], date.isoformat(), date.isoformat())
        elif 0 < horizon <= 16:
            obs = weather_client.forecast_daily(area["lat"], area["lon"], days=horizon + 1)
            obs = obs[obs["date"] == date.isoformat()]
        else:
            obs = pd.DataFrame()
        if not obs.empty:
            row = _apply_observation(row, obs.iloc[0])
    except Exception as exc:  # live weather is best-effort; the climatological normal is always a valid fallback
        log.warning("Live weather fetch failed (%s); falling back to the climate normal for day %s.", exc, row["day_of_year"])
    return row


def _weather_features_for_dates(area: dict, dates: list[dt.date]) -> list[dict]:
    """
    Batched version of `_weather_features_for_date`: ONE historical-archive
    call and ONE forecast call cover every date in `dates` that falls in
    their respective windows, instead of one live call per date. Used by
    best_window(), which asks about ~70 dates spanning a year at once.
    """
    area_id = area.get("area_id", area.get("id"))
    rows = {d: _normal_row_for_date(area_id, d) for d in dates}

    today = dt.date.today()
    hist_dates = sorted(d for d in dates if -30 <= (d - today).days <= 0)
    fcst_dates = sorted(d for d in dates if 0 < (d - today).days <= 16)

    try:
        if hist_dates:
            obs = weather_client.historical_daily(area["lat"], area["lon"], hist_dates[0].isoformat(), hist_dates[-1].isoformat())
            obs_by_date = {row_["date"]: row_ for _, row_ in obs.iterrows()} if not obs.empty else {}
            for d in hist_dates:
                if d.isoformat() in obs_by_date:
                    rows[d] = _apply_observation(rows[d], obs_by_date[d.isoformat()])
        if fcst_dates:
            obs = weather_client.forecast_daily(area["lat"], area["lon"], days=16)
            obs_by_date = {row_["date"]: row_ for _, row_ in obs.iterrows()} if not obs.empty else {}
            for d in fcst_dates:
                if d.isoformat() in obs_by_date:
                    rows[d] = _apply_observation(rows[d], obs_by_date[d.isoformat()])
    except Exception as exc:
        log.warning("Batched live weather fetch failed (%s); falling back to climate normals.", exc)

    return [rows[d] for d in dates]


def predict(area_id: str, date: str, species_key: str | None = None) -> list[dict]:
    """
    Ranked encounter-probability predictions for every plausible species in
    `area_id` on `date` ('YYYY-MM-DD'), or just `species_key` if given.
    """
    bundle = _load_model()
    area = get_area(area_id)
    target_date = dt.date.fromisoformat(date)
    weather_row = _weather_features_for_date({**area, "area_id": area_id}, target_date)

    candidates = species_for_area(area_id)
    if species_key:
        candidates = [c for c in candidates if c["scientific_name"] == species_key]
        if not candidates:
            raise KeyError(f"'{species_key}' has no training support in area '{area_id}' (plausibility gate).")

    rows = []
    for sp in candidates:
        rows.append(
            {
                "species": sp["scientific_name"],
                "taxon_class": sp["taxon_class"],
                "area_id": area_id,
                "lat": area["lat"],
                "lon": area["lon"],
                **weather_row,
            }
        )
    frame = pd.DataFrame(rows)
    features = build_feature_frame(frame)
    proba = bundle["model"].predict_proba(features)[:, 1]

    seasonality = bundle.get("species_seasonality", {})
    results = []
    for sp, p in zip(candidates, proba):
        support = sp["training_support"]
        confidence = "high" if support >= 150 else "medium" if support >= 40 else "low"
        factors = _explain(sp["scientific_name"], target_date, weather_row, seasonality, support, confidence)
        results.append(
            {
                "scientific_name": sp["scientific_name"],
                "common_name": sp["common_name"],
                "taxon_class": sp["taxon_class"],
                "probability": round(float(p), 4),
                "confidence": confidence,
                "training_records": support,
                "factors": factors,
            }
        )
    results.sort(key=lambda r: r["probability"], reverse=True)
    return results


def best_window(area_id: str, species_key: str) -> list[dict]:
    """
    One probability per day-of-year (sampled every 5 days) for the
    best-window planner -- built as ONE batched model call rather than ~73
    separate `predict()` calls. The naive per-date loop was measured at
    ~5 seconds for a year's worth of points (mostly per-call DataFrame and
    10-fold-calibrated-model overhead, paid 73 times over); batching pays
    that fixed cost once and lets sklearn vectorize across rows.
    """
    bundle = _load_model()
    area = get_area(area_id)
    candidates = [c for c in species_for_area(area_id) if c["scientific_name"] == species_key]
    if not candidates:
        raise KeyError(f"'{species_key}' has no training support in area '{area_id}' (plausibility gate).")
    sp = candidates[0]

    today = dt.date.today()
    dates = [dt.date(today.year, 1, 1) + dt.timedelta(days=offset) for offset in range(0, 366, 5)]
    weather_rows = _weather_features_for_dates({**area, "area_id": area_id}, dates)

    rows = []
    for d, weather_row in zip(dates, weather_rows):
        rows.append(
            {
                "species": sp["scientific_name"],
                "taxon_class": sp["taxon_class"],
                "area_id": area_id,
                **weather_row,
            }
        )
    frame = pd.DataFrame(rows)
    features = build_feature_frame(frame)
    proba = bundle["model"].predict_proba(features)[:, 1]

    return [
        {"date": d.isoformat(), "day_of_year": i * 5 + 1, "probability": round(float(p), 4)}
        for i, (d, p) in enumerate(zip(dates, proba))
    ]

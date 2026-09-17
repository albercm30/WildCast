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
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# app.config must be imported before joblib: it sets LOKY_MAX_CPU_COUNT,
# which must be set before loky's cpu-count detection runs (see the comment
# in app/config.py for why).
from app.config import CACHE_DIR, DATA_DIR, MODEL_DIR, WEATHER_DAILY_VARS

import joblib
import pandas as pd

from app.ml.features import build_feature_frame
from app.services import ebird_client, gbif_client, inaturalist_client, weather_client

log = logging.getLogger("wildcast.prediction")

# --- "Explore Anywhere" mode -------------------------------------------
# Curated areas (species_for_area) are a static, pre-vetted plausibility
# gate: fast, reliable, zero live calls. "Anywhere" mode trades that for
# coverage: any point on Earth, gated by a live GBIF presence check per
# species instead of a fixed seed list. Same n=50 curated species (no new
# ones are invented), just asked "has this one ever actually been recorded
# near this exact point?" instead of "is this one of the 5 areas' presets?"
_ANYWHERE_DEFAULT_RADIUS_KM = 150.0
_ANYWHERE_MAX_WORKERS = 12
_ANYWHERE_PRESENCE_TIMEOUT_S = 8.0
_PRESENCE_CACHE_TTL_SECONDS = 24 * 60 * 60
_presence_cache: dict[tuple[str, float, float, int], tuple[float, dict]] = {}

# Same breakpoints used everywhere a raw sighting count is turned into a
# high/medium/low confidence label -- curated areas' home-area training
# support (predict()) and "Explore Anywhere"'s real local GBIF count
# (predict_at_location()) alike, so the label means the same thing in both
# modes: "how much real evidence backs this number," not "how the app feels."
_CONFIDENCE_HIGH_THRESHOLD = 150
_CONFIDENCE_MEDIUM_THRESHOLD = 40

# "Explore Anywhere" evidence dampening. The model's raw predict_proba is
# trained on curated-area data and has no idea how many real sightings
# exist near an arbitrary clicked point -- species_for_location's presence
# gate only asks "at least one," which lets a single incidental record
# (or a handful) pass exactly as freely as a well-documented population.
# This scales the raw probability down when local evidence is thin, per an
# explicit product requirement: a species can be genuinely present in an
# area (passes the plausibility gate) while still being rare enough there
# that a prediction should not read as high-confidence. sqrt (not linear)
# so the penalty is steep at very low counts (1-10 records) but eases off
# well before _CONFIDENCE_HIGH_THRESHOLD, rather than requiring hundreds of
# local records just to stop being penalized at all.
_EVIDENCE_DAMPENING_FLOOR = _CONFIDENCE_HIGH_THRESHOLD

# How much extra credit an independent second/third source gives the
# evidence factor, on top of the primary count's own volume-based factor --
# capped at 2 corroborating sources (iNaturalist + eBird) so a very heavily
# double-reported species can't multiply its way past a real cap. Kept
# deliberately modest: corroboration should nudge confidence, not replace
# volume as the main signal, and it can never push the factor above 1.0.
_CORROBORATION_BONUS_PER_SOURCE = 0.15
_MAX_CORROBORATING_SOURCES_COUNTED = 2

# Plain-language notes for the "species ability to evade" factor the user
# asked for. Deliberately NOT a fabricated numeric "evasion score" -- this
# is real natural-history metadata (see data/seed_species.json's
# `activity_pattern` field) presented honestly as qualitative context.
# `diurnal` gets no note: it's the default "easiest to encounter" pattern,
# so calling it out on every single prediction would just be noise.
_ACTIVITY_PATTERN_NOTES = {
    "nocturnal": "This species is nocturnal (mainly active at night), so it's naturally harder to encounter than a day-active species even where it's common.",
    "crepuscular": "This species is crepuscular (most active around dawn and dusk) -- encounters are more likely near those hours.",
    "cathemeral": "This species is cathemeral (active at irregular times across day and night), which can make timing harder to predict.",
}

# Plain-language labels for the population-trend field IUCN returns.
_POPULATION_TREND_LABELS = {
    "Increasing": "increasing",
    "Decreasing": "decreasing",
    "Stable": "stable",
}


def _confidence_label(count: int) -> str:
    return "high" if count >= _CONFIDENCE_HIGH_THRESHOLD else "medium" if count >= _CONFIDENCE_MEDIUM_THRESHOLD else "low"


def _evidence_factor(primary_count: int, corroborating_count: int = 0) -> float:
    """
    0..1 multiplier applied to "Explore Anywhere"'s raw model probability,
    based on how much real local evidence backs this species at this exact
    point. `primary_count` drives the base factor: 1 record -> ~0.08x; 40
    -> ~0.52x; >=150 (the same bar as "high confidence" elsewhere) -> 1.0x.
    `corroborating_count` (independent sources beyond the primary one that
    also confirm presence -- see `_local_evidence_cached`) adds a modest
    bonus on top, capped so the result never exceeds 1.0. A species with
    zero local evidence from every source never reaches this function at
    all -- it's excluded by species_for_location's presence gate first.
    """
    if primary_count <= 0:
        return 0.0
    base = min(1.0, math.sqrt(primary_count / _EVIDENCE_DAMPENING_FLOOR))
    bonus = 1.0 + _CORROBORATION_BONUS_PER_SOURCE * min(corroborating_count, _MAX_CORROBORATING_SOURCES_COUNTED)
    return min(1.0, base * bonus)


# Every live weather call made while a person is waiting on a response
# (both curated-area predict() and anywhere-mode) uses this fast-fail
# budget instead of weather_client's patient ingestion defaults -- an
# outage should surface in seconds, not make the page hang for minutes.
# See weather_client.INTERACTIVE_* for why the two schedules differ.
_INTERACTIVE_RETRY = {
    "max_retries": weather_client.INTERACTIVE_MAX_RETRIES,
    "base_backoff": weather_client.INTERACTIVE_BASE_BACKOFF,
    "max_backoff": weather_client.INTERACTIVE_MAX_BACKOFF,
    "request_timeout": weather_client.INTERACTIVE_REQUEST_TIMEOUT,
}

SEED_SPECIES = json.loads((DATA_DIR / "seed_species.json").read_text())
SPECIES_BY_KEY = {s["scientific_name"]: s for s in SEED_SPECIES}


class ModelNotTrained(RuntimeError):
    pass


class LiveDataUnavailable(RuntimeError):
    """
    Raised only by "Explore Anywhere" mode, when the one live call it has no
    offline fallback for -- fetching climate normals for an arbitrary point
    -- itself fails (e.g. Open-Meteo unreachable). Curated areas never hit
    this: their normals are pre-cached CSVs with no network dependency.
    Kept distinct from ModelNotTrained so routers can return a clear,
    specific error instead of an unhandled 500.
    """

    pass


def _climate_normals_for_location(lat: float, lon: float) -> pd.DataFrame:
    try:
        normals = weather_client.climate_normals(round(lat, 2), round(lon, 2), **_INTERACTIVE_RETRY)
    except Exception as exc:
        raise LiveDataUnavailable(
            f"Couldn't fetch live weather data for ({lat:.2f}, {lon:.2f}) right now: {exc}"
        ) from exc
    if normals.empty:
        raise LiveDataUnavailable(f"No weather data available for ({lat:.2f}, {lon:.2f}).")
    return normals


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
    support_source: str | None = None,
    evidence_note: str | None = None,
    activity_pattern: str | None = None,
    conservation_status: dict | None = None,
) -> list[str]:
    """
    Short, plain-language reasons behind a prediction: data support,
    seasonal timing, the day's most unusual weather factor (if any), and
    -- the user's explicit "population" and "species ability to evade"
    asks -- real conservation status and activity-pattern context when
    available. This is Phase 0's version of the design doc's Product & UX
    "why" explanation text -- deliberately simple (no free-text LLM
    generation, so it's fast, deterministic, and always traceable to a
    concrete number).

    `support_source`: set only by "Explore Anywhere" mode, where
    `seasonality`'s peak/trough curve comes from the species' curated home
    area's training data, not the actual clicked point -- without this,
    "here" would misleadingly read as being about the click itself. Found
    via a real report: a lion showing "500 historical sighting records...
    in this area" near Bangkok read as GBIF evidence for Bangkok, when it
    was really Kruger's training count carried over as the closest analog.

    `evidence_note`: set only by "Explore Anywhere" mode, which now has a
    REAL local evidence count for the clicked point (see
    `_local_evidence_cached`) and builds its own fully-formed confidence
    sentence from it rather than the generic home-area-support sentence
    below -- see predict_at_location. When given, it replaces the first
    (data-support) factor line entirely.

    `activity_pattern`: the species' real diel activity pattern (see
    `data/seed_species.json`) -- honest qualitative "ability to evade"
    context, not a fabricated numeric statistic. No line is added for
    `diurnal` (the default, "easiest to encounter" case) or when unknown.

    `conservation_status`: the species' latest IUCN Red List assessment
    (see `app.services.iucn_client`), when IUCN_API_KEY was configured at
    training time -- real "population" context: a rare/declining species
    should be expected to have fewer sightings than a common one,
    independent of local search effort. Silently omitted when unavailable
    (no key configured, or no IUCN assessment found for this name) rather
    than showing a misleading "no data" line.
    """
    location_phrase = "in this area" if support_source is None else f"in {support_source} (closest trained analog)"
    seasonal_phrase = "here" if support_source is None else f"in {support_source}"
    factors = []

    if evidence_note is not None:
        factors.append(evidence_note)
    elif support >= _CONFIDENCE_HIGH_THRESHOLD:
        factors.append(f"High confidence: {support} historical sighting records for this species {location_phrase}.")
    elif support >= _CONFIDENCE_MEDIUM_THRESHOLD:
        factors.append(f"Medium confidence: {support} historical sighting records for this species {location_phrase}.")
    else:
        factors.append(f"Low confidence: only {support} historical sighting records for this species {location_phrase}.")

    season = seasonality.get(species_key)
    if season:
        doy = target_date.timetuple().tm_yday
        dist_peak = _circular_day_distance(doy, season["peak_doy"])
        dist_trough = _circular_day_distance(doy, season["trough_doy"])
        peak_month = _month_label(season["peak_doy"])
        trough_month = _month_label(season["trough_doy"])
        if dist_peak <= 35 and dist_peak <= dist_trough:
            factors.append(f"Near this species' peak season {seasonal_phrase} (historically strongest around {peak_month}).")
        elif dist_trough <= 35 and dist_trough < dist_peak:
            factors.append(f"Near this species' quietest time of year {seasonal_phrase} (weakest around {trough_month}).")
        else:
            factors.append(f"Shoulder season {seasonal_phrase} -- peak is around {peak_month}, quietest around {trough_month}.")

    best_var, best_z = None, 0.0
    for var in _WEATHER_VAR_META:
        z = weather_row.get(f"{var}_anomaly") or 0.0
        if abs(z) > abs(best_z):
            best_var, best_z = var, z
    if best_var and abs(best_z) >= 0.75:
        meta = _WEATHER_VAR_META[best_var]
        word = meta["hi"] if best_z > 0 else meta["lo"]
        factors.append(f"Today looks {word} than typical for this date (unusual {meta['name']}).")

    activity_note = _ACTIVITY_PATTERN_NOTES.get(activity_pattern or "")
    if activity_note:
        factors.append(activity_note)

    if conservation_status and conservation_status.get("category_label"):
        trend = _POPULATION_TREND_LABELS.get(conservation_status.get("population_trend") or "")
        trend_phrase = f" (population trend: {trend})" if trend else ""
        factors.append(
            f"IUCN Red List status: {conservation_status['category_label']}{trend_phrase} -- "
            "real conservation data, not sighting data; rarer/declining species are naturally harder to encounter."
        )

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


def get_data_source() -> dict:
    """
    Whether the currently-loaded model was trained on REAL ingested data
    (`scripts/ingest_gbif.py` + `ingest_weather.py`) or the offline
    SYNTHETIC demo fixtures (`scripts/generate_sample_fixtures.py`) -- see
    `app.services.data_source_marker`. `{"source": "unknown"}` for a model
    bundle trained before this field existed, so an old real model is never
    mislabeled as synthetic. Surfaced on `/api/health` and every prediction
    response so a user (or operator) never has to guess which one they're
    looking at.
    """
    try:
        return _load_model().get("data_source", {"source": "unknown"})
    except ModelNotTrained:
        return {"source": "unknown"}


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


def _normal_row_from_df(normals: pd.DataFrame, date: dt.date) -> dict:
    """Climate-normal baseline only (no live call) -- the always-available fallback.

    Pure function over an already-loaded normals DataFrame, so it works
    identically whether that DataFrame came from a pre-cached per-area CSV
    (curated areas) or a live `weather_client.climate_normals()` call for an
    arbitrary point ("Explore Anywhere" mode) -- both have the same
    day_of_year / <var>_normal_mean / <var>_normal_std schema.
    """
    day_of_year = date.timetuple().tm_yday
    normal_row = normals[normals["day_of_year"] == day_of_year] if not normals.empty else normals
    row: dict = {"day_of_year": day_of_year}
    for var in WEATHER_DAILY_VARS:
        mean = float(normal_row[f"{var}_normal_mean"].iloc[0]) if len(normal_row) else 0.0
        std = float(normal_row[f"{var}_normal_std"].iloc[0]) if len(normal_row) else 1.0
        row[f"{var}_normal_mean"] = mean
        row[f"{var}_normal_std"] = std
        row[var] = mean  # no anomaly unless a live observation overrides it
        row[f"{var}_anomaly"] = 0.0
    return row


def _normal_row_for_date(area_id: str, date: dt.date) -> dict:
    return _normal_row_from_df(_load_climate_normals(area_id), date)


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


def _weather_features_for_date(lat: float, lon: float, normals: pd.DataFrame, date: dt.date) -> dict:
    """
    Cached climate-normal baseline + a best-effort live observation/forecast
    for the anomaly term, falling back gracefully (anomaly=0) when the date
    is outside the live archive/forecast window or the network call fails --
    which also covers this being run somewhere with restricted egress. For a
    single date; see `_weather_features_for_dates` for the batched version
    best_window() uses (one live call for the whole date range instead of
    one per date). Takes `normals` directly (rather than an area_id) so the
    same function serves both curated areas (pre-cached CSV normals) and
    "Explore Anywhere" locations (a live climate_normals() call).
    """
    row = _normal_row_from_df(normals, date)
    today = dt.date.today()
    horizon = (date - today).days
    try:
        if -30 <= horizon <= 0:
            obs = weather_client.historical_daily(lat, lon, date.isoformat(), date.isoformat(), **_INTERACTIVE_RETRY)
        elif 0 < horizon <= 16:
            obs = weather_client.forecast_daily(lat, lon, days=horizon + 1, **_INTERACTIVE_RETRY)
            obs = obs[obs["date"] == date.isoformat()]
        else:
            obs = pd.DataFrame()
        if not obs.empty:
            row = _apply_observation(row, obs.iloc[0])
    except Exception as exc:  # live weather is best-effort; the climatological normal is always a valid fallback
        log.warning("Live weather fetch failed (%s); falling back to the climate normal for day %s.", exc, row["day_of_year"])
    return row


def _weather_features_for_dates(lat: float, lon: float, normals: pd.DataFrame, dates: list[dt.date]) -> list[dict]:
    """
    Batched version of `_weather_features_for_date`: ONE historical-archive
    call and ONE forecast call cover every date in `dates` that falls in
    their respective windows, instead of one live call per date. Used by
    best_window(), which asks about ~70 dates spanning a year at once.
    """
    rows = {d: _normal_row_from_df(normals, d) for d in dates}

    today = dt.date.today()
    hist_dates = sorted(d for d in dates if -30 <= (d - today).days <= 0)
    fcst_dates = sorted(d for d in dates if 0 < (d - today).days <= 16)

    try:
        if hist_dates:
            obs = weather_client.historical_daily(
                lat, lon, hist_dates[0].isoformat(), hist_dates[-1].isoformat(), **_INTERACTIVE_RETRY
            )
            obs_by_date = {row_["date"]: row_ for _, row_ in obs.iterrows()} if not obs.empty else {}
            for d in hist_dates:
                if d.isoformat() in obs_by_date:
                    rows[d] = _apply_observation(rows[d], obs_by_date[d.isoformat()])
        if fcst_dates:
            obs = weather_client.forecast_daily(lat, lon, days=16, **_INTERACTIVE_RETRY)
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
    normals = _load_climate_normals(area_id)
    weather_row = _weather_features_for_date(area["lat"], area["lon"], normals, target_date)

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
    conservation_by_species = bundle.get("species_conservation_status", {})
    results = []
    for sp, p in zip(candidates, proba):
        support = sp["training_support"]
        confidence = _confidence_label(support)
        factors = _explain(
            sp["scientific_name"], target_date, weather_row, seasonality, support, confidence,
            activity_pattern=sp.get("activity_pattern"),
            conservation_status=conservation_by_species.get(sp["scientific_name"]),
        )
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
    normals = _load_climate_normals(area_id)
    weather_rows = _weather_features_for_dates(area["lat"], area["lon"], normals, dates)

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


# --- "Explore Anywhere" mode --------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearest_area(lat: float, lon: float) -> dict:
    """The curated area geographically closest to an arbitrary point.

    Used as the model's `area_id` categorical value for anywhere-mode
    predictions (the model has never seen a made-up area_id, but it *has*
    seen this one, so it keeps whatever location-flavored signal that
    category carries) and surfaced to the caller so the approximation is
    always visible rather than silently assumed.
    """
    areas = _load_model()["areas"]
    return min(areas.values(), key=lambda a: _haversine_km(lat, lon, a["lat"], a["lon"]))


def _local_evidence_cached(
    scientific_name: str,
    lat: float,
    lon: float,
    radius_km: float,
    taxon_class: str,
    plausible_countries: list[str] | None = None,
) -> dict:
    """In-process, time-boxed cache in front of live multi-source presence checks.

    Queries GBIF (always) and iNaturalist (always, no key needed) for real
    records within `radius_km`, plus eBird for Aves species when
    EBIRD_API_KEY is configured. This is the "many sources, for quality"
    implementation of "Explore Anywhere"'s evidence: GBIF already
    aggregates a large share of iNaturalist's own research-grade
    observations as one of its contributing datasets, so the sources are
    NOT summed as if independent record counts (that would double-count a
    lot of the same underlying sightings) -- instead one source is the
    `primary_count` (GBIF's count if it found anything, otherwise whichever
    other source did) and the rest are `corroborating_count`: how many
    ADDITIONAL independent sources also confirm presence, which nudges the
    evidence factor up a bit (see `_evidence_factor`) without inflating the
    volume number itself.

    24h TTL (occurrence data does not change meaningfully within a day).
    Keyed on coordinates rounded to ~1km so nearby clicks share a cache
    entry. `plausible_countries` is not part of the key: it's fixed,
    curated metadata per species, not a per-call variable.

    Returns {"sources": {"gbif": int, "inaturalist": int, "ebird": int (Aves,
    keyed only when checked)}, "primary_count": int, "corroborating_count": int}.
    A source that errors out is recorded as 0 for that source (fails closed,
    same as the old boolean version did) rather than crashing the whole
    lookup or making that species's evidence look artificially strong.
    """
    key = (scientific_name, round(lat, 2), round(lon, 2), int(radius_km))
    now = time.monotonic()
    cached = _presence_cache.get(key)
    if cached is not None and now - cached[0] < _PRESENCE_CACHE_TTL_SECONDS:
        return cached[1]

    def _fetch_gbif() -> int:
        try:
            return gbif_client.presence_count(
                scientific_name, lat, lon, radius_km,
                timeout=_ANYWHERE_PRESENCE_TIMEOUT_S, plausible_countries=plausible_countries,
                max_retries=gbif_client.INTERACTIVE_MAX_RETRIES,
                base_backoff=gbif_client.INTERACTIVE_BASE_BACKOFF,
                max_backoff=gbif_client.INTERACTIVE_MAX_BACKOFF,
            )
        except Exception as exc:
            log.warning("Live GBIF presence check failed for %s near (%.2f, %.2f): %s", scientific_name, lat, lon, exc)
            return 0

    def _fetch_inaturalist() -> int:
        try:
            return inaturalist_client.presence_count(
                scientific_name, lat, lon, radius_km, timeout=_ANYWHERE_PRESENCE_TIMEOUT_S,
            )
        except Exception as exc:
            log.warning(
                "Live iNaturalist presence check failed for %s near (%.2f, %.2f): %s", scientific_name, lat, lon, exc,
            )
            return 0

    def _fetch_ebird() -> int:
        try:
            return ebird_client.presence_count(
                scientific_name, lat, lon, radius_km, timeout=_ANYWHERE_PRESENCE_TIMEOUT_S,
            )
        except Exception as exc:
            log.warning("Live eBird presence check failed for %s near (%.2f, %.2f): %s", scientific_name, lat, lon, exc)
            return 0

    fetchers = {"gbif": _fetch_gbif, "inaturalist": _fetch_inaturalist}
    if taxon_class == "Aves" and ebird_client.is_configured():
        fetchers["ebird"] = _fetch_ebird

    # Run every source concurrently, not one after another. These used to
    # run sequentially -- fine when it was just GBIF + iNaturalist, but a
    # real 2026-09-17 incident found the actual cost of that design once
    # EBIRD_API_KEY was first configured in production: three sequential
    # ~8-20s-timeout calls for one species could sum past gunicorn's default
    # 30s worker timeout, and Render's proxy returned a bare 502 to the
    # user instead of a slow-but-working prediction. Fetching in parallel
    # bounds one species' worst-case wait to its single slowest source,
    # not the sum of all of them -- this also means a future 4th source
    # doesn't reintroduce the same failure mode by construction.
    sources: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=len(fetchers)) as pool:
        future_to_name = {pool.submit(fn): name for name, fn in fetchers.items()}
        for future in as_completed(future_to_name):
            sources[future_to_name[future]] = future.result()

    primary_count = sources["gbif"] if sources["gbif"] > 0 else max(sources.values(), default=0)
    corroborating_count = sum(1 for v in sources.values() if v > 0)
    if primary_count > 0:
        corroborating_count -= 1  # don't count the primary source as its own corroboration

    evidence = {
        "sources": sources,
        "primary_count": primary_count,
        "corroborating_count": max(corroborating_count, 0),
    }
    _presence_cache[key] = (now, evidence)
    return evidence


def species_for_location(lat: float, lon: float, radius_km: float = _ANYWHERE_DEFAULT_RADIUS_KM) -> list[dict]:
    """
    'Explore Anywhere' mode: instead of a fixed curated area's seed list,
    checks every species WildCast already knows (the same 50 curated
    species -- no new ones are invented) for at least one real record
    (GBIF, iNaturalist, or -- for birds -- eBird) within `radius_km` of
    this exact point. This is the live version of the plausibility gate
    `species_for_area` applies statically -- the concrete upgrade
    CONTRIBUTING.md flagged as the single highest-value next step, now
    wired in and now multi-source (see `_local_evidence_cached`). A
    species with zero live records from every source nearby is never
    offered, no matter how close the click is to a curated area. Each
    match also carries `local_evidence_count` (the primary source's real
    record count), `local_corroborating_count` (how many additional
    independent sources also confirm presence), and `local_evidence_sources`
    (the raw per-source breakdown) -- used downstream by
    predict_at_location to calibrate confidence and probability to actual
    local evidence rather than the species' curated home area's training
    count.
    """
    support = _load_model()["species_support"]
    matches: list[dict] = []
    with ThreadPoolExecutor(max_workers=_ANYWHERE_MAX_WORKERS) as pool:
        future_to_species = {
            pool.submit(
                _local_evidence_cached,
                s["scientific_name"], lat, lon, radius_km, s["taxon_class"], s.get("plausible_countries"),
            ): s
            for s in SEED_SPECIES
        }
        for future in as_completed(future_to_species):
            s = future_to_species[future]
            try:
                evidence = future.result()
            except Exception:
                evidence = {"sources": {}, "primary_count": 0, "corroborating_count": 0}
            if evidence["primary_count"] > 0:
                matches.append(
                    {
                        **s,
                        "training_support": support.get(s["scientific_name"], 0),
                        "local_evidence_count": evidence["primary_count"],
                        "local_corroborating_count": evidence["corroborating_count"],
                        "local_evidence_sources": evidence["sources"],
                    }
                )
    matches.sort(key=lambda s: s["scientific_name"])
    return matches


def predict_at_location(
    lat: float,
    lon: float,
    date: str,
    radius_km: float = _ANYWHERE_DEFAULT_RADIUS_KM,
    species_key: str | None = None,
) -> dict:
    """
    The 'Explore Anywhere' counterpart to `predict()`. Candidates come from
    a live GBIF presence check (`species_for_location`) instead of a fixed
    area's seed list, and weather comes from a live call for these exact
    coordinates instead of a pre-cached per-area normals file. The
    underlying model is not retrained per location -- its categorical
    `area_id` feature is set to whichever curated area is geographically
    nearest, so it still has some location signal to key off -- and the
    result always names that nearest area and the real distance to it, so
    the approximation is never hidden from the caller.
    """
    bundle = _load_model()
    target_date = dt.date.fromisoformat(date)
    nearest = _nearest_area(lat, lon)
    distance_km = _haversine_km(lat, lon, nearest["lat"], nearest["lon"])

    normals = _climate_normals_for_location(lat, lon)
    weather_row = _weather_features_for_date(lat, lon, normals, target_date)

    candidates = species_for_location(lat, lon, radius_km)
    if species_key:
        candidates = [c for c in candidates if c["scientific_name"] == species_key]
        if not candidates:
            raise KeyError(
                f"No live GBIF record of '{species_key}' within {radius_km:.0f}km of ({lat:.2f}, {lon:.2f})."
            )

    results = []
    if candidates:
        rows = [
            {
                "species": sp["scientific_name"],
                "taxon_class": sp["taxon_class"],
                "area_id": nearest["id"],
                "lat": lat,
                "lon": lon,
                **weather_row,
            }
            for sp in candidates
        ]
        frame = pd.DataFrame(rows)
        features = build_feature_frame(frame)
        proba = bundle["model"].predict_proba(features)[:, 1]

        seasonality = bundle.get("species_seasonality", {})
        conservation_by_species = bundle.get("species_conservation_status", {})
        _SOURCE_LABELS = {"gbif": "GBIF", "inaturalist": "iNaturalist", "ebird": "eBird"}
        for sp, p in zip(candidates, proba):
            # Real local evidence, not the species' curated home area's
            # training count -- see species_for_location and
            # _local_evidence_cached. This is what fixed a real report: a
            # Leopard reading 91% "high confidence" near a point in
            # Thailand off the strength of Kruger, South Africa's 500
            # training records, with nothing about that number actually
            # measured near the click.
            local_count = sp["local_evidence_count"]
            corroborating_count = sp["local_corroborating_count"]
            local_sources = sp["local_evidence_sources"]
            confidence = _confidence_label(local_count)
            evidence_factor = _evidence_factor(local_count, corroborating_count)
            dampened_probability = float(p) * evidence_factor

            if confidence == "high":
                evidence_note = (
                    f"High confidence: {local_count} confirmed wild sighting records found within "
                    f"{radius_km:.0f}km of this exact point."
                )
            elif confidence == "medium":
                evidence_note = (
                    f"Medium confidence: {local_count} confirmed wild sighting records found within "
                    f"{radius_km:.0f}km of this exact point."
                )
            else:
                record_word = "record" if local_count == 1 else "records"
                evidence_note = (
                    f"Low confidence: only {local_count} confirmed wild sighting {record_word} found within "
                    f"{radius_km:.0f}km of this exact point -- probability has been scaled down accordingly."
                )

            factors = _explain(
                sp["scientific_name"], target_date, weather_row, seasonality, local_count, confidence,
                support_source=nearest["name"], evidence_note=evidence_note,
                activity_pattern=sp.get("activity_pattern"),
                conservation_status=conservation_by_species.get(sp["scientific_name"]),
            )

            # "many sources, for quality" -- named explicitly so a user can
            # see this wasn't just one dataset's word for it.
            other_sources = [
                _SOURCE_LABELS.get(name, name) for name, count in local_sources.items() if name != "gbif" and count > 0
            ]
            if corroborating_count > 0 and other_sources:
                sources_phrase = " and ".join(other_sources)
                factors.append(f"Also independently confirmed by {sources_phrase} in this area -- not just one data source.")

            factors.append(
                f"Verified live against real wildlife-sighting databases (wild records only, native-range "
                f"countries only) within {radius_km:.0f}km of this exact point (closest flagship area for "
                f"seasonal/weather modeling: {nearest['name']}, {distance_km:.0f}km away)."
            )
            results.append(
                {
                    "scientific_name": sp["scientific_name"],
                    "common_name": sp["common_name"],
                    "taxon_class": sp["taxon_class"],
                    "probability": round(dampened_probability, 4),
                    "raw_model_probability": round(float(p), 4),
                    "confidence": confidence,
                    "training_records": local_count,
                    "local_evidence_count": local_count,
                    "local_corroborating_count": corroborating_count,
                    "local_evidence_sources": local_sources,
                    "factors": factors,
                }
            )
        results.sort(key=lambda r: r["probability"], reverse=True)

    return {
        "lat": lat,
        "lon": lon,
        "radius_km": radius_km,
        "nearest_area": {"id": nearest["id"], "name": nearest["name"], "distance_km": round(distance_km, 1)},
        "predictions": results,
    }


def best_window_at_location(
    lat: float, lon: float, species_key: str, radius_km: float = _ANYWHERE_DEFAULT_RADIUS_KM
) -> list[dict]:
    """
    Anywhere-mode counterpart to `best_window()` -- see its docstring for
    the batching rationale. Applies the same local-evidence dampening as
    `predict_at_location` (see `_evidence_factor`) so a species with only a
    handful of real GBIF records near this point doesn't show a
    misleadingly confident probability curve across the whole year either.
    """
    bundle = _load_model()
    nearest = _nearest_area(lat, lon)
    candidates = [c for c in species_for_location(lat, lon, radius_km) if c["scientific_name"] == species_key]
    if not candidates:
        raise KeyError(f"No live GBIF record of '{species_key}' within {radius_km:.0f}km of ({lat:.2f}, {lon:.2f}).")
    sp = candidates[0]
    evidence_factor = _evidence_factor(sp["local_evidence_count"], sp["local_corroborating_count"])

    today = dt.date.today()
    dates = [dt.date(today.year, 1, 1) + dt.timedelta(days=offset) for offset in range(0, 366, 5)]
    normals = _climate_normals_for_location(lat, lon)
    weather_rows = _weather_features_for_dates(lat, lon, normals, dates)

    rows = [
        {"species": sp["scientific_name"], "taxon_class": sp["taxon_class"], "area_id": nearest["id"], **weather_row}
        for weather_row in weather_rows
    ]
    frame = pd.DataFrame(rows)
    features = build_feature_frame(frame)
    proba = bundle["model"].predict_proba(features)[:, 1]

    return [
        {"date": d.isoformat(), "day_of_year": i * 5 + 1, "probability": round(float(p) * evidence_factor, 4)}
        for i, (d, p) in enumerate(zip(dates, proba))
    ]

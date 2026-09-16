"""
Generates SYNTHETIC occurrence + weather fixtures with the exact same file
schema that scripts/ingest_gbif.py and scripts/ingest_weather.py produce
from the real GBIF and Open-Meteo APIs.

Why this file exists: it lets the rest of the pipeline (feature engineering,
pseudo-absence generation, model training, the serving API, the frontend)
be exercised end-to-end without live internet access -- useful in locked-down
sandboxes/CI, or for a five-second first run before you've pointed the real
ingestion scripts at the live APIs.

THE DATA IN THESE FILES IS NOT REAL. Species' monthly activity weights below
are set from well-established, general natural-history knowledge (e.g. brown
bears hibernate roughly December-February; Kruger's dry season concentrates
animals near water June-September) so that the synthetic seasonal curves are
directionally realistic and the demo model has something meaningful to
learn -- but they are illustrative approximations, not sourced records, and
must never be presented to an end user as observation data. Replace this
cache by running ingest_gbif.py + ingest_weather.py against the live APIs
before showing predictions to anyone.

Run as: python -m scripts.generate_sample_fixtures
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CACHE_DIR, DATA_DIR, RANDOM_SEED  # noqa: E402  (needs the sys.path.insert above first)

PILOT_AREAS = {a["id"]: a for a in json.loads((DATA_DIR / "pilot_areas.json").read_text())}
SEED_SPECIES = json.loads((DATA_DIR / "seed_species.json").read_text())

YEARS_BACK = 15
TODAY = pd.Timestamp.today().normalize()
START = TODAY - pd.DateOffset(years=YEARS_BACK)

rng = np.random.default_rng(RANDOM_SEED)

# Relative month-by-month detectability weight (Jan..Dec), and roughly how many
# occurrence records over YEARS_BACK years a species of that detectability gets.
# These are illustrative, not measured.
SPECIES_PROFILE = {
    "Ursus arctos":               {"weights": [1, 1, 2, 6, 9, 9, 8, 8, 9, 9, 4, 1], "n_records": 220},
    "Canis lupus":                {"weights": [7, 7, 6, 5, 5, 5, 5, 5, 5, 6, 7, 8], "n_records": 180},
    "Bison bison":                {"weights": [6, 6, 6, 7, 8, 8, 7, 7, 7, 7, 6, 6], "n_records": 400},
    "Haliaeetus leucocephalus":   {"weights": [9, 9, 6, 4, 3, 2, 2, 2, 3, 5, 7, 9], "n_records": 260},
    "Cervus canadensis":          {"weights": [5, 5, 5, 6, 7, 6, 6, 7, 9, 9, 6, 5], "n_records": 380},
    "Loxodonta africana":         {"weights": [5, 5, 6, 6, 7, 8, 9, 9, 8, 7, 6, 5], "n_records": 450},
    "Panthera leo":                {"weights": [5, 5, 6, 6, 7, 8, 8, 8, 8, 7, 6, 5], "n_records": 260},
    "Giraffa camelopardalis":     {"weights": [6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 6, 6], "n_records": 420},
    "Panthera pardus":            {"weights": [4, 4, 5, 5, 6, 7, 7, 7, 7, 6, 5, 4], "n_records": 140},
    "Tragelaphus strepsiceros":   {"weights": [5, 5, 6, 6, 7, 8, 9, 9, 8, 7, 6, 5], "n_records": 300},
    "Macropus giganteus":         {"weights": [5, 5, 6, 7, 8, 8, 8, 8, 7, 6, 5, 5], "n_records": 500},
    "Pteropus poliocephalus":     {"weights": [9, 8, 7, 5, 3, 2, 2, 3, 5, 7, 9, 9], "n_records": 260},
    "Gymnorhina tibicen":         {"weights": [5, 5, 5, 6, 6, 6, 6, 8, 9, 8, 6, 5], "n_records": 520},
    "Trichoglossus moluccanus":   {"weights": [6, 6, 7, 6, 5, 5, 5, 7, 9, 9, 7, 6], "n_records": 480},
    "Dacelo novaeguineae":        {"weights": [7, 7, 7, 6, 6, 6, 6, 8, 9, 9, 8, 7], "n_records": 400},
    # Svalbard: polar bears are least visible in deep winter/spring while out
    # hunting on distant sea ice, and most visible ashore in the ice-free
    # season while waiting for the sea to refreeze (roughly Jul-Oct).
    "Ursus maritimus":            {"weights": [2, 2, 2, 3, 4, 6, 8, 9, 9, 8, 5, 3], "n_records": 90},
    "Vulpes lagopus":             {"weights": [4, 4, 5, 6, 8, 9, 9, 8, 7, 6, 5, 4], "n_records": 260},
    "Ovibos moschatus":           {"weights": [5, 5, 5, 6, 7, 7, 7, 7, 6, 6, 5, 5], "n_records": 340},
    "Bubo scandiacus":            {"weights": [3, 3, 4, 5, 7, 9, 9, 8, 6, 4, 3, 3], "n_records": 150},
    "Rangifer tarandus":          {"weights": [4, 4, 4, 5, 7, 9, 9, 8, 7, 5, 4, 4], "n_records": 300},
    # Amazon: many species concentrate near shrinking water bodies and are
    # easier to detect in the relatively drier, clearer months (roughly
    # Jun-Nov here); the wet season disperses animals and obscures canopy/sky
    # sightings.
    "Panthera onca":              {"weights": [3, 3, 3, 4, 5, 7, 9, 9, 8, 7, 5, 4], "n_records": 110},
    "Ara macao":                  {"weights": [6, 6, 6, 6, 6, 7, 7, 8, 9, 9, 8, 6], "n_records": 420},
    "Hydrochoerus hydrochaeris":  {"weights": [4, 4, 4, 5, 6, 8, 9, 9, 8, 7, 5, 4], "n_records": 480},
    "Pteronura brasiliensis":     {"weights": [4, 4, 4, 5, 6, 8, 9, 9, 8, 6, 5, 4], "n_records": 150},
    "Harpia harpyja":             {"weights": [5, 5, 5, 5, 6, 7, 8, 8, 7, 6, 5, 5], "n_records": 70},
}

# Per-species same-day weather sensitivity: how much a day's temperature and
# precipitation anomaly (in standard deviations from that calendar day's
# normal -- exactly the *_anomaly features the model trains on) shifts that
# day's chance of producing a sighting, on top of the baseline seasonal
# weight above. Positive = more likely on warmer/wetter-than-normal days;
# negative = more likely on cooler/drier-than-normal days. Illustrative
# natural-history judgement calls (e.g. water-dependent savanna species
# concentrate and become MORE visible on hot days near rivers; species that
# avoid heat stress become LESS visible), not measured coefficients -- this
# is what gives the synthetic demo genuine, learnable weather signal on top
# of the month-level seasonality, so the pipeline's "does weather add lift
# over a season-only baseline" evaluation has something real to detect.
SPECIES_WEATHER_SENSITIVITY = {
    "Ursus arctos":               {"temp": -0.35, "precip": -0.30},
    "Canis lupus":                {"temp": -0.25, "precip": -0.15},
    "Bison bison":                {"temp": -0.10, "precip": -0.10},
    "Haliaeetus leucocephalus":   {"temp": -0.30, "precip": -0.35},
    "Cervus canadensis":          {"temp": -0.30, "precip": -0.10},
    "Loxodonta africana":         {"temp": 0.35, "precip": -0.30},
    "Panthera leo":                {"temp": -0.20, "precip": -0.15},
    "Giraffa camelopardalis":     {"temp": 0.15, "precip": -0.10},
    "Panthera pardus":            {"temp": -0.15, "precip": -0.35},
    "Tragelaphus strepsiceros":   {"temp": 0.30, "precip": -0.30},
    "Macropus giganteus":         {"temp": -0.40, "precip": -0.10},
    "Pteropus poliocephalus":     {"temp": 0.35, "precip": -0.30},
    "Gymnorhina tibicen":         {"temp": 0.05, "precip": -0.25},
    "Trichoglossus moluccanus":   {"temp": 0.25, "precip": -0.20},
    "Dacelo novaeguineae":        {"temp": 0.05, "precip": -0.35},
    "Ursus maritimus":            {"temp": -0.30, "precip": -0.15},
    "Vulpes lagopus":             {"temp": 0.20, "precip": -0.20},
    "Ovibos moschatus":           {"temp": -0.15, "precip": -0.20},
    "Bubo scandiacus":            {"temp": 0.10, "precip": -0.30},
    "Rangifer tarandus":          {"temp": 0.15, "precip": -0.15},
    "Panthera onca":              {"temp": 0.10, "precip": -0.35},
    "Ara macao":                  {"temp": 0.05, "precip": -0.25},
    "Hydrochoerus hydrochaeris":  {"temp": 0.20, "precip": -0.30},
    "Pteronura brasiliensis":     {"temp": 0.10, "precip": -0.30},
    "Harpia harpyja":             {"temp": 0.05, "precip": -0.30},
}

# Simple sinusoidal seasonal climate model per area: base + amplitude*cos(2pi*(doy-peak)/365).
# peak_doy ~200 (mid-July) = northern-hemisphere summer peak; ~20 (mid-Jan) = southern-hemisphere summer peak.
AREA_CLIMATE = {
    "yellowstone": {
        "peak_doy": 200,
        "temperature_2m_max": (12, 18, 4.5),
        "temperature_2m_min": (-2, 15, 4.0),
        "precipitation_sum": (2.0, 1.3, 2.5),
        "windspeed_10m_max": (16, 4, 5),
        "cloudcover_mean": (42, 12, 15),
    },
    "kruger": {
        "peak_doy": 20,
        "temperature_2m_max": (29, 6, 3.0),
        "temperature_2m_min": (16, 6, 2.5),
        "precipitation_sum": (3.5, 4.0, 4.0),
        "windspeed_10m_max": (11, 2, 3),
        "cloudcover_mean": (38, 20, 15),
    },
    "brisbane_urban": {
        "peak_doy": 20,
        "temperature_2m_max": (25, 5.5, 2.5),
        "temperature_2m_min": (15, 6, 2.2),
        "precipitation_sum": (3.2, 3.0, 4.5),
        "windspeed_10m_max": (13, 2.5, 3.5),
        "cloudcover_mean": (48, 10, 15),
    },
    # Svalbard: polar/arctic climate, peak (mildest) around day 200 (~mid-July,
    # N-hemisphere summer). Genuinely cold-to-below-freezing most of the year.
    "svalbard_arctic": {
        "peak_doy": 200,
        "temperature_2m_max": (-8, 13, 3.0),
        "temperature_2m_min": (-12, 13, 3.0),
        "precipitation_sum": (1.0, 0.3, 1.0),
        "windspeed_10m_max": (15, 3, 4),
        "cloudcover_mean": (65, 10, 15),
    },
    # Central Amazon: near-equatorial, so temperature is nearly flat all year;
    # the seasonal signal here is mostly rainfall (wetter Dec-May, drier and
    # clearer Jun-Nov). peak_doy=60 (~early March) models the wet-season peak.
    "amazon_rainforest": {
        "peak_doy": 60,
        "temperature_2m_max": (31, 1.5, 2.0),
        "temperature_2m_min": (22, 1.0, 1.5),
        "precipitation_sum": (6.0, 4.5, 5.0),
        "windspeed_10m_max": (8, 1.5, 2.0),
        "cloudcover_mean": (70, 8, 12),
    },
}


def _jitter_point(lat: float, lon: float, radius_km: float) -> tuple[float, float]:
    angle = rng.uniform(0, 2 * math.pi)
    dist_km = rng.uniform(0, radius_km)
    dlat = (dist_km * math.cos(angle)) / 111.0
    dlon = (dist_km * math.sin(angle)) / (111.320 * max(math.cos(math.radians(lat)), 0.1))
    return lat + dlat, lon + dlon


def generate_weather() -> dict[str, pd.DataFrame]:
    """Writes the weather + climate-normal caches, and returns each area's daily frame
    (including temp/precip z-score anomalies) in memory for generate_occurrences to use."""
    date_range = pd.date_range(START, TODAY, freq="D")
    doy = date_range.dayofyear
    in_memory: dict[str, pd.DataFrame] = {}

    for area_id, climate in AREA_CLIMATE.items():
        peak = climate["peak_doy"]
        radians = 2 * math.pi * (doy - peak) / 365.0

        daily = {"area_id": area_id, "date": date_range.date.astype(str)}
        z_scores = {}
        for var, spec in climate.items():
            if var == "peak_doy":
                continue
            base, amp, noise_sd = spec
            mean_curve = base + amp * np.cos(radians)
            noise = rng.normal(0, noise_sd, size=len(date_range))
            values = mean_curve + noise
            if var in ("precipitation_sum", "cloudcover_mean"):
                values = np.clip(values, 0, None)
            if var == "cloudcover_mean":
                values = np.clip(values, 0, 100)
            daily[var] = np.round(values, 2)
            z_scores[f"{var}_z"] = noise / noise_sd  # exact anomaly this day was generated with

        weather_df = pd.DataFrame(daily)
        weather_df.to_csv(CACHE_DIR / f"weather_daily_{area_id}.csv", index=False)
        print(f"  [synthetic] wrote {len(weather_df)} daily-weather rows -> weather_daily_{area_id}.csv")

        # Climate normals: per day-of-year mean/std, straight from the same generating curve.
        unique_doy = np.arange(1, 367)
        rad_u = 2 * math.pi * (unique_doy - peak) / 365.0
        normal_data = {"area_id": area_id, "day_of_year": unique_doy}
        for var, spec in climate.items():
            if var == "peak_doy":
                continue
            base, amp, noise_sd = spec
            normal_data[f"{var}_normal_mean"] = np.round(base + amp * np.cos(rad_u), 2)
            normal_data[f"{var}_normal_std"] = round(noise_sd, 2)
        normals_df = pd.DataFrame(normal_data)
        normals_df.to_csv(CACHE_DIR / f"climate_normals_{area_id}.csv", index=False)
        print(f"  [synthetic] wrote {len(normals_df)} climate-normal rows -> climate_normals_{area_id}.csv")

        in_memory[area_id] = pd.DataFrame({"date": date_range, "month": date_range.month, **z_scores})

    return in_memory


def generate_occurrences(weather_by_area: dict[str, pd.DataFrame]) -> None:
    """
    Sample occurrence dates from a weight that combines each species'
    baseline month-of-year seasonality with its same-day temperature/
    precipitation sensitivity (SPECIES_WEATHER_SENSITIVITY) applied to that
    day's *actual* generated weather anomaly -- so a species genuinely does
    show up more often on the kind of day its sensitivity favors, and the
    training pipeline's "does weather add lift over season alone" check has
    real signal to find.
    """
    for area_id, area in PILOT_AREAS.items():
        weather = weather_by_area[area_id]
        rows = []
        for sp in [s for s in SEED_SPECIES if s["area_id"] == area_id]:
            profile = SPECIES_PROFILE[sp["scientific_name"]]
            sens = SPECIES_WEATHER_SENSITIVITY[sp["scientific_name"]]
            month_weight = np.array(profile["weights"], dtype=float)[weather["month"].to_numpy() - 1]
            weather_multiplier = np.exp(
                sens["temp"] * weather["temperature_2m_max_z"].to_numpy()
                + sens["precip"] * weather["precipitation_sum_z"].to_numpy()
            )
            day_weight = month_weight * weather_multiplier
            day_weight = day_weight / day_weight.sum()

            n = profile["n_records"]
            picked_idx = rng.choice(len(weather), size=n, replace=True, p=day_weight)
            for i in picked_idx:
                d = weather["date"].iloc[i]
                lat, lon = _jitter_point(area["lat"], area["lon"], area["radius_km"])
                rows.append(
                    {
                        "species": sp["scientific_name"],
                        "common_name": sp["common_name"],
                        "taxon_class": sp["taxon_class"],
                        "area_id": area_id,
                        "lat": round(lat, 5),
                        "lon": round(lon, 5),
                        "date": d.date().isoformat(),
                    }
                )
        df = pd.DataFrame(rows).sort_values("date")
        out = CACHE_DIR / f"occurrences_{area_id}.csv"
        df.to_csv(out, index=False)
        print(f"  [synthetic] wrote {len(df)} occurrence rows -> {out}")


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating SYNTHETIC offline fixtures (not real observation data -- see module docstring).")
    weather_by_area = generate_weather()
    generate_occurrences(weather_by_area)
    print("Done. Run `python -m app.ml.train` next.")


if __name__ == "__main__":
    main()

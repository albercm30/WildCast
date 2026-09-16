"""
Central configuration for WildCast.

Nothing here requires a paid account. eBird, IUCN and Protected Planet keys
are optional -- the app runs on GBIF + Open-Meteo alone (both fully free,
no key) and simply skips the extra sources if their keys are unset.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "app" / "ml" / "artifacts"
CACHE_DIR = BASE_DIR / "data" / "cache"

for d in (MODEL_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- External API base URLs (verified live 2026-09-16) ---
GBIF_API_BASE = "https://api.gbif.org/v1"
OPEN_METEO_ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
OPEN_ELEVATION_BASE = "https://api.open-elevation.com/api/v1/lookup"
EBIRD_API_BASE = "https://api.ebird.org/v2"

# --- Optional keys, read from environment / .env ---
EBIRD_API_KEY = os.environ.get("EBIRD_API_KEY", "")
IUCN_API_KEY = os.environ.get("IUCN_API_KEY", "")
PROTECTED_PLANET_API_KEY = os.environ.get("PROTECTED_PLANET_API_KEY", "")

# --- Model / training parameters ---
CLIMATE_NORMAL_YEARS = 15          # how many past years to average for a "normal" day
PLAUSIBILITY_RADIUS_KM_DEFAULT = 200.0
MIN_OCCURRENCES_FOR_PLAUSIBILITY = 1
RANDOM_SEED = 42

# --- Flask ---
DEBUG = os.environ.get("WILDCAST_DEBUG", "1") == "1"
PORT = int(os.environ.get("PORT", "8000"))

# Daily weather variables pulled from Open-Meteo for both training and inference.
WEATHER_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "windspeed_10m_max",
    "cloudcover_mean",
]

"""
REAL data ingestion: pulls live historical daily weather + climate normals
from Open-Meteo for every pilot area and writes
data/cache/weather_daily_<area_id>.csv and data/cache/climate_normals_<area_id>.csv.

Run AFTER scripts/ingest_gbif.py (it reads the occurrence dates that need
weather). Requires outbound internet access to https://archive-api.open-meteo.com
(no API key). Run as: python -m scripts.ingest_weather
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CACHE_DIR, DATA_DIR  # noqa: E402  (needs the sys.path.insert above first)
from app.services import weather_client  # noqa: E402

PILOT_AREAS = json.loads((DATA_DIR / "pilot_areas.json").read_text())


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for area in PILOT_AREAS:
        area_id = area["id"]
        occ_path = CACHE_DIR / f"occurrences_{area_id}.csv"
        if not occ_path.exists():
            print(f"Skipping {area_id}: run scripts/ingest_gbif.py first.")
            continue

        occ = pd.read_csv(occ_path)
        if occ.empty:
            print(f"Skipping {area_id}: no occurrence records.")
            continue

        dates = pd.to_datetime(occ["date"])
        start, end = dates.min().date(), dates.max().date()
        print(f"Fetching daily weather for {area['name']}: {start} -> {end} ...")
        daily = weather_client.historical_daily(area["lat"], area["lon"], start.isoformat(), end.isoformat())
        daily.insert(0, "area_id", area_id)
        daily.to_csv(CACHE_DIR / f"weather_daily_{area_id}.csv", index=False)
        print(f"  wrote {len(daily)} rows -> weather_daily_{area_id}.csv")

        print(f"Computing climate normals for {area['name']} (this fetches ~15 years of history)...")
        normals = weather_client.climate_normals(area["lat"], area["lon"])
        normals.insert(0, "area_id", area_id)
        normals.to_csv(CACHE_DIR / f"climate_normals_{area_id}.csv", index=False)
        print(f"  wrote {len(normals)} rows -> climate_normals_{area_id}.csv")

        # A courtesy pause between areas. weather_client's own retry-with-
        # backoff (see _get_with_retry) is what actually rides out a 429 if
        # one happens anyway -- this is just to make one less likely in the
        # first place, since a shared CI-runner IP can trip Open-Meteo's
        # rate limit even at this low a request volume (real incident:
        # 2026-09-17, failed on the 3rd area's 5th request).
        time.sleep(2)


if __name__ == "__main__":
    main()

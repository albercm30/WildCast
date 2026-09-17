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
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CACHE_DIR, DATA_DIR  # noqa: E402  (needs the sys.path.insert above first)
from app.services import data_source_marker, weather_client  # noqa: E402

PILOT_AREAS = json.loads((DATA_DIR / "pilot_areas.json").read_text())

_FULL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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

        # This is a defensive second layer, not the real fix -- that's
        # scripts/ingest_gbif.py's _record_date, which now only ever writes
        # a real day-precision date. This just means a still-unparseable
        # date (a stale CSV committed before that fix, or a future GBIF
        # quirk nobody's hit yet) gets skipped with a warning instead of
        # crashing the whole ingestion run, the way a bare-year "2008" from
        # a Scottish Highlands record did on 2026-09-17 (pd.to_datetime,
        # with no format hint, locked onto "%Y-%m-%d" from the surrounding
        # good rows and then raised on that one bad one).
        #
        # The regex pre-filter matters: pd.to_datetime with errors="coerce"
        # alone does NOT reject a bare year like "2008" -- its flexible
        # parser happily fills in a fabricated "2008-01-01" instead of
        # rejecting it, which is exactly the fabricated-date problem this
        # whole fix is trying to avoid. Only a string that already looks
        # like a full YYYY-MM-DD date is handed to to_datetime at all; a
        # value that matches that shape but still isn't a real date (e.g.
        # "2008-13-40") is what errors="coerce" is there to catch.
        date_strs = occ["date"].astype(str)
        looks_like_a_full_date = date_strs.str.match(_FULL_DATE_RE)
        dates = pd.to_datetime(date_strs.where(looks_like_a_full_date), format="%Y-%m-%d", errors="coerce")
        n_bad = int(dates.isna().sum())
        if n_bad:
            print(f"  ({n_bad} of {len(dates)} occurrence dates for {area_id} were unparseable and skipped)")
            dates = dates.dropna()
        if dates.empty:
            print(f"Skipping {area_id}: no occurrence records with a parseable date.")
            continue
        start, end = dates.min().date(), dates.max().date()
        print(f"Fetching daily weather for {area['name']}: {start} -> {end} ...")
        daily = weather_client.historical_daily(area["lat"], area["lon"], start.isoformat(), end.isoformat())
        daily.insert(0, "area_id", area_id)
        daily.to_csv(CACHE_DIR / f"weather_daily_{area_id}.csv", index=False)
        print(f"  wrote {len(daily)} rows -> weather_daily_{area_id}.csv")

        # A courtesy pause between the two calls this area makes.
        time.sleep(5)

        print(f"Computing climate normals for {area['name']} (this fetches ~15 years of history)...")
        normals = weather_client.climate_normals(area["lat"], area["lon"])
        normals.insert(0, "area_id", area_id)
        normals.to_csv(CACHE_DIR / f"climate_normals_{area_id}.csv", index=False)
        print(f"  wrote {len(normals)} rows -> climate_normals_{area_id}.csv")

        # A courtesy pause between areas. weather_client's own retry-with-
        # backoff (see _get_with_retry) is what actually rides out a 429 if
        # one happens anyway (patiently -- up to ~4 minutes per request as of
        # this version) -- this pause is just to make one less likely in the
        # first place, since a shared CI-runner IP can trip Open-Meteo's
        # rate limit even at this low a request volume (real incidents,
        # 2026-09-17: one run failed with no retry logic at all; a second,
        # with a shorter/5-retry schedule, still exhausted retries on the
        # last, largest area).
        time.sleep(8)

    data_source_marker.mark("real", {"ingested_by": "scripts.ingest_weather"})


if __name__ == "__main__":
    main()

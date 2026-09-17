"""
REAL data ingestion: pulls live occurrence records from GBIF for every seed
species in every pilot area and writes data/cache/occurrences_<area_id>.csv.

Requires outbound internet access to https://api.gbif.org (no API key).
Run as: python -m scripts.ingest_gbif
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CACHE_DIR, DATA_DIR  # noqa: E402  (needs the sys.path.insert above first)
from app.services import data_source_marker, gbif_client  # noqa: E402

PILOT_AREAS = {a["id"]: a for a in json.loads((DATA_DIR / "pilot_areas.json").read_text())}
SEED_SPECIES = json.loads((DATA_DIR / "seed_species.json").read_text())

_FULL_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _record_date(rec: dict[str, Any]) -> str | None:
    """
    A single day-precision 'YYYY-MM-DD' date for one GBIF occurrence record,
    or None if the record doesn't actually have day-level precision. Used to
    join a sighting to the one day's weather it happened on
    (scripts/ingest_weather.py + app.ml.train's positive-example
    construction), so a date without a real day is worse than useless --
    it would either crash downstream or silently fabricate a day the
    sighting may not have happened on.

    GBIF's `eventDate` is NOT reliably a full ISO date. This function used
    to just take `rec["eventDate"] or f"{year}-{month or 1:02d}-{day or 1:02d}"`
    and blindly slice the first 10 characters -- which crashed a real
    `retrain.yml` run on 2026-09-17: `pd.to_datetime` in
    scripts/ingest_weather.py choked on the literal string "2008" (a bare
    year, no month/day at all) that had come from a Scottish Highlands
    occurrence record whose `eventDate` was just "2008" -- `str("2008")[:10]`
    is still "2008", not a valid date, but the old code never checked for
    that and happily wrote it into occurrences_scottish_highlands.csv.
    `eventDate` can also be year-month ("2008-05"), an ISO8601 interval
    ("2008-05-01/2008-05-03", for which the interval's start is used), or a
    full timestamp ("2008-05-01T00:00:00"). And the old year/month/day
    fallback path had a quieter version of the same problem: it defaulted a
    missing month or day to 1, fabricating a plausible-looking "January 1st"
    date for a record that might only have year-level precision at all.
    """
    raw = rec.get("eventDate")
    if raw:
        start = str(raw).split("/", 1)[0]  # take the start of an interval, if any
        match = _FULL_DATE_RE.match(start)
        if match:
            return match.group(0)
        # Not day-precision (bare year, year-month, or something malformed)
        # -- fall through to the year/month/day fields below rather than
        # guessing.
    year, month, day = rec.get("year"), rec.get("month"), rec.get("day")
    if year and month and day:
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return None


def ingest_area(area_id: str) -> pd.DataFrame:
    area = PILOT_AREAS[area_id]
    species_here = [s for s in SEED_SPECIES if s["area_id"] == area_id]
    rows = []
    for sp in species_here:
        print(f"  fetching GBIF occurrences: {sp['scientific_name']} near {area['name']} ...")
        records = gbif_client.occurrences_near(
            sp["scientific_name"], area["lat"], area["lon"], area["radius_km"], max_records=500
        )
        for rec in records:
            date = _record_date(rec)
            lat, lon = rec.get("decimalLatitude"), rec.get("decimalLongitude")
            if not date or lat is None or lon is None:
                continue
            rows.append(
                {
                    "species": sp["scientific_name"],
                    "common_name": sp["common_name"],
                    "taxon_class": sp["taxon_class"],
                    "area_id": area_id,
                    "lat": lat,
                    "lon": lon,
                    "date": date,
                }
            )
        print(f"    -> {len(records)} records")
        time.sleep(0.2)
    return pd.DataFrame(rows)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for area_id in PILOT_AREAS:
        print(f"Ingesting {area_id}...")
        df = ingest_area(area_id)
        out_path = CACHE_DIR / f"occurrences_{area_id}.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {len(df)} rows -> {out_path}")
    data_source_marker.mark("real", {"ingested_by": "scripts.ingest_gbif"})


if __name__ == "__main__":
    main()

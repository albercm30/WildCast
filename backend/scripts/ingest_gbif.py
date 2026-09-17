"""
REAL data ingestion: pulls live occurrence records from GBIF for every seed
species in every pilot area and writes data/cache/occurrences_<area_id>.csv.

Requires outbound internet access to https://api.gbif.org (no API key).
Run as: python -m scripts.ingest_gbif
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CACHE_DIR, DATA_DIR  # noqa: E402  (needs the sys.path.insert above first)
from app.services import data_source_marker, gbif_client  # noqa: E402

PILOT_AREAS = {a["id"]: a for a in json.loads((DATA_DIR / "pilot_areas.json").read_text())}
SEED_SPECIES = json.loads((DATA_DIR / "seed_species.json").read_text())


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
            date = rec.get("eventDate") or (
                f"{rec.get('year')}-{rec.get('month', 1):02d}-{rec.get('day', 1):02d}"
                if rec.get("year") else None
            )
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
                    "date": str(date)[:10],
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

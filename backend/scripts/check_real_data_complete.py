"""
Exit 0 if EVERY pilot area has a complete, committed real data cache
(occurrences + daily weather + climate normals); exit 1 otherwise. Used by
docker/Dockerfile.backend's build step to decide whether to trust
data/cache/ as-is or fall back to generating the synthetic demo dataset for
all areas.

Why this exists: the previous check was `[ -f data/cache/occurrences_yellowstone.csv ]`
-- true the moment ANY real data had ever been committed, even if it only
covered the original 5 areas from an early round. Once
`backend/data/pilot_areas.json` grew to 10 areas (see the README's "Catalog
expansion" section) without a fresh `scripts.ingest_gbif`/`ingest_weather`
run covering all of them, that check kept passing (Yellowstone's real file
was still there) while 5 areas had no cache at all -- so the Docker build
skipped the synthetic-fixtures fallback, trusted the incomplete cache, and
`app.ml.train`'s `_load_area_caches()` crashed the build with a
`FileNotFoundError` for the first missing area it hit. A real, observed
failure, not a hypothetical.

This is deliberately all-or-nothing, matching `app.services.data_source_marker`'s
own model: training needs one consistent cache across every area, so a
partial real dataset always falls back to a full synthetic dataset rather
than silently mixing real data for some areas with synthetic for others
(which would also make the single "real" vs "synthetic_demo" data-source
label a lie for whichever areas didn't match it).

Run as: python -m scripts.check_real_data_complete
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CACHE_DIR, DATA_DIR  # noqa: E402  (needs the sys.path.insert above first)

PILOT_AREAS = json.loads((DATA_DIR / "pilot_areas.json").read_text())


def missing_areas() -> list[str]:
    missing = []
    for area in PILOT_AREAS:
        area_id = area["id"]
        required = (
            CACHE_DIR / f"occurrences_{area_id}.csv",
            CACHE_DIR / f"weather_daily_{area_id}.csv",
            CACHE_DIR / f"climate_normals_{area_id}.csv",
        )
        if not all(path.exists() for path in required):
            missing.append(area_id)
    return missing


def main() -> int:
    missing = missing_areas()
    if missing:
        print(f"Incomplete real data cache -- missing area(s): {', '.join(missing)}", file=sys.stderr)
        return 1
    print(f"Real data cache is complete for all {len(PILOT_AREAS)} pilot areas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

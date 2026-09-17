"""
OPTIONAL real data ingestion: pulls live IUCN Red List conservation status
(category + population trend) for every seed species and writes
data/cache/conservation_status.json.

Why this is a separate script, not just a live call inside app.ml.train:
GBIF/weather ingestion follows a "fetch once here (real network + secrets
available), commit the result to data/cache/, then app.ml.train just reads
the committed file" pattern -- see ingest_gbif.py/ingest_weather.py. IUCN
conservation status originally skipped that pattern and called
iucn_client.py directly from inside app.ml.train.train() instead. That's a
real bug, not just an inconsistency: the model actually shipped to
production is trained by `RUN python -m app.ml.train` inside
docker/Dockerfile.backend's build step, and a Docker build has no access to
GitHub repo secrets (like IUCN_API_KEY) unless explicitly wired through as
a build arg -- which nothing here does. So even a correctly-configured
IUCN_API_KEY repo secret would silently never reach the deployed model.

The fix: run this script from .github/workflows/retrain.yml (same runner
as ingest_gbif.py/ingest_weather.py, where the secret genuinely is
available as an env var and real network access exists), commit
conservation_status.json alongside the rest of data/cache/, and have
app.ml.train._compute_conservation_status() prefer that committed file over
a live call. app.ml.train still falls back to a live iucn_client call when
no cache file exists yet, purely as a local-dev convenience for someone
with IUCN_API_KEY set in their own .env running `python -m app.ml.train`
directly.

Requires outbound internet access to https://api.iucnredlist.org and
IUCN_API_KEY set (get a free key: https://api.iucnredlist.org). Skips
cleanly (exit 0, cache untouched) when no key is configured -- this is
optional enrichment, not a required ingestion step, so it must never fail
`retrain.yml`'s run just because the user hasn't gotten a key yet.
Run as: python -m scripts.ingest_iucn
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CACHE_DIR, DATA_DIR  # noqa: E402  (needs the sys.path.insert above first)
from app.services import iucn_client  # noqa: E402

SEED_SPECIES = json.loads((DATA_DIR / "seed_species.json").read_text())
CONSERVATION_STATUS_PATH = CACHE_DIR / "conservation_status.json"


def main():
    if not iucn_client.is_configured():
        print("IUCN_API_KEY not set -- skipping (optional; see .env.example). Leaving any existing cache untouched.")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    status: dict[str, dict] = {}
    for sp in SEED_SPECIES:
        name = sp["scientific_name"]
        try:
            assessment = iucn_client.species_assessment(name)
        except Exception as exc:
            print(f"  IUCN lookup failed for {name}, skipping: {exc}")
            continue
        if not assessment or not assessment.get("category"):
            print(f"  No usable IUCN assessment for {name}.")
            continue
        status[name] = {
            **assessment,
            "category_label": iucn_client.CATEGORY_LABELS.get(assessment["category"], assessment["category"]),
        }
        print(f"  {name}: {status[name]['category_label']} (population trend: {assessment.get('population_trend')})")

    if not status:
        # A key IS configured but nothing came back for any species -- more
        # likely a transient IUCN outage or an API-shape mismatch (see this
        # module's docstring about unverified field names) than "none of 50
        # species have an assessment." Don't clobber a previously-good
        # committed cache with an empty one over a bad day.
        if CONSERVATION_STATUS_PATH.exists():
            print("No species returned a usable assessment this run -- leaving the existing committed cache as-is.")
        else:
            print("No species returned a usable assessment this run, and no cache exists yet -- nothing written.")
        return

    CONSERVATION_STATUS_PATH.write_text(json.dumps(status, indent=2))
    print(f"Wrote conservation status for {len(status)}/{len(SEED_SPECIES)} species -> {CONSERVATION_STATUS_PATH}")


if __name__ == "__main__":
    main()

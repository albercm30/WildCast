"""
Optional IUCN Red List API v4 client. Skipped automatically when
IUCN_API_KEY is unset -- request a free key at https://api.iucnredlist.org
(account signup; approval is not always instant, unlike eBird's). Adds real
conservation-status context (Red List category + population trend) per
species: not a sighting-evidence source like GBIF/iNaturalist/eBird, but a
real, authoritative signal that a rare/declining species should be expected
to have fewer sightings than a common one, independent of local search
effort -- used only at TRAINING time, since a species' Red List status
doesn't change per location or date the way live occurrence counts do. In
production this module is called from `scripts/ingest_iucn.py` (run on
GitHub's runners via .github/workflows/retrain.yml, where the
IUCN_API_KEY secret and real network access both exist) rather than from
`app.ml.train` directly -- see that script's docstring for why: a Docker
build (where `app.ml.train` actually runs to produce the deployed model)
has neither secrets nor a guaranteed live-API path by default.
`app.ml.train._compute_conservation_status()` still calls this module
directly too, as a local-dev convenience when IUCN_API_KEY is set in your
own `.env` and no committed cache file exists yet.

Built from the IUCN Red List API v4's public documentation -- this
sandbox's locked-down egress cannot reach api.iucnredlist.org to verify
live (the same restriction gbif_client.py/inaturalist_client.py already
document for their own APIs), and the exact response field names below
(`red_list_category`/`population_trend`) are the best read of v4's public
docs, not confirmed against a real response. Re-verify field names against
a real call (once you have a key) before trusting this in production --
`species_assessment()` is written defensively (checks a couple of
plausible field-name variants) but may still need a small fix.
"""
from __future__ import annotations

from typing import Any

import requests

from app.config import IUCN_API_BASE, IUCN_API_KEY


class IUCNUnavailable(RuntimeError):
    """Raised when IUCN integration is used without an API key configured."""


# IUCN Red List category codes -> plain-language labels, for the factor
# text in prediction_service._explain(). Codes per IUCN's own standard
# categories. Lives here (not app.ml.train) so both app.ml.train (live
# fallback path) and scripts.ingest_iucn (the real production path -- see
# that script's docstring) can share one definition.
CATEGORY_LABELS = {
    "EX": "Extinct",
    "EW": "Extinct in the Wild",
    "CR": "Critically Endangered",
    "EN": "Endangered",
    "VU": "Vulnerable",
    "NT": "Near Threatened",
    "LC": "Least Concern",
    "DD": "Data Deficient",
    "NE": "Not Evaluated",
}


def is_configured() -> bool:
    return bool(IUCN_API_KEY)


def _headers() -> dict[str, str]:
    if not IUCN_API_KEY:
        raise IUCNUnavailable(
            "IUCN_API_KEY is not set. Request a free key at https://api.iucnredlist.org "
            "and add it to your .env to enable real conservation-status context in predictions."
        )
    return {"Authorization": f"Bearer {IUCN_API_KEY}"}


def species_assessment(scientific_name: str, timeout: float = 15.0) -> dict[str, Any] | None:
    """
    The current (latest published) Red List assessment for a species, or
    None if IUCN has no assessment findable under this exact scientific
    name (could be a real "not assessed" species, or a taxonomic name
    mismatch -- IUCN's own accepted names can differ from GBIF's; this is
    not distinguished here, it's simply "no usable status available").
    """
    genus, _, epithet = scientific_name.partition(" ")
    resp = requests.get(
        f"{IUCN_API_BASE}/taxa/scientific_name",
        headers=_headers(),
        params={"genus_name": genus, "species_name": epithet},
        timeout=timeout,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    assessments = payload.get("assessments") or []
    if not assessments:
        return None
    # Sort explicitly rather than assume the API returns newest-first.
    latest = max(assessments, key=lambda a: a.get("year_published") or 0)
    category = latest.get("red_list_category")
    category_code = category.get("code") if isinstance(category, dict) else latest.get("red_list_category_code")
    return {
        "category": category_code,
        "population_trend": latest.get("population_trend"),
        "year_published": latest.get("year_published"),
    }

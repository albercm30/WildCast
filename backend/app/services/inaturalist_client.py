"""
Thin client for the iNaturalist API (public, no key needed for read-only
search -- see app/config.py's docstring). Built the same way gbif_client.py
was: from the API's public documentation, not a live call this sandbox's
locked-down egress can make (same restriction gbif_client.py's module
docstring already documents for GBIF). Re-verify against a real call (or
via .github/workflows/retrain.yml, which runs on GitHub's own unrestricted
runners) before trusting this in production -- the unit tests here check
the CODE (URL/param construction, response parsing), not that iNaturalist's
API still matches this exact shape.

Why iNaturalist alongside GBIF: GBIF already aggregates a large share of
iNaturalist's "research grade" (community-vetted) observations as one of
its many contributing datasets, so the two are NOT independent record
counts -- summing them would double-count a lot of the same underlying
sightings. Instead, WildCast uses iNaturalist as an INDEPENDENT
CORROBORATION check (does a second real database also confirm this species
here, separately from whatever GBIF returned), not as more volume to add to
GBIF's count -- see app.ml.prediction_service._local_evidence_cached for
how the two are combined without double-counting.

iNaturalist's own `captive` observation field is more direct than GBIF's
basisOfRecord for excluding zoo/garden animals: it's a first-class boolean
on every observation (`captive_cultivated` in the API's filter terms),
not something you have to infer from institutional metadata quality.
"""
from __future__ import annotations

from typing import Any

import requests

from app.config import INATURALIST_API_BASE

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "WildCast/0.1 (wildlife-encounter-forecast)"})


def _get(path: str, params: dict[str, Any], timeout: float = 15.0) -> dict[str, Any]:
    resp = _SESSION.get(f"{INATURALIST_API_BASE}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def search_observations(
    *,
    scientific_name: str,
    lat: float,
    lon: float,
    radius_km: float,
    wild_only: bool = True,
    quality_grade: str = "research",
    per_page: int = 1,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """
    `quality_grade="research"` restricts to community-vetted, ID-confirmed
    observations (iNaturalist's own highest-confidence tier) -- deliberately
    stricter than GBIF's default, since this is a cross-check on GBIF's own
    evidence, not a replacement for it; a "casual" grade observation is not
    strong-enough independent corroboration to be worth much here.
    """
    params: dict[str, Any] = {
        "taxon_name": scientific_name,
        "lat": lat,
        "lng": lon,
        "radius": radius_km,  # kilometers
        "per_page": per_page,
    }
    if wild_only:
        params["captive"] = "false"
    if quality_grade:
        params["quality_grade"] = quality_grade
    return _get("/observations", params, timeout=timeout)


def presence_count(
    scientific_name: str, lat: float, lon: float, radius_km: float, timeout: float = 15.0,
) -> int:
    """The real number of research-grade, non-captive iNaturalist observations
    of `scientific_name` within `radius_km` of (lat, lon). Same shape as
    gbif_client.presence_count -- see app.ml.prediction_service for how the
    two are combined as independent corroboration, not summed as raw volume."""
    page = search_observations(
        scientific_name=scientific_name, lat=lat, lon=lon, radius_km=radius_km, timeout=timeout,
    )
    return int(page.get("total_results", 0) or 0)

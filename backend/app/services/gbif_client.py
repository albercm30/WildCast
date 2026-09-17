"""
Thin, real client for the GBIF (Global Biodiversity Information Facility) API.

Endpoints and response fields here were verified live against
https://api.gbif.org/v1 on 2026-09-16 (species/match, species/search,
occurrence/search). No API key is required for these read endpoints.

GBIF's occurrence/search does not take a "radius" parameter directly, so
`occurrences_near` builds a simple lat/lon bounding box (adequate at the
tens-to-hundreds-of-km scale WildCast pilots operate at; a proper geodesic
buffer via the `geometry` WKT parameter is a drop-in upgrade for Phase 1).
"""
from __future__ import annotations

import math
import time
from typing import Any

import requests

from app.config import GBIF_API_BASE

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "WildCast/0.1 (wildlife-encounter-forecast)"})

_MAX_PAGE_SIZE = 300  # GBIF's hard per-request cap

# GBIF's basisOfRecord vocabulary includes LIVING_SPECIMEN (an individual
# held alive in a collection -- in practice this is dominated by zoo,
# aquarium, and botanical-garden holdings) and FOSSIL_SPECIMEN. Neither
# means "this species lives here": a zoo lion a few km from a user's click
# is a real GBIF record, correctly geocoded, and a completely wrong answer
# to "would you encounter this animal in the wild near this point?" --
# WildCast's whole premise. Excluding both from presence/plausibility
# checks is the standard mitigation for exactly this class of false
# positive (found via a real report: a lion and leopard both showing up as
# "verified" near Bangkok, Thailand -- Bangkok has multiple zoos and safari
# parks; the leopard is very plausibly a genuine wild Indochinese leopard
# record, but the lion almost certainly is not, since lions have no wild
# range anywhere near Southeast Asia).
_WILD_BASIS_OF_RECORD = [
    "HUMAN_OBSERVATION",
    "OBSERVATION",
    "MACHINE_OBSERVATION",
    "PRESERVED_SPECIMEN",
    "MATERIAL_SAMPLE",
    "OCCURRENCE",
]


def _get(path: str, params: dict[str, Any], timeout: float = 20.0) -> dict[str, Any]:
    resp = _SESSION.get(f"{GBIF_API_BASE}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def match_species(scientific_name: str) -> dict[str, Any]:
    """Resolve a scientific name to GBIF's canonical taxon record (usageKey, rank, kingdom..species)."""
    return _get("/species/match", {"name": scientific_name})


def search_species(query: str, rank: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """Free-text species search, e.g. for an autocomplete box in the UI."""
    params: dict[str, Any] = {"q": query, "limit": limit}
    if rank:
        params["rank"] = rank
    return _get("/species/search", params).get("results", [])


def _bbox_from_point(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon) for a simple equirectangular bounding box."""
    dlat = radius_km / 111.0  # ~111 km per degree of latitude everywhere
    # degrees of longitude per km shrinks toward the poles
    dlon = radius_km / (111.320 * max(math.cos(math.radians(lat)), 0.1))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def occurrence_search(
    *,
    scientific_name: str | None = None,
    taxon_key: int | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float | None = None,
    year_range: tuple[int, int] | None = None,
    country: str | None = None,
    limit: int = 300,
    offset: int = 0,
    timeout: float = 20.0,
    wild_only: bool = False,
) -> dict[str, Any]:
    """One page of GBIF occurrence records matching the given filters.

    `wild_only=True` excludes captive (LIVING_SPECIMEN) and fossil
    (FOSSIL_SPECIMEN) records -- see `_WILD_BASIS_OF_RECORD`'s comment.
    Off by default (existing callers like `occurrences_near`, used to build
    real training examples from dated sighting records, want everything);
    `has_any_presence` turns it on, since a zoo record is exactly the wrong
    answer to the plausibility question it's asking.
    """
    params: dict[str, Any] = {
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "limit": min(limit, _MAX_PAGE_SIZE),
        "offset": offset,
    }
    if scientific_name:
        params["scientificName"] = scientific_name
    if taxon_key:
        params["taxonKey"] = taxon_key
    if country:
        params["country"] = country
    if year_range:
        params["year"] = f"{year_range[0]},{year_range[1]}"
    if wild_only:
        params["basisOfRecord"] = _WILD_BASIS_OF_RECORD
    if lat is not None and lon is not None and radius_km:
        min_lat, max_lat, min_lon, max_lon = _bbox_from_point(lat, lon, radius_km)
        params["decimalLatitude"] = f"{min_lat:.4f},{max_lat:.4f}"
        params["decimalLongitude"] = f"{min_lon:.4f},{max_lon:.4f}"
    return _get("/occurrence/search", params, timeout=timeout)


def occurrences_near(
    scientific_name: str,
    lat: float,
    lon: float,
    radius_km: float,
    max_records: int = 1000,
    years_back: int = 20,
    polite_delay_s: float = 0.15,
) -> list[dict[str, Any]]:
    """
    Page through GBIF occurrence/search for one species near a point, up to
    `max_records`. Used both to build the plausibility gate (>=1 record ever)
    and to supply positive training examples (record dates -> weather join).
    """
    import datetime as dt

    this_year = dt.date.today().year
    results: list[dict[str, Any]] = []
    offset = 0
    while len(results) < max_records:
        page = occurrence_search(
            scientific_name=scientific_name,
            lat=lat,
            lon=lon,
            radius_km=radius_km,
            year_range=(this_year - years_back, this_year),
            limit=_MAX_PAGE_SIZE,
            offset=offset,
        )
        batch = page.get("results", [])
        results.extend(batch)
        if page.get("endOfRecords", True) or not batch:
            break
        offset += _MAX_PAGE_SIZE
        time.sleep(polite_delay_s)  # be a good API citizen
    return results[:max_records]


def has_any_presence(
    scientific_name: str, lat: float, lon: float, radius_km: float, timeout: float = 20.0
) -> bool:
    """Cheap plausibility check: does GBIF have >=1 *wild* record of this species near this point, ever?

    `wild_only=True` (excludes zoo/captive and fossil records -- see
    `_WILD_BASIS_OF_RECORD`): a captive record passing this check would
    mean WildCast telling someone they might encounter a lion near a
    Bangkok zoo, which is real GBIF data but a wrong answer to the
    question this function exists to answer.

    `timeout` is lower by default for interactive "Explore Anywhere" callers
    (see app.ml.prediction_service.species_for_location, which runs many of
    these concurrently against a real user's click) than for the batch
    ingestion scripts, which can afford to wait longer per call.
    """
    page = occurrence_search(
        scientific_name=scientific_name,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        limit=1,
        timeout=timeout,
        wild_only=True,
    )
    return page.get("count", 0) > 0

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
) -> dict[str, Any]:
    """One page of GBIF occurrence records matching the given filters."""
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
    if lat is not None and lon is not None and radius_km:
        min_lat, max_lat, min_lon, max_lon = _bbox_from_point(lat, lon, radius_km)
        params["decimalLatitude"] = f"{min_lat:.4f},{max_lat:.4f}"
        params["decimalLongitude"] = f"{min_lon:.4f},{max_lon:.4f}"
    return _get("/occurrence/search", params)


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


def has_any_presence(scientific_name: str, lat: float, lon: float, radius_km: float) -> bool:
    """Cheap plausibility check: does GBIF have >=1 record of this species near this point, ever?"""
    page = occurrence_search(scientific_name=scientific_name, lat=lat, lon=lon, radius_km=radius_km, limit=1)
    return page.get("count", 0) > 0

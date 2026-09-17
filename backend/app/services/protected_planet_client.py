"""
Optional Protected Planet (WDPA -- World Database on Protected Areas) v4 API
client. Skipped automatically when PROTECTED_PLANET_API_KEY is unset --
request a free key at https://api.protectedplanet.net/request (same
opt-in-key pattern as eBird/IUCN; see app.services.ebird_client and
app.services.iucn_client).

Powers the "Browse Parks" feature (Task #20): a country + filter browser
over the world's ~313,000 protected areas, NOT a free-text "type a park
name" search box. That distinction is deliberate, not a shortcut -- real
research against the live v4 documentation (https://api.protectedplanet.net/documentation,
fetched 2026-09-17) found:

  - Neither the deprecated v3 API nor the current v4 API has a free-text
    name-search endpoint. `GET /v4/protected_areas` (the bare list) only
    takes `with_geometry`/`page`/`per_page`. Filtering by country,
    designation, governance, IUCN category, or marine status lives on a
    SEPARATE endpoint, `GET /v4/protected_areas/search`, used below.
  - `country` on that search endpoint is a 3-letter ISO 3166-1 alpha-3 code
    (see data/countries.py for the static code->name table used to build a
    country picker, since the API itself has no "list countries" endpoint).
  - `designation`, `governance`, and `iucn_category` are documented as
    INTEGER IDs, not free-text strings or codes, and the docs page didn't
    give the id->label mapping. Exposing those as user-facing filters would
    mean either hardcoding a guessed id table (fragile -- exactly the kind
    of unverified guess this project's history (see iucn_client.py's
    docstring) has burned us on before) or a separate lookup call this API
    doesn't offer. So this module accepts them as pass-through *args for
    future use once verified against a real key, but the current Browse
    Parks UI only wires up `country` and `marine` (both unambiguous).
  - A global bulk-download alternative (the monthly WDPA geodatabase dump)
    exists and WOULD support real name search via a locally-built index,
    but it's ~1.1GB as a File Geodatabase only (no CSV), needs a GDAL
    pipeline this sandbox cannot install (PyPI is blocked here, same as the
    SQLAlchemy issue elsewhere in this project), has no pre-computed
    centroid field, and is licensed NON-COMMERCIAL ONLY by UNEP-WCMC --
    which conflicts with WildCast's stated monetization plans and would
    need a separate license resolved before shipping. The user chose the
    lighter-weight country+filter browsing approach instead (2026-09-17)
    specifically to avoid that cost and licensing risk; revisit only if
    that decision changes.
  - List/single-record responses are wrapped (`{"protected_areas": [...]}`
    / `{"protected_area": {...}}`), not bare arrays/objects, and the docs
    page did not show a fully worked example response, so field names below
    (`name`, `site_id`, `designation`, `iucn_category`, `marine`, `geojson`,
    `countries`) are read from the docs' per-field descriptions, not a
    single confirmed real payload -- same "verified params, best-read
    fields" confidence level iucn_client.py already flags for itself.
    `_parse_protected_area` is written defensively (tries a couple of
    plausible key variants) but may need a small fix once a real
    PROTECTED_PLANET_API_KEY is available to test against. No documented
    rate limit was found for v4.
"""
from __future__ import annotations

from typing import Any

import requests

from app.config import PROTECTED_PLANET_API_BASE, PROTECTED_PLANET_API_KEY

_DEFAULT_TIMEOUT = 20.0
_MAX_PER_PAGE = 50  # documented API cap -- requesting more silently gets clamped server-side, so clamp here too


class ProtectedPlanetUnavailable(RuntimeError):
    """Raised when Protected Planet integration is used without an API key configured."""


def is_configured() -> bool:
    return bool(PROTECTED_PLANET_API_KEY)


def _token() -> str:
    if not PROTECTED_PLANET_API_KEY:
        raise ProtectedPlanetUnavailable(
            "PROTECTED_PLANET_API_KEY is not set. Request a free key at "
            "https://api.protectedplanet.net/request and add it to your .env to enable park browsing."
        )
    return PROTECTED_PLANET_API_KEY


def _centroid_from_geojson(geojson: dict[str, Any] | None) -> dict[str, float] | None:
    """
    Approximate centroid (plain average of every polygon vertex, NOT a true
    area-weighted centroid) for pointing WildCast's existing "Explore
    Anywhere" lat/lon prediction flow somewhere inside a park. A vertex
    average is a reasonable approximation for typical park shapes and is
    the right tradeoff here: a real area-weighted centroid needs a GIS
    library (Shapely) this sandbox can't pip-install (see module
    docstring), and this value is only ever used to seed a "jump to this
    area" click, not for anything that needs geometric precision.
    """
    if not geojson:
        return None
    geometry = geojson.get("geometry", geojson)  # tolerate a bare geometry or a full Feature
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return None

    points: list[tuple[float, float]] = []

    def _walk(node: Any, depth: int) -> None:
        # GeoJSON coordinate nesting depth: Polygon rings are depth 2
        # (list of rings -> list of [lon, lat] pairs), MultiPolygon adds one
        # more level (list of polygons). Recurse until we hit a raw
        # [lon, lat] pair (a 2-element list of numbers).
        if depth == 0 and isinstance(node, (list, tuple)) and len(node) >= 2 and all(
            isinstance(v, (int, float)) for v in node[:2]
        ):
            points.append((float(node[0]), float(node[1])))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child, max(depth - 1, 0))

    start_depth = {"Polygon": 2, "MultiPolygon": 3, "Point": 0}.get(gtype, 2)
    _walk(coords, start_depth)
    if not points:
        return None
    lon = sum(p[0] for p in points) / len(points)
    lat = sum(p[1] for p in points) / len(points)
    return {"lat": lat, "lon": lon}


def _parse_protected_area(raw: dict[str, Any]) -> dict[str, Any]:
    designation = raw.get("designation") or {}
    iucn_category = raw.get("iucn_category") or {}
    countries = raw.get("countries") or []
    geojson = raw.get("geojson")
    return {
        "site_id": raw.get("site_id") or raw.get("wdpa_id") or raw.get("id"),
        "name": raw.get("name") or raw.get("name_english") or raw.get("original_name"),
        "designation": designation.get("name") if isinstance(designation, dict) else designation,
        "iucn_category": iucn_category.get("name") if isinstance(iucn_category, dict) else iucn_category,
        "marine": bool(raw.get("marine")),
        "countries": [
            {"iso3": c.get("iso_3") or c.get("iso3"), "name": c.get("name")}
            for c in countries
            if isinstance(c, dict)
        ],
        "centroid": _centroid_from_geojson(geojson),
    }


def search_protected_areas(
    country_iso3: str | None = None,
    marine: bool | None = None,
    page: int = 1,
    per_page: int = 25,
    with_geometry: bool = True,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    Country + marine-status filtered browse over WDPA protected areas
    (see module docstring for why this isn't a free-text name search).
    Returns {"protected_areas": [...normalized...], "page": int, "per_page": int}.
    `with_geometry=True` is needed to get a usable centroid back per result
    (see _centroid_from_geojson) -- set False only for a lighter-weight
    list call that doesn't need "explore this park" coordinates.
    """
    params: dict[str, Any] = {
        "token": _token(),
        "page": max(1, page),
        "per_page": max(1, min(per_page, _MAX_PER_PAGE)),
        "with_geometry": str(with_geometry).lower(),
    }
    if country_iso3:
        params["country"] = country_iso3.upper()
    if marine is not None:
        params["marine"] = str(marine).lower()

    resp = requests.get(f"{PROTECTED_PLANET_API_BASE}/protected_areas/search", params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    raw_areas = payload.get("protected_areas") or []
    return {
        "protected_areas": [_parse_protected_area(a) for a in raw_areas],
        "page": params["page"],
        "per_page": params["per_page"],
        # The API doesn't document a total-count/total-pages field (see
        # module docstring), so callers can only infer "there's probably a
        # next page" from a full page of results, not a real total.
        "has_more": len(raw_areas) >= params["per_page"],
    }


def get_protected_area(site_id: str, timeout: float = _DEFAULT_TIMEOUT) -> dict[str, Any] | None:
    """A single protected area by its WDPA site_id, or None if not found."""
    resp = requests.get(
        f"{PROTECTED_PLANET_API_BASE}/protected_areas/{site_id}",
        params={"token": _token(), "with_geometry": "true"},
        timeout=timeout,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    raw = payload.get("protected_area")
    if not raw:
        return None
    return _parse_protected_area(raw)

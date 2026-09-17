"""
Optional eBird API 2.0 client. Skipped automatically when EBIRD_API_KEY is
unset (get a free key in seconds at https://ebird.org/api/keygen -- no
approval wait). Adds bird-specific detection-frequency signal that is much
denser than GBIF alone in well-birded regions.

Base URL and auth header verified against eBird API 2.0's public docs
(https://documenter.getpostman.com/view/664302/S1ENwy59): every request
carries an `X-eBirdApiToken` header.
"""
from __future__ import annotations

from typing import Any

import requests

from app.config import EBIRD_API_BASE, EBIRD_API_KEY


class EBirdUnavailable(RuntimeError):
    """Raised when eBird integration is used without an API key configured."""


def _headers() -> dict[str, str]:
    if not EBIRD_API_KEY:
        raise EBirdUnavailable(
            "EBIRD_API_KEY is not set. Get a free key at https://ebird.org/api/keygen "
            "and add it to your .env to enable eBird-enriched bird predictions."
        )
    return {"X-eBirdApiToken": EBIRD_API_KEY}


def is_configured() -> bool:
    return bool(EBIRD_API_KEY)


def recent_observations(region_code: str, back_days: int = 14) -> list[dict[str, Any]]:
    """Recent bird observations for an eBird region code (e.g. 'US-WY' or a lat/lon-derived code)."""
    resp = requests.get(
        f"{EBIRD_API_BASE}/data/obs/{region_code}/recent",
        headers=_headers(),
        params={"back": back_days},
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


def nearby_observations(lat: float, lon: float, dist_km: int = 25, back_days: int = 14) -> list[dict[str, Any]]:
    """Recent observations within `dist_km` of a point -- the call WildCast actually uses per-area."""
    resp = requests.get(
        f"{EBIRD_API_BASE}/data/obs/geo/recent",
        headers=_headers(),
        params={"lat": lat, "lng": lon, "dist": dist_km, "back": back_days},
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


def presence_count(scientific_name: str, lat: float, lon: float, radius_km: float, back_days: int = 30) -> int:
    """
    Count of distinct recent eBird checklist reports of `scientific_name`
    within `radius_km` of (lat, lon) in the last `back_days` -- used as an
    independent, bird-specific corroboration source alongside GBIF and
    iNaturalist (see app.ml.prediction_service._local_evidence_cached).
    Raises EBirdUnavailable if no key is configured -- callers that want a
    graceful skip instead should check `is_configured()` first, the same
    pattern the rest of this module already uses.

    Reuses `nearby_observations` (all species near a point) and filters
    client-side by scientific name, rather than adding a new endpoint: eBird
    API 2.0 does have a species-specific nearby-observations endpoint
    (`/data/obs/geo/recent/{speciesCode}`), but that needs a scientific-name
    -> eBird species-code lookup this client doesn't yet do, and this
    approach is correct with code already written and already tested.
    eBird's own `dist` parameter caps at 50km, unlike GBIF/iNaturalist's
    much larger search radii, so this is a tighter-radius signal by nature
    of the API, not a WildCast choice.
    """
    dist_km = min(int(round(radius_km)), 50)
    observations = nearby_observations(lat, lon, dist_km=dist_km, back_days=back_days)
    return sum(1 for obs in observations if obs.get("sciName") == scientific_name)


def historic_observations(region_code: str, year: int, month: int, day: int) -> list[dict[str, Any]]:
    """What was reported in this region on this exact calendar date in a past year -- used to build phenology."""
    resp = requests.get(
        f"{EBIRD_API_BASE}/data/obs/{region_code}/historic/{year}/{month}/{day}",
        headers=_headers(),
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()

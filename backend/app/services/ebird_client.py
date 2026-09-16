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


def historic_observations(region_code: str, year: int, month: int, day: int) -> list[dict[str, Any]]:
    """What was reported in this region on this exact calendar date in a past year -- used to build phenology."""
    resp = requests.get(
        f"{EBIRD_API_BASE}/data/obs/{region_code}/historic/{year}/{month}/{day}",
        headers=_headers(),
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()

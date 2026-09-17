"""
Real client for Open-Meteo's free Historical Weather (archive) and Forecast
APIs. Endpoint and parameters verified live on 2026-09-16 against
https://archive-api.open-meteo.com/v1/archive. No API key needed for
non-commercial use.

Also derives per-calendar-day "climate normals" (mean/std over the last N
years) purely from the historical archive, since Open-Meteo does not expose
day-specific normals directly -- this is what lets WildCast score a day's
weather as "unusually warm/cold/wet for this place and time of year" rather
than just reading the raw numbers.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from functools import lru_cache
from typing import Any

import pandas as pd
import requests

from app.config import (
    CLIMATE_NORMAL_YEARS,
    OPEN_METEO_ARCHIVE_BASE,
    OPEN_METEO_FORECAST_BASE,
    WEATHER_DAILY_VARS,
)

log = logging.getLogger("wildcast.weather")

_SESSION = requests.Session()

# Open-Meteo's free tier is generous in absolute terms, but a shared CI
# runner IP (many different GitHub Actions jobs, from many different repos,
# all sharing the same small pool of outbound IPs) can trip its rate limit
# even at low request volume from any one job -- and how long that
# contention lasts varies run to run (real incidents, both 2026-09-17: one
# run failed outright on its 5th request with no retry logic at all; after
# adding retries with a 5-attempt/3s-doubling schedule, a second run got
# through 4 of 5 areas before still exhausting all 5 retries on the 5th).
# The fix is more patience, not smarter logic: more retries, a higher cap
# per wait, so a longer contention window gets ridden out rather than timed
# out on.
_MAX_RETRIES = 8
_BASE_BACKOFF_SECONDS = 4.0
_MAX_BACKOFF_SECONDS = 60.0

# The patient retry schedule above (~8 attempts, up to ~4 minutes of total
# backoff) is right for scripts/ingest_weather.py: an unattended batch job
# where riding out a contention window matters more than speed. It is
# wrong for a live request a person is actually waiting on in a browser
# (predict(), predict_at_location(), and everything under "Explore
# Anywhere") -- a live outage there should fail fast with a clear error,
# not hang the page for minutes. Interactive call sites pass these
# individually (not as a dict: historical_daily/climate_normals are
# lru_cache'd, which requires hashable arguments) via the *_retry kwargs
# threaded through below.
INTERACTIVE_MAX_RETRIES = 2
INTERACTIVE_BASE_BACKOFF = 1.5
INTERACTIVE_MAX_BACKOFF = 4.0
INTERACTIVE_REQUEST_TIMEOUT = 6.0


def _get_with_retry(
    url: str,
    params: dict,
    *,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BASE_BACKOFF_SECONDS,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
    request_timeout: float = 30.0,
) -> requests.Response:
    delay = base_backoff
    for attempt in range(1, max_retries + 1):
        resp = _SESSION.get(url, params=params, timeout=request_timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries:
                resp.raise_for_status()  # out of retries -- surface the real error
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(delay, max_backoff)
            log.warning(
                "Open-Meteo returned %s; retrying in %.1fs (attempt %d/%d).",
                resp.status_code, wait, attempt, max_retries,
            )
            time.sleep(wait)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp
    raise AssertionError("unreachable")  # loop always returns or raises above


def _round_coord(x: float) -> float:
    # Open-Meteo's grid is ~9-25km; rounding keeps our on-disk cache from
    # exploding into one entry per pixel-different lat/lon.
    return round(x, 2)


def historical_daily(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    *,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BASE_BACKOFF_SECONDS,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
    request_timeout: float = 30.0,
) -> pd.DataFrame:
    """Daily weather observations for one location and date range. Dates are 'YYYY-MM-DD'.

    Pass the `INTERACTIVE_*` module constants for a live, user-facing call
    (fails fast); the defaults are the patient, batch-ingestion schedule --
    see that constant's docstring for why they differ.
    """
    params = {
        "latitude": _round_coord(lat),
        "longitude": _round_coord(lon),
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(WEATHER_DAILY_VARS),
        "timezone": "auto",
    }
    resp = _get_with_retry(
        OPEN_METEO_ARCHIVE_BASE,
        params,
        max_retries=max_retries,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
        request_timeout=request_timeout,
    )
    payload = resp.json()
    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame(columns=["date", *WEATHER_DAILY_VARS])
    df = pd.DataFrame({"date": daily["time"]})
    for var in WEATHER_DAILY_VARS:
        df[var] = daily.get(var, [None] * len(df))
    return df


def forecast_daily(
    lat: float,
    lon: float,
    days: int = 16,
    *,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BASE_BACKOFF_SECONDS,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
    request_timeout: float = 30.0,
) -> pd.DataFrame:
    """Upcoming forecast, used when a user asks about a near-future date. See `historical_daily` for the retry kwargs."""
    params = {
        "latitude": _round_coord(lat),
        "longitude": _round_coord(lon),
        "daily": ",".join(WEATHER_DAILY_VARS),
        "forecast_days": min(days, 16),
        "timezone": "auto",
    }
    resp = _get_with_retry(
        OPEN_METEO_FORECAST_BASE,
        params,
        max_retries=max_retries,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
        request_timeout=request_timeout,
    )
    daily = resp.json().get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame(columns=["date", *WEATHER_DAILY_VARS])
    df = pd.DataFrame({"date": daily["time"]})
    for var in WEATHER_DAILY_VARS:
        df[var] = daily.get(var, [None] * len(df))
    return df


@lru_cache(maxsize=64)
def climate_normals(
    lat: float,
    lon: float,
    years: int = CLIMATE_NORMAL_YEARS,
    *,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BASE_BACKOFF_SECONDS,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
    request_timeout: float = 30.0,
) -> pd.DataFrame:
    """
    Mean and std of each weather variable per day-of-year, computed from the
    last `years` of the historical archive. Cached in-process (and callers
    are expected to also persist this to disk -- see scripts/ingest_weather.py)
    since it is one of the more expensive calls in the pipeline. Pass the
    `INTERACTIVE_*` module constants for a live, user-facing call (see
    `historical_daily`'s docstring); all keyword args are part of the
    `lru_cache` key, so an interactive and a batch caller for the same
    coordinates get separate cache entries -- correct, since they're really
    different requests (different patience for the same data).
    """
    today = dt.date.today()
    start = dt.date(today.year - years, 1, 1)
    end = today - dt.timedelta(days=6)  # archive lags a few days behind real time
    df = historical_daily(
        lat,
        lon,
        start.isoformat(),
        end.isoformat(),
        max_retries=max_retries,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
        request_timeout=request_timeout,
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear
    agg = df.groupby("day_of_year")[WEATHER_DAILY_VARS].agg(["mean", "std"])
    agg.columns = [f"{var}_normal_{stat}" for var, stat in agg.columns]
    return agg.reset_index()


def weather_anomaly_features(lat: float, lon: float, date: str, normals: pd.DataFrame | None = None) -> dict[str, Any]:
    """
    Raw + normal-relative ('anomaly') weather features for one location and
    date. Falls back to climate normals alone (anomaly = 0) if the raw daily
    value for that exact date can't be fetched (e.g. far-future date beyond
    the forecast window).
    """
    target = dt.date.fromisoformat(date)
    day_of_year = target.timetuple().tm_yday
    if normals is None:
        normals = climate_normals(lat, lon)

    row = {}
    normal_row = normals[normals["day_of_year"] == day_of_year] if not normals.empty else normals
    for var in WEATHER_DAILY_VARS:
        mean_col, std_col = f"{var}_normal_mean", f"{var}_normal_std"
        row[f"{var}_normal_mean"] = float(normal_row[mean_col].iloc[0]) if len(normal_row) else 0.0
        row[f"{var}_normal_std"] = float(normal_row[std_col].iloc[0]) if len(normal_row) else 1.0

    today = dt.date.today()
    horizon = (target - today).days
    if -30 <= horizon <= 0:
        obs = historical_daily(lat, lon, date, date)
    elif 0 < horizon <= 16:
        obs = forecast_daily(lat, lon, days=horizon + 1)
        obs = obs[obs["date"] == date]
    else:
        obs = pd.DataFrame()

    for var in WEATHER_DAILY_VARS:
        if not obs.empty and var in obs.columns and pd.notna(obs[var].iloc[0]):
            value = float(obs[var].iloc[0])
        else:
            value = row[f"{var}_normal_mean"]  # best guess: fall back to the seasonal normal
        row[var] = value
        std = max(row[f"{var}_normal_std"], 1e-6)
        row[f"{var}_anomaly"] = (value - row[f"{var}_normal_mean"]) / std

    row["day_of_year"] = day_of_year
    return row

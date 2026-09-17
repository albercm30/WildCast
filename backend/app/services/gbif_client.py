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

import logging
import math
import time
from typing import Any

import requests

from app.config import GBIF_API_BASE

log = logging.getLogger("wildcast.gbif")

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "WildCast/0.1 (wildlife-encounter-forecast)"})

_MAX_PAGE_SIZE = 300  # GBIF's hard per-request cap

# `_get` used to be a plain `_SESSION.get(...).raise_for_status()` with no
# retry logic at all -- a single transient failure (a GBIF 5xx, a 429, or a
# raw connection/read timeout) killed whichever call made it. Hardened
# proactively after the same class of bug (an unretried ReadTimeout) took
# down scripts/ingest_weather.py on 2026-09-17 -- see weather_client.py's
# _get_with_retry -- since scripts/ingest_gbif.py runs earlier in the same
# retrain.yml pipeline and was structurally even more exposed (weather_client
# at least already retried 429/5xx). Same idea, a smaller schedule: GBIF
# occurrence pages are far smaller requests than Open-Meteo's multi-year
# archive fetches, so there's less to wait out.
_MAX_RETRIES = 5
_BASE_BACKOFF_SECONDS = 3.0
_MAX_BACKOFF_SECONDS = 30.0

# Interactive "Explore Anywhere" callers (species_for_location, run many at
# once via a ThreadPoolExecutor against a real user's click -- see
# app.ml.prediction_service) need to fail fast, not ride out the patient
# batch-ingestion schedule above and leave someone staring at a spinner for
# a minute-plus. Same rationale as weather_client.INTERACTIVE_*.
INTERACTIVE_MAX_RETRIES = 2
INTERACTIVE_BASE_BACKOFF = 1.0
INTERACTIVE_MAX_BACKOFF = 3.0

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


def _get(
    path: str,
    params: dict[str, Any],
    timeout: float = 20.0,
    *,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BASE_BACKOFF_SECONDS,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
) -> dict[str, Any]:
    url = f"{GBIF_API_BASE}{path}"
    delay = base_backoff
    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            # See the module-level comment above _MAX_RETRIES: a raw
            # transport failure never reaches the status-code check below,
            # so without this it bypasses the retry schedule entirely.
            if attempt == max_retries:
                raise
            wait = min(delay, max_backoff)
            log.warning(
                "GBIF request failed (%s: %s); retrying in %.1fs (attempt %d/%d).",
                type(exc).__name__, exc, wait, attempt, max_retries,
            )
            time.sleep(wait)
            delay *= 2
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries:
                resp.raise_for_status()  # out of retries -- surface the real error
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(delay, max_backoff)
            log.warning(
                "GBIF returned %s; retrying in %.1fs (attempt %d/%d).",
                resp.status_code, wait, attempt, max_retries,
            )
            time.sleep(wait)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise AssertionError("unreachable")  # loop always returns or raises above


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
    country: str | list[str] | None = None,
    limit: int = 300,
    offset: int = 0,
    timeout: float = 20.0,
    wild_only: bool = False,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BASE_BACKOFF_SECONDS,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """One page of GBIF occurrence records matching the given filters.

    Pass the `INTERACTIVE_*` module constants for `max_retries`/`base_backoff`/
    `max_backoff` from a live, user-facing call (fails fast); the defaults
    are the patient, batch-ingestion schedule -- see those constants'
    docstring for why they differ.

    `country` accepts a single ISO 3166-1 alpha-2 code or a list of them
    (GBIF ORs multiple `country` query params together) -- a list is how
    `has_any_presence`'s `plausible_countries` restricts a search to a
    species' real native range.

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
    return _get(
        "/occurrence/search",
        params,
        timeout=timeout,
        max_retries=max_retries,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
    )


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


def presence_count(
    scientific_name: str,
    lat: float,
    lon: float,
    radius_km: float,
    timeout: float = 20.0,
    plausible_countries: list[str] | None = None,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BASE_BACKOFF_SECONDS,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
) -> int:
    """The real number of *wild* GBIF records of this species within radius_km of this point.

    Same two stacked filters as `has_any_presence` (wild_only +
    plausible_countries -- see that function's docstring for why both are
    needed), but returns GBIF's exact `count` field instead of collapsing
    it to a boolean. GBIF returns the TOTAL matching count on every
    occurrence/search response regardless of `limit` -- even a `limit=1`
    presence check already gets the real number back for free -- so this
    costs no extra request over a plain presence check.

    This real local count is what "Explore Anywhere" mode's confidence and
    probability calibration are built on (see
    app.ml.prediction_service._evidence_factor). Before this existed, a
    species passing the plausibility gate (wild_only + plausible_countries)
    would still display its home curated area's training count as if it
    were local evidence -- e.g. a Leopard showing "500 historical sighting
    records" and 91% confidence near a point with only a handful of real
    nearby records. Found via a real user report that a 91% Leopard
    prediction near Bangkok, Thailand was not credible even though Leopard
    correctly passes the plausibility gate there (Indochinese leopards are
    a real native subspecies in Thailand) -- the number backing the
    confidence just wasn't about that point at all.

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
        country=plausible_countries,
        max_retries=max_retries,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
    )
    return int(page.get("count", 0) or 0)


def total_wild_occurrence_count(
    lat: float,
    lon: float,
    radius_km: float,
    timeout: float = 20.0,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BASE_BACKOFF_SECONDS,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
) -> int:
    """Total *wild* GBIF records of ANY species within radius_km of this point.

    Not a species-specific count -- a rough "how much wildlife-observation
    activity happens here at all" proxy (area-level observer effort), used
    to correct a real data-reliability problem: a raw per-species record
    count alone conflates true local abundance with how much anyone
    happens to be looking or reporting there. A heavily-touristed spot
    inflates counts for whatever's charismatic there even if it's
    genuinely hard to find (a real concern raised about this app: "5000
    polar bear observations doesn't mean they're easy to find"), while a
    rarely-visited spot deflates counts for everything, common species
    included ("5 kangaroo observations doesn't mean it's hard"). See
    app.ml.prediction_service._local_effort_index_cached for where this
    gets used.

    This mirrors, at area level, the "target-group background" bias
    correction app.ml.pseudo_absence.py already uses when TRAINING the
    curated-area models (Phillips et al. 2009: other species' records
    stand in for "someone was out looking here"). A version scoped to the
    same taxonomic class (mirroring pseudo_absence.py's area+taxon_class
    grouping exactly) would be more precise, but GBIF's occurrence search
    takes a numeric `classKey`, not a plain class name like "Mammalia",
    and this sandbox has no live GBIF access to verify each taxon_class's
    correct key -- shipping a guessed mapping risks silently corrupting
    the effort baseline, which would be worse than this coarser
    area-level proxy. Left as a documented follow-up once classKey values
    are verified against a real GBIF call (e.g. during a retrain.yml run).

    Same "free" trick as `presence_count`: a `limit=1` request still gets
    GBIF's real total `count` back, so this costs no extra request beyond
    the search itself.
    """
    page = occurrence_search(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        limit=1,
        timeout=timeout,
        wild_only=True,
        max_retries=max_retries,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
    )
    return int(page.get("count", 0) or 0)


def has_any_presence(
    scientific_name: str,
    lat: float,
    lon: float,
    radius_km: float,
    timeout: float = 20.0,
    plausible_countries: list[str] | None = None,
    max_retries: int = _MAX_RETRIES,
    base_backoff: float = _BASE_BACKOFF_SECONDS,
    max_backoff: float = _MAX_BACKOFF_SECONDS,
) -> bool:
    """Cheap plausibility check: does GBIF have >=1 *wild* record of this species near this point, ever?

    A thin wrapper around `presence_count` for callers that only need a
    yes/no plausibility gate (e.g. species_for_location's "is this species
    offered at all here" filter) and don't need the real count.

    Two independent filters, stacked, because they catch different failure
    modes of the same underlying problem:

    1. `wild_only=True` excludes captive (LIVING_SPECIMEN) and fossil
       (FOSSIL_SPECIMEN) records -- see `_WILD_BASIS_OF_RECORD`'s comment.
       This only catches formal institutional living-collection records
       (zoo/garden accession databases published straight to GBIF); it does
       NOT catch a casual visitor's photo of a zoo animal, which typically
       still carries basisOfRecord=HUMAN_OBSERVATION -- the same value a
       genuine wild sighting has. GBIF simply has no reliable per-record
       "this individual was captive" flag for that far more common case.
    2. `plausible_countries` (from the species' curated entry in
       seed_species.json, a hand-checked list of its real native-range
       countries) restricts the GBIF query itself to those countries. This
       is what actually closes the gap #1 leaves open: whatever basisOfRecord
       a Bangkok zoo lion's citizen-science photo carries, Thailand is
       simply never in Panthera leo's plausible-country list, so the query
       cannot return it. Found and added after a real report: a lion and a
       gray wolf both still passed the wild_only-only filter near Bangkok.

    `timeout` is lower by default for interactive "Explore Anywhere" callers
    (see app.ml.prediction_service.species_for_location, which runs many of
    these concurrently against a real user's click) than for the batch
    ingestion scripts, which can afford to wait longer per call.
    """
    return presence_count(
        scientific_name, lat, lon, radius_km, timeout=timeout, plausible_countries=plausible_countries,
        max_retries=max_retries, base_backoff=base_backoff, max_backoff=max_backoff,
    ) > 0

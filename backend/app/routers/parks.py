"""
"Browse Parks" endpoints (Task #20): country + filter discovery over the
world's protected areas, backed by app.services.protected_planet_client.
See that module's docstring for why this is a browse/filter UI rather than
a free-text park-name search box, and for the real research (v4 API docs,
2026-09-17) behind that call.

Requires PROTECTED_PLANET_API_KEY -- unset, these endpoints return a clean
503 (same "optional integration, clean unavailable response" shape the
rest of the app already uses for e.g. a live weather outage, see
predictions.py) rather than 500ing or silently returning nothing, so the
frontend can show "connect a Protected Planet key" instead of a broken UI.
"""
from __future__ import annotations

import json

from flask import Blueprint, jsonify, request

from app.config import BASE_DIR
from app.services import protected_planet_client as ppc

bp = Blueprint("parks", __name__, url_prefix="/api/parks")

_COUNTRIES_PATH = BASE_DIR / "data" / "countries.json"
_countries_cache: list[dict[str, str]] | None = None


def _load_countries() -> list[dict[str, str]]:
    global _countries_cache
    if _countries_cache is None:
        raw = json.loads(_COUNTRIES_PATH.read_text())
        _countries_cache = [{"iso3": code, "name": name} for code, name in raw.items()]
    return _countries_cache


@bp.get("/countries")
def countries():
    """
    Static ISO3 code/name list for the country picker -- not a Protected
    Planet API call (that API has no "list countries" endpoint; see
    data/countries.py's docstring), so this works even without a
    PROTECTED_PLANET_API_KEY configured.
    """
    return jsonify(_load_countries())


@bp.get("")
def search_parks():
    if not ppc.is_configured():
        return jsonify({"error": "Protected Planet is not configured (PROTECTED_PLANET_API_KEY unset)"}), 503

    country_iso3 = request.args.get("country")
    marine_param = request.args.get("marine")
    marine = {"true": True, "false": False}.get((marine_param or "").lower())

    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        return jsonify({"error": "'page' must be an integer"}), 400
    try:
        per_page = int(request.args.get("per_page", 25))
    except ValueError:
        return jsonify({"error": "'per_page' must be an integer"}), 400

    try:
        result = ppc.search_protected_areas(
            country_iso3=country_iso3, marine=marine, page=page, per_page=per_page,
        )
    except ppc.ProtectedPlanetUnavailable as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify(result)


@bp.get("/<site_id>")
def get_park(site_id: str):
    if not ppc.is_configured():
        return jsonify({"error": "Protected Planet is not configured (PROTECTED_PLANET_API_KEY unset)"}), 503
    try:
        area = ppc.get_protected_area(site_id)
    except ppc.ProtectedPlanetUnavailable as exc:
        return jsonify({"error": str(exc)}), 503
    if area is None:
        return jsonify({"error": f"no protected area found for site_id {site_id!r}"}), 404
    return jsonify(area)

from __future__ import annotations

from flask import Blueprint, g, jsonify, request

from app.routers.auth import require_auth
from app.services import saved_search_service

bp = Blueprint("saved_searches", __name__, url_prefix="/api/saved-searches")


@bp.get("")
@require_auth
def list_saved_searches():
    return jsonify(saved_search_service.list_saved_searches(g.user["id"]))


@bp.post("")
@require_auth
def add_saved_search():
    payload = request.get_json(silent=True) or {}
    try:
        search = saved_search_service.add_saved_search(
            g.user["id"],
            kind=payload.get("kind"),
            label=payload.get("label", ""),
            area_id=payload.get("area_id"),
            lat=payload.get("lat"),
            lon=payload.get("lon"),
            radius_km=payload.get("radius_km"),
            species_key=payload.get("species_key"),
            min_probability=payload.get("min_probability", 0.5),
        )
    except saved_search_service.SavedSearchError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(search), 201


@bp.delete("/<int:search_id>")
@require_auth
def remove_saved_search(search_id: int):
    try:
        saved_search_service.remove_saved_search(g.user["id"], search_id)
    except saved_search_service.SavedSearchError as exc:
        return jsonify({"error": str(exc)}), 404
    return "", 204

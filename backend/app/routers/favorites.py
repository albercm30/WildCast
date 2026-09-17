from __future__ import annotations

from flask import Blueprint, g, jsonify, request

from app.routers.auth import require_auth
from app.services import favorites_service

bp = Blueprint("favorites", __name__, url_prefix="/api/favorites")


@bp.get("")
@require_auth
def list_favorites():
    return jsonify(favorites_service.list_favorites(g.user["id"]))


@bp.post("")
@require_auth
def add_favorite():
    payload = request.get_json(silent=True) or {}
    try:
        favorite = favorites_service.add_favorite(
            g.user["id"],
            kind=payload.get("kind"),
            label=payload.get("label", ""),
            area_id=payload.get("area_id"),
            lat=payload.get("lat"),
            lon=payload.get("lon"),
            radius_km=payload.get("radius_km"),
            species_key=payload.get("species_key"),
        )
    except favorites_service.FavoritesError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(favorite), 201


@bp.delete("/<int:favorite_id>")
@require_auth
def remove_favorite(favorite_id: int):
    try:
        favorites_service.remove_favorite(g.user["id"], favorite_id)
    except favorites_service.FavoritesError as exc:
        return jsonify({"error": str(exc)}), 404
    return "", 204

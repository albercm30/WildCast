from flask import Blueprint, jsonify

from app.ml import prediction_service

bp = Blueprint("areas", __name__, url_prefix="/api/areas")


@bp.get("")
def list_areas():
    return jsonify(prediction_service.list_areas())


@bp.get("/<area_id>")
def get_area(area_id: str):
    try:
        return jsonify(prediction_service.get_area(area_id))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404


@bp.get("/<area_id>/species")
def area_species(area_id: str):
    try:
        return jsonify(prediction_service.species_for_area(area_id))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404

import datetime as dt

from flask import Blueprint, jsonify, request

from app.ml import prediction_service

bp = Blueprint("predictions", __name__, url_prefix="/api")


@bp.get("/predict")
def predict():
    area_id = request.args.get("area_id")
    date = request.args.get("date")
    species_key = request.args.get("species")  # optional: scientific name, narrows to one species

    if not area_id or not date:
        return jsonify({"error": "'area_id' and 'date' query parameters are required"}), 400
    try:
        dt.date.fromisoformat(date)
    except ValueError:
        return jsonify({"error": "'date' must be YYYY-MM-DD"}), 400

    try:
        results = prediction_service.predict(area_id, date, species_key=species_key)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except prediction_service.ModelNotTrained as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify({"area_id": area_id, "date": date, "predictions": results})


@bp.get("/best-window")
def best_window():
    area_id = request.args.get("area_id")
    species_key = request.args.get("species")
    if not area_id or not species_key:
        return jsonify({"error": "'area_id' and 'species' query parameters are required"}), 400
    try:
        points = prediction_service.best_window(area_id, species_key)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except prediction_service.ModelNotTrained as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"area_id": area_id, "species": species_key, "points": points})

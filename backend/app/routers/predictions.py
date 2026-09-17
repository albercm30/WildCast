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


def _parse_lat_lon_radius():
    """Shared query-param parsing for the two 'Explore Anywhere' endpoints below.
    Returns (lat, lon, radius_km, error_response_or_None)."""
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return None, None, None, (jsonify({"error": "'lat' and 'lon' query parameters (numbers) are required"}), 400)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None, None, None, (jsonify({"error": "'lat' must be in [-90, 90] and 'lon' in [-180, 180]"}), 400)
    try:
        radius_km = float(request.args.get("radius_km", 150))
    except ValueError:
        return None, None, None, (jsonify({"error": "'radius_km' must be a number"}), 400)
    radius_km = max(10.0, min(radius_km, 300.0))  # bounded so a bad/huge value can't trigger a runaway GBIF query
    return lat, lon, radius_km, None


@bp.get("/predict-location")
def predict_location():
    """
    'Explore Anywhere' mode: same shape of response as /predict, but for any
    lat/lon instead of one of the 5 curated area IDs. Candidates are gated
    by a live GBIF presence check instead of a fixed seed list -- see
    app.ml.prediction_service.predict_at_location's docstring for how the
    approximation (nearest curated area supplies the model's area_id
    feature) is surfaced rather than hidden.
    """
    lat, lon, radius_km, err = _parse_lat_lon_radius()
    if err:
        return err
    date = request.args.get("date")
    species_key = request.args.get("species")
    if not date:
        return jsonify({"error": "'date' query parameter is required"}), 400
    try:
        dt.date.fromisoformat(date)
    except ValueError:
        return jsonify({"error": "'date' must be YYYY-MM-DD"}), 400

    try:
        result = prediction_service.predict_at_location(lat, lon, date, radius_km=radius_km, species_key=species_key)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except (prediction_service.ModelNotTrained, prediction_service.LiveDataUnavailable) as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify(result)


@bp.get("/best-window-location")
def best_window_location():
    lat, lon, radius_km, err = _parse_lat_lon_radius()
    if err:
        return err
    species_key = request.args.get("species")
    if not species_key:
        return jsonify({"error": "'species' query parameter is required"}), 400

    try:
        points = prediction_service.best_window_at_location(lat, lon, species_key, radius_km=radius_km)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except (prediction_service.ModelNotTrained, prediction_service.LiveDataUnavailable) as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify({"lat": lat, "lon": lon, "species": species_key, "points": points})

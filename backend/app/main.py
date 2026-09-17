"""
WildCast serving API. Deliberately built on Flask + the packages already
provable to work end-to-end in this project (see README's "Why Flask, not
FastAPI" note) -- swapping in FastAPI later is a small, mechanical change
since all the real logic lives in app/ml and app/services, not in the route
handlers themselves.

Run for local dev: python -m app.main
Run for production: gunicorn -w 2 -b 0.0.0.0:8000 app.main:app  (see docker/Dockerfile.backend)
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify

from app.config import DEBUG, PORT
from app.ml import prediction_service
from app.routers.areas import bp as areas_bp
from app.routers.predictions import bp as predictions_bp

logging.basicConfig(level=logging.INFO)


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(areas_bp)
    app.register_blueprint(predictions_bp)

    @app.after_request
    def add_cors_headers(response):
        # Minimal manual CORS (no flask-cors dependency needed) so a
        # statically-served frontend on a different origin/port can call
        # this API directly during local dev and simple deployments.
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.get("/api/health")
    def health():
        try:
            areas = prediction_service.list_areas()
            model_loaded = True
        except prediction_service.ModelNotTrained:
            areas, model_loaded = [], False
        return jsonify(
            {
                "status": "ok",
                "model_loaded": model_loaded,
                "areas": [a["id"] for a in areas],
                # Real ingested data vs. the offline synthetic demo fixtures --
                # see app.services.data_source_marker. The frontend uses this
                # to render an unmissable "demo data" banner rather than
                # letting synthetic predictions pass as real ones silently.
                "data_source": prediction_service.get_data_source(),
            }
        )

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "not found"}), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)

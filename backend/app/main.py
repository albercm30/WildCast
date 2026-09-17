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

from app import db
from app.config import DEBUG, PORT
from app.ml import prediction_service
from app.routers.areas import bp as areas_bp
from app.routers.auth import bp as auth_bp
from app.routers.favorites import bp as favorites_bp
from app.routers.internal import bp as internal_bp
from app.routers.notifications import bp as notifications_bp
from app.routers.predictions import bp as predictions_bp
from app.routers.saved_searches import bp as saved_searches_bp

logging.basicConfig(level=logging.INFO)


def create_app() -> Flask:
    app = Flask(__name__)
    db.init_db()  # accounts/favorites/saved-searches -- fails fast at boot rather than on first request
    app.register_blueprint(areas_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(favorites_bp)
    app.register_blueprint(saved_searches_bp)
    app.register_blueprint(notifications_bp)
    app.register_blueprint(internal_bp)

    @app.after_request
    def add_cors_headers(response):
        # Minimal manual CORS (no flask-cors dependency needed) so a
        # statically-served frontend on a different origin/port can call
        # this API directly during local dev and simple deployments.
        response.headers["Access-Control-Allow-Origin"] = "*"
        # Widened from "GET, OPTIONS" now that accounts/favorites/saved
        # searches add real write endpoints, and "Authorization" added
        # alongside Content-Type so the frontend's Bearer token can
        # actually reach these routes cross-origin.
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    # No explicit OPTIONS route needed for CORS preflight: Flask
    # auto-registers an OPTIONS handler for every route unless a view opts
    # out, and add_cors_headers (above) attaches the Allow-* headers to
    # that automatic response too. An earlier version of this file added
    # an explicit catch-all OPTIONS route here, which broke unknown-route
    # 404s (Werkzeug returns 405, not 404, for a URL that matches a
    # registered rule's pattern but not its method) -- see
    # NotFoundTests.test_unknown_route_is_json_404.

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

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return jsonify({"error": "method not allowed"}), 405

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)

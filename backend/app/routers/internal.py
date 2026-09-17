"""
Internal-only endpoints, meant to be called by an external scheduler --
NEVER by the frontend or a logged-in user. Protected by a shared-secret
header rather than user auth, because the caller here is a machine on a
schedule, not a person: see README's "Accounts & saved-search alerts"
section for a ready-made scheduled GitHub Actions workflow that calls this
with the CRON_SECRET_KEY stored as a repo secret, the same pattern already
used for retrain.yml's EBIRD_API_KEY/IUCN_API_KEY.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.config import CRON_SECRET_KEY
from app.services import saved_search_service

bp = Blueprint("internal", __name__, url_prefix="/api/internal")


@bp.post("/check-saved-searches")
def check_saved_searches():
    if not CRON_SECRET_KEY:
        # Fails closed: an unset secret must never mean "anyone can trigger
        # this," so an empty CRON_SECRET_KEY refuses every request rather
        # than falling back to some default value.
        return jsonify({"error": "CRON_SECRET_KEY is not configured on this server."}), 503
    provided = request.headers.get("X-Cron-Key", "")
    if provided != CRON_SECRET_KEY:
        return jsonify({"error": "Invalid or missing X-Cron-Key header."}), 401
    created = saved_search_service.check_due_saved_searches()
    return jsonify({"notifications_created": created})

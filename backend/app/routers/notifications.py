from __future__ import annotations

from flask import Blueprint, g, jsonify, request

from app.routers.auth import require_auth
from app.services import notification_service

bp = Blueprint("notifications", __name__, url_prefix="/api/notifications")


@bp.get("")
@require_auth
def list_notifications():
    unread_only = request.args.get("unread_only") == "1"
    return jsonify(notification_service.list_notifications(g.user["id"], unread_only=unread_only))


@bp.post("/<int:notification_id>/read")
@require_auth
def mark_read(notification_id: int):
    ok = notification_service.mark_read(g.user["id"], notification_id)
    if not ok:
        return jsonify({"error": "Notification not found or already read."}), 404
    return jsonify({"ok": True})

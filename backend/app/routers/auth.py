from __future__ import annotations

import functools

from flask import Blueprint, g, jsonify, request

from app.services import auth_service

bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def require_auth(view):
    """Decorator for any endpoint that needs a logged-in user: reads
    'Authorization: Bearer <token>', verifies it, and sets g.user -- or
    returns 401 without calling the view at all. Defined here (not a
    separate app.auth module) since this file is already the one place
    token verification happens; other routers (favorites, saved_searches,
    notifications) import require_auth from here rather than duplicating
    it."""

    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header."}), 401
        token = header[len("Bearer "):]
        try:
            g.user = auth_service.user_from_token(token)
        except auth_service.AuthError as exc:
            return jsonify({"error": str(exc)}), 401
        return view(*args, **kwargs)

    return wrapped


@bp.post("/register")
def register():
    payload = request.get_json(silent=True) or {}
    try:
        user, token = auth_service.register(
            payload.get("email", ""), payload.get("password", ""), payload.get("display_name", "")
        )
    except auth_service.AuthError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"user": user, "token": token}), 201


@bp.post("/login")
def login():
    payload = request.get_json(silent=True) or {}
    try:
        user, token = auth_service.login(payload.get("email", ""), payload.get("password", ""))
    except auth_service.AuthError as exc:
        return jsonify({"error": str(exc)}), 401
    return jsonify({"user": user, "token": token})


@bp.get("/me")
@require_auth
def me():
    return jsonify({"user": g.user})


@bp.patch("/me")
@require_auth
def update_me():
    payload = request.get_json(silent=True) or {}
    user = auth_service.update_display_name(g.user["id"], payload.get("display_name", g.user["display_name"]))
    return jsonify({"user": user})

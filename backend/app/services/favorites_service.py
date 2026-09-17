"""
Server-synced favorites -- replaces the frontend's original localStorage-only
favorites (see frontend/index.html's existing `FAVORITES_KEY` logic) with
real per-account storage, so a favorite follows the user across devices
instead of living in one browser only.

A favorite is either a curated area (kind='area', area_id set) or an
arbitrary Explore Anywhere point (kind='location', lat/lon/radius_km set),
optionally narrowed to one species via species_key -- the same shape the
frontend already builds for its localStorage favorites (see its
`favoriteKey`/`jumpToFavorite` helpers), so wiring the frontend to this API
instead is a small, mechanical change rather than a redesign.
"""
from __future__ import annotations

from app import db


class FavoritesError(Exception):
    """Raised for a malformed favorite payload or a favorite that doesn't
    belong to the caller -- routers turn this into a 400/404."""


def _validate(kind, area_id, lat, lon) -> None:
    if kind not in ("area", "location"):
        raise FavoritesError("'kind' must be 'area' or 'location'.")
    if kind == "area" and not area_id:
        raise FavoritesError("'area_id' is required when kind is 'area'.")
    if kind == "location":
        if lat is None or lon is None:
            raise FavoritesError("'lat' and 'lon' are required when kind is 'location'.")
        if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
            raise FavoritesError("'lat' must be in [-90, 90] and 'lon' in [-180, 180].")


def list_favorites(user_id: int) -> list[dict]:
    with db.get_connection() as conn:
        rows = conn.execute("SELECT * FROM favorites WHERE user_id = ? ORDER BY created_at DESC", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def add_favorite(
    user_id: int,
    kind: str,
    label: str = "",
    area_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float | None = None,
    species_key: str | None = None,
) -> dict:
    _validate(kind, area_id, lat, lon)
    with db.get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO favorites (user_id, kind, area_id, lat, lon, radius_km, species_key, label, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, kind, area_id, lat, lon, radius_km, species_key, (label or "").strip(), db.now_iso()),
        )
        row = conn.execute("SELECT * FROM favorites WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def remove_favorite(user_id: int, favorite_id: int) -> None:
    with db.get_connection() as conn:
        cur = conn.execute("DELETE FROM favorites WHERE id = ? AND user_id = ?", (favorite_id, user_id))
        deleted = cur.rowcount
    if deleted == 0:
        raise FavoritesError("Favorite not found.")

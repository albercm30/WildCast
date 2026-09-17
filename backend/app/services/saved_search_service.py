"""
Saved-search alerts: "notify me if <species> is predicted with encounter
probability >= X near <area or point>." check_due_saved_searches() is the
one function that actually evaluates these -- it's called from two places
that both need to keep working independently of this sandbox's own
limitations:

  - scripts/check_saved_searches.py, a standalone CLI entry point, for
    whoever runs the check via a scheduler that can exec into the same
    codebase/container as the deployed service (e.g. a Render Cron Job --
    though see that script's docstring for why Render Cron Jobs can't
    actually reach this project's database, and what to use instead).
  - app.routers.internal's POST /api/internal/check-saved-searches, an
    HTTP endpoint meant to be hit by an external scheduler (a scheduled
    GitHub Actions workflow, the same free mechanism already used for
    retrain.yml) that has no direct access to the deployed service's
    filesystem/database at all -- only the deployed service itself does,
    so the trigger has to reach it over HTTP.

Notification delivery itself is a single INSERT into the `notifications`
table (an in-app inbox, read via app.services.notification_service) --
deliberately NOT wired to real email/push in this round, since this sandbox
has no email or push credentials to test against (the same reason eBird/
IUCN keys were obtained via the live user rather than guessed at here). See
README's "Accounts & saved-search alerts" section for how to add a real
delivery channel later: the natural extension point is right where this
module inserts into `notifications` -- add a call there to whatever
provider you pick (e.g. SendGrid for email, a web-push library for
browser push), guarded by its own optional API key the same way
EBIRD_API_KEY/IUCN_API_KEY are optional.
"""
from __future__ import annotations

import datetime as dt
import logging

from app import db
from app.ml import prediction_service

logger = logging.getLogger("wildcast.saved_searches")

# Slightly under 24h so a daily scheduler trigger never skips a day due to
# drift (e.g. a workflow that fires at a slightly different minute each
# day). Both cooldowns are independent: last_checked_at governs how often a
# saved search is even evaluated (keeps a busy account from hammering the
# live prediction pipeline every time the scheduler fires); last_notified_at
# separately governs how often a search that's ALREADY above its threshold
# is allowed to re-notify, so a condition that stays true for a week
# doesn't produce seven identical notifications.
_RECHECK_COOLDOWN_HOURS = 20
_NOTIFY_COOLDOWN_HOURS = 20


class SavedSearchError(Exception):
    """Raised for a malformed saved-search payload or one that doesn't
    belong to the caller -- routers turn this into a 400/404."""


def _validate(kind, area_id, lat, lon, min_probability) -> None:
    if kind not in ("area", "location"):
        raise SavedSearchError("'kind' must be 'area' or 'location'.")
    if kind == "area" and not area_id:
        raise SavedSearchError("'area_id' is required when kind is 'area'.")
    if kind == "location":
        if lat is None or lon is None:
            raise SavedSearchError("'lat' and 'lon' are required when kind is 'location'.")
        if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
            raise SavedSearchError("'lat' must be in [-90, 90] and 'lon' in [-180, 180].")
    try:
        ok = min_probability is not None and 0.0 <= float(min_probability) <= 1.0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise SavedSearchError("'min_probability' must be a number between 0 and 1.")


def list_saved_searches(user_id: int) -> list[dict]:
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM saved_searches WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def add_saved_search(
    user_id: int,
    kind: str,
    label: str = "",
    area_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float | None = None,
    species_key: str | None = None,
    min_probability: float = 0.5,
) -> dict:
    _validate(kind, area_id, lat, lon, min_probability)
    with db.get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO saved_searches
               (user_id, kind, area_id, lat, lon, radius_km, species_key, min_probability, label, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id, kind, area_id, lat, lon, radius_km, species_key,
                float(min_probability), (label or "").strip(), db.now_iso(),
            ),
        )
        row = conn.execute("SELECT * FROM saved_searches WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def remove_saved_search(user_id: int, search_id: int) -> None:
    with db.get_connection() as conn:
        cur = conn.execute("DELETE FROM saved_searches WHERE id = ? AND user_id = ?", (search_id, user_id))
        deleted = cur.rowcount
    if deleted == 0:
        raise SavedSearchError("Saved search not found.")


def _today_iso() -> str:
    return dt.date.today().isoformat()


def _best_probability_for(search: dict) -> tuple[float, str] | None:
    """Runs the existing prediction pipeline for this saved search's place
    and today's date, and returns (best_probability, headline_species_name)
    -- or None if nothing could be predicted (model not trained, live data
    temporarily unavailable, no species plausible there, etc.). Any of
    those is treated as "nothing to alert on right now," not an error: a
    saved search should degrade quietly on a transient failure rather than
    surface it to the user as a broken notification."""
    date = _today_iso()
    try:
        if search["kind"] == "area":
            results = prediction_service.predict(search["area_id"], date, species_key=search["species_key"])
        else:
            result = prediction_service.predict_at_location(
                search["lat"], search["lon"], date,
                radius_km=search["radius_km"] or 150.0, species_key=search["species_key"],
            )
            results = result["predictions"]
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        logger.info("Saved search #%s could not be evaluated today: %s", search["id"], exc)
        return None
    if not results:
        return None
    best = max(results, key=lambda r: r["probability"])
    return best["probability"], best.get("common_name") or best.get("scientific_name", "that species")


def _hours_since(iso_timestamp: str | None) -> float:
    if not iso_timestamp:
        return float("inf")
    then = dt.datetime.fromisoformat(iso_timestamp)
    if then.tzinfo is None:
        then = then.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - then).total_seconds() / 3600.0


def check_due_saved_searches() -> int:
    """Evaluates every saved search whose recheck cooldown has elapsed, and
    creates a notification for any that now clear their probability
    threshold (subject to its own separate notify cooldown). Returns how
    many notifications were created. Safe to call repeatedly -- idempotent
    within the cooldown windows -- and one search's failure never stops the
    others from being checked (see _best_probability_for)."""
    with db.get_connection() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM saved_searches").fetchall()]

    created = 0
    for search in rows:
        if _hours_since(search["last_checked_at"]) < _RECHECK_COOLDOWN_HOURS:
            continue

        outcome = _best_probability_for(search)
        with db.get_connection() as conn:
            conn.execute("UPDATE saved_searches SET last_checked_at = ? WHERE id = ?", (db.now_iso(), search["id"]))

        if outcome is None:
            continue
        probability, headline = outcome
        if probability < search["min_probability"]:
            continue
        if _hours_since(search["last_notified_at"]) < _NOTIFY_COOLDOWN_HOURS:
            continue

        message = (
            f"{headline}: {probability:.0%} predicted encounter probability today near "
            f"'{search['label']}' -- at or above your {search['min_probability']:.0%} alert threshold."
        )
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO notifications (user_id, saved_search_id, message, created_at) VALUES (?, ?, ?, ?)",
                (search["user_id"], search["id"], message, db.now_iso()),
            )
            conn.execute("UPDATE saved_searches SET last_notified_at = ? WHERE id = ?", (db.now_iso(), search["id"]))
        created += 1

    return created

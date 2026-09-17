"""
The in-app notification inbox: notifications created by
app.services.saved_search_service.check_due_saved_searches, read by the
frontend's profile/notifications panel via app.routers.notifications.
"""
from __future__ import annotations

from app import db


def list_notifications(user_id: int, unread_only: bool = False) -> list[dict]:
    query = "SELECT * FROM notifications WHERE user_id = ?"
    if unread_only:
        query += " AND read_at IS NULL"
    query += " ORDER BY created_at DESC"
    with db.get_connection() as conn:
        rows = conn.execute(query, (user_id,)).fetchall()
    return [dict(r) for r in rows]


def mark_read(user_id: int, notification_id: int) -> bool:
    with db.get_connection() as conn:
        cur = conn.execute(
            "UPDATE notifications SET read_at = ? WHERE id = ? AND user_id = ? AND read_at IS NULL",
            (db.now_iso(), notification_id, user_id),
        )
        updated = cur.rowcount
    return updated > 0

"""
Account creation, login, and stateless auth tokens.

Tokens are itsdangerous-signed strings (itsdangerous ships with Flask, so
this needs no new dependency -- see app/db.py's docstring for why that
matters in this sandbox), not JWTs: same idea (a signed, tamper-proof,
expiring claim), reached without adding a dependency this sandbox can't
install or test. This is NOT a real session store -- a valid token alone
proves the holder was issued it by this server within its lifetime;
revocation (e.g. "log out everywhere") isn't supported, since that needs
server-side session state, deliberately out of scope for this round. See
README's "Accounts & saved-search alerts" section for that follow-up note.
"""
from __future__ import annotations

import re
import sqlite3

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

from app import db
from app.config import SECRET_KEY

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_TOKEN_MAX_AGE_SECONDS = 30 * 24 * 60 * 60  # 30 days
_TOKEN_SALT = "wildcast-auth-v1"

_PUBLIC_USER_COLUMNS = "id, email, display_name, created_at"


class AuthError(Exception):
    """Raised for any user-facing auth failure (bad credentials, duplicate
    email, invalid/expired token) -- routers turn this into a 400/401."""


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(SECRET_KEY, salt=_TOKEN_SALT)


def register(email: str, password: str, display_name: str = "") -> tuple[dict, str]:
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise AuthError("Enter a valid email address.")
    if not password or len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")
    password_hash = generate_password_hash(password)
    with db.get_connection() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                (email, password_hash, (display_name or "").strip(), db.now_iso()),
            )
        except sqlite3.IntegrityError:
            raise AuthError("An account with this email already exists.") from None
        user_id = cur.lastrowid
        row = conn.execute(f"SELECT {_PUBLIC_USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
    user = dict(row)
    return user, _serializer().dumps({"user_id": user_id})


def login(email: str, password: str) -> tuple[dict, str]:
    email = (email or "").strip().lower()
    with db.get_connection() as conn:
        row = conn.execute(
            f"SELECT {_PUBLIC_USER_COLUMNS}, password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()
    if not row or not check_password_hash(row["password_hash"], password or ""):
        # Deliberately the same message for "no such account" and "wrong
        # password" -- distinguishing them lets an attacker enumerate
        # which emails have accounts.
        raise AuthError("Incorrect email or password.")
    user = {k: row[k] for k in ("id", "email", "display_name", "created_at")}
    return user, _serializer().dumps({"user_id": row["id"]})


def user_from_token(token: str) -> dict:
    try:
        data = _serializer().loads(token, max_age=_TOKEN_MAX_AGE_SECONDS)
    except SignatureExpired:
        raise AuthError("Session expired -- please log in again.") from None
    except BadSignature:
        raise AuthError("Invalid session token.") from None
    with db.get_connection() as conn:
        row = conn.execute(
            f"SELECT {_PUBLIC_USER_COLUMNS} FROM users WHERE id = ?", (data.get("user_id"),)
        ).fetchone()
    if not row:
        raise AuthError("Account no longer exists.")
    return dict(row)


def update_display_name(user_id: int, display_name: str) -> dict:
    with db.get_connection() as conn:
        conn.execute("UPDATE users SET display_name = ? WHERE id = ?", ((display_name or "").strip(), user_id))
        row = conn.execute(f"SELECT {_PUBLIC_USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row)

"""
Tracks whether backend/data/cache/ currently holds REAL ingested data or the
offline SYNTHETIC demo fixtures -- see app/config.DATA_SOURCE_MARKER_PATH.

Why this exists: before this, nothing in the repo distinguished the two
(app/ml/train.py's own docstring used to say so explicitly: "does not know
or care which one produced its inputs"). That was fine while the app was a
local prototype, but it means a deployed instance could be silently serving
predictions built entirely from illustrative, non-observational synthetic
data with no way for a user -- or an operator -- to tell from the API or
UI alone. `scripts/ingest_gbif.py` / `ingest_weather.py` (real) and
`scripts/generate_sample_fixtures.py` (synthetic) now each call `mark()` at
the end of a successful run; `app/ml/train.py` reads it into the model
bundle so `/api/health` and every prediction response can say, honestly,
which one backs the current model -- see the README's "Is this real data?"
section.
"""
from __future__ import annotations

import datetime as dt
import json

from app.config import DATA_SOURCE_MARKER_PATH


def mark(source: str, extra: dict | None = None) -> None:
    """`source` is 'real' or 'synthetic_demo'. Overwrites any previous marker --
    the marker always reflects whichever ingestion path ran most recently."""
    if source not in ("real", "synthetic_demo"):
        raise ValueError(f"Unknown data source '{source}' -- must be 'real' or 'synthetic_demo'.")
    payload = {"source": source, "written_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if extra:
        payload.update(extra)
    DATA_SOURCE_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_SOURCE_MARKER_PATH.write_text(json.dumps(payload, indent=2))


def read() -> dict:
    """Never raises -- a missing or corrupt marker means 'unknown', not a crash.
    'unknown' also covers cache/ directories populated before this marker
    existed, so an older real dataset isn't mislabeled as synthetic."""
    if not DATA_SOURCE_MARKER_PATH.exists():
        return {"source": "unknown"}
    try:
        payload = json.loads(DATA_SOURCE_MARKER_PATH.read_text())
        if not isinstance(payload, dict) or payload.get("source") not in ("real", "synthetic_demo"):
            return {"source": "unknown"}
        return payload
    except (json.JSONDecodeError, OSError):
        return {"source": "unknown"}

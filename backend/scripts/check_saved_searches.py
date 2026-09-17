"""
Standalone CLI entry point for `app.services.saved_search_service.check_due_saved_searches`
-- evaluates every user's saved searches and creates notifications for the
ones that now clear their alert threshold.

IMPORTANT -- this script only makes sense to run somewhere that has direct
access to the SAME database file the deployed backend uses (WILDCAST_DB_PATH,
see app/db.py). That rules out Render's own Cron Jobs feature: Render Cron
Jobs "can't provision or access a persistent disk" (confirmed against
Render's docs, 2026-09-17), so a Cron Job would only ever see an empty,
freshly-created database, never the real one the web service is using. Two
options that actually work:

  1. Run this script as a one-off/manual task on the same machine/container
     as the deployed backend (e.g. via `render exec` against the web
     service itself, if your plan supports it), so it shares that service's
     filesystem -- including its persistent disk, if you've attached one.
  2. Prefer this in production: don't run this script remotely at all.
     Instead, trigger the deployed backend's own
     POST /api/internal/check-saved-searches endpoint (app/routers/internal.py)
     from an external scheduler -- a scheduled GitHub Actions workflow is
     the free option this project already uses elsewhere (see .github/
     workflows/retrain.yml). That endpoint runs check_due_saved_searches()
     IN-PROCESS on the actual deployed service, so it naturally has access
     to whatever database that service is using -- no shared filesystem
     needed at all. See README's "Accounts & saved-search alerts" section
     for a ready-made workflow file.

This script remains useful for local development/testing and for option 1
above. Usage: `python -m scripts.check_saved_searches` from backend/.
"""
from __future__ import annotations

import logging

from app.services.saved_search_service import check_due_saved_searches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wildcast.check_saved_searches")


def main() -> None:
    created = check_due_saved_searches()
    logger.info("Checked saved searches: %d notification(s) created.", created)


if __name__ == "__main__":
    main()

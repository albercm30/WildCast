# Contributing to WildCast

Thanks for considering it. This project is most useful when it covers more
places and more species than one person can curate alone -- contributions
that expand real-world coverage are the ones that matter most.

## The easiest contribution: add your area

1. Fork the repo and create a branch.
2. Add an entry to `backend/data/pilot_areas.json`:
   `{id, name, type, lat, lon, radius_km, gbif_country, timezone}`.
   `type` is currently one of `national_park`, `urban`, `arctic_tundra`,
   `rainforest`, `steppe`, `moorland`, `cloud_forest` -- add a new one if
   yours doesn't fit (and a matching label in `frontend/index.html`'s
   `AREA_TYPE_LABELS`, or the frontend will just fall back to labeling it
   "National park").
3. Add 3-5 species to `backend/data/seed_species.json` for that area:
   `{scientific_name, common_name, taxon_class, area_id}`. Confirm the
   scientific name resolves on GBIF first --
   `python -c "from app.services import gbif_client; print(gbif_client.match_species('Your species name'))"`
   from `backend/` -- so it's not silently added with zero training data.
4. Run the real ingestion + training locally (`python -m scripts.ingest_gbif`,
   `python -m scripts.ingest_weather`, `python -m app.ml.train` -- see the
   README's "Switching to real data"), or just open the PR and let
   `.github/workflows/retrain.yml` handle it after merge.
5. Before opening the PR: `pip install -r requirements-dev.txt`, then
   `ruff check .`, `flake8 --max-line-length=130 --extend-ignore=E501,W503,E127 app scripts tests`,
   and `python -m unittest discover -s tests`, all from `backend/`. CI runs
   these too, but catching it locally first is faster for everyone.

## Bigger contributions

**Done, as of recent rounds:** live GBIF presence checking is wired in as
"Explore Anywhere" mode (`app/ml/prediction_service.species_for_location` /
`predict_at_location`, `/api/predict-location`), its confidence/probability
is now calibrated to real local evidence near the clicked point rather than
a borrowed home-area number (see the README's "Confidence is calibrated to
real local evidence" section), and the catalog was doubled from 5 areas/25
species to 10 areas/50 species with the same hand-checked curation standard
(see the README's "Catalog expansion" section). Most recently: iNaturalist
and eBird are now wired in as independent corroboration sources alongside
GBIF (see "Multi-source data quality"), IUCN Red List conservation status
and each species' real `activity_pattern` are surfaced as new "why" factors
(see "Population and ability to evade factors"), the app now honestly labels
real vs. synthetic-demo data everywhere (`/api/health`, every prediction
response, and a persistent frontend banner -- see "Is this real data?"), and
the frontend is now an installable PWA (see "Installing WildCast as an
app"). What's still open, in rough order of value:

1. **True "any species, anywhere" discovery, and/or a real protected-areas
   database for area expansion.** Explore Anywhere today still only checks
   WildCast's 50 curated species against a clicked point, and there are
   still only 10 curated areas. The original ask was 500+ of each; getting
   there for real needs live GBIF species discovery by location (rather
   than checking a known list one by one) and wiring up Protected Planet's
   API (key already scaffolded in `app/config.py` as
   `PROTECTED_PLANET_API_KEY`, unused so far) instead of hand-picking areas.
   This is a real architecture decision, not just more curation -- see the
   README's "Catalog expansion" section before starting on it.
2. A scientific-name -> eBird-species-code lookup, so
   `ebird_client.presence_count()` can use eBird's per-species nearby
   endpoint directly instead of filtering the all-species endpoint
   client-side -- correct today, just not the most efficient shape at a
   much larger bird catalog.
3. Live-verifying `inaturalist_client.py` and `iucn_client.py` against a
   real call (this sandbox's egress can't reach either host -- see "What's
   not done yet" in the README) -- run `retrain.yml` on GitHub's runners and
   confirm the field names/response shapes still match.
4. Per-taxon-class models instead of one shared model.
5. Real accounts with synced saved locations -- today's "favorites" are a
   per-browser `localStorage` list on the frontend only (see
   `frontend/index.html`), not synced across devices.

## Reporting a bug or a bad prediction

Open a GitHub Issue. If it's a prediction that looks wrong (e.g. a species
scored implausibly high or low for a date), include the area, species,
and date -- that's usually enough to reproduce it against the trained model
directly.

## Code of conduct

Be respectful, assume good faith, focus feedback on the work. This is a
small project maintained by volunteers; patience with response times is
appreciated.

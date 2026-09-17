# Contributing to WildCast

Thanks for considering it. This project is most useful when it covers more
places and more species than one person can curate alone -- contributions
that expand real-world coverage are the ones that matter most.

## The easiest contribution: add your area

1. Fork the repo and create a branch.
2. Add an entry to `backend/data/pilot_areas.json`:
   `{id, name, type, lat, lon, radius_km, gbif_country, timezone}`.
   `type` is currently one of `national_park`, `urban`, `arctic_tundra`,
   `rainforest` -- add a new one if yours doesn't fit (and a matching label
   in `frontend/index.html`'s `AREA_TYPE_LABELS`).
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

**Done, as of this round:** live GBIF presence checking is now wired in as
"Explore Anywhere" mode (`app/ml/prediction_service.species_for_location` /
`predict_at_location`, `/api/predict-location`) -- see the README's "Explore
Anywhere mode" section. What's still open, in rough order of value:

1. **True "any species, anywhere" discovery.** Explore Anywhere today still
   only checks WildCast's 25 curated species against a clicked point. A
   bigger step would let it discover species it doesn't already know about
   near a point (GBIF occurrence search by location rather than by a known
   name), with a review/curation step before anything untested gets a
   trained-model score.
2. Wiring `app/services/ebird_client.py` (already written, just unused)
   into feature engineering for richer bird data.
3. Per-taxon-class models instead of one shared model.
4. Real accounts with synced saved locations -- today's "favorites" are a
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

# WildCast

A daily wildlife-encounter forecast: for a place and a date, a ranked list of
species with a calibrated probability of encounter, factoring in season and
that day's weather. See the full design doc (architecture, data strategy,
roadmap) here: **[Wildlife Encounter Prediction Platform](https://claude.ai/code/artifact/833b0c2f-87b4-40e1-9262-a5ff433907a8)**.

This repo is the **Phase 0 prototype**: five pilot areas spanning five very
different biomes -- Yellowstone (temperate national park), Kruger (savanna
national park), urban Brisbane, Arctic tundra (Svalbard), and Amazon
rainforest -- 25 species, a real GBIF + Open-Meteo data pipeline, a trained,
calibrated model, a plain-language "why" explanation for every prediction, a
Flask API, and a map-based web frontend. It runs end-to-end today, on an
offline demo dataset if you have no internet yet, or on live data if you do.

The five areas were deliberately chosen to make the project's own founding
plausibility requirement checkable, not just assumed: Svalbard (polar bear,
arctic fox, muskox, snowy owl, Svalbard reindeer) and the Amazon (jaguar,
scarlet macaw, capybara, giant otter, harpy eagle) sit at opposite climate
extremes, and `backend/tests/test_prediction_service.py` has a test that
pins this literally -- a polar bear must never be offered as a candidate for
the Amazon rainforest area, and nothing is ever offered outside the one area
it has real support in.

## Repo layout

```
wildcast/
  backend/
    app/
      main.py              Flask app (routes)
      config.py             All settings / API base URLs / pilot areas paths
      routers/               areas.py, predictions.py -- the HTTP layer
      services/               gbif_client.py, weather_client.py, ebird_client.py
      ml/
        features.py           Shared train/serve feature engineering
        pseudo_absence.py     Target-group background sampling
        train.py               Trains + calibrates the model
        prediction_service.py  Loads the model, answers prediction requests
    data/
      pilot_areas.json        The 5 pilot areas
      seed_species.json       The 25 seed species (Phase 0's curated catalog)
      cache/                  Ingested/generated data lands here (gitignored)
    scripts/
      ingest_gbif.py          REAL: pulls live GBIF occurrence records
      ingest_weather.py       REAL: pulls live Open-Meteo weather + normals
      generate_sample_fixtures.py   SYNTHETIC offline demo data (see below)
    tests/                     65 unit/integration tests, see "Testing & CI"
    requirements.txt
    requirements-dev.txt       ruff, flake8, pytest (optional runner)
  frontend/
    index.html                 Single-file map + forecast UI (no build step)
  docker/
    Dockerfile.backend, Dockerfile.frontend, docker-compose.yml
  .github/workflows/ci.yml      Lint + test + Docker build, on every push/PR
```

## Quickstart (offline demo, ~2 minutes, no internet needed)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt

# 1. Generate the offline demo dataset (synthetic, but schema-identical to
#    the real ingestion output -- see the big warning in that file).
python -m scripts.generate_sample_fixtures

# 2. Train the model
python -m app.ml.train

# 3. Run the API
python -m app.main        # http://localhost:8000
```

Then open `frontend/index.html` directly in a browser (or `python -m
http.server 8080` inside `frontend/` and visit `http://localhost:8080`). The
API base URL field at the bottom of the page defaults to
`http://localhost:8000` -- change it if your backend runs elsewhere.

Pick an area from the dropdown (or click a marker on the map), pick a date,
and you'll get a ranked probability list. Click a species to see its
best-months-to-visit chart for that location.

**This offline demo runs on synthetic data.** `scripts/generate_sample_fixtures.py`
generates occurrence dates and daily weather from hand-set, illustrative
seasonal/weather-sensitivity profiles per species -- directionally realistic
(grounded in real natural history: bears hibernate, savanna herbivores
concentrate near water in the dry season, and so on) but **not observation
records**. It exists so the whole pipeline is runnable and testable with zero
setup. Move to the next section before showing predictions to anyone.

## Switching to real data

```bash
cd backend
python -m scripts.ingest_gbif      # live GBIF occurrence records, no API key needed
python -m scripts.ingest_weather   # live Open-Meteo historical weather + climate normals
python -m app.ml.train             # retrain on the real cache
```

This needs outbound internet access to `api.gbif.org` and
`archive-api.open-meteo.com` (both free, no key). If you're running this
from a sandboxed/locked-down environment, these calls will fail with a
connection or proxy error -- run it from a normal machine or CI runner
instead.

Optional: copy `backend/.env.example` to `backend/.env` and add an
[eBird API key](https://ebird.org/api/keygen) (instant, free) to enable
`app/services/ebird_client.py`'s bird-specific detection-frequency data.
It's wired up but not yet pulled into the training pipeline -- see
"What's not done yet" below.

## Adding your own areas or species

Edit `backend/data/pilot_areas.json` (add `{id, name, type, lat, lon,
radius_km, gbif_country, timezone}`) and `backend/data/seed_species.json`
(add `{scientific_name, common_name, taxon_class, area_id}` -- scientific
names are matched against GBIF, so use `app/services/gbif_client.match_species()`
to confirm a name resolves before adding it). Then re-run ingestion +
training. Phase 0's plausibility gate is exactly this seed list -- a species
never appears for an area it wasn't added to. Phase 1 replaces the seed list
with a live GBIF presence check (`gbif_client.has_any_presence`) so new areas
don't need manual curation; the function already exists, it just isn't
wired into `species_for_area()` yet.

## Testing & CI

```bash
cd backend
pip install -r requirements-dev.txt
ruff check .
flake8 --max-line-length=130 --extend-ignore=E501,W503,E127 app scripts tests
python -m unittest discover -s tests -v      # 65 tests, ~3 seconds
```

The suite is plain `unittest.TestCase` (no pytest dependency required to run
it, though pytest picks it up fine too). It splits into two kinds:

- **Pure unit tests** (`test_features.py`, `test_pseudo_absence.py`,
  `test_gbif_client.py`, `test_weather_client.py`) -- no network, no trained
  model. The GBIF/weather client tests mock HTTP responses to check request
  construction and response parsing without needing live access; one of
  them (`test_weather_client.py`'s `WeatherAnomalyFeaturesTests`) caught a
  real bug while being written -- `weather_anomaly_features()` was reading
  normals columns under their pre-refactor names and would have raised a
  `KeyError` the first time anything actually called it. Fixed; the test
  now pins the correct names so it can't silently regress.
- **Integration tests** (`test_prediction_service.py`, `test_api.py`) --
  exercise the trained model and Flask API together, using the
  `model.joblib` and cache files that ship in the repo. `PredictTests`
  includes four tests that pin a specific *direction* (grizzly bears score
  higher in July than January; kangaroos score higher in a Brisbane winter
  than a Brisbane summer; polar bears score higher in the ice-free season
  than deep winter; jaguars score higher in the dry season than the wet
  season) rather than only checking that a number comes back -- generic "did
  it 200 OK and return a float in [0,1]" tests would not have caught the
  calibration-collapse bug described below; probing a concrete, checkable
  claim is what did. `SpeciesForAreaTests` includes a direct, literal test of
  the project's own founding plausibility example: every species must appear
  in exactly the one area it has support in and nowhere else, checked
  pairwise across all five areas, plus a named assertion that a polar bear
  is never offered for the Amazon rainforest.

CI (`.github/workflows/ci.yml`) runs the same lint + test steps on every
push/PR, plus a separate job that builds both Docker images -- the one
verification this project could not run locally (no Docker daemon in the
sandbox it was built in). Push this repo to GitHub and that job is the real
build test; a green run there is what "Docker images work" actually rests on.

## API reference

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | Liveness + whether a trained model is loaded |
| `GET /api/areas` | The pilot areas |
| `GET /api/areas/<id>` | One area |
| `GET /api/areas/<id>/species` | Plausibility-gated species catalog for an area |
| `GET /api/predict?area_id=&date=YYYY-MM-DD[&species=<scientific name>]` | Ranked probabilities, each with a `factors` list of 2-3 plain-language reasons (data support, seasonal timing, unusual weather) |
| `GET /api/best-window?area_id=&species=<scientific name>` | Probability by day-of-year, for trip planning |

## Deploying

**Backend:** `docker build -f docker/Dockerfile.backend -t wildcast-backend .`
from the repo root, then push to any container host (Render, Fly.io, AWS
ECS/App Runner, ...). The image bakes in the offline demo model at build
time so it's immediately useful; swap the `RUN` line in the Dockerfile for
the real ingestion + training commands once you're ready, or run training
as a separate scheduled job and mount/fetch the resulting `model.joblib`
instead of baking it into the image (better for a model that gets retrained
regularly).

**Frontend:** it's one static HTML file with no build step -- deploy it as-is
to Vercel, Netlify, Cloudflare Pages, or GitHub Pages, or serve it via
`docker/Dockerfile.frontend`. Point its "API base URL" field (or hardcode
the default in `frontend/index.html`) at your deployed backend's URL.

**Local full stack:** `docker compose -f docker/docker-compose.yml up --build`
then open `http://localhost:8080`.

*(Docker images were written to standard conventions, and `docker compose
config` was used to validate the compose file's syntax/structure -- but
neither image has been through an actual `docker build` yet: the sandbox
this project was built in has the Docker CLI but no reachable daemon
(`docker info` fails with "no such file or directory" on the socket). A
second path was tried -- running the build on the developer's own linked
computer via its local shell bridge, which has a separate network egress
from the sandbox -- but that shell currently fails outright on this machine
due to a Windows update (released 2026-09-08) that Anthropic has already
identified and is tracking; it isn't specific to this project. The
`docker-build` job in `.github/workflows/ci.yml` runs both builds on
GitHub's runners, which do have a working daemon -- push this repo to
GitHub and that job's result is the real build test. Everything else
[ingestion, training, the Flask API, the frontend calling it, and the full
test suite] was actually run and tested end-to-end in this sandbox,
including a headless-browser click-through of the UI, for all five pilot
areas.)*

## Design decisions worth knowing about

- **Flask, not FastAPI.** The design doc's architecture section describes a
  FastAPI service; this prototype uses Flask instead. The environment this
  was built in could not install new packages (locked-down egress), so
  every dependency here is one that was already provable to work rather
  than assumed to work. All the real logic lives in `app/ml` and
  `app/services`, not in route handlers, so switching frameworks later is
  mechanical, not a rewrite.
- **HistGradientBoostingClassifier (scikit-learn), not XGBoost**, for the
  same reason -- already installed, fully tested here. It's a very similar
  algorithm (native categorical-feature support, comparable accuracy on
  tabular data); swapping is a few lines in `app/ml/train.py` if you'd
  rather standardize on XGBoost/LightGBM.
- **Sigmoid calibration, not isotonic**, despite the design doc naming
  isotonic. Building this prototype's calibration step surfaced a concrete
  failure mode worth knowing about: with isotonic regression and few
  calibration folds, one badly-split fold produced a base classifier that
  predicted ~99% for *every* input regardless of season, and averaging it
  into the ensemble inverted the model's whole seasonal curve for at least
  one species (checked by probing grizzly-bear probability across months
  and finding summer scored *below* winter, backwards from the training
  data's own seasonality). The fix was two-fold: shuffle the data before
  the calibration split (it was previously grouped by area/species, so an
  unshuffled K-fold handed some folds unrepresentative slices) and switch
  to sigmoid calibration, which needs less data per fold to fit stably.
  Revisit isotonic once real training data is orders of magnitude larger.
- **No raw lat/lon as a model feature**, on purpose. See the comment in
  `app/ml/features.py` -- with only 5 pilot areas, a record's lat/lon is
  still close to a per-area constant (area center plus jitter), so it mostly
  gives the model something to overfit to rather than generalize from.
  `area_id` captures location at the resolution this dataset supports;
  reintroduce lat/lon once the catalog spans many distinct locations.
- **Deterministic, template-based "why" explanations, not an LLM.** Each
  prediction's `factors` list (data support, seasonal position, unusual
  weather) is generated from concrete numbers already in the model bundle
  and the day's weather -- see `_explain()` and `_compute_species_seasonality()`
  in `app/ml/prediction_service.py` / `app/ml/train.py`. This was a
  deliberate choice over free-text generation: it's instant, has no
  external dependency, and every sentence is directly traceable to a number
  you could look up yourself, which matters for a scientific-leaning
  audience. A future version could layer richer natural-language generation
  on top of these same underlying facts.

## What's not done yet (see the design doc's roadmap for the full plan)

- eBird, IUCN range polygons, and land-cover data are designed for and
  partly wired (`ebird_client.py` is a complete, working client) but not
  yet pulled into feature engineering or the plausibility gate.
- The plausibility gate is today's curated seed list, not a live GBIF range
  check -- fine for 5 pilot areas spanning very different biomes, not yet
  for "any point on Earth." `gbif_client.has_any_presence()` already exists
  and is unit-tested; it just isn't wired into `species_for_area()` yet.
- One shared model across all species/areas, not per-taxon-class models --
  reasonable at this data volume, worth revisiting as the catalog grows.
- No accounts or saved locations.
- Live GBIF/Open-Meteo ingestion has been reasoned through against verified
  API schemas but never actually executed end-to-end against the live
  APIs -- both this sandbox's network policy and (this round) a currently-
  broken local-shell bridge on the developer's linked computer have blocked
  it. `docker build` is in the same position -- see the Deploying section's
  caveat. Both are one `git push` to GitHub away from being genuinely
  verified by CI.

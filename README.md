# WildCast

A daily wildlife-encounter forecast: for a place and a date, a ranked list of
species with a calibrated probability of encounter, factoring in season and
that day's weather. See the full design doc (architecture, data strategy,
roadmap) here: **[Wildlife Encounter Prediction Platform](https://claude.ai/code/artifact/833b0c2f-87b4-40e1-9262-a5ff433907a8)**.

This repo is the **Phase 0/1 prototype**: ten curated pilot areas spanning
ten very different biomes -- Yellowstone (temperate national park), Kruger
(savanna national park), urban Brisbane, Arctic tundra (Svalbard), Amazon
rainforest, Serengeti (East African savanna), Borneo rainforest, Patagonia
(cold steppe), the Scottish Highlands (temperate moorland/mountain), and a
Costa Rican cloud forest -- 50 species, a real multi-source data pipeline
(GBIF + iNaturalist + eBird sightings, real Open-Meteo weather, and optional
real IUCN Red List conservation status), a trained, calibrated model, a
plain-language "why" explanation for every prediction that now also covers
population/conservation status and a species' natural activity pattern
("ability to evade"), a Flask API, and a map-based web frontend -- installable
as a real app on desktop and mobile (see "Installing WildCast as an app"
below) -- with two browsing modes: **Curated areas** (instant, pre-trained)
and **Explore anywhere** (click any point on Earth; gated live against real
sighting records from multiple independent sources instead of a fixed area
list -- see "Explore Anywhere mode" below). It runs end-to-end today, on an
offline demo dataset if you have no internet yet, or on live data if you do
-- and it always tells you, unmistakably, which one you're looking at (see
"Is this real data?" below).

The first five areas were deliberately chosen to make the project's own
founding plausibility requirement checkable, not just assumed: Svalbard
(polar bear, arctic fox, muskox, snowy owl, Svalbard reindeer) and the
Amazon (jaguar, scarlet macaw, capybara, giant otter, harpy eagle) sit at
opposite climate extremes, and `backend/tests/test_prediction_service.py`
has a test that pins this literally -- a polar bear must never be offered as
a candidate for the Amazon rainforest area, and nothing is ever offered
outside the one area it has real support in. The second five (Serengeti,
Borneo, Patagonia, Scottish Highlands, Costa Rica) were added in a later
round to broaden geographic and biome coverage -- see "Catalog expansion"
below for how they were curated and what's intentionally still out of scope.

## Repo layout

```
wildcast/
  backend/
    app/
      main.py              Flask app (routes)
      config.py             All settings / API base URLs / pilot areas paths
      routers/               areas.py, predictions.py, auth.py, favorites.py,
                               saved_searches.py, notifications.py, internal.py,
                               parks.py -- the HTTP layer
      services/               gbif_client.py, inaturalist_client.py, ebird_client.py,
                               iucn_client.py, weather_client.py, data_source_marker.py,
                               protected_planet_client.py, auth_service.py,
                               favorites_service.py, saved_search_service.py,
                               notification_service.py
      db.py                   stdlib-sqlite3 accounts/favorites/saved-searches storage
      ml/
        features.py           Shared train/serve feature engineering
        pseudo_absence.py     Target-group background sampling
        train.py               Trains + calibrates the model
        prediction_service.py  Loads the model, answers prediction requests
    data/
      pilot_areas.json        The 10 pilot areas
      seed_species.json       The 50 seed species (Phase 0/1's curated catalog),
                               each with a real activity_pattern (diurnal/
                               nocturnal/crepuscular/cathemeral)
      countries.py / .json    Static ISO3 code/name table for the Browse Parks
                               country picker (regenerate with `python data/countries.py`)
      cache/                  Ingested/generated data lands here (gitignored,
                               except when .github/workflows/retrain.yml
                               force-commits real data back to the repo);
                               data_source.json marks it real vs. synthetic_demo
    scripts/
      ingest_gbif.py          REAL: pulls live GBIF occurrence records
      ingest_weather.py       REAL: pulls live Open-Meteo weather + normals
      ingest_iucn.py           REAL, OPTIONAL: pulls live IUCN conservation status
      generate_sample_fixtures.py   SYNTHETIC offline demo data (see below)
      check_real_data_complete.py   Used by Dockerfile.backend: is every area's
                                     real cache present, or fall back to synthetic?
    tests/                     283 unit/integration tests, see "Testing & CI"
    requirements.txt
    requirements-dev.txt       ruff, flake8, pytest (optional runner)
  frontend/
    index.html                 Single-file map + forecast UI (no build step);
                                mode switch (curated / explore anywhere / browse
                                parks), favorites, area comparison, coordinate
                                entry, an unmissable real-vs-demo-data banner
    manifest.json, sw.js, icons/   PWA: installable as a real app -- see
                                "Installing WildCast as an app" below
  docker/
    Dockerfile.backend, Dockerfile.frontend, docker-compose.yml
  .github/workflows/
    ci.yml                       Lint + test + Docker build, on every push/PR
    retrain.yml                  Fetches real GBIF/Open-Meteo data + commits
                                  it; manual trigger or weekly schedule
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

**Recommended: let GitHub do it.** `.github/workflows/retrain.yml` runs the
real ingestion scripts on GitHub's own runners (which have real internet,
unlike a locked-down sandbox) and commits the result back to `backend/data/cache/`.
Trigger it from the repo's **Actions** tab -> **Retrain on live data** ->
**Run workflow**, or just let its weekly schedule (Mondays, 06:00 UTC) keep
it fresh. Once it's run at least once **and produced a complete cache for
every pilot area**, `docker/Dockerfile.backend` automatically detects that
and trains on it instead of the synthetic demo fixtures -- so the very next
Docker build (including a redeploy on Render/Fly/etc.) bakes in real
predictions, no other change needed. The completeness check
(`scripts/check_real_data_complete.py`) falls back to generating synthetic
fixtures for every area whenever any single area's real cache is missing
or partial (e.g. right after a new area is added to `pilot_areas.json` but
before the next `retrain.yml` run covers it) -- this is deliberate and
all-or-nothing, so the Docker build never crashes and never silently mixes
real data for some areas with synthetic for others.

**Or run it yourself, locally:**

```bash
cd backend
python -m scripts.ingest_gbif      # live GBIF occurrence records, no API key needed
python -m scripts.ingest_weather   # live Open-Meteo historical weather + climate normals
python -m scripts.ingest_iucn      # optional: live IUCN conservation status, needs IUCN_API_KEY in .env
python -m app.ml.train             # retrain on the real cache
```

Either way needs outbound internet access to `api.gbif.org` and
`archive-api.open-meteo.com` (both free, no key). If you're running this
locally from a sandboxed/locked-down environment, these calls will fail with
a connection or proxy error -- use the GitHub Actions path above instead, or
run it from a normal machine.

Both `ingest_gbif.py` and `ingest_weather.py` retry transient failures with
exponential backoff -- not just HTTP 429/5xx responses, but raw transport
failures too (`ReadTimeout`, `ConnectionError`, ...), since a shared
GitHub-runner IP or a slow response for a large request (e.g. fetching ~20
years of history for one area) can trip either failure mode. A real
`retrain.yml` run hit a `ReadTimeout` fetching Kruger National Park's full
historical range and crashed the whole ingestion script before this was
added -- see `weather_client._get_with_retry` / `gbif_client._get` for the
retry schedule (patient for batch ingestion, fast-fail via `INTERACTIVE_*`
for live "Explore Anywhere" calls -- see below).

`ingest_gbif.py` also only ever writes a real, day-precision date
(`scripts/ingest_gbif._record_date`) for each occurrence record. GBIF's
`eventDate` is not reliably a full date -- it can be year-only, year-month,
an ISO8601 interval, or occasionally malformed text -- and a record without
real day precision is worse than useless for joining a sighting to one
day's weather, so it's dropped rather than guessed at (no more silently
defaulting a missing month/day to "the 1st"). A real `retrain.yml` run hit
exactly this: a Scottish Highlands record's `eventDate` was a bare "2008",
which used to pass straight through into `occurrences_scottish_highlands.csv`
and crash `ingest_weather.py`'s date parsing downstream. `ingest_weather.py`
also has a defensive second layer (an unparseable date is skipped with a
warning rather than crashing the run) in case a stale cache file or a future
GBIF quirk slips past the first fix.

iNaturalist needs no key at all and is queried automatically in Explore
Anywhere mode alongside GBIF (see "Multi-source data quality" below).
Optional: copy `backend/.env.example` to `backend/.env` and add
[an eBird API key](https://ebird.org/api/keygen) (instant, free) to add
eBird as a third independent corroboration source for birds, and/or
[an IUCN Red List API key](https://api.iucnredlist.org) (free, approval not
always instant) to enable real conservation-status context in every
prediction's "why" explanation (see "Population and ability-to-evade
factors" below). Both are skipped automatically, with a clear log message,
when unset -- nothing breaks without them.

**IUCN specifically needs one extra step to actually reach production,**
unlike eBird: add `IUCN_API_KEY` as a **GitHub repo secret** (Settings ->
Secrets and variables -> Actions -> New repository secret) so
`.github/workflows/retrain.yml`'s `scripts/ingest_iucn.py` step can see it
and commit `data/cache/conservation_status.json`. A key only in your local
`.env` reaches conservation status if you run `python -m app.ml.train`
yourself, but not the deployed model -- see "Population and ability to
evade factors" below for exactly why (Docker build steps, where the
deployed model is actually trained, have no access to secrets).

## Adding your own areas or species

Edit `backend/data/pilot_areas.json` (add `{id, name, type, lat, lon,
radius_km, gbif_country, timezone}`) and `backend/data/seed_species.json`
(add `{scientific_name, common_name, taxon_class, area_id}` -- scientific
names are matched against GBIF, so use `app/services/gbif_client.match_species()`
to confirm a name resolves before adding it). Then re-run ingestion +
training. A new species you add here also immediately becomes checkable in
Explore Anywhere mode everywhere on Earth, not just its home area -- see
below.

## Catalog expansion (5 areas/25 species -> 10 areas/50 species)

The catalog started at 5 areas / 25 species and was deliberately doubled in
a later round, not jumped straight to hundreds. Real reason: curating a
species' `plausible_countries` list (or an area's climate/seasonality
profile for the offline demo) is manual natural-history work, and manual
work at scale makes mistakes -- a self-check while building the *first* 25
species' country lists already caught a real one (Svalbard's GBIF country
code, `SJ` not `NO`; see "Explore Anywhere mode" below). Jumping straight
from 25 to 500 species multiplies that error surface by 20x with no
corresponding increase in how carefully any one entry gets checked. Five
new areas -- Serengeti (Tanzania), Borneo rainforest (Malaysian Sabah),
Patagonia (Chile), the Scottish Highlands, and a Costa Rican cloud forest --
and 5 species per area (25 new species total) is a deliberately bounded
second batch: large enough to meaningfully broaden geographic/biome
coverage (savanna, tropical rainforest, cold steppe, temperate moorland,
tropical cloud forest -- five biomes with zero overlap with the original
five), small enough that every single new entry's `plausible_countries`
list and offline-demo seasonal/weather profile could be checked by hand
against real natural-history knowledge, the same standard the original 25
were held to.

The same automated self-check used for the original 25 was re-run against
all 50 (see `scripts/` -- or just: every species' own curated home area's
`gbif_country` must appear in that species' own `plausible_countries` list)
and passed clean this time -- no Svalbard-style mistake in this batch.

**This is intentionally not the full 500+ species / 500+ national parks
originally asked for.** That scale needs a real architecture change, not
more hand-curation: live GBIF *species discovery* per clicked point instead
of checking a fixed list one by one, and a real public protected-areas
database (Protected Planet's API key is scaffolded in `app/config.py` as
`PROTECTED_PLANET_API_KEY` but not yet wired up) instead of hand-picking
areas one at a time. Both would need to go through the same GitHub Actions
pipeline `retrain.yml` already uses, since this sandbox can't reach GBIF or
Protected Planet directly. That's a real product/architecture decision, not
a default to make silently -- flagged for a future round.

## Explore Anywhere mode

Curated areas are the fast path: 10 pre-trained places, plausibility-gated
by a fixed seed list, zero live calls per request. **Explore Anywhere** is the
live path, wired in this round (`app/ml/prediction_service.species_for_location`
/ `predict_at_location` / `best_window_at_location`, and the
`/api/predict-location` + `/api/best-window-location` endpoints) -- the
concrete upgrade this README and `CONTRIBUTING.md` previously flagged as the
single highest-value next step (`gbif_client.has_any_presence()` existed and
was unit-tested for a while before anything actually called it).

For an arbitrary lat/lon: every one of WildCast's 50 curated species (no new
ones are invented) is checked live against GBIF for at least one real
occurrence record within a radius of that exact point (parallelized across
species, cached in-process for 24h so repeat visits to the same spot are
fast). Species that pass are scored using real live weather for those exact
coordinates and the trained model of whichever curated area is geographically
closest -- the response always names that nearest area and the real distance
to it, so the approximation is never hidden. All live calls on this path
(GBIF presence checks and the live weather calls) use a fast-fail retry
budget (a couple of seconds, not the ~4-minute-worst-case patience
`retrain.yml`'s batch ingestion uses) so a live outage surfaces as a clear
error in seconds rather than hanging the page -- see
`weather_client.INTERACTIVE_*` / `gbif_client.INTERACTIVE_*` and the tests in
`AnywhereModeTests` / `PredictLocationEndpointTests` that pin this.

For one species, GBIF, iNaturalist, and (for Aves, when `EBIRD_API_KEY` is
set) eBird are all fetched **concurrently**, not one after another --
`LiveEvidenceConcurrencyTests` pins this. This isn't just an optimization:
a real 2026-09-17 production incident found that fetching them sequentially
let their timeouts SUM, and the moment `EBIRD_API_KEY` was first configured
live, a single bird-species request could exceed gunicorn's default 30s
worker timeout and Render's proxy returned a bare 502 instead of a slow
prediction. `docker/Dockerfile.backend` also now sets gunicorn's
`--timeout` to 60s (was the 30s default) as extra headroom on top of the
concurrency fix.

The presence check stacks two independent filters against false positives
from zoo/captive animals, because one alone was not enough:

1. It excludes GBIF's `LIVING_SPECIMEN` and `FOSSIL_SPECIMEN` `basisOfRecord`
   values (see `gbif_client._WILD_BASIS_OF_RECORD`). This only catches
   formal institutional living-collection records (a zoo's own accession
   database, published straight to GBIF).
2. Each curated species also carries a hand-checked `plausible_countries`
   list (ISO country codes, in `seed_species.json`) that the live GBIF
   query is restricted to. This is what closes the gap #1 leaves open: a
   casual visitor's photo of a zoo animal (e.g. posted to iNaturalist)
   typically carries the exact same `basisOfRecord` as a genuine wild
   sighting -- `HUMAN_OBSERVATION` -- so metadata filtering alone can't
   tell them apart. Restricting the query to a species' real native-range
   countries can: whatever basis-of-record a Bangkok zoo lion's photo
   carries, Thailand is simply never in `Panthera leo`'s country list, so
   the query cannot return it, full stop.

Found via two rounds of real user reports: a lion and a leopard first
showed up as "verified" a few km from Bangkok, Thailand (fix #1, above, was
built for that); after redeploying, the lion (and now a gray wolf too)
*still* showed up, which is what led to fix #2. The leopard was correctly
left alone throughout -- Indochinese leopards are a real native subspecies
in that region, so it's a genuinely plausible result, not a bug. (A
self-check while building the country lists also caught a real mistake
before it shipped: Svalbard's GBIF country code is `SJ`, not `NO` -- using
`NO` alone would have made the fix wrongly exclude Svalbard's own polar
bear/arctic fox/muskox/snowy owl/reindeer records from their actual home
turf.) This remains a real GBIF data-quality caveat worth knowing, not
fully eliminated: GBIF occurrence coordinates can still be mis-geocoded on
rare occasions (e.g. a museum specimen geocoded to a country centroid), so
"verified live against GBIF" means "GBIF has a wild-labeled record in a
plausible country here," not an absolute guarantee.

Honest limits: this is still bounded to the 50 curated species (a real
"any species anywhere" mode needs live GBIF *species discovery* near a
point, not just presence-checking a known list -- a bigger follow-on), and
reusing the nearest curated area's model for the categorical `area_id`
feature is an approximation, not a location-specific retrain.

### Confidence is calibrated to real local evidence, not borrowed counts

A third real user report, after the two data-quality fixes above: a Leopard
near a point in Thailand showed "91% high confidence" backed by "500
historical sighting records" -- and Leopard genuinely passes both
plausibility filters there (Indochinese leopards are a real native
subspecies), but that 500-record figure was Kruger, South Africa's home-area
training count, reused as the closest analog. It had nothing to do with how
many leopards had actually been recorded near that exact point. The user's
own framing of the underlying bug: *"if there is a species in an area, but
there are barely sightings, you cannot have a high probability."*

The fix: `gbif_client.presence_count()` (species_for_location's presence
check was already making this exact GBIF call -- GBIF returns the true
`count` on every response regardless of `limit`, even `limit=1`, so this
required zero extra requests, just stopping at throwing the number away).
Explore Anywhere now carries this real local count end to end:

- **Confidence** (`high`/`medium`/`low`) is computed from the real local
  count within `radius_km` of the click, using the same 150/40 thresholds
  curated areas use for their real per-area training counts -- not the
  species' curated home area's number.
- **Probability is scaled down** when local evidence is thin:
  `probability = raw_model_probability * min(1, sqrt(local_count / 150))`.
  One confirmed local record scales a prediction to ~8% of the model's raw
  output; 40 records to ~52%; 150+ records is unscaled. Square root (not
  linear) so the penalty is steep at the low end but eases off well before
  the "high confidence" bar, rather than needing hundreds of local records
  just to stop being penalized at all. Both the scaled `probability` and
  the unscaled `raw_model_probability` are returned, so the calibration is
  never hidden. `best_window_at_location` applies the same per-species
  scaling factor across its whole year-long curve.
- The confidence factor sentence itself now reads e.g. *"Low confidence:
  only 2 confirmed wild GBIF sighting records found within 150km of this
  exact point -- probability has been scaled down accordingly"* instead of
  citing a number from a different place entirely.

Curated areas (`predict()` / `best_window()`) are not changed by this: their
`training_records` figure was already the real per-area count the model was
literally trained on, and the model's own calibrated probability already
reflects how much (or little) of that real data backs each species -- there
was never a borrowed-number problem there to begin with. This fix is scoped
to Explore Anywhere, where the borrowing was the actual bug.

See `AnywhereModeTests.test_predict_at_location_dampens_probability_for_thin_local_evidence`
and `..._best_window_at_location_dampens_for_thin_local_evidence` in
`test_prediction_service.py` for the regression tests: same inputs, same raw
model probability, and a 1-record species scores far below a 1000-record
species once evidence-scaled.

## Multi-source data quality (GBIF + iNaturalist + eBird)

Explicit product requirement: *"real data only, from many sources, to
ensure quality."* Explore Anywhere now queries three independent sighting
databases per species per click -- GBIF (always), iNaturalist (always, no
key needed), and eBird (birds only, when `EBIRD_API_KEY` is configured) --
instead of GBIF alone.

The tempting-but-wrong approach is to just sum every source's count as
"more evidence." That would double-count: GBIF already aggregates a large
share of iNaturalist's own research-grade (community-vetted) observations
as one of its many contributing datasets, so a raw sum inflates the same
underlying sightings twice. Instead (`app/ml/prediction_service._local_evidence_cached`):

- One source becomes the **primary count** -- GBIF's count if it found
  anything, otherwise whichever other source did. This is what confidence
  labels and the evidence-dampening scaling (see above) are computed from,
  exactly as before.
- The other sources that also found something become **corroborating
  count** -- real, additional, independent confirmation, which nudges the
  evidence factor up by a modest, capped bonus
  (`min(1.0, base * (1 + 0.15 * min(corroborating_count, 2)))`) rather than
  inflating the volume number itself. A species backed by two independent
  databases is more trustworthy evidence than the same raw count from one,
  but it should nudge confidence, not multiply it.
- When a corroborating source confirms a species, the prediction's
  `factors` list says so explicitly (e.g. *"Also independently confirmed
  by iNaturalist in this area -- not just one data source"*), and the API
  response exposes the full per-source breakdown
  (`local_evidence_sources: {"gbif": 12, "inaturalist": 3}`) rather than
  hiding it behind a single opaque number.
- A source that errors out (rate limit, outage, blocked egress) is recorded
  as 0 for that source and logged, not treated as a crash -- consistent
  with the fail-closed design the original GBIF presence check already
  used.

See `EvidenceFactorTests` and the `AnywhereModeTests` corroboration tests in
`test_prediction_service.py`, plus `test_inaturalist_client.py` and
`test_ebird_client.py`, for the pinned behavior.

## Population and "ability to evade" factors

Explicit product requirement: predictions should factor in *"population,
species ability to evade, and add more relevant ones,"* shown clearly to
the user -- and, per the same requirement, only using **real data**, never
an invented statistic.

- **Population / conservation status**: `app/ml/train.py`'s
  `_compute_conservation_status()` looks up each species' latest IUCN Red
  List assessment (category + population trend) at training time -- real,
  authoritative data, not sighting counts, and training-time only since a
  species' Red List status doesn't change per location or date the way live
  occurrence counts do. **Production path:** `scripts/ingest_iucn.py` runs
  on GitHub's runners via `.github/workflows/retrain.yml` (where the
  `IUCN_API_KEY` secret and real network access both exist) and commits
  `data/cache/conservation_status.json`; `train()` just reads that file --
  the same "fetch with real access on GitHub, commit as cache, train from
  the cache" pattern GBIF/weather already use. This matters because the
  model that actually ships is trained inside
  `docker/Dockerfile.backend`'s build step, which has **no access to repo
  secrets** unless explicitly wired through -- calling the IUCN API
  directly from inside `train()` (an earlier version of this feature) would
  have meant conservation status could never reach a deployed model no
  matter how correctly `IUCN_API_KEY` was configured as a secret. `train()`
  still falls back to a live IUCN call when no cache file exists yet and a
  key happens to be set locally, purely as a convenience for running
  `python -m app.ml.train` directly during development. When available, it's
  surfaced as a real prediction factor, e.g. *"IUCN Red List status:
  Endangered (population trend: decreasing) -- real conservation data, not
  sighting data; rarer/declining species are naturally harder to
  encounter."* Skipped automatically (with a clear log message, not a
  silent gap) when no key is configured -- see `app/services/iucn_client.py`.
- **Ability to evade / detectability**: rather than inventing a numeric
  "evasion score" with no real backing, `data/seed_species.json` now
  carries each species' real natural-history `activity_pattern`
  (`diurnal` / `nocturnal` / `crepuscular` / `cathemeral`), hand-curated to
  the same standard as the rest of the catalog. It's surfaced as honest
  qualitative context (e.g. *"This species is nocturnal (mainly active at
  night), so it's naturally harder to encounter than a day-active species
  even where it's common"*) rather than a fabricated statistic, and is
  deliberately **not** fed into the ML model itself -- with only 50 species
  and 10 areas, adding it as a training feature risked the model picking up
  a spurious correlation from a handful of nocturnal species rather than a
  real signal. `diurnal` (the default, easiest-to-encounter case) gets no
  extra line, so the note only appears when it's actually informative.

Both factors are additive and independent: a prediction can have zero, one,
or both, and neither ever replaces the core evidence-based confidence
figure -- they're context, not a replacement for real sighting data. See
`ExplainFactorsTests` in `test_prediction_service.py`.

**Deliberately deferred, not half-built:** true time-of-day encounter
timing (e.g. "most sightings happen 6-9am") would need real
timestamp-derived diel-activity data per species per area, which isn't
reliably available from GBIF/iNaturalist/eBird at this catalog's scale yet
-- `activity_pattern` above is the honest, real-data-backed stand-in for
now.

## Is this real data?

Explicit product requirement: keep the offline synthetic demo mode (it's
what makes the app explorable with zero internet access or API keys), but
label which one is currently live *unmistakably* -- never let a user
mistake demo predictions for real ones.

`scripts/ingest_gbif.py` / `ingest_weather.py` (real ingestion) and
`scripts/generate_sample_fixtures.py` (synthetic demo) each write
`backend/data/cache/data_source.json` via `app/services/data_source_marker.py`
when they finish a run -- whichever ran most recently is what's marked.
`app/ml/train.py` reads it into the trained model bundle, so:

- `GET /api/health` always includes a `data_source` field
  (`{"source": "real" | "synthetic_demo" | "unknown"}`, plus a timestamp).
- Every `/api/predict` and `/api/predict-location` response includes the
  same field.
- The frontend shows a persistent, color-coded banner at the top of the
  page built directly from `/api/health`'s answer -- green "Live data" for
  real, amber "Demo mode" for synthetic, computed fresh on load and whenever
  the API base URL changes (see `loadDataSourceBanner()` in
  `frontend/index.html`).

`"unknown"` covers a model bundle trained before this marker existed, so an
older real dataset is never mislabeled as synthetic by omission.

## Installing WildCast as an app

The frontend is a installable Progressive Web App: `frontend/manifest.json`
+ `frontend/sw.js` + `frontend/icons/` (generated paw-print icons at every
required size, including a maskable variant for Android's adaptive-icon
mask). On any modern desktop or mobile browser (Chrome, Edge, Safari), open
the deployed frontend URL and use the browser's own "Install app" / "Add to
Home Screen" prompt -- no app-store submission needed. Once installed, it
opens in its own window/icon like a native app and the app shell (HTML/CSS/
JS/icons) loads instantly from the service worker's cache, even offline.

**What's cached and what isn't, on purpose:** only the static app shell is
cached. `sw.js` explicitly never caches API responses -- predictions must
always be a live network call, both because they change with the date/
weather and because caching them would risk showing stale, wrongly-labeled
(real vs. synthetic-demo) data long after it was fetched, directly
undermining the real-data labeling above. Cross-origin requests (the
configured API host, the Leaflet CDN, OpenStreetMap tiles) are never
intercepted by the service worker at all -- see `sw.js`'s comments.

This ships the "make it an app already" requirement as a real, working,
installable app today, on every platform with a browser, with zero app-store
review wait and zero extra backend infrastructure. A wrapped native
(React Native / Capacitor) build remains a possible future step if
app-store distribution specifically becomes a requirement, but a PWA is the
faster, lower-risk path to "a real app" from this single-file frontend
architecture.

## Testing & CI

```bash
cd backend
pip install -r requirements-dev.txt
ruff check .
flake8 --max-line-length=130 --extend-ignore=E501,W503,E127 app scripts tests
python -m unittest discover -s tests -v      # 178 tests, ~15 seconds
```

The suite is plain `unittest.TestCase` (no pytest dependency required to run
it, though pytest picks it up fine too). It splits into two kinds:

- **Pure unit tests** (`test_features.py`, `test_pseudo_absence.py`,
  `test_gbif_client.py`, `test_inaturalist_client.py`, `test_ebird_client.py`,
  `test_iucn_client.py`, `test_weather_client.py`, `test_data_source_marker.py`)
  -- no network, no trained model. The client tests mock HTTP responses to
  check request construction and response parsing without needing live
  access; one of them (`test_weather_client.py`'s
  `WeatherAnomalyFeaturesTests`) caught a real bug while being written --
  `weather_anomaly_features()` was reading normals columns under their
  pre-refactor names and would have raised a `KeyError` the first time
  anything actually called it. Fixed; the test now pins the correct names so
  it can't silently regress.
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
push/PR, plus a separate job that builds both Docker images. **Both jobs are
confirmed green on GitHub's runners** -- this is the real, independent proof
that the test suite passes and both images actually build, since the
sandbox this project was developed in has no Docker daemon of its own.
(One real bug surfaced and got fixed via this: an early CI run failed on an
unpinned `ruff` install picking up a newer default ruleset than the version
used during development -- see "Design decisions" below.)

A separate workflow, `.github/workflows/retrain.yml`, fetches real GBIF/
Open-Meteo data on GitHub's runners and commits it back to the repo -- see
"Switching to real data" above.

## API reference

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | Liveness + whether a trained model is loaded + `data_source` (real vs. synthetic_demo, see "Is this real data?") |
| `GET /api/areas` | The pilot areas |
| `GET /api/areas/<id>` | One area |
| `GET /api/areas/<id>/species` | Plausibility-gated species catalog for an area |
| `GET /api/predict?area_id=&date=YYYY-MM-DD[&species=<scientific name>]` | Ranked probabilities, each with a `factors` list of plain-language reasons (data support, seasonal timing, unusual weather, and -- when available -- conservation status and activity pattern); response also includes `data_source` |
| `GET /api/best-window?area_id=&species=<scientific name>` | Probability by day-of-year, for trip planning |
| `GET /api/predict-location?lat=&lon=&date=YYYY-MM-DD[&species=&radius_km=]` | Explore Anywhere: same shape as `/predict`, gated by a live multi-source check (GBIF + iNaturalist + eBird, see "Multi-source data quality") instead of a fixed area; `radius_km` defaults to 150, clamped to [10, 300]. `probability`/`training_records`/`confidence` are calibrated to the real local evidence near the click (see "Confidence is calibrated to real local evidence" above); `raw_model_probability` is the unscaled figure; `local_evidence_sources` breaks the count down per source |
| `GET /api/best-window-location?lat=&lon=&species=<scientific name>[&radius_km=]` | Explore Anywhere counterpart to `/best-window` |
| `POST /api/auth/register` / `POST /api/auth/login` | Create an account / sign in -- returns `{user, token}`; every endpoint below needs `Authorization: Bearer <token>` |
| `GET /api/auth/me` / `PATCH /api/auth/me` | Current user's profile / update display name |
| `GET /api/favorites` / `POST /api/favorites` / `DELETE /api/favorites/<id>` | Server-synced saved places (area or Explore Anywhere point, optionally narrowed to one species) |
| `GET /api/saved-searches` / `POST /api/saved-searches` / `DELETE /api/saved-searches/<id>` | "Alert me if `<species>` reaches >= X% probability near `<place>`" |
| `GET /api/notifications[?unread_only=1]` / `POST /api/notifications/<id>/read` | The in-app notification inbox saved-search alerts land in |
| `POST /api/internal/check-saved-searches` (needs `X-Cron-Key: <CRON_SECRET_KEY>`) | Triggers a saved-search evaluation pass -- see "Accounts & saved-search alerts" below, not meant for the frontend |
| `GET /api/parks/countries` | Static ISO3 code/name list for the Browse Parks country picker -- works with no `PROTECTED_PLANET_API_KEY` set |
| `GET /api/parks?country=<ISO3>[&marine=true\|false][&page=&per_page=]` | Country + marine-status filtered browse over WDPA protected areas (see "Browse Parks" below for why this isn't free-text search); `503` if `PROTECTED_PLANET_API_KEY` is unset |
| `GET /api/parks/<site_id>` | One protected area by its WDPA site_id |

## Accounts & saved-search alerts

Real per-account storage (see `backend/app/db.py`, `backend/app/services/{auth,favorites,saved_search,notification}_service.py`)
for the favorites/alerts/notifications features above -- built on Python's
stdlib `sqlite3` rather than an ORM, for the same "only ship what's actually
been run here" reason this project avoids other unprovable dependencies (see
"Design decisions worth knowing about"). Auth tokens are itsdangerous-signed
strings (ships with Flask already), not a session cookie.

**Before relying on this in production, three things need real values, not
the defaults:**

1. **`WILDCAST_SECRET_KEY`** (Render env var) -- signs login tokens. The
   fallback in `app/config.py` is a fixed, publicly-known string (it's in
   this repo); using it in production would let anyone forge a valid login
   for any account. Generate one long random value, e.g. `openssl rand -hex 32`.
2. **`CRON_SECRET_KEY`** (Render env var, and the *same* value as a
   `CRON_SECRET_KEY` GitHub repo secret) -- gates `POST /api/internal/check-saved-searches`
   so only your own scheduler can trigger it. No fallback at all: unset, that
   endpoint refuses every request. See `.github/workflows/check_alerts.yml`
   for a ready-made scheduled GitHub Actions workflow that calls it daily
   (free, same mechanism `retrain.yml` already uses) -- it needs that repo
   secret plus, optionally, a `WILDCAST_BACKEND_URL` repo *variable* if your
   backend isn't at `https://wildcast.onrender.com`.
3. **Persistent storage for the database.** Render's web services have an
   **ephemeral filesystem by default** (confirmed against Render's docs,
   2026-09-17) -- the SQLite file at `WILDCAST_DB_PATH` (default
   `backend/data/wildcast.db`) is wiped on every redeploy/restart otherwise,
   taking every account, favorite, and alert with it. Fix: attach a paid
   Render persistent disk (not available on the free tier) and point
   `WILDCAST_DB_PATH` at a file inside its mount path. Render's Cron Jobs
   feature can't help here either -- it explicitly can't access a persistent
   disk, which is exactly why the alert check is triggered over HTTP against
   the already-running web service instead of run as its own scheduled job
   (see `backend/scripts/check_saved_searches.py`'s docstring for the full
   reasoning).

**Not done yet:** real email/push delivery for alerts -- notifications are
in-app only (`GET /api/notifications`) since this project has no email/push
provider credentials to build and test against, the same reason eBird/IUCN
API keys were obtained through the live user rather than guessed at here.
The extension point is a single `INSERT` in
`app/services/saved_search_service.py`'s `check_due_saved_searches` -- add a
call there to whichever provider you pick (e.g. SendGrid for email), guarded
by its own optional API key the same way `EBIRD_API_KEY`/`IUCN_API_KEY` are
optional. Also not done: session revocation ("log out everywhere") and
password reset, both needing server-side session state or outbound email,
respectively -- both deliberate deferrals, not oversights.

Migrating to a managed Postgres database instead of SQLite (so persistence
doesn't depend on a paid Render disk at all) is a natural next step, but
wasn't done here: it needs a driver such as `psycopg2`, which -- like every
other new dependency this session considered -- this sandbox's blocked
egress couldn't actually install or test, so it's left as a documented
follow-up rather than shipped unverified. `app/db.py` is the only file that
migration would touch; every service function above already goes through it
alone.

## Browse Parks

A "Browse parks" mode (alongside Curated areas / Explore anywhere) lets you
pick a country and browse its protected areas -- national parks, reserves,
marine protected areas, and more -- from the [World Database on Protected
Areas](https://www.protectedplanet.net/) (WDPA), then jump straight into
Explore Anywhere at that park's approximate location. Backed by
`app/services/protected_planet_client.py` and `app/routers/parks.py`.

**This is deliberately a country + filter browser, not a "type a park name
and jump to it" search box.** Real research against Protected Planet's live
v4 API documentation (2026-09-17) found that *neither* the deprecated v3 API
nor the current v4 API has a free-text/name-search endpoint -- only
structured filters (country, marine status, designation, governance, IUCN
category) on a separate `/v4/protected_areas/search` endpoint. A true
name-search feature would need a locally-built index from Protected
Planet's bulk WDPA geodatabase download instead, which was investigated and
explicitly **not** built here: it's ~1.1GB as a File Geodatabase only (no
CSV), needs a GDAL processing pipeline this sandbox can't pip-install (same
blocked-PyPI issue documented elsewhere in this README), ships no
pre-computed centroid field, and -- most importantly -- is licensed
**non-commercial use only** by UNEP-WCMC, which would need a separate
license resolved before shipping given WildCast's monetization plans. The
country+filter approach avoids all of that and is fully buildable today; if
real name search is wanted later, revisit that bulk-download path (or check
whether UNEP-WCMC will grant a commercial license) rather than assuming the
live API grew a search endpoint.

**Requires `PROTECTED_PLANET_API_KEY`** (free, request at
<https://api.protectedplanet.net/request>, same opt-in-key pattern as
eBird/IUCN) -- unset, `/api/parks` and `/api/parks/<id>` return a clean
`503` and the frontend shows a "not configured" message rather than a
broken UI; `/api/parks/countries` always works (it's a static local file,
not a live API call, since Protected Planet has no "list countries"
endpoint either).

A park's `centroid` is a plain average of its polygon's vertices, not a
true area-weighted centroid (that needs a GIS library like Shapely, same
install problem as GDAL above) -- close enough to seed an "explore this
area" click, not meant to be geometrically precise. `designation`,
`governance`, and `iucn_category` come back from the API as **integer IDs**
per the docs, not names/codes, and the docs page didn't give the ID->label
mapping, so those three aren't exposed as user-facing filters yet (only the
unambiguous `country` and `marine` are) -- see
`protected_planet_client.py`'s module docstring before wiring them up.

## Deploying

**Backend:** `docker build -f docker/Dockerfile.backend -t wildcast-backend .`
from the repo root, then push to any container host (Render, Fly.io, AWS
ECS/App Runner, ...) -- or just point the host at this GitHub repo and let
it build from the Dockerfile directly, which is simpler and means every new
push (including a `retrain.yml` run) redeploys automatically. The image
trains on whatever's in `backend/data/cache/` at build time: real data if
`retrain.yml` has committed it, the synthetic demo dataset otherwise (see
"Switching to real data" above) -- so run that workflow at least once before
deploying if you want real predictions from the start. If you're using the
accounts/favorites/alerts features, see "Accounts & saved-search alerts"
above for the environment variables and persistent-storage setup those need
before going to production.

**Frontend:** it's one static HTML file with no build step -- deploy it as-is
to Vercel, Netlify, Cloudflare Pages, or GitHub Pages, or serve it via
`docker/Dockerfile.frontend`. Point its "API base URL" field (or hardcode
the default in `frontend/index.html`) at your deployed backend's URL.

**Local full stack:** `docker compose -f docker/docker-compose.yml up --build`
then open `http://localhost:8080`.

*(The sandbox this project was originally developed in has the Docker CLI
but no reachable daemon, so `docker compose config` was used there to
validate the compose file's syntax only, and a real `docker build` genuinely
could not run there. That gap is now closed: both Docker images have been
built successfully on GitHub's own runners via `.github/workflows/ci.yml`'s
`docker-build` job, confirmed green. Everything else [ingestion, training,
the Flask API, the frontend calling it, and the full test suite] was also
run and tested end-to-end along the way, including a headless-browser
click-through of the UI for all ten pilot areas.)*

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
  `app/ml/features.py` -- with only 10 pilot areas, a record's lat/lon is
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

- IUCN range polygons and land-cover data are still not wired in (only
  Red List category/population trend, via `iucn_client.py`, is).
- eBird's own species-specific nearby-observations endpoint isn't used yet
  -- `ebird_client.presence_count()` reuses the all-species nearby endpoint
  and filters client-side by scientific name, since a scientific-name ->
  eBird-species-code lookup hasn't been built. Correct today, just not the
  most efficient shape for a much larger bird catalog.
- `iucn_client.py` and `inaturalist_client.py` were built from each API's
  public documentation, not verified against a live call (this sandbox's
  egress can't reach either host) -- re-verify field names/response shape
  against a real call, or via `retrain.yml` on GitHub's runners, before
  fully trusting them in production. `gbif_client.py` and `ebird_client.py`
  were already live-verified in earlier rounds.
- True time-of-day encounter timing (not just the coarse `activity_pattern`
  qualitative note) needs real timestamp-derived diel-activity data this
  catalog's scale doesn't reliably have yet -- see "Population and ability
  to evade factors" above.
- Explore Anywhere mode (see above) covers any point on Earth for live
  plausibility, but only for WildCast's existing 50 curated species -- true
  "any species, anywhere" needs live GBIF species *discovery* near a point,
  a bigger follow-on than the presence-check gate this round wired in.
- One shared model across all species/areas, not per-taxon-class models --
  reasonable at this data volume, worth revisiting as the catalog grows.
- No accounts; saved locations are a per-browser `localStorage` favorites
  list on the frontend, not synced anywhere. The PWA (see "Installing
  WildCast as an app") makes it installable, but still browser-local, not
  cross-device-synced.
- Catalog scale: 10 areas / 50 species today (see "Catalog expansion"
  above), deliberately short of a 500+/500+ ask because that scale needs
  live species-discovery + a real protected-areas database, not more
  hand-curation -- a genuine architecture decision still awaiting a call.

Live GBIF/Open-Meteo ingestion, `docker build`, and `retrain.yml` are all
now genuinely verified against real data -- see "Testing & CI" and the
Deploying section. iNaturalist, eBird corroboration, and IUCN conservation
status are new this round and unit-tested against mocked responses, but (per
the point above) not yet live-verified the same way GBIF/Open-Meteo were --
run `retrain.yml` at least once to genuinely exercise them against live
APIs.

## Contributing

Contributions that expand real-world coverage -- more areas, more species --
are the most valuable kind this project can get; see
[CONTRIBUTING.md](CONTRIBUTING.md) for how to add one, plus a couple of
bigger, high-value open items. Licensed under [MIT](LICENSE).

## Data attribution

Species occurrence data from [GBIF.org](https://www.gbif.org) and
[iNaturalist](https://www.inaturalist.org); bird observation data from
[eBird](https://ebird.org) (Cornell Lab of Ornithology), when configured;
conservation status from the [IUCN Red List](https://www.iucnredlist.org),
when configured; weather data from [Open-Meteo.com](https://open-meteo.com);
map tiles from [OpenStreetMap](https://www.openstreetmap.org/copyright)
contributors. Please keep this attribution if you fork or redeploy this
project -- it's both a courtesy to the data providers this project depends
on and a condition of their free/non-commercial terms of use.

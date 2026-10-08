# EffOne

Machine-learning predictions for the current Formula 1 season: qualifying, race results, and the championships.

> **Status: early development.** Data ingestion is in progress; there are no models or UI yet. See [Project status](#project-status).

## What it will do

- Predict **qualifying**, then chain it into a **race prediction**. Users can enter their own starting grid instead.
- Output per-driver probabilities for **winning, podium and top-10** finishes, plus DNF risk.
- Let users pick the race, weather (what-if) and grid.
- Simulate the rest of the season to give **drivers' and constructors' championship** odds.
- Update automatically as new sessions finish, and log each prediction before the race so accuracy can be scored afterwards.

Planned for later: pit-stop strategy and safety-car prediction.

## Design principles

- **No leakage.** A feature for a session uses only data available before that session.
- **Time-ordered validation.** Models are tested on seasons they were not trained on (walk-forward), never with random splits.
- **Baselines first.** A model only counts if it clearly beats the simple baselines on held-out seasons: "grid = finish", "previous race = finish" and "average of the last 5 results". "Clearly" uses an interval that resamples whole seasons.
- **Probabilities, not single picks.** Results come from Monte Carlo simulation and are checked for calibration.
- **Honest limits.** The model cannot see car upgrades, new drivers or new regulations. It extrapolates from recent form and shows lower confidence when data is thin.

## Project status

| Milestone | Description | Status |
|---|---|---|
| 0 | Project setup (Python 3.12, tooling, tests) | Done |
| 1 | Data ingestion, 2018 to present | Done for races, qualifying and sprints (practice sessions still downloading) |
| 2 | Evaluation harness and baselines | Done |
| 3 | Leakage-safe features | Done |
| 4 | Qualifying and race models (model zoo, honest leaderboards) | Done, with an honest negative result: for qualifying no model clearly beats a simple form average (see the report) |
| 5 | Win / podium / top-10 probabilities and calibration | Not started |
| 6 | Season simulation (both championships) | Not started |
| 7 | API and prediction log | Not started |
| 8 | Web UI | Not started |
| 9 | Docker, AWS deployment, scheduled refresh | Not started |

The full plan, decisions and a dated progress log are in [`HANDOFF.md`](HANDOFF.md). Contributor and agent rules are in [`AGENTS.md`](AGENTS.md).

## Architecture

```
scheduled refresh -> ingest (fastf1) -> Parquet -> features -> models -> simulation
                                                                  |
                                              JSON / API  <-------+
                                                   |
                                          Next.js web app (Cloudflare Pages)
```

| Part | Technology |
|---|---|
| Data source | [fastf1](https://docs.fastf1.dev/) |
| Storage | Parquet files (S3 in production) |
| ML | pandas, scikit-learn, LightGBM |
| API | FastAPI (planned) |
| Web | Next.js + TypeScript, static export (planned) |
| Deployment | Docker on AWS, frontend on Cloudflare Pages (planned) |

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
pip install --user uv            # once; if `uv` is not on PATH, use `python -m uv ...`
python -m uv sync                # create .venv and install dependencies
python -m uv run pytest          # run tests
python -m uv run ruff check .    # lint
```

### Ingest data

```powershell
# everything finished from 2018 to today (resumable)
python -m uv run python -m ml.ingest

# a subset
python -m uv run python -m ml.ingest --years 2024 --sessions Race Qualifying

# what is still missing? lists pending sessions, downloads none; exit code 1 if any are pending
python -m uv run python -m ml.ingest --dry-run --sessions Race Qualifying Sprint "Sprint Qualifying" "Sprint Shootout"

# the race calendar (needed for circuits), lookup tables, and a data-quality report
python -m uv run python -m ml.ingest.events
python -m uv run python -m ml.ingest.dimensions     # data/dim/{drivers,teams,circuits}.parquet
python -m uv run python -m ml.ingest.audit          # data/reports/data_quality.csv
```

- Output goes to `data/raw/<table>/year=YYYY/round=RR/<session>.parquet`, with tables `results`, `laps`, `weather` and `track_status`, plus `data/raw/events/` for the calendar.
- Ingestion is **idempotent**: finished sessions are recorded in `data/raw/_done/` and skipped on rerun. Failed sessions are retried. All writes are atomic (temp file, then rename).
- A table the source lacks is only accepted for sessions on the explicit allow-list in `ml/ingest/sessions.py` (`KNOWN_MISSING`, currently the 2018 Italian GP laps); anything else fails and is retried.
- fastf1 allows 500 API calls per hour, so a full backfill takes many hours. The job waits and resumes by itself when the limit is hit.
- Data and the fastf1 cache live in `data/` (git-ignored). Set `EFFONE_DATA_DIR` to store them elsewhere, for example on a different drive.
- The audit flags problems per session and sets `results_ok` / `laps_ok`; it never deletes anything.

### Build features

```powershell
python -m uv run python -m ml.features    # data/features/race_features.parquet + feature_missing.csv
```

One row per driver per race. Each feature is computed from races strictly before the target race plus
its entry list, never from its outcome; tests prove this by scrambling everything from the target
onward and checking the features do not move. Rules for anyone training on the table:

- Rows are ordered by driver id, never by finishing position. Keep it that way.
- Use `feature_columns(target)` to get the features legitimately known for a target. The starting
  grid is excluded for qualifying and grid targets. Weather is opt-in (`include_scenario=True`)
  because history holds the realised race-day weather while a prediction only has a forecast.
- `race_ok` marks races that carried no performance information (the 2021 Belgian GP). Use it to
  drop training rows only: it is derived from that race's own results, so it is not a feature and
  not a test filter.
- `features_for_entry(history, entries, key)` builds features for a race that has not happened.

### Evaluate

```powershell
# the model zoo on every target (about 3 minutes): qualifying, race with the real grid, and race
# from a predicted grid. Writes data/reports/leaderboard_<target>.csv and per_race_<target>.csv.
python -m uv run python -m ml.evaluation --models all

# just the baselines, or one target
python -m uv run python -m ml.evaluation
python -m uv run python -m ml.evaluation --models all --target quali
```

Every model is compared with the best baseline on the same races. Each row shows the gain with a 95%
interval from a season-clustered bootstrap (races in a season are alike, so whole seasons are
resampled), plus the number of seasons the model wins. "Clearly beats" means that interval is above
zero. The exit code is 3 if no model clearly beats the best baseline on the race or qualifying
board. The locked 2025 season is never scored unless `--allow-locked` is given.

Tied scores are scored by expectation (tied drivers are equally likely to take any position the tie
covers), so results never depend on row order, and a model that cannot tell drivers apart gets
exactly chance-level credit.

### Tests

```powershell
python -m uv run pytest                    # all tests
python -m uv run pytest -m "not realdata"  # only tests that need no ingested data
```

Tests marked `realdata` read the real `data/` directory and are skipped on a fresh clone. While the backfill is incomplete they fail on purpose.

## Repository layout

```
ml/
  ingest/       fastf1 -> Parquet (incremental, idempotent)
  features/     leakage-safe feature builders
  models/       qualifying and race models
  evaluation/   walk-forward validation, metrics, calibration
  simulation/   race and season Monte Carlo
tests/          pytest suite
data/           raw data and caches (not committed)
HANDOFF.md      plan, decisions, progress log
AGENTS.md       rules for contributors and coding agents
```

## Development workflow

- Work is split into the phases above; each has a "done when" criterion in `HANDOFF.md`.
- Run `pytest` and `ruff` before committing. Every feature builder needs a test showing it does not use future data.
- Notebooks are for exploration only; pipeline code lives in `ml/` and must run from the command line.
- Secrets and credentials are never committed. Cloud resources are created only with explicit approval.

## Data and licensing

Race data comes from fastf1, which uses the official F1 timing feed and the Jolpica (Ergast-compatible) API. This is an unofficial, non-commercial project and is not affiliated with Formula 1. Check the data providers' terms before deploying publicly.

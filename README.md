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
- **Baselines first.** A model only counts if it beats "grid = finish" and "previous race = finish" on held-out seasons.
- **Probabilities, not single picks.** Results come from Monte Carlo simulation and are checked for calibration.
- **Honest limits.** The model cannot see car upgrades, new drivers or new regulations. It extrapolates from recent form and shows lower confidence when data is thin.

## Project status

| Phase | Description | Status |
|---|---|---|
| 0 | Project setup (Python 3.12, tooling, tests) | Done |
| 1 | Data ingestion, 2018 to present | **In progress** (code written, backfill running; rate-limited) |
| 2 | Evaluation harness and baselines | Done |
| 3 | Leakage-safe features | Not started |
| 4 | Qualifying model | Not started |
| 5 | Race model (predicted or user grid) | Not started |
| 6 | Win / podium / top-10 probabilities and calibration | Not started |
| 7 | Season simulation (both championships) | Not started |
| 8 | API and prediction log | Not started |
| 9 | Web UI | Not started |
| 10 | Docker, AWS deployment, scheduled refresh | Not started |

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
```

### Evaluate

```powershell
# score the baselines with walk-forward validation; writes data/reports/leaderboard.csv
python -m uv run python -m ml.evaluation
```

- Output goes to `data/raw/<table>/year=YYYY/round=RR/<session>.parquet`, with tables `results`, `laps`, `weather` and `track_status`.
- Ingestion is **idempotent**: finished sessions are recorded in `data/raw/_done/` and skipped on rerun. Failed sessions are retried.
- fastf1 allows 500 API calls per hour, so the full backfill takes many hours. The job waits and resumes by itself when the limit is hit.
- Data and the fastf1 cache live in `data/` (git-ignored). Set `EFFONE_DATA_DIR` to store them elsewhere, for example on a different drive.

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

# EffOne: Project Plan and Handoff

Status: **planning complete, no code written yet.** Last updated 2026-10-07.
Read `AGENTS.md` first for working rules. This file holds the decisions, architecture, phases and progress log.

## 1. Goal

A full-stack F1 prediction app, focused on the **current season** and updating as new data arrives.

- Predict qualifying, then chain it into a race prediction.
- Let the user override the grid; the race model must turn any grid into a race prediction.
- Outputs per driver: **P(win), P(podium), P(top 10)**, plus P(DNF) and expected position.
- Simulate the rest of the season to give **drivers' champion** and **constructors' champion** probabilities.
- User-selectable inputs: year, circuit, weather (what-if), optional grid.
- Must not overfit to past data.
- Out of scope for now (later phases): race strategy (1-stop, 2-stop) and safety-car prediction.

## 2. Decisions (confirmed by the user)

| Topic | Decision |
|---|---|
| Focus | Current season's upcoming races; continuously updated |
| Grid | Predict qualifying first and chain into race; user may enter a grid instead |
| Training data | **2018 onward only** (fastf1). No pre-2018 data in training. |
| Pre-2018 | Possible later as a held-out **test set** (simulated old races), not as features |
| Returning circuits | A past race at a circuit counts toward track-specific features only if at least 25% of the current grid's drivers raced it that year. Fallback: circuit-type features, shrunk toward the global mean, lower confidence shown. |
| Python | 3.12, pinned via `uv` (the machine has 3.14, which may lack wheels) |
| Frontend | Next.js + TypeScript, **static export**, hosted on **Cloudflare Pages** (the user's site is on Cloudflare Pages, linked to GitHub) |
| Backend | FastAPI in Docker on AWS |
| Refresh | Incremental update only (features, driver and team form ratings, simulation). **No full retrain after every race**; full retrains are rare and manual. |
| Outputs | Win/podium/top-10 are the priority |
| AWS | The user has an account but is new to AWS. **Guide them step by step**, explain costs, set a billing alarm first. |

## 3. Key modelling principles (anti-overfitting)

- **Walk-forward validation only** (train on seasons up to N, test on N+1). Never random k-fold.
- Every feature for a race uses only data available before that session. Test for leakage.
- Predict relative quantities (gap to field, team-relative pace). Regulation eras reset absolute times.
- Weight the current season heavily with recency decay. Early in the season, shrink toward last year's form and show lower confidence.
- Start simple and require each step to beat the last on held-out seasons:
  1. baseline "grid = finish" and "previous race = finish"
  2. regularised LightGBM (shallow trees)
  3. ranking or Bayesian model (e.g. Plackett-Luce) only if it earns its place
- Output **distributions** (Monte Carlo over model scores plus DNF risk), not a single order.
- Check calibration with reliability curves.
- The grid is an input feature; the race model must not just copy it.
- Be honest about limits: the model cannot see upgrades, new drivers or new regulations (the 2026 rules reset form).

## 4. Prediction stages in a weekend

1. **Pre-weekend:** form and circuit only.
2. **After practice:** add long-run race pace from FP2/FP3 where available.
3. **After qualifying:** the real grid replaces the predicted one.
4. For sprint weekends, handle the sprint session and sprint points.

Each stage re-predicts with whatever data exists. Log every prediction **before** the race and score it afterwards (running accuracy record).

## 5. Season simulation

- Each run plays through all remaining races, with completed real results fixed.
- Include the points system (sprint points, fastest-lap rule for the season in question), DNFs, and **teammate correlation** (otherwise the constructors' odds come out too narrow).
- Use the remaining calendar and current lineups from fastf1; handle mid-season driver swaps as an edge case.
- A few thousand runs give P(drivers' champion), P(constructors' champion), expected final points.
- Cache results after each refresh; the API serves cached output.

## 6. Architecture

```
EffOne/
├── AGENTS.md
├── HANDOFF.md
├── ml/            # ingest, features, qualifying + race models, simulation, evaluation
├── api/           # FastAPI: /next-race, /predict, /season, /history, /meta
├── web/           # Next.js + TypeScript (static export -> Cloudflare Pages)
├── infra/         # Dockerfiles, Terraform for AWS
├── data/          # raw fastf1 cache + parquet (gitignored)
├── tests/
└── docker-compose.yml
```

Data flow: scheduled job pulls new sessions → idempotent append to Parquet → update features and ratings → rerun qualifying/race/season prediction → write artifacts to S3 → API reads latest → web displays.

Storage: Parquet and model artifacts in S3; no database to start.

Weather: Open-Meteo forecast for the circuit (no key). The user can override for what-if.

### Hosting notes (AWS, beginner-friendly)

- The static Next.js site lives on Cloudflare Pages; it calls the API on AWS, so the API needs **CORS** for the site's domain, plus HTTPS.
- Suggested starting setup: API container on **AWS App Runner** (simplest), S3 bucket for data and artifacts, EventBridge schedule triggering an ECS Fargate task for the refresh job, all in Terraform.
- Verify App Runner's current availability and pricing before committing; **ECS Fargate** is the fallback.
- First AWS steps for the user: create a billing alarm, use an IAM user or role with limited permissions (not root), pick one region.
- Do not commit AWS credentials. Do not run `terraform apply` or create paid resources without the user's explicit go-ahead.

## 7. Phases

| # | Phase | Done when |
|---|---|---|
| 0 | Setup: git init, `uv` with Python 3.12, lint (ruff), pytest, `.gitignore`, README | `pytest` runs, fastf1 imports |
| 1 | Ingestion: incremental and idempotent, 2018 to present, fastf1 cache, Parquet tables (`races`, `results`, `laps`, `weather`, `drivers`, `teams`, `circuits`), circuit-history rule | Row counts match known races and drivers; rerun adds nothing |
| 2 | Eval harness and baselines: walk-forward CV, metrics (rank correlation, top-N accuracy, log loss, calibration) | Baselines reproducible |
| 3 | Features: leakage-safe, form ratings, circuit features, weather | Leakage tests pass |
| 4 | Qualifying model | Beats baseline on held-out seasons |
| 5 | Race model, accepting predicted or user grid | Beats baselines |
| 6 | Probabilities: win, podium, top 10, DNF, calibration. **Review checkpoint with the user.** | Reliability curves look sane |
| 7 | Season simulation: both championships. **Review checkpoint.** | Matches known standings logic |
| 8 | API and prediction log | `/predict` returns distributions; tests |
| 9 | Next.js UI: next-race view, grid override, what-if weather, championship page | End-to-end demo |
| 10 | Docker, AWS deploy, scheduler (guide the user) | Live and refreshing |
| 11 | Later: pre-2018 test set; strategy and safety-car models | Separate scope |

## 8. Risks

- Small data (about 400 races since 2018): expect modest accuracy. Beating "grid = finish" clearly is realistic; reliably predicting winners is not.
- fastf1 first loads are slow and can be incomplete right after a session: cache, resume and retry.
- Regulation changes break historical relationships.
- Cost: set a billing alarm before creating AWS resources.

## 9. Open items

- Exact domain or subpath where the site will call the API (needed for CORS).
- Whether to use App Runner or Fargate for the API (decide in phase 10).
- Whether the user wants sprint handling in the first model version or after.

## 10. Progress log

- 2026-10-07: Requirements gathered, plan written. `D:\repos\EffOne` is empty apart from these docs and is not yet a git repository. Python 3.14.3, Node 24 and git 2.53 are installed. `fastf1` is not installed. Next step: phase 0.
- 2026-10-07: **Phase 0 done.** Git repo initialised (branch `main`, no commits yet). `uv` installed via pip (invoke as `python -m uv`). Python 3.12 pinned, `.venv` created, deps installed: fastf1, pandas, numpy, pyarrow, scikit-learn, lightgbm; dev: pytest, ruff. `ml/` skeleton (ingest, features, models, evaluation, simulation) and `tests/test_smoke.py` exist; pytest and ruff pass. Note: always `Set-Location D:\repos\EffOne` in the same command before running `uv`; the shell can reset to `D:\repos`. Next step: phase 1 (incremental ingestion of 2018 to present into Parquet).
- 2026-10-07: **Phase 1 ingestion code written and started** (not yet verified complete). `python -m uv run python -m ml.ingest [--years ...] [--sessions Race Qualifying] [--limit N]` ingests all finished sessions (practice, qualifying, sprint, race) with tables `results`, `laps`, `weather`, `track_status` into `data/raw/<table>/year=YYYY/round=RR/<session>.parquet`. Sessions are marked done in `data/raw/_done/`; reruns skip them and retry failures (also waits 6h after a session ends). All data and the fastf1 cache are under `D:\repos\EffOne\data` (C: has about 8 GB free, so keep it there; `EFFONE_DATA_DIR` overrides). Full backfill was launched in the background, logging to `data/ingest.log`; about 4 MB of cache per session. Still to do for phase 1: check row counts against known race and driver counts, add tests, add `circuits`, `drivers`, `teams` tables and the 25% circuit-history rule. The uv package cache is still on C:; set `UV_CACHE_DIR` to a D: path if it grows.

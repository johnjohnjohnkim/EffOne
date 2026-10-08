# Building the scoreboard before the players: evaluation harness and baselines

Date: 2026-10-08 | Scope: Milestone 2 / Phase 2 (plus the data-quality audit that followed) | Outcome: done

## What

There is now a way to score any F1 race predictor honestly, and two simple predictors to compare against. `python -m uv run python -m ml.evaluation` runs walk-forward validation (explained below) and writes `data/reports/leaderboard.csv`. Key files, all in `ml/evaluation/`: `splits.py` (which seasons train and test), `harness.py` (the loop), `metrics.py` (the scores), `baselines.py` (the two baselines and the interface models must follow), `data.py` (loading results). `tests/test_evaluation.py` has 8 tests.

Afterwards I added `ml/ingest/audit.py`, a read-only report of which ingested sessions have missing or odd data (`python -m ml.ingest.audit`).

Result on the 99 races ingested so far (test seasons 2019 to 2023):

| Baseline | Spearman | Winner right | Podium overlap | Top-10 overlap | Avg position error |
|---|---|---|---|---|---|
| Grid = finish | 0.602 | 51.5% | 0.690 | 0.752 | 3.59 |
| Previous race = finish | 0.444 | 39.4% | 0.545 | 0.682 | 4.40 |

## Why

With about 400 races of data, it is easy to build a model that looks great because it memorised the past. Before writing any model, we need a scoreboard that cannot be fooled, and a floor that any model must clearly beat. Otherwise a score like "0.65" means nothing.

## How

1. **Load** every ingested race result into one table (`data.py`). Columns are split into *pre-race* (grid, driver, team) and *outcome* (finish position, status, points).
2. **Split by season** (`splits.py`): for each test season, train only on earlier seasons. Season 2025 is "locked" and skipped unless you pass `--allow-locked`.
3. **Predict one race at a time** (`harness.py`). The model receives only the pre-race columns for that race, plus `history`: the full results of every race strictly before it. That includes earlier rounds of the same season, because that information exists before the race.
4. **Score** each race (`metrics.py`): rank correlation between predicted and actual order, how many of the actual top 1, 3 and 10 were predicted, and average position error. Then average per season and overall.
5. **Baselines** (`baselines.py`) follow a small interface: `fit(train)` and `predict_race(history, race)`, returning a score per driver where lower means a better finish. Milestone 4 models plug into the same interface.

The key idea is step 3: the harness *removes* what a model must not see, so cheating is impossible unless someone edits the harness.

## Concepts to know

**Data leakage and walk-forward validation.** Leakage is when information from the future (or from the answer) sneaks into what a model learns from. The usual cause is random train/test splits on time-ordered data: the model trains on races from after the ones it is tested on. Walk-forward validation always trains on the past and tests on the next period, then moves forward, which is how the model will actually be used. *Search for:* time series cross-validation, data leakage.

**Baselines.** A baseline is a deliberately simple predictor, such as "everyone finishes where they started". It sets the bar. A fancy model that scores 0.58 where the baseline scores 0.60 has learned nothing useful, and without the baseline you would not know. *Search for:* naive baseline, benchmark model.

**Rank metrics.** Finishing order is a ranking, so plain accuracy is the wrong tool. Spearman correlation measures how similar two orderings are (1 is identical, -1 is reversed). Top-N overlap asks the practical question: of the real podium, how many did we name? *Search for:* Spearman rank correlation, precision at k.

## Hard parts

**1. A test that could never fail**
- *Problem:* my first spy-model test contained `assert train["Year"].max() < 2021 or True`.
- *Cause:* I wrote a placeholder assertion and moved on. `or True` makes any assertion pass.
- *Fix:* the model now records the newest year it was fitted on, and `predict_race` asserts that year is before the race being predicted.
- *Lesson:* after writing a test, ask how it could fail. A test that cannot fail gives false confidence, which is worse than no test.

**2. `0.9999999999999999 == 1.0`**
- *Symptom:* the perfect-prediction test failed with a Spearman of 0.9999999999999999.
- *Cause:* floating-point rounding. Computers store most decimals approximately.
- *Fix:* compare with `pytest.approx(1.0)`.
- *Lesson:* never test floats with `==`.

**3. My first audit flagged half the data**
- *Symptom:* the audit flagged 149 of 294 sessions.
- *Cause:* I applied race rules to every session type. In qualifying and practice, 20 to 40% of laps legitimately have no lap time (out-laps, in-laps, aborted runs), and practice results have no finishing position.
- *Fix:* checks now depend on session type. Flags dropped to 28, all about lap data in races and sprints.
- *Lesson:* a validation rule is only as good as your understanding of normal data. If a check fires on half of everything, suspect the check before the data.

**4. New lint rules**
- `ruff` passed in phase 0 but flagged 9 issues in the ingestion code later. The tool's default rules are broader than they were, or there was more code to check. The only substantive finding was `B023`: lambdas inside a loop capture the loop variable by reference. It was harmless here because they are called immediately, but `functools.partial` removes the risk.

## Where a beginner goes wrong

- Using random k-fold on race data, which lets the model train on the future.
- Computing a feature such as "average finish" over the whole dataset, then splitting. The test rows have already influenced it.
- Picking the model with the best test score and reporting that score as if it were honest. That is why 2025 is locked: the number is only trustworthy if looked at once.
- Comparing a model to nothing, or to another fancy model, instead of a dumb baseline.
- Treating a flagged row as a bad row and deleting it. The 2018 Italian GP has no lap data, but its finishing order is fine. Flag, then decide per use.

## Decisions

| Decision | Options | Why | Cost |
|---|---|---|---|
| Hide outcomes from models in the harness | trust each model to behave | Leakage becomes impossible by construction, and a test checks it | Every model must fit the `history` + `race` interface |
| History includes earlier rounds of the test season | only prior seasons | That data exists before the race, and a real forecaster would use it | Slightly harder to reason about than a clean season boundary |
| Ties broken by row order (`rank(method="first")`) | average ties | Needs a unique predicted order for top-N | Arbitrary choice among tied drivers; unlikely to matter, since grids have no ties |
| Pit-lane or missing grid slots go to the back | drop them | The 41 pit-lane starts in the data really do start last | Approximation when several drivers start from the pit lane |
| Lock 2025 as the final test set | lock the latest ingested year | 2025 is the latest complete season | It is not ingested yet, so the lock guards nothing today |
| Audit flags, never deletes | delete bad sessions | Re-downloading is slow, and results can be fine when laps are not | Later code must remember to filter on `results_ok` / `laps_ok` |
| Audit thresholds chosen by me (5% untimed laps in races, 18 to 24 result rows) | statistical thresholds | Simple and explainable | They are guesses; the 5% line decides whether 27 races are flagged |

## Dead ends

- A patch script to make the audit session-aware crashed on a string that didn't match, after `ruff format` had reflowed the file. Nothing was written, so I rewrote the file cleanly instead of patching it again.

## Verified vs. not

Verified:
- 8 tests pass and `ruff` is clean.
- The leaderboard CSV is byte-identical across two runs.
- The harness test uses a spy model to confirm models never see outcome columns, later races, or training data from their own test season.
- Real-data sanity check: pole-sitter wins about half the time, which matches F1 history as I remember it.

Not verified:
- The audit has no tests yet.
- The new-driver fallback in `PreviousRaceBaseline` is not covered by a test.
- 2024 and 2025 are not ingested, so the locked-season guard has only run against synthetic data.
- The numbers are from partial data and will change when the backfill finishes.
- Log loss and calibration are not computed (they need probabilities).
- I haven't looked into why 27 races have many untimed laps. Red flags are my guess, not a finding.

## What to look for in review

- `ml/evaluation/harness.py`, the `earlier` mask: it must include earlier rounds of the same year but never the current one. An off-by-one here leaks the answer.
- `ml/evaluation/baselines.py`, `PreviousRaceBaseline.predict_race`: the `prev.fillna(prev.max() + fallback ...)` line ranks drivers with no history behind everyone else. Check it does what you expect and that it handles an empty history.
- `ml/ingest/audit.py`, `audit_session`: the thresholds and which flags make a session not `results_ok` or `laps_ok`.

## Open questions / next

- Should the 2021 Belgian GP (60 lap records, 67% untimed; I believe it barely ran) be excluded from training?
- Rerun the leaderboard and audit when the backfill finishes, and note how much the baselines move.
- Milestone 3 is the features; every builder needs a test proving it cannot see the future.

## Glossary

- **Walk-forward validation:** train on the past, test on the next period, repeat.
- **Data leakage:** future or answer information reaching the training process.
- **Baseline:** a simple predictor that any real model must beat.
- **Spearman correlation:** measure of how similar two rankings are, from -1 to 1.
- **Top-N overlap:** share of the real top N that were predicted in the top N.
- **Locked test set:** data held back and evaluated once, so the final score is honest.
- **Pit-lane start:** a driver starting from the pit lane; stored as grid position 0.
- **Spy model:** a test double that checks what it is given.
- **False positive:** a check that flags normal data.

## Try it yourself

Introduce a deliberate leak and see what catches it.

1. In `ml/evaluation/harness.py`, change `(results["Round"] < rnd)` to `(results["Round"] <= rnd)` in the `earlier` mask.
2. Before running anything, predict: which test fails, and does the leaderboard change?
3. Run `python -m uv run pytest`, then `python -m uv run python -m ml.evaluation`. Then undo the edit.

.

.

**Answer:** `test_harness_hides_outcomes_and_future_from_models` fails with "history contains the current or a later race", because the spy model checks for exactly that. The leaderboard would change too: "previous race = finish" would read the current race's own result and score close to perfect, which is the signature of a leak. A suspiciously good score is a warning sign, not good news.

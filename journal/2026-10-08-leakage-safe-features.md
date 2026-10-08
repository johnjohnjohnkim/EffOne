# Leakage-safe features: making the wrong thing impossible, and the leak nobody was testing for

Date: 2026-10-08 | Scope: Milestone 3 (Leakage-safe features) | Outcome: done

## What

There is now a feature table, `data/features/race_features.parquet`, with one row per driver per race (3,810 rows, 189 races, 2018 to 2026), written by `python -m uv run python -m ml.features`. It has 24 features for predicting finishing order and DNF, 21 for predicting qualifying or the grid, four targets, and a count of missing values per column in `feature_missing.csv`.

Key files in `ml/features/`: `tables.py` (turns raw files into one tidy history table), `builders.py` (the five feature builders), `build.py` (`features_for_entry`, `features_for_race`, `feature_columns`, `training_frame`), `lineage.py` (team renames). 226 tests pass.

Along the way I found and fixed a bug in the already-pushed Milestone 2 evaluation code: a model that gave every driver the same score earned a perfect result. That is the most important part of this entry.

## Why

Wrong features, ones that peek at the future, make every later score meaningless, and a model with leaked features looks brilliant until it meets a real race. This milestone exists to make that kind of mistake hard to make, not just to avoid it once.

## How

1. **One history table** (`tables.py::build_history`): per driver per race, the race result, qualifying, a rough pace figure from clean laps, and the weather. Lap-based figures are blanked where the audit says the laps are unusable, and a `race_ok` flag marks races that carry no information (the 2021 Belgian GP).
2. **Builders see only what they may.** A builder is called as `compute(history, race)`. `history` is every race strictly before the target. `race` is the entry list with only the columns its *stage* allows: `pre_weekend` gets who is entered and where; `scenario` adds weather; `post_quali` adds the grid. The target race's finish, points, qualifying and pace are simply not in the frame, so they cannot be read by accident.
3. **Two entry points.** `features_for_entry(history, entries, key)` needs no results at all, which is what predicting a future race looks like. `features_for_race` calls it and then attaches the real targets.
4. **Stage-aware column selection.** `feature_columns(target)` returns only features that are known when that target is decided. The grid is left out for qualifying targets; weather is opt-in.
5. **Tests that attack it** (below).

## Concepts to know

**Make the mistake impossible, not unlikely.** I could have written feature code carefully and promised not to peek. Instead the function signature hands each builder only past data and the allowed columns, so a builder that tries `race["Position"]` fails with a `KeyError`. A test then proves it. This is the same idea as a type system: move the rule from "someone must remember" to "the code won't run otherwise". *Search for:* make illegal states unrepresentable, defensive design.

**Properties, not just examples.** "Prefix invariance" is a property test: take a race, delete everything after it, blank its outcomes, and rebuild its features. They must be identical. It says nothing about particular values, yet it would break if any feature used the future. The same style of test appends a *different* future and checks nothing earlier changes. *Search for:* property-based testing, metamorphic testing.

**Information hides in the layout of the data.** Leakage is not only wrong numbers. Row order, index order, tie-breaks and `groupby().first()` can all carry the answer if the data was arranged using the answer. Anything that depends on "which row comes first" is reading the arrangement. *Search for:* target leakage, implicit ordering.

## Hard parts

**1. The leak nobody was testing for**
- *Problem:* round 2 of the review found that every race in my table, and in the Milestone 2 evaluation data, was stored in exact finishing order. The metric broke ties by row order. So a model that scored every driver identically got Spearman 1.0 and picked the winner 100% of the time. I confirmed it myself before fixing.
- *Cause:* I had sorted by `Position` for readability. Then the metric's `rank(method="first")` treated row order as the tie-breaker. Neither piece looked wrong alone.
- *Why the tests missed it:* they compared feature *values*. Row order is not a value.
- *Fix, in two steps:* first sort rows by `DriverId` (a pre-race key). Then fix the metric itself. My first attempt used a seeded random shuffle for ties, which is neutral on average but noisy: round 3 measured a standard deviation of about 0.05 in Spearman depending on how a model happened to order its output. The real fix scores ties *by expectation*: tied drivers are equally likely to take any of the positions the tie covers, so every metric is deterministic and independent of order. A constant score now gets exactly chance (Spearman 0.0, winner 1 over the field size).
- *Impact:* the Milestone 2 baselines barely moved (previous race 0.449 to 0.448), because ties are rare for them. A model that produced ties would have been rewarded.
- *Lesson:* "deterministic" is not "neutral". Neutral means the answer doesn't change when you reorder the input. Test that directly, with adversarially ordered data.

**2. Leakage tests that can pass while a leak exists**
- The first scramble test randomised outcomes but left driver, team and circuit of later races alone. A reviewer pointed out that a builder reacting to *who* raced later would not be caught. I widened the scramble, and added prefix invariance, an append-a-different-future test, and "peeker" builders that try to read withheld columns.
- A negative control shows the test is able to fail: break the history slice three ways and every history-reading builder must be caught. Once I added input validation, the broken variants started failing with a `ValueError` instead of a changed value, so the control now counts a loud rejection as caught too.

**3. A train/serve gap in types, found by a leakage test**
- With no history, a column came back as `object` instead of `float`, so the prediction-time path would not match the training table. The prefix test found it. The fix was one rule in one place: every feature is cast to float64 in `features_for_entry`.

**4. My own tooling mistakes**
- My "break the code, check the tests fail" script restored files with `git checkout`, which silently does nothing for files git doesn't track yet. One file stayed broken, and an odd failure in the "restored" run gave it away. I fixed it and now compare restored files byte for byte with `cmp`.
- In one heredoc, `\\n` was altered by the shell, so a text replacement matched nothing and raised no error. Always assert that the old text exists before replacing it.

## Where a beginner goes wrong

- Computing features over the whole dataset and then splitting, or with `groupby(...).shift()` written once and trusted without a test.
- Training a qualifying model on the starting grid, which is simply the qualifying result copied (Spearman 0.96).
- Sorting data by the answer "for tidiness", then using order-dependent operations.
- Using the real weather in training and a forecast at prediction time, so the model scores better on paper than it will in practice.
- Believing "no feature correlates too strongly with the target" is proof. The check I added (nothing above 0.85) catches blatant copies only.
- Trusting a flag derived from the target (`race_ok`) as a feature or to filter the test set.

## Decisions

| Decision | Options | Why | Cost |
|---|---|---|---|
| Stage-limited builder inputs | pass the full row, rely on care | leaks fail loudly | each new column needs a stage |
| `features_for_entry` for future races | build only from the table | prediction needs it | extra validation code |
| Weather is opt-in; never for qualifying targets | always include | train/serve skew; race-day weather isn't known at qualifying | lose a (weak) signal by default |
| `race_ok` is a training filter only | use as a feature | it's derived from the race's own results | must be kept out by convention, so `training_frame` enforces it |
| Treat the 2021 Belgian GP as not a race | count it | finish order equals the grid, half points | also drops its real qualifying; my default, needs your confirmation |
| Points as a share of the field's, not a running total | cumulative points | totals just grow with the round number | undefined at round 1 |
| Circuit history uses the 25% rule on the race's own entry list | latest race's grid | entry list is known before the weekend | includes late substitutes for past races |
| Team lineage table | treat renames as new teams | keeps form across renames | a judgement from public knowledge, not data |
| Ties scored by expectation | seeded shuffle | order-independent and exactly neutral | slightly more complex metric |
| 3 reviewer rounds, then commit | until perfect | reviewers can always find more | deferred items are written down |

## Dead ends

- The seeded-shuffle tie-break: neutral on average, noisy in practice (see Hard part 1).
- Allowing weather for every target by default: a reviewer showed it was not available when qualifying is decided.

## Verified vs. not

Verified:
- 226 tests pass (181 without the real-data ones); lint clean.
- Scrambling everything from the target onward leaves every builder's features unchanged, on synthetic and real races.
- A table cut at the target with outcomes blanked rebuilds identical features (random tables, 3 real races).
- I broke the code ten ways (by hand): letting history include the target (37 failures), removing the `race_ok` filter, letting qualifying use the grid, removing the lap-quality gate, and three row-order/tie changes. Each was caught.
- A feature recomputed straight from the raw result files matches, including the race right after the Belgian GP.
- Constant score: Spearman 0.0 exactly. Leaderboard on 144 races: grid = finish 0.623, previous race = finish 0.448.

Not verified:
- No model has used these features yet, so I don't know if they predict anything.
- The 0.85 correlation guard is a smoke test, not proof.
- The team lineage table and the 2026 calendar oddities (`kuala_lumpur`, `madrid`) are unverified.
- Weather skew is documented, not eliminated.
- The Milestone 2 harness still reads raw results, not this table.

## What to look for in review

- `ml/evaluation/metrics.py::race_metrics`: the `better`/`tied` arithmetic. Check the expected top-N overlap and the expected absolute error against a hand example in `tests/test_evaluation.py`.
- `ml/features/build.py::usable_history` and `_check_entries`: the validation that stops silent fallbacks. Check that no legitimate prediction-time call is rejected.
- `ml/features/builders.py::CircuitHistory`: it combines the 25% rule, shrinkage and the entry list; it's the most intricate builder.

## Open questions / next

- Confirm or reverse the 2021 Belgian GP exclusion.
- Milestone 4 must: wrap `features_for_entry` inside `predict_race`, switch the harness to this history table, impute NaNs, report with and without weather, and never use `race_ok` as a feature.
- Per-race-constant features (weather, round, circuit rates) can act as race IDs for tree models.

## Glossary

- **Prefix invariance:** features for a race are unchanged if everything after it is deleted and its outcomes blanked.
- **Stage:** when information becomes known (before the weekend, scenario inputs, after qualifying).
- **Train/serve skew:** the model sees different inputs while training than it will when used.
- **Peeker:** a deliberately cheating test builder that tries to read withheld columns.
- **Negative control:** a test that deliberately uses broken code to prove the check can fail.
- **Expected-value tie handling:** scoring tied predictions as the average over every way of breaking the tie.
- **Lineage:** the chain of renamed ids that belong to the same team entry.

## Try it yourself

Reintroduce the row-order bug and see which tests notice.

1. In `ml/features/tables.py`, near the end of `build_history`, change `["Year", "Round", "DriverId"]` in the final `sort_values` back to `["Year", "Round", "Position"]`.
2. Before running anything, predict: do the feature values change? How many tests fail, and which?
3. Run `python -m uv run pytest`, then undo the change.

.

.

**Answer:** exactly one test fails (`test_rows_are_ordered_by_driver_id_never_by_finishing_position`) and the other 225 pass. The feature *values* don't change at all, only the row order, which is why the many value-based leakage tests stay green. That one test is the only thing standing between the table and the leak, which is why it exists. I ran this experiment while writing the entry and restored the file afterwards.

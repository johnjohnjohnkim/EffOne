# The model zoo, and how one extra baseline changed the headline

Date: 2026-10-08 | Scope: Milestone 4 (Qualifying and race models) | Outcome: done, with a negative result for qualifying

## What

Eight kinds of model now run through the same walk-forward harness, on two targets: the qualifying order and the race finishing order. `python -m uv run python -m ml.evaluation --models all` (about 3 minutes) prints and saves three leaderboards: `quali`, `race` (given the real or a user-typed grid) and `race_predicted_grid` (a forecast before qualifying). The models are ridge, random forest, LightGBM, XGBoost, LightGBM LambdaRank, XGBoost pairwise ranking, Elo and Plackett-Luce, plus a weather variant and "no grid" variants.

The result is modest and honest:
- **Qualifying:** the best models (ridge, Elo) score Spearman 0.751; a rule that just averages each driver's last 5 qualifying positions scores 0.740. The gain, +0.011, is inconclusive (interval [-0.001, 0.024]).
- **Race with the real grid:** ridge (+0.040), random forest (+0.037) and LambdaRank (+0.031, marginal) clearly beat "grid = finish" (0.623). Ridge and random forest win in all 7 seasons.
- **Race before qualifying:** nothing beats "predicted grid = result".

Key files: `ml/models/` (`base.py`, `factories.py`, `elo.py`, `plackett_luce.py`, `chain.py`, `zoo.py`), and in `ml/evaluation/` the new `compare.py`, a harness that takes a `target`, and a CLI that prints the three boards. 331 tests pass.

## Why

The plan said to try many models and judge them fairly. The real work was in "fairly": most of this milestone was building a comparison that could not flatter the models, and then being surprised by what survived.

## How

1. **One interface.** Every model has `fit(train)` and `predict_race(history, race)` and returns a score per driver (lower is better). Feature models wrap an estimator; they train on rows of the leakage-safe feature table for the training races only, and predict through `features_for_entry`. Elo and Plackett-Luce are written by hand.
2. **Two targets.** The harness gives each target only what is known when it is decided: qualifying gets no grid and no race-day weather; the race gets the grid.
3. **A chain for forecasting before qualifying.** `PredictedGridRace` asks a qualifying model for scores, turns them into a grid, and feeds that to a race model. `DropGrid` hides the grid altogether.
4. **Fair comparison.** Each model is compared with the best baseline on the same races, with an interval from a bootstrap that resamples whole seasons, and a count of how many of the 7 seasons it wins.
5. **A stop rule that can fire.** The CLI exits with code 3 if no model *clearly* beats the best baseline on the race or qualifying board.

## Concepts to know

**Compare against the best naive predictor, not the easiest.** My first qualifying baselines were "repeat the last result". Every model crushed them (+0.09, all 7 seasons) and I wrote "every model clearly beats both baselines". A reviewer then computed "average the last five results": no model, no features, 0.739. The true gain was about +0.01. A weak baseline manufactures a good headline, so pick the strongest simple rule you can think of. *Search for:* strong baselines, "beat the benchmark".

**Races in a season are not independent.** Comparing 144 races as if each were a fresh coin flip makes intervals too narrow, because races in a season share rules and car pecking order. The fix is a *cluster bootstrap*: resample whole seasons, then races within them. With only 7 seasons the interval is wide, which is the honest answer. *Search for:* cluster bootstrap, clustered standard errors.

**Researcher degrees of freedom.** Every choice made after looking at the results (adding a variant, changing a setting) quietly uses the test data. They are not forbidden, but they must be disclosed, and the numbers treated as optimistic. *Search for:* forking paths, p-hacking, multiple comparisons.

## Hard parts

**1. The headline that one baseline reversed**
- *Problem:* round 1 of the review reproduced my numbers but pointed out the weak baseline. After adding it, my own stop rule fired: no qualifying model clearly beats the form average.
- *What I did:* I did not weaken the rule or hunt for a model that wins. I reported it, with the caveat that +0.011 with an interval touching zero is "inconclusive", not "the models add nothing". I ran the experiment in "Try it yourself" to confirm that the baseline alone flips the verdict.
- *Lesson:* a disappointing result stated clearly is worth more than a flattering one that falls apart.

**2. A default setting that handicapped one model**
- *Problem:* LambdaRank came last. A reviewer suspected LightGBM's default exponential label gain, which cares almost only about the top few places.
- *Test:* a linear gain lifts it from 0.723 to 0.742 (qualifying) and 0.632 to 0.654 (race). The suspicion was right.
- *The catch:* I found this by looking at the test seasons, so the improved number is optimistic. I adopted it and disclosed it as post-hoc. *Lesson:* a fix motivated by test results is still a fix, but it must be labelled.

**3. Models trained on real grids fail on predicted grids**
- *Problem:* chained models got worse, XGBoost clearly (-0.022 [-0.053, -0.001]).
- *Cause:* train/serve mismatch. During training the grid is the real, highly informative one; at prediction time it is a smoothed guess. Models that lean on it are miscalibrated.
- *What I tried:* "no grid" models. They only tie the baseline. The honest conclusion is that a forecast before qualifying is capped near the qualifying forecast itself (Spearman about 0.6).
- *Lesson:* a feature that is cleaner in training than in use is a trap.

**4. Ties on the *result* side**
- I had fixed ties in predictions. A reviewer showed a perfect predictor got 0% winner credit when the result had a tied pole, because ties in the actual positions were still being broken by row order. Both sides are now handled by expectation (a tied pole gives 0.5).

**5. A filtered entry list**
- The harness handed models only drivers with a known result, so the field itself was chosen by the outcome. Now models see the full entry list and only drivers with a known outcome are scored. It barely moved the numbers (the largest baseline change was 0.0003), but "never select by outcome" has no exceptions.

**6. Tooling**
- Long shell heredocs failed to parse several times; I wrote text to a file with the file tool and spliced it in. A delete command was blocked by a safety check; I didn't work around it and left stale files, noted in the report.

## Where a beginner goes wrong

- Reporting a gain over a baseline that is trivially weak.
- Reading the order of the top models as meaningful when the gaps are smaller than the noise.
- Tuning or adding variants after seeing test results without saying so.
- Assuming a more complex model (boosting, rankers) will beat a simple one with 3,000 rows.
- Calling something "untuned" without a log that proves it.
- Treating "average of ties" and "break ties by row order" as the same.

## Decisions

| Decision | Options | Why | Cost |
|---|---|---|---|
| Add an average-of-last-5 baseline | keep two baselines | the honest reference | the headline shrinks |
| Cluster bootstrap by season | race-level bootstrap | races in a season are alike | wide, rough interval (7 clusters) |
| Exit code 3 when no model clearly beats the best baseline (race, quali only) | positive mean; every board | a positive mean is luck with 12 models; the predicted-grid board's baseline is itself a model | conservative: flags an inconclusive result |
| Keep the linear LambdaRank gain | revert | the default was clearly handicapping it | its race result is optimistic |
| Weather opt-in and kept off the pre-qualifying board | always on | history holds realised weather | lose a (nil) signal |
| Fit once per season on earlier seasons | refit after every race | matches the plan (no full retrain per race) | models don't learn within the season |
| Hand-set hyperparameters, no search | tune | no validation scheme yet | provenance unlogged, so numbers are mildly optimistic |
| Plackett-Luce implemented by hand | a library | learn whole orders; small | more code to test |
| Don't delete stale reports after a blocked command | work around | respect the guard | clutter in `data/reports/` |

## Dead ends

- "Beats the best baseline" meaning a positive mean (could never fail).
- The race-level bootstrap as the headline interval.
- "No grid" variants as the fix for the predicted-grid result (they tie, they don't win).

## Verified vs. not

Verified:
- 331 tests pass (286 without the real-data ones); lint clean.
- Reviewer re-derived every headline number from the CSVs and recomputed features for 189 races from strictly earlier history: 0 mismatches.
- 2025 never appears in any per-race file.
- Ten deliberate breakages of the model and metric paths were each caught.
- The "try it yourself" result below was observed.

Not verified:
- Whether any hyperparameter was influenced by the test seasons.
- Whether the qualifying call would pass under a different interval method (a one-stage bootstrap over season means would just pass, a t-interval just fail).
- Anything about calibration, probabilities, DNF, CatBoost, Bayesian or neural models.
- Results on 2025, which stays locked.

## What to look for in review

- `ml/evaluation/compare.py::paired_bootstrap`: the two-stage resampling and the pooled mean.
- `ml/evaluation/metrics.py::race_metrics`: the expected overlap and expected error when both sides have ties.
- `ml/models/base.py::FeatureModel.fit` and `training_rows`: it trains on precomputed table rows for the training races, which is only safe because features are prefix-invariant.

## Open questions / next

- **Your call:** continue to Milestone 5 (turn scores into probabilities), or first do a 4b aimed at qualifying (better features, a proper validation scheme, out-of-fold predicted grids to train race models on).
- Candidates left out: CatBoost, a Bayesian hierarchical model, neural nets, stacking.

## Glossary

- **Cluster bootstrap:** resampling groups (seasons) first, so correlated observations stay together.
- **Clearly beats:** the 95% interval for the gain lies entirely above zero.
- **Post-hoc change:** a modelling choice made after seeing the test results.
- **Form baseline:** predict each driver by the average of their last few results.
- **Chained model:** one model's output is the next model's input.
- **Label gain:** how much each relevance level counts in a ranking loss.
- **Seasons better:** the number of test seasons in which a model beats the best baseline.

## Try it yourself

Remove the strongest baseline and see what the leaderboard says.

1. In `ml/evaluation/baselines.py`, in `BASELINES_BY_TARGET["quali"]`, delete `MeanLast5QualiBaseline` (keep the other two).
2. Before running, predict: what is the best baseline now, how many models "clearly beat" it, and what is the exit code?
3. Run `python -m uv run python -m ml.evaluation --models all --target quali --n-boot 300`, then undo the change.

.

.

**Answer:** the best baseline becomes `previous_quali_equals_quali` (0.660), all eight models clearly beat it (about +0.082 to +0.091, in 7 of 7 seasons) and the exit code is 0. Same models, same data, opposite verdict. I ran this while writing the entry, restored the file, and regenerated the real leaderboard (exit code 3).

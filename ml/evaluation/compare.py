"""Compare models with the best baseline, with an uncertainty estimate.

With only a hundred or so test races, a model that is 0.01 better on average may just be lucky. For
each model we resample the test data (paired with the best baseline's result on the same race) and
report a 95% interval for the difference in mean Spearman.

Races in the same season are alike (same rules, same car pecking order), so treating every race as
independent makes the interval too narrow. The reported interval is therefore a SEASON-CLUSTERED
bootstrap: resample whole seasons, then resample races inside each chosen season. The plain
race-level interval is kept alongside for reference. With only seven seasons the clustered interval
is honest but wide.
"""

import numpy as np
import pandas as pd

METRICS = ["spearman", "mae_position", "winner_acc", "podium_overlap", "top10_overlap"]


def _race_scores(per_race: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per race, one column per model; refuses duplicates and uneven coverage."""
    if per_race.duplicated(["model", "Year", "Round"]).any():
        raise ValueError("a model has more than one score for the same race")
    scores = per_race.pivot(index=["Year", "Round"], columns="model", values=metric)
    if scores.isna().any().any():
        raise ValueError("models were not scored on the same races")
    return scores


def paired_bootstrap(
    per_race: pd.DataFrame,
    model: str,
    baseline: str,
    metric: str = "spearman",
    n_boot: int = 2000,
    seed: int = 0,
    clustered: bool = True,
) -> tuple[float, float, float]:
    """(mean difference, 2.5th percentile, 97.5th percentile) of model minus baseline, per race.

    `clustered=True` resamples seasons and then races within each season; False resamples races.
    """
    scores = _race_scores(per_race, metric)
    diff = scores[model] - scores[baseline]
    rng = np.random.default_rng(seed)
    if clustered:
        seasons = [group.to_numpy() for _, group in diff.groupby(level="Year")]
        means = np.empty(n_boot)
        for b in range(n_boot):
            chosen = rng.integers(0, len(seasons), size=len(seasons))
            draws = [rng.choice(seasons[i], size=len(seasons[i]), replace=True) for i in chosen]
            means[b] = np.concatenate(draws).mean()
    else:
        values = diff.to_numpy()
        means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def leaderboard(per_race: pd.DataFrame, baselines: list[str], n_boot: int = 2000) -> pd.DataFrame:
    """One row per model: mean metrics, and the gain over the best baseline with its interval.

    `clearly_beats` means the season-clustered 95% interval for the gain lies above zero.
    `seasons_better` counts the test seasons in which the model beats the best baseline.
    """
    means = per_race.groupby("model")[METRICS].mean()
    means["races"] = per_race.groupby("model").size()
    best = means.loc[baselines, "spearman"].idxmax()
    by_season = per_race.groupby(["Year", "model"])["spearman"].mean().unstack("model")
    rows = []
    for model, row in means.sort_values("spearman", ascending=False).iterrows():
        entry = {"model": model, "is_baseline": model in baselines, **row.to_dict()}
        if model == best:
            entry.update(vs_best_baseline=0.0, ci_low=np.nan, ci_high=np.nan)
            entry.update(iid_ci_low=np.nan, iid_ci_high=np.nan)
        else:
            delta, low, high = paired_bootstrap(per_race, model, best, n_boot=n_boot)
            _, iid_low, iid_high = paired_bootstrap(
                per_race, model, best, n_boot=n_boot, clustered=False
            )
            entry.update(vs_best_baseline=delta, ci_low=low, ci_high=high)
            entry.update(iid_ci_low=iid_low, iid_ci_high=iid_high)
        entry["seasons_better"] = (
            int((by_season[model] > by_season[best]).sum()) if model != best else 0
        )
        entry["seasons"] = len(by_season)
        entry["beats_best_baseline"] = bool(model != best and entry["vs_best_baseline"] > 0)
        entry["clearly_beats"] = bool(model != best and entry["ci_low"] > 0)
        rows.append(entry)
    out = pd.DataFrame(rows)
    out.attrs["best_baseline"] = best
    return out

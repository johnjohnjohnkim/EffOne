"""Per-race ranking metrics. Predictions are scores where lower means a better finish.

Tied scores are handled by EXPECTATION, not by a tie-break: drivers with equal scores are treated as
equally likely to take any of the positions the tie covers. Every metric is therefore deterministic
and independent of the order of the rows and of the order a model returns its scores in. A model
that cannot tell drivers apart (a constant score) gets exactly chance-level credit.
"""

import numpy as np
import pandas as pd


def _tie_structure(scores: pd.Series) -> tuple[pd.Series, pd.Series]:
    """For each driver: how many drivers score strictly better, and how many share their score."""
    values = scores.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("scores must not contain NaN: a model has to score every driver")
    better = pd.Series((values[None, :] < values[:, None]).sum(axis=1), index=scores.index)
    tied = pd.Series((values[None, :] == values[:, None]).sum(axis=1), index=scores.index)
    return better, tied


def race_metrics(scores: pd.Series, actual_position: pd.Series) -> dict[str, float]:
    """Compare one race's predicted scores with the actual finishing positions.

    - spearman: rank correlation, with tied scores given their average rank (0.0 if all tied).
    - mae_position: expected absolute difference between predicted and actual position.
    - winner_acc, podium_overlap, top10_overlap: expected share of the real top 1 / 3 / 10 that the
      predicted top 1 / 3 / 10 contains (a tie that straddles the cut counts fractionally).
    """
    actual = actual_position.rank(method="first").astype(int)
    scores = scores.reindex(actual.index)
    better, tied = _tie_structure(scores)

    average_rank = better + (tied + 1) / 2
    if average_rank.std() == 0 or actual.std() == 0:
        spearman = 0.0  # every driver tied: the ranking says nothing
    else:
        spearman = float(average_rank.corr(actual))

    # A driver in a tie covering positions better+1 .. better+tied takes each with equal chance.
    expected_error = pd.Series(
        [
            np.abs(np.arange(b + 1, b + t + 1) - a).mean()
            for b, t, a in zip(better, tied, actual, strict=True)
        ],
        index=actual.index,
    )

    def top_overlap(n: int) -> float:
        in_top = ((n - better) / tied).clip(0, 1)  # chance of being inside the predicted top n
        return float(
            in_top[actual <= n].sum() / min(n, len(actual))
        )  # a short field has fewer slots

    return {
        "spearman": spearman,
        "mae_position": float(expected_error.mean()),
        "winner_acc": top_overlap(1),
        "podium_overlap": top_overlap(3),
        "top10_overlap": top_overlap(10),
    }


def summarize(per_race: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric per (model, test year), plus an 'all' row per model."""
    metrics = [c for c in per_race.columns if c not in ("model", "Year", "Round")]
    by_year = per_race.groupby(["model", "Year"])[metrics].mean().reset_index()
    overall = per_race.groupby("model")[metrics].mean().reset_index()
    overall.insert(1, "Year", "all")
    overall["races"] = per_race.groupby("model").size().to_numpy()
    return pd.concat([by_year.astype({"Year": object}), overall], ignore_index=True)

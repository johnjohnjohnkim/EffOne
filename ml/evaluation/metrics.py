"""Per-race ranking metrics. Predictions are scores where lower means a better finish.

Ties are handled by EXPECTATION, on both sides, never by a tie-break: drivers with equal predicted
scores are equally likely to take any position the tie covers, and so are drivers with equal ACTUAL
positions (a shared qualifying time, say). Every metric is therefore deterministic and independent
of the order of the rows and of the order a model returns its scores in. A model that cannot tell
drivers apart (a constant score) gets exactly chance-level credit, and a perfect model is not
penalised for a tie in the result.
"""

import numpy as np
import pandas as pd


def _tie_structure(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """For each driver: how many drivers are strictly better, and how many share their value."""
    array = values.to_numpy(dtype=float)
    if np.isnan(array).any():
        raise ValueError("scores and positions must not contain NaN: every driver needs a value")
    better = pd.Series((array[None, :] < array[:, None]).sum(axis=1), index=values.index)
    tied = pd.Series((array[None, :] == array[:, None]).sum(axis=1), index=values.index)
    return better, tied


def race_metrics(scores: pd.Series, actual_position: pd.Series) -> dict[str, float]:
    """Compare one race's predicted scores with the actual finishing positions.

    - spearman: rank correlation, with ties given their average rank (0.0 if either side is all tied).
    - mae_position: expected absolute difference between predicted and actual position.
    - winner_acc, podium_overlap, top10_overlap: expected share of the real top 1 / 3 / 10 that the
      predicted top 1 / 3 / 10 contains. A tie that straddles the cut counts fractionally.
    """
    scores = scores.reindex(actual_position.index)
    better_p, tied_p = _tie_structure(scores)
    better_a, tied_a = _tie_structure(actual_position)

    predicted_rank = better_p + (tied_p + 1) / 2
    actual_rank = better_a + (tied_a + 1) / 2
    if predicted_rank.std() == 0 or actual_rank.std() == 0:
        spearman = 0.0  # one side says nothing about the order
    else:
        spearman = float(predicted_rank.corr(actual_rank))

    # Each driver takes each position of their tie with equal chance, independently on both sides.
    expected_error = pd.Series(
        [
            np.abs(
                np.arange(bp + 1, bp + tp + 1)[:, None] - np.arange(ba + 1, ba + ta + 1)[None, :]
            ).mean()
            for bp, tp, ba, ta in zip(better_p, tied_p, better_a, tied_a, strict=True)
        ],
        index=actual_position.index,
    )

    def top_overlap(n: int) -> float:
        in_predicted = ((n - better_p) / tied_p).clip(
            0, 1
        )  # chance of being in the predicted top n
        in_actual = ((n - better_a) / tied_a).clip(0, 1)  # chance of being in the real top n
        return float((in_predicted * in_actual).sum() / min(n, len(actual_position)))

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

"""Per-race ranking metrics. Predictions are scores where lower means a better finish."""

import pandas as pd


def predicted_order(scores: pd.Series) -> pd.Series:
    """Unique predicted finishing positions (1 = best). Ties are broken by row order."""
    return scores.rank(method="first").astype(int)


def race_metrics(scores: pd.Series, actual_position: pd.Series) -> dict[str, float]:
    """Compare one race's predicted scores with the actual finishing positions."""
    pred = predicted_order(scores)
    actual = actual_position.rank(method="first").astype(int)

    def top_overlap(n: int) -> float:
        return len(set(pred[pred <= n].index) & set(actual[actual <= n].index)) / n

    return {
        "spearman": float(pred.corr(actual, method="spearman")),
        "mae_position": float((pred - actual).abs().mean()),
        "winner_acc": float(top_overlap(1)),
        "podium_overlap": float(top_overlap(3)),
        "top10_overlap": float(top_overlap(10)),
    }


def summarize(per_race: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric per (model, test year), plus an 'all' row per model."""
    metrics = [c for c in per_race.columns if c not in ("model", "Year", "Round")]
    by_year = per_race.groupby(["model", "Year"])[metrics].mean().reset_index()
    overall = per_race.groupby("model")[metrics].mean().reset_index()
    overall.insert(1, "Year", "all")
    overall["races"] = per_race.groupby("model").size().to_numpy()
    return pd.concat([by_year.astype({"Year": object}), overall], ignore_index=True)

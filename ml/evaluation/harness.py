"""Walk-forward evaluation of race predictors."""

import pandas as pd

from ml.evaluation.baselines import RacePredictor
from ml.evaluation.data import KEY, PRE_RACE_COLUMNS
from ml.evaluation.metrics import race_metrics
from ml.evaluation.splits import LOCKED_SEASON, check_not_locked, walk_forward_splits


def evaluate(
    models: list[RacePredictor],
    results: pd.DataFrame,
    min_train_years: int = 1,
    locked_season: int | None = LOCKED_SEASON,
    allow_locked: bool = False,
) -> pd.DataFrame:
    """Per-race metrics for every model over every walk-forward test season."""
    results = results.dropna(subset=["Position"])
    splits = walk_forward_splits(
        sorted(results["Year"].unique()), min_train_years, locked_season, allow_locked
    )
    rows = []
    for split in splits:
        check_not_locked(split.test_year, locked_season, allow_locked)
        train = results[results["Year"].isin(split.train_years)]
        test = results[results["Year"] == split.test_year]
        for model in models:
            model.fit(train.copy())
            for (year, rnd), race in test.groupby(KEY, sort=True):
                # History is every race strictly before this one, including earlier rounds of
                # the test season (that information exists before the race starts).
                earlier = (results["Year"] < year) | (
                    (results["Year"] == year) & (results["Round"] < rnd)
                )
                history = results[earlier].copy()
                scores = model.predict_race(history, race[PRE_RACE_COLUMNS].copy())
                metrics = race_metrics(scores, race["Position"])
                rows.append({"model": model.name, "Year": year, "Round": rnd, **metrics})
    return pd.DataFrame(rows)

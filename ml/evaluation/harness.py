"""Walk-forward evaluation of race predictors."""

import pandas as pd

from ml.evaluation.baselines import RacePredictor
from ml.evaluation.data import ENTRY_COLUMNS, KEY, TARGET_COLUMNS
from ml.evaluation.metrics import race_metrics
from ml.evaluation.splits import LOCKED_SEASON, check_not_locked, walk_forward_splits


def evaluate(
    models: list[RacePredictor],
    results: pd.DataFrame,
    target: str = "race",
    min_train_years: int = 1,
    locked_season: int | None = LOCKED_SEASON,
    allow_locked: bool = False,
) -> pd.DataFrame:
    """Per-race metrics for every model over every walk-forward test season.

    target "race" scores predicted finishing order against `Position`; "quali" scores predicted
    qualifying order against `QualiPosition`. Each model is given only the entry columns that are
    known when that target is decided (see ENTRY_COLUMNS) plus the results of earlier races.

    A model is handed the FULL entry list of each race, including drivers whose outcome is unknown
    (they were entered), so the field a model sees is never chosen by the result. Only drivers with
    a known outcome are scored.
    """
    actual_column = TARGET_COLUMNS[target]
    entry_columns = [c for c in ENTRY_COLUMNS[target] if c in results.columns]
    scored = results.dropna(subset=[actual_column])  # used only to find which seasons can be tested
    splits = walk_forward_splits(
        sorted(scored["Year"].unique()), min_train_years, locked_season, allow_locked
    )
    rows = []
    for split in splits:
        check_not_locked(split.test_year, locked_season, allow_locked)
        train = results[results["Year"].isin(split.train_years)]
        test = results[results["Year"] == split.test_year]
        for model in models:
            model.fit(train.copy())
            for (year, rnd), race in test.groupby(KEY, sort=True):
                known = race[actual_column].notna()
                if known.sum() < 2:
                    continue  # nothing to rank
                # History is every race strictly before this one, including earlier rounds of
                # the test season (that information exists before the race starts).
                earlier = (results["Year"] < year) | (
                    (results["Year"] == year) & (results["Round"] < rnd)
                )
                history = results[earlier].copy()
                scores = model.predict_race(history, race[entry_columns].copy())
                metrics = race_metrics(scores[known], race.loc[known, actual_column])
                rows.append({"model": model.name, "Year": year, "Round": rnd, **metrics})
    return pd.DataFrame(rows)

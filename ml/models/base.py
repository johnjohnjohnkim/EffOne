"""Wrap a scikit-learn style estimator as a RacePredictor that uses the leakage-safe features.

Training reads rows of the precomputed feature table for the training races only. That is safe
because every feature is built from strictly earlier races (prefix invariance, tested), so a row's
features do not change when later races are added. Prediction recomputes the features from
`history` and the entry list with `features_for_entry`, exactly as it would for a future race.
"""

import pandas as pd

from ml.features.build import TARGET_STAGES, features_for_entry, training_frame
from ml.features.builders import BUILDERS, HALF_LIVES


def feature_plan(
    target: str,
    include_scenario: bool = False,
    with_grid: bool = True,
    with_practice: bool = False,
    recency: int | None = None,
) -> tuple[list, list[str]]:
    """The builders to run and the feature columns they produce for `target`.

    `with_grid=False` drops the post-qualifying stage, giving a model that never sees a grid: the
    right shape for a forecast made before qualifying. `with_practice=True` adds the same-weekend
    features, which only exist once practice has been run. `recency=h` adds the recency-weighted
    form columns for half-life `h` (one of HALF_LIVES) and no other half-life.
    """
    stages = set(TARGET_STAGES[target])
    if not include_scenario:
        stages.discard("scenario")
    if not with_practice:
        stages.discard("post_practice")
    if not with_grid:
        stages.discard("post_quali")
    if recency is None:
        stages.discard("recency")
    elif recency not in HALF_LIVES:
        raise ValueError(f"recency half-life must be one of {HALF_LIVES}, not {recency!r}")
    builders = [b for b in BUILDERS if b.stage in stages]
    columns = [column for b in builders for column in b.columns]
    if recency is not None:
        columns = [c for c in columns if "_ew" not in c or c.endswith(f"_ew{recency}")]
    return builders, columns


def training_rows(
    table: pd.DataFrame,
    train: pd.DataFrame,
    target: str,
    include_scenario: bool,
    include_practice: bool = False,
    include_recency: bool = False,
) -> tuple[frozenset, pd.DataFrame]:
    """(training races, safe training frame): rows of `table` for the races present in `train`.

    `train` is used only to say WHICH races to learn from; the numbers come from the feature table.
    Races missing from the table, or no usable rows at all, are errors, not silent omissions.
    """
    races = frozenset(map(tuple, train[["Year", "Round"]].drop_duplicates().to_numpy().tolist()))
    in_table = frozenset(map(tuple, table[["Year", "Round"]].drop_duplicates().to_numpy().tolist()))
    if races - in_table:
        raise ValueError(
            f"training races missing from the feature table: {sorted(races - in_table)[:3]}"
        )
    rows = pd.MultiIndex.from_frame(table[["Year", "Round"]]).isin(list(races))
    frame = training_frame(table[rows], target, include_scenario, include_practice, include_recency)
    if frame.empty:
        raise ValueError("no usable training rows (no earlier seasons to learn from?)")
    return races, frame


class FeatureModel:
    """A model trained on `training_frame(table, target)` and scored through `features_for_entry`.

    `make_estimator()` returns an object with `fit(X, y)` and `predict(X)` (lower prediction means a
    better finish). Estimators with `needs_groups = True` are fitted as `fit(X, y, groups)`, where
    `groups` holds the number of drivers in each consecutive race.
    """

    def __init__(
        self,
        name: str,
        make_estimator,
        table: pd.DataFrame,
        target: str = "y_finish",
        include_scenario: bool = False,
        with_grid: bool = True,
        with_practice: bool = False,
        recency: int | None = None,
    ):
        self.name = name
        self.target = target
        self.include_scenario = include_scenario
        self.with_practice = with_practice
        self.recency = recency
        self.builders, self.columns = feature_plan(
            target, include_scenario, with_grid, with_practice, recency
        )
        self._make = make_estimator
        self._table = table
        self._estimator = None
        self._fitted_on: frozenset | None = None

    def fit(self, train: pd.DataFrame) -> None:
        races = frozenset(
            map(tuple, train[["Year", "Round"]].drop_duplicates().to_numpy().tolist())
        )
        if races == self._fitted_on:
            return  # already trained on exactly these races
        races, frame = training_rows(
            self._table,
            train,
            self.target,
            self.include_scenario,
            self.with_practice,
            self.recency is not None,
        )
        estimator = self._make()
        if getattr(estimator, "needs_groups", False):
            groups = frame.groupby(["Year", "Round"], sort=False).size().to_numpy()
            estimator.fit(frame[self.columns], frame[self.target], groups)
        else:
            estimator.fit(frame[self.columns], frame[self.target])
        self._estimator, self._fitted_on = estimator, races

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        if self._estimator is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        key = (int(race["Year"].iloc[0]), int(race["Round"].iloc[0]))
        features = features_for_entry(history, race, key, self.builders)
        return pd.Series(self._estimator.predict(features[self.columns]), index=race.index)

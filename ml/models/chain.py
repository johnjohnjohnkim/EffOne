"""Predict the race from a PREDICTED grid: a qualifying model's output feeds the race model.

The race model is trained on real starting grids, so this is the realistic "before qualifying"
forecast. The same race model also works with a user-supplied grid: just pass the grid in the entry
list and do not wrap it.
"""

import warnings

import pandas as pd

from ml.evaluation.data import ENTRY_COLUMNS
from ml.features.builders import Grid


class PredictedGridRace:
    """Replace the entry list's grid with the one the qualifying model predicts, then predict."""

    def __init__(self, race_model, grid_model, name: str | None = None):
        self.race_model = race_model
        self.grid_model = grid_model
        self.name = name or f"{race_model.name} [predicted grid]"

    def fit(self, train: pd.DataFrame) -> None:
        self.race_model.fit(train)
        self.grid_model.fit(train)

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        # The grid is exactly what is being predicted, so the qualifying model must not receive it.
        grid = predicted_grid(self.grid_model, history, race)
        return self.race_model.predict_race(history, race.assign(GridPosition=grid))


def predicted_grid(grid_model, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
    """The grid a qualifying model predicts for `race`: ranks of its scores, ties sharing a slot."""
    scores = grid_model.predict_race(history, race.drop(columns=["GridPosition"], errors="ignore"))
    return scores.rank(method="average")


def out_of_fold_grids(
    train: pd.DataFrame, make_grid_model, cache: dict | None = None
) -> pd.DataFrame:
    """Walk-forward predicted grids for every race in `train` that has an earlier season to learn from.

    Each season is predicted by a qualifying model fitted on the seasons BEFORE it only, never on
    itself, so a training race's predicted grid is as honest as one made for a future race. The first
    season has nothing earlier and is absent from the result. Returns Year, Round, DriverId and
    GridPosition (the predicted slot). `cache` reuses a season's grids when its earlier seasons are
    unchanged, which they are for every fit of an expanding walk-forward.
    """
    entry_columns = [c for c in ENTRY_COLUMNS["quali"] if c in train.columns]
    parts = []
    for season in sorted(train["Year"].unique())[1:]:
        earlier = train[train["Year"] < season]
        # The key names the season's own races too, so a longer partial season is never served stale.
        in_season = train.loc[train["Year"] == season, ["Year", "Round"]].drop_duplicates()
        key = (
            season,
            frozenset(map(tuple, earlier[["Year", "Round"]].drop_duplicates().to_numpy().tolist())),
            frozenset(map(tuple, in_season.to_numpy().tolist())),
        )
        if cache is not None and key in cache:
            parts.append(cache[key])
            continue
        model = make_grid_model()
        model.fit(earlier)
        grids = []
        for rnd in sorted(train.loc[train["Year"] == season, "Round"].unique()):
            race = train[(train["Year"] == season) & (train["Round"] == rnd)]
            history = train[
                (train["Year"] < season) | ((train["Year"] == season) & (train["Round"] < rnd))
            ]
            slots = predicted_grid(model, history, race[entry_columns].copy())
            grids.append(
                race[["Year", "Round", "DriverId"]].assign(GridPosition=slots.reindex(race.index))
            )
        frame = pd.concat(grids, ignore_index=True)
        if cache is not None:
            cache[key] = frame
        parts.append(frame)
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["Year", "Round", "DriverId", "GridPosition"])
    )


def with_grids(table: pd.DataFrame, grids: pd.DataFrame) -> pd.DataFrame:
    """`table` restricted to the races in `grids`, with the grid features recomputed from them."""
    keys = ["Year", "Round", "DriverId"]
    races = pd.MultiIndex.from_frame(grids[["Year", "Round"]].drop_duplicates())
    out = table[pd.MultiIndex.from_frame(table[["Year", "Round"]]).isin(races)].copy()
    out["GridPosition"] = out.merge(grids, on=keys, how="left")["GridPosition"].to_numpy()
    for _, race in out.groupby(["Year", "Round"], sort=False):
        out.loc[race.index, list(Grid.columns)] = Grid().compute(None, race).to_numpy()
    return out.drop(columns="GridPosition")


class OutOfFoldGridRace:
    """A race model trained on walk-forward PREDICTED grids, then fed a predicted grid.

    PredictedGridRace trains the race model on real grids but uses it on predicted ones, which are
    blurrier; the model over-trusts the grid. Here training and use match: the grid feature is a
    qualifying model's honest prediction both times. `make_race_model(table)` and
    `make_grid_model()` build fresh models; the race model is built on a table whose grid columns
    hold the out-of-fold predictions.
    """

    def __init__(self, make_race_model, make_grid_model, table: pd.DataFrame, name: str):
        self.name = name
        self._make_race, self._make_grid, self._table = make_race_model, make_grid_model, table
        self._grid_model = None
        self._race_model = None
        self.fell_back = False
        self._fitted_on: frozenset | None = None
        self._cache: dict = {}

    def fit(self, train: pd.DataFrame) -> None:
        races = frozenset(
            map(tuple, train[["Year", "Round"]].drop_duplicates().to_numpy().tolist())
        )
        if races == self._fitted_on:
            return
        grids = out_of_fold_grids(train, self._make_grid, self._cache)
        self.fell_back = grids.empty
        if self.fell_back:
            # One training season (the first split) has no earlier season to predict it from, so
            # this split trains on real grids like PredictedGridRace; `fell_back` records that.
            warnings.warn(
                f"{self.name}: one training season, so this split trains on real grids",
                stacklevel=2,
            )
            race_model = self._make_race(self._table)
            race_model.fit(train)
        else:
            seen = pd.MultiIndex.from_frame(grids[["Year", "Round"]].drop_duplicates())
            usable = train[pd.MultiIndex.from_frame(train[["Year", "Round"]]).isin(seen)]
            race_model = self._make_race(with_grids(self._table, grids))
            race_model.fit(usable)
        grid_model = self._make_grid()
        grid_model.fit(train)
        self._race_model, self._grid_model, self._fitted_on = race_model, grid_model, races

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        if self._race_model is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        grid = predicted_grid(self._grid_model, history, race)
        return self._race_model.predict_race(history, race.assign(GridPosition=grid))


class DropGrid:
    """Hide the starting grid from a model: a race forecast made before qualifying."""

    def __init__(self, model, name: str | None = None):
        self.model = model
        self.name = name or f"{model.name} [no grid]"

    def fit(self, train: pd.DataFrame) -> None:
        self.model.fit(train)

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        return self.model.predict_race(history, race.drop(columns=["GridPosition"]))

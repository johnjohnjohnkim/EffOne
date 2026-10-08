"""Predict the race from a PREDICTED grid: a qualifying model's output feeds the race model.

The race model is trained on real starting grids, so this is the realistic "before qualifying"
forecast. The same race model also works with a user-supplied grid: just pass the grid in the entry
list and do not wrap it.
"""

import pandas as pd


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
        scores = self.grid_model.predict_race(history, race.drop(columns=["GridPosition"]))
        grid = scores.rank(method="average")  # tied scores share a slot, not an arbitrary order
        return self.race_model.predict_race(history, race.assign(GridPosition=grid))


class DropGrid:
    """Hide the starting grid from a model: a race forecast made before qualifying."""

    def __init__(self, model, name: str | None = None):
        self.model = model
        self.name = name or f"{model.name} [no grid]"

    def fit(self, train: pd.DataFrame) -> None:
        self.model.fit(train)

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        return self.model.predict_race(history, race.drop(columns=["GridPosition"]))

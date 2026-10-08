"""Model interface plus the baselines every real model must beat.

A model returns one score per driver for a race (lower = better finish). It is given only
pre-race columns for the race itself and the full results of all earlier races as `history`.
"""

from typing import Protocol

import pandas as pd

from ml.features.build import as_flags


class RacePredictor(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> None:
        """Learn from complete seasons before the test season. Baselines do nothing here."""

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        """Scores indexed like `race`. `history` holds only races strictly before this one."""


def _grid_score(race: pd.DataFrame) -> pd.Series:
    """Grid slot, with pit-lane starts (0) and missing slots placed at the back."""
    grid = race["GridPosition"].where(race["GridPosition"] > 0)
    return grid.fillna(grid.max() + 1 if grid.notna().any() else 1)


def usable(history: pd.DataFrame) -> pd.DataFrame:
    """Earlier races that carried real information (strict about `race_ok`, like the models)."""
    if "race_ok" in history.columns:
        return history[as_flags(history["race_ok"], "history race_ok")]
    return history


class GridBaseline:
    """Everyone finishes where they started."""

    name = "grid_equals_finish"

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        return _grid_score(race)


class PreviousResultBaseline:
    """Everyone repeats their last result in `column`. Drivers with none go behind everyone else.

    Tied scores (several debutants) are scored by expectation, so there is no hidden tie-break.
    """

    def __init__(self, column: str, name: str):
        self.column = column
        self.name = name

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        # Drivers without a previous result are ordered by their grid slot when the entry list has
        # one (race target), and are otherwise tied behind everyone else (qualifying target).
        fallback = (
            _grid_score(race) if "GridPosition" in race.columns else pd.Series(1.0, race.index)
        )
        if history.empty or self.column not in history.columns:
            return fallback
        ordered = usable(history).dropna(subset=[self.column]).sort_values(["Year", "Round"])
        last = ordered.groupby("Abbreviation")[self.column].last()
        previous = race["Abbreviation"].map(last)
        return previous.fillna(previous.max() + fallback) if previous.notna().any() else fallback


class PreviousRaceBaseline(PreviousResultBaseline):
    """Everyone finishes where they finished last race."""

    def __init__(self):
        super().__init__("Position", "previous_race_equals_finish")


class PreviousQualiBaseline(PreviousResultBaseline):
    """Everyone qualifies where they qualified last time."""

    def __init__(self):
        super().__init__("QualiPosition", "previous_quali_equals_quali")


class PreviousRaceForQualiBaseline(PreviousResultBaseline):
    """Everyone qualifies where they finished last race."""

    def __init__(self):
        super().__init__("Position", "previous_race_equals_quali")


class MeanLastKBaseline:
    """Everyone repeats the average of their last `k` results in `column`: a form baseline.

    A single previous result is a noisy guess; the average of a few is the obvious "no model"
    reference a real model has to beat. Like the previous-result baselines, it uses only races that
    carried real information.
    """

    def __init__(self, column: str, name: str, k: int = 5):
        self.column, self.name, self.k = column, name, k

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        fallback = (
            _grid_score(race) if "GridPosition" in race.columns else pd.Series(1.0, race.index)
        )
        if history.empty or self.column not in history.columns:
            return fallback
        recent = usable(history).dropna(subset=[self.column]).sort_values(["Year", "Round"])
        recent = recent.groupby("Abbreviation").tail(self.k)
        form = race["Abbreviation"].map(recent.groupby("Abbreviation")[self.column].mean())
        return form.fillna(form.max() + fallback) if form.notna().any() else fallback


class MeanLast5FinishBaseline(MeanLastKBaseline):
    def __init__(self):
        super().__init__("Position", "mean_last5_finish")


class MeanLast5QualiBaseline(MeanLastKBaseline):
    def __init__(self):
        super().__init__("QualiPosition", "mean_last5_quali")


BASELINES_BY_TARGET = {
    "race": [GridBaseline, PreviousRaceBaseline, MeanLast5FinishBaseline],
    "quali": [PreviousQualiBaseline, PreviousRaceForQualiBaseline, MeanLast5QualiBaseline],
}

"""Model interface plus the two baselines every real model must beat.

A model returns one score per driver for a race (lower = better finish). It is given only
pre-race columns for the race itself and the full results of all earlier races as `history`.
"""

from typing import Protocol

import pandas as pd


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


class GridBaseline:
    """Everyone finishes where they started."""

    name = "grid_equals_finish"

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        return _grid_score(race)


class PreviousRaceBaseline:
    """Everyone finishes where they finished last race. New drivers fall back to their grid slot."""

    name = "previous_race_equals_finish"

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        fallback = _grid_score(race)
        if history.empty:
            return fallback
        last = history.sort_values(["Year", "Round"]).groupby("Abbreviation")["Position"].last()
        prev = race["Abbreviation"].map(last)
        # Drivers without a previous result are ranked by grid slot after those who have one.
        return prev.fillna(prev.max() + fallback if prev.notna().any() else fallback)


BASELINES = [GridBaseline, PreviousRaceBaseline]

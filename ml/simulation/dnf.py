"""Probability that a driver does not finish (DNF).

A DNF here is `dnf_flag`: not classified as a finisher. Development-season experiments (see the
Milestone 5 report) showed two things: the DNF features carry very little signal, and the DNF rate
drifts down over the years (19% of entries in 2018, about 10% in 2024), so a model fitted on all
earlier seasons is systematically too pessimistic and scored WORSE than a flat rate. The forecast is
therefore built in two parts:

- the level: the DNF rate of races that carried real information since RECENT_SEASONS seasons before
  the race's season (so up to two full seasons plus the part of the current one already run), which
  tracks the drift;
- the relative risk: a strongly regularised (C=0.01) logistic regression on the driver, team and
  circuit DNF rates plus the grid, rescaled so its average over the race is 1 and applied in full.

`RateBaseline` is the level alone, so the gap between the two is the value of the features.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.models.base import FeatureModel

DNF_FLOOR = 0.005  # nobody is certain to finish; keeps the simulation from dividing by zero odds
RECENT_SEASONS = 2
DEFAULT_RATE = 0.12  # only used before any race has been run
RISK_FEATURES = ("dnf", "grid")  # feature columns whose name contains one of these


def recent_rate(history: pd.DataFrame, year: int) -> float:
    """Share of entries that did not finish in the last RECENT_SEASONS seasons up to `year`."""
    usable = history[history["race_ok"]] if "race_ok" in history.columns else history
    recent = usable[usable["Year"] >= year - RECENT_SEASONS]
    pool = recent if len(recent) else usable
    return float(pool["dnf"].astype(float).mean()) if len(pool) else DEFAULT_RATE


class RelativeRisk:
    """Logistic regression on the DNF-related features; `predict` returns P(DNF)."""

    def __init__(self, c: float = 0.01):
        self._keep: list[str] = []
        self._pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=c, max_iter=1000)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._keep = [c for c in X.columns if any(k in c for k in RISK_FEATURES)]
        self._pipe.fit(X[self._keep], y.astype(int))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipe.predict_proba(X[self._keep])[:, 1]


class DnfModel:
    """Recent DNF rate times a damped relative risk. Same interface as the other predictors."""

    name = "dnf_rate_x_risk"

    def __init__(self, table: pd.DataFrame, c: float = 0.01):
        self._risk = FeatureModel("dnf_risk", lambda: RelativeRisk(c), table, "y_dnf")

    def fit(self, train: pd.DataFrame) -> None:
        self._risk.fit(train)

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        risk = self._risk.predict_race(history, race)
        relative = risk / risk.mean()
        level = recent_rate(history, int(race["Year"].iloc[0]))
        return (level * relative).clip(DNF_FLOOR, 1 - DNF_FLOOR)


class RateBaseline:
    """Every driver gets the recent DNF rate: the level without any driver or team information."""

    name = "dnf_recent_rate"

    def fit(self, train: pd.DataFrame) -> None:
        return None

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        level = recent_rate(history, int(race["Year"].iloc[0]))
        return pd.Series(np.clip(level, DNF_FLOOR, 1 - DNF_FLOOR), index=race.index)

"""Plackett-Luce: a ranking model fitted on whole finishing orders.

Each driver gets a utility u = w . features. The model says the winner is chosen with probability
proportional to exp(u) among everyone, then the runner-up among the rest, and so on. The weights w
are found by maximum likelihood over all training races (with an L2 penalty), so it learns from the
full ordering of each race, not from isolated position numbers.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features.build import features_for_entry
from ml.models.base import feature_plan, training_rows


def race_log_likelihood(utilities: np.ndarray) -> tuple[float, np.ndarray]:
    """Log-likelihood of one finishing order (best driver first) and its gradient in the utilities."""
    s = utilities - utilities.max()  # the model ignores a constant shift; this keeps exp() safe
    # lse[k] = log(sum of exp(s[j]) for j >= k): the pool the k-th place was drawn from.
    lse = np.logaddexp.accumulate(s[::-1])[::-1]
    log_likelihood = float((s - lse).sum())
    # d/ds_m = 1 - sum over k <= m of exp(s_m - lse_k)
    pairs = np.triu(np.exp(s[None, :] - lse[:, None]))  # entry [k, m] kept only for k <= m
    return log_likelihood, 1.0 - pairs.sum(axis=0)


class PlackettLuceModel:
    def __init__(
        self,
        name: str,
        table: pd.DataFrame,
        target: str = "y_finish",
        include_scenario: bool = False,
        with_grid: bool = True,
        l2: float = 1.0,
    ):
        self.name = name
        self.target = target
        self.l2 = l2
        self.include_scenario = include_scenario
        self.builders, self.columns = feature_plan(target, include_scenario, with_grid)
        self._table = table
        self._prep: Pipeline | None = None
        self.weights: np.ndarray | None = None
        self.converged: bool | None = None
        self._fitted_on: frozenset | None = None

    def fit(self, train: pd.DataFrame) -> None:
        races = frozenset(
            map(tuple, train[["Year", "Round"]].drop_duplicates().to_numpy().tolist())
        )
        if races == self._fitted_on:
            return
        races, frame = training_rows(self._table, train, self.target, self.include_scenario)
        self._prep = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )
        x = self._prep.fit_transform(frame[self.columns])
        # Row indices of each race, best finisher first.
        orders = [
            group.sort_values(self.target).index.to_numpy()
            for _, group in frame.reset_index(drop=True).groupby(["Year", "Round"], sort=False)
        ]

        def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
            utilities = x @ w
            total, grad_u = 0.0, np.zeros(len(x))
            for idx in orders:
                ll, g = race_log_likelihood(utilities[idx])
                total += ll
                grad_u[idx] = g
            return -total + self.l2 * float(w @ w), -(x.T @ grad_u) + 2 * self.l2 * w

        result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
        self.converged = bool(result.success)
        if not self.converged:
            warnings.warn(
                f"{self.name}: optimiser did not converge ({result.message})", stacklevel=2
            )
        self.weights, self._fitted_on = result.x, races

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        if self.weights is None or self._prep is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        key = (int(race["Year"].iloc[0]), int(race["Round"].iloc[0]))
        features = features_for_entry(history, race, key, self.builders)
        utilities = self._prep.transform(features[self.columns]) @ self.weights
        return pd.Series(-utilities, index=race.index)  # lower score = better finish

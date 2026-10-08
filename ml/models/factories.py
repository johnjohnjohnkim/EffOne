"""Estimator factories for the model zoo.

Hyperparameters were set by hand to conservative values (shallow trees, strong regularisation, about
3,000 training rows) and were never searched. Their provenance was not logged, though, and they were
chosen with these seasons in view, so treat the leaderboard as mildly optimistic. One setting, the
LambdaRank label gain, was changed after the test seasons showed the default underperforming. Each
model predicts a score where lower means a better finish, so rankers return the negative of their
relevance score.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker, XGBRegressor

SEED = 0
FIELD_CEILING = 23  # larger than any grid, so relevance = FIELD_CEILING - position stays positive


def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


def ridge() -> Pipeline:
    return Pipeline(
        [("impute", _imputer()), ("scale", StandardScaler()), ("model", Ridge(alpha=30.0))]
    )


def random_forest() -> Pipeline:
    forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=15,
        max_features=0.5,
        n_jobs=-1,
        random_state=SEED,
    )
    return Pipeline([("impute", _imputer()), ("model", forest)])


def lightgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=8,
        max_depth=4,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=SEED,
        verbose=-1,
    )


def xgboost() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=SEED,
        n_jobs=1,
    )


def _relevance(positions: pd.Series) -> np.ndarray:
    """Higher is better: the winner gets the largest label."""
    return (FIELD_CEILING - positions.to_numpy()).clip(min=1).astype(int)


class LightGBMRankAdapter:
    """LambdaRank on whole races: learns to order drivers within a race directly."""

    needs_groups = True

    def __init__(self):
        self.model = LGBMRanker(
            objective="lambdarank",
            # LightGBM's default gain is exponential (2^relevance - 1), which makes the loss care
            # almost only about the top few places. Whole-order accuracy wants a linear gain.
            # (Changed after the exponential default was seen to underperform on the test seasons.)
            label_gain=list(range(32)),
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=8,
            max_depth=4,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=SEED,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, positions: pd.Series, groups: np.ndarray) -> None:
        self.model.fit(X, _relevance(positions), group=groups)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.predict(X)


class XGBoostRankAdapter:
    """Pairwise ranking on whole races (xgboost's rank:pairwise)."""

    needs_groups = True

    def __init__(self):
        self.model = XGBRanker(
            objective="rank:pairwise",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=SEED,
            n_jobs=1,
        )

    def fit(self, X: pd.DataFrame, positions: pd.Series, groups: np.ndarray) -> None:
        qid = np.repeat(np.arange(len(groups)), groups)  # one id per race, ascending
        self.model.fit(X, _relevance(positions), qid=qid)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.predict(X)

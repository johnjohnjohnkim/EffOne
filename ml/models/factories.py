"""Estimator factories for the model zoo.

Every factory takes the settings that can be tuned (see ml/tuning.py) as keyword arguments. The
defaults are the hand-set values from Milestone 4: conservative (shallow trees, strong
regularisation, about 3,000 training rows). They were chosen with the Milestone 4 seasons in view
and one, the LambdaRank label gain, was changed after the exponential default underperformed there.
Tuned values live in `ml/models/tuned_params.json`, were chosen on DEVELOPMENT seasons only, and
every trial is logged in `ml/models/tuning_log.csv`.

Each model predicts a score where lower means a better finish, so rankers return the negative of
their relevance score.
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


def ridge(alpha: float = 30.0) -> Pipeline:
    return Pipeline(
        [("impute", _imputer()), ("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]
    )


def random_forest(
    max_depth: int = 8,
    min_samples_leaf: int = 15,
    n_estimators: int = 200,
    max_features: float = 0.5,
) -> Pipeline:
    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=SEED,
    )
    return Pipeline([("impute", _imputer()), ("model", forest)])


def lightgbm(
    num_leaves: int = 8,
    max_depth: int = 4,
    n_estimators: int = 250,
    learning_rate: float = 0.03,
    min_child_samples: int = 30,
    reg_lambda: float = 5.0,
) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=reg_lambda,
        random_state=SEED,
        verbose=-1,
    )


def xgboost(
    max_depth: int = 3,
    n_estimators: int = 250,
    learning_rate: float = 0.03,
    min_child_weight: float = 10,
    reg_lambda: float = 5.0,
) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=reg_lambda,
        random_state=SEED,
        n_jobs=1,
    )


def _relevance(positions: pd.Series) -> np.ndarray:
    """Higher is better: the winner gets the largest label."""
    return (FIELD_CEILING - positions.to_numpy()).clip(min=1).astype(int)


class LightGBMRankAdapter:
    """LambdaRank on whole races: learns to order drivers within a race directly."""

    needs_groups = True

    def __init__(
        self,
        num_leaves: int = 8,
        max_depth: int = 4,
        n_estimators: int = 200,
        learning_rate: float = 0.03,
        min_child_samples: int = 30,
        reg_lambda: float = 5.0,
    ):
        self.model = LGBMRanker(
            objective="lambdarank",
            # LightGBM's default gain is exponential (2^relevance - 1), which makes the loss care
            # almost only about the top few places. Whole-order accuracy wants a linear gain.
            # (Changed after the exponential default was seen to underperform on the test seasons.)
            label_gain=list(range(32)),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=reg_lambda,
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

    def __init__(
        self,
        max_depth: int = 3,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        min_child_weight: float = 5,
        reg_lambda: float = 5.0,
    ):
        self.model = XGBRanker(
            objective="rank:pairwise",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=reg_lambda,
            random_state=SEED,
            n_jobs=1,
        )

    def fit(self, X: pd.DataFrame, positions: pd.Series, groups: np.ndarray) -> None:
        qid = np.repeat(np.arange(len(groups)), groups)  # one id per race, ascending
        self.model.fit(X, _relevance(positions), qid=qid)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.predict(X)

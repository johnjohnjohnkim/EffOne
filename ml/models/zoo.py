"""The model zoo: every model that runs through the walk-forward harness."""

import pandas as pd

from ml.evaluation.baselines import (
    BASELINES_BY_TARGET,
    GridBaseline,
    MeanLast5FinishBaseline,
    PreviousRaceBaseline,
)
from ml.models.base import FeatureModel
from ml.models.chain import DropGrid, PredictedGridRace
from ml.models.elo import EloModel
from ml.models.factories import (
    LightGBMRankAdapter,
    XGBoostRankAdapter,
    lightgbm,
    random_forest,
    ridge,
    xgboost,
)
from ml.models.plackett_luce import PlackettLuceModel

FEATURE_TARGET = {"race": "y_finish", "quali": "y_quali"}


def baselines(target: str) -> list:
    return [cls() for cls in BASELINES_BY_TARGET[target]]


def make_models(target: str, table: pd.DataFrame) -> list:
    """Every implemented model for `target` ("race" or "quali"), not including baselines."""
    y = FEATURE_TARGET[target]
    models = [
        FeatureModel("ridge", ridge, table, y),
        FeatureModel("random_forest", random_forest, table, y),
        FeatureModel("lightgbm", lightgbm, table, y),
        FeatureModel("xgboost", xgboost, table, y),
        FeatureModel("lightgbm_lambdarank", LightGBMRankAdapter, table, y),
        FeatureModel("xgboost_rank", XGBoostRankAdapter, table, y),
        EloModel("elo", target),
        PlackettLuceModel("plackett_luce", table, y),
    ]
    if target == "race":
        # Weather is opt-in: history holds the realised weather, a prediction only has a forecast.
        models.append(
            FeatureModel("lightgbm_with_weather", lightgbm, table, y, include_scenario=True)
        )
    return models


def make_no_grid_models(table: pd.DataFrame) -> list:
    """Race models trained and used WITHOUT any grid feature (a forecast before qualifying).

    These variants were added after the chained models were seen to do badly; see the report.
    Elo never uses a grid, so it only needs relabelling.
    """
    y = FEATURE_TARGET["race"]
    return [
        DropGrid(FeatureModel("ridge", ridge, table, y, with_grid=False)),
        DropGrid(FeatureModel("random_forest", random_forest, table, y, with_grid=False)),
        DropGrid(FeatureModel("lightgbm", lightgbm, table, y, with_grid=False)),
        DropGrid(PlackettLuceModel("plackett_luce", table, y, with_grid=False)),
        DropGrid(EloModel("elo", "race")),
    ]


def make_predicted_grid_models(table: pd.DataFrame) -> tuple[list, list[str]]:
    """Race forecasts made before qualifying: (models, names of the baselines among them).

    Race models trained on real grids are fed a grid predicted by a LightGBM qualifying model; the
    "[no grid]" models never use a grid at all. The baselines are "the predicted grid is the result"
    and two naive references that need neither a grid nor a model: the last result and the
    average of the last five.
    """
    grid_model = FeatureModel("lightgbm_quali", lightgbm, table, "y_quali")
    chained = [
        PredictedGridRace(m, grid_model)
        for m in make_models("race", table)
        if m.name not in ("elo", "lightgbm_with_weather")  # no grid or realised weather yet
    ]
    naive = [DropGrid(PreviousRaceBaseline()), DropGrid(MeanLast5FinishBaseline())]
    grid_baseline = PredictedGridRace(GridBaseline(), grid_model)
    models = [grid_baseline, *naive, *chained, *make_no_grid_models(table)]
    return models, [grid_baseline.name, *[m.name for m in naive]]

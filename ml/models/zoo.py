"""The model zoo: every model that runs through the walk-forward harness.

`build_model` is the single place a model is constructed, so the leaderboards and the tuner always
use exactly the same thing. Settings come from `ml/models/tuned_params.json` when it exists.
"""

from collections.abc import Mapping
from functools import partial

import pandas as pd

from ml.evaluation.baselines import (
    BASELINES_BY_TARGET,
    GridBaseline,
    MeanLast5FinishBaseline,
    PreviousRaceBaseline,
)
from ml.models.base import FeatureModel
from ml.models.chain import DropGrid, OutOfFoldGridRace, PredictedGridRace
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
from ml.models.tuned import load_tuned, params_for

FEATURE_TARGET = {"race": "y_finish", "quali": "y_quali", "quali_after_practice": "y_quali"}
AFTER_PRACTICE = "quali_after_practice"  # qualifying forecast made after practice
MODEL_NAMES = (
    "ridge",
    "random_forest",
    "lightgbm",
    "xgboost",
    "lightgbm_lambdarank",
    "xgboost_rank",
    "elo",
    "plackett_luce",
)
_ESTIMATORS = {
    "ridge": ridge,
    "random_forest": random_forest,
    "lightgbm": lightgbm,
    "xgboost": xgboost,
    "lightgbm_lambdarank": LightGBMRankAdapter,
    "xgboost_rank": XGBoostRankAdapter,
}


def baselines(target: str) -> list:
    return [cls() for cls in BASELINES_BY_TARGET[target]]


def build_model(
    name: str,
    target: str,
    table: pd.DataFrame,
    params: Mapping | None = None,
    **options,
):
    """One model for `target` ("race" or "quali") with the given settings (defaults if None).

    `options` are structural, not tuned: `include_scenario` and `with_grid` for feature models.
    """
    settings = dict(params or {})
    y = FEATURE_TARGET[target]
    if options.pop("gap_target", False):
        if y != "y_quali" or name in ("elo", "plackett_luce"):
            raise ValueError(
                f"the gap-to-pole target is for regression models on qualifying, not {name}/{target}"
            )
        y = "y_qgap"
    if target == AFTER_PRACTICE:
        if name == "elo":
            raise ValueError("Elo has no practice information; it is not part of this board")
        options = {"with_practice": True, **options}
    if name == "elo":
        return EloModel("elo", target, **settings)
    if name == "plackett_luce":
        return PlackettLuceModel("plackett_luce", table, y, **settings, **options)
    return FeatureModel(name, partial(_ESTIMATORS[name], **settings), table, y, **options)


def make_models(target: str, table: pd.DataFrame, tuned: Mapping | None = None) -> list:
    """Every implemented model for `target`, not including baselines.

    `tuned` maps target -> model -> settings; None loads ml/models/tuned_params.json, {} forces
    the defaults.
    """
    tuned = load_tuned() if tuned is None else tuned
    names = [n for n in MODEL_NAMES if not (target == AFTER_PRACTICE and n == "elo")]
    models = [build_model(n, target, table, params_for(tuned, target, n)) for n in names]
    if target == "race":
        # Weather is opt-in: history holds the realised weather, a prediction only has a forecast.
        models.append(
            FeatureModel(
                "lightgbm_with_weather",
                partial(lightgbm, **params_for(tuned, target, "lightgbm")),
                table,
                FEATURE_TARGET[target],
                include_scenario=True,
            )
        )
    return models


FEATURE_EXPERIMENT_MODELS = ("ridge", "random_forest", "lightgbm", "xgboost")  # 4b.3


def make_feature_variants(
    table: pd.DataFrame, tuned: Mapping | None = None, target: str = "quali"
) -> list:
    """Qualifying models with the gap target and/or recency form that development tuning chose.

    Settings come from tuned["quali_features"][model] = {"gap_target": bool, "recency": int|None};
    a model whose choice is the plain one is left out (it is already on the board as itself). Each
    variant is named after what it changes, e.g. "ridge+gap+ew4".
    """
    tuned = load_tuned() if tuned is None else tuned
    models = []
    for name in FEATURE_EXPERIMENT_MODELS:
        choice = tuned.get("quali_features", {}).get(name, {})
        gap, recency = bool(choice.get("gap_target", False)), choice.get("recency")
        if not gap and recency is None:
            continue
        model = build_model(
            name,
            target,
            table,
            params_for(tuned, target, name),
            gap_target=gap,
            recency=recency,
        )
        model.name = name + ("+gap" if gap else "") + (f"+ew{recency}" if recency else "")
        models.append(model)
    return models


def make_no_grid_models(table: pd.DataFrame, tuned: Mapping | None = None) -> list:
    """Race models trained and used WITHOUT any grid feature (a forecast before qualifying).

    These variants were added after the chained models were seen to do badly; see the report.
    Elo never uses a grid, so it only needs relabelling.
    """
    tuned = load_tuned() if tuned is None else tuned

    def race(name: str, **options):
        return build_model(name, "race", table, params_for(tuned, "race", name), **options)

    return [
        DropGrid(race("ridge", with_grid=False)),
        DropGrid(race("random_forest", with_grid=False)),
        DropGrid(race("lightgbm", with_grid=False)),
        DropGrid(race("plackett_luce", with_grid=False)),
        DropGrid(race("elo")),
    ]


def make_predicted_grid_models(
    table: pd.DataFrame, tuned: Mapping | None = None
) -> tuple[list, list[str]]:
    """Race forecasts made before qualifying: (models, names of the baselines among them).

    Race models trained on real grids are fed a grid predicted by a LightGBM qualifying model; the
    "[no grid]" models never use a grid at all. The baselines are "the predicted grid is the result"
    and two naive references that need neither a grid nor a model: the last result and the
    average of the last five.
    """
    tuned = load_tuned() if tuned is None else tuned
    grid_model = build_model("lightgbm", "quali", table, params_for(tuned, "quali", "lightgbm"))
    chained = [
        PredictedGridRace(m, grid_model)
        for m in make_models("race", table, tuned)
        if m.name not in ("elo", "lightgbm_with_weather")  # no grid or realised weather yet
    ]
    naive = [DropGrid(PreviousRaceBaseline()), DropGrid(MeanLast5FinishBaseline())]
    grid_baseline = PredictedGridRace(GridBaseline(), grid_model)
    # 4b.4: the same race models, trained on walk-forward predicted grids instead of real ones.
    oof = [
        OutOfFoldGridRace(
            partial(build_model, m.name, "race", params=params_for(tuned, "race", m.name)),
            partial(
                build_model, "lightgbm", "quali", table, params_for(tuned, "quali", "lightgbm")
            ),
            table,
            f"{m.name} [oof grid]",
        )
        for m in make_models("race", table, tuned)
        if m.name not in ("elo", "lightgbm_with_weather")
    ]
    models = [grid_baseline, *naive, *chained, *oof, *make_no_grid_models(table, tuned)]
    return models, [grid_baseline.name, *[m.name for m in naive]]

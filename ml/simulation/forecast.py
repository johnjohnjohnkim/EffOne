"""A race forecaster: finishing-order scores + DNF risk + noise -> win/podium/top-10/DNF odds.

Two modes, because they carry different uncertainty:
- "real_grid": the grid is known (or typed in by the user) and goes straight into the models.
- "predicted_grid": the forecast is made before qualifying; a LightGBM qualifying model supplies
  the grid, so the noise has to be wider.

The noise settings (`sim_params.json`) are chosen on the development seasons only; the models are
refitted on seasons before the race, exactly as in the walk-forward evaluation.
"""

import json
from functools import partial
from pathlib import Path

import pandas as pd

from ml.models.chain import predicted_grid
from ml.models.tuned import load_tuned, params_for
from ml.models.zoo import build_model
from ml.simulation.dnf import DnfModel
from ml.simulation.race import SimParams, simulate_race

PARAMS_PATH = Path(__file__).with_name("sim_params.json")
MODES = ("real_grid", "predicted_grid")
RACE_MODEL = "ridge"  # best Spearman on the race board; see the Milestone 5 report
GRID_MODEL = "lightgbm"  # the qualifying model behind a predicted grid (as in Milestone 4)


def load_sim_params(mode: str, path: Path = PARAMS_PATH, **overrides) -> SimParams:
    """Noise settings for `mode` from sim_params.json; defaults when none were tuned yet."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    found = {}
    if path.exists():
        found = json.loads(path.read_text(encoding="utf-8")).get(mode, {})
    return SimParams(**{**found, **overrides})


class RaceForecaster:
    """Fit on earlier seasons, then turn an entry list into per-driver outcome probabilities."""

    def __init__(
        self, table: pd.DataFrame, mode: str = "real_grid", params: SimParams | None = None
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        tuned = load_tuned()
        self.mode = mode
        self.params = params or load_sim_params(mode)
        self.race_model = build_model(
            RACE_MODEL, "race", table, params_for(tuned, "race", RACE_MODEL)
        )
        self.dnf_model = DnfModel(table)
        # A factory, so the baselines can fit their own copies of the qualifying model.
        self.make_grid_model = (
            partial(build_model, GRID_MODEL, "quali", table, params_for(tuned, "quali", GRID_MODEL))
            if mode == "predicted_grid"
            else None
        )
        self.grid_model = self.make_grid_model() if self.make_grid_model else None

    def fit(self, train: pd.DataFrame) -> None:
        self.race_model.fit(train)
        self.dnf_model.fit(train)
        if self.grid_model is not None:
            self.grid_model.fit(train)

    def raw_predictions(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        """Expected-finish score, DNF chance and the grid used, per driver, before any noise."""
        if self.grid_model is not None:
            race = race.assign(GridPosition=predicted_grid(self.grid_model, history, race))
        return pd.DataFrame(
            {
                "score": self.race_model.predict_race(history, race),
                "p_dnf": self.dnf_model.predict_race(history, race),
                "grid": race["GridPosition"],
            }
        )

    def forecast(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        """One row per entered driver with p_win, p_podium, p_top10, p_dnf and expected_position."""
        raw = self.raw_predictions(history, race)
        out = simulate_race(raw["score"].to_numpy(), raw["p_dnf"].to_numpy(), self.params)
        out.index = race.index
        return pd.concat([race[["DriverId", "Abbreviation"]], raw[["grid"]], out], axis=1)

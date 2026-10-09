"""Collect walk-forward forecasts, turn them into probabilities, tune the noise, and score them.

The expensive part (fitting the models and predicting each race) is done once by `collect`, which
returns one row per driver per race: the expected-finish score, the DNF chance, the grid used and
the actual outcome. Everything after that (choosing the noise settings, simulating, scoring) works
on that table, so a parameter search never refits a model.

Selection of the noise settings is allowed on the development seasons only (`tune`). The report
seasons are only ever scored with settings that were already frozen.
"""

import json
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.evaluation.data import ENTRY_COLUMNS, KEY
from ml.evaluation.splits import (
    DEV_SEASONS,
    LOCKED_SEASON,
    check_development_only,
    check_not_locked,
    walk_forward_splits,
)
from ml.models.chain import out_of_fold_grids
from ml.simulation.dnf import RateBaseline
from ml.simulation.forecast import MODES, PARAMS_PATH, RaceForecaster
from ml.simulation.race import SimParams, draw_noise, simulate_positions, summarize

EVENTS = {"win": 1, "podium": 3, "top10": 10}  # event name -> finishing position it covers
CLIP = 1e-3  # probabilities are clipped to [CLIP, 1 - CLIP] before log-loss, for every model alike
TUNE_SIMS = 4000
FINAL_SIMS = 20000
SIGMA_STEPS = (0.4, 0.55, 0.75, 1.0, 1.3, 1.7, 2.2, 3.0)
GAMMAS = (0.0, 0.5, 1.0)
REFERENCE_SPREAD = 3.0  # finishing places of noise at gamma 0; other gammas are scaled to match
REFERENCE_SCORE = 8.0  # a mid-field expected finish


def _slots(race: pd.DataFrame) -> pd.Series:
    """Grid slot with pit-lane starts and gaps placed at the back (as the baselines do)."""
    grid = race["grid"].where(race["grid"] > 0)
    return grid.fillna(grid.max() + 1 if grid.notna().any() else 1.0)


class GridPrior:
    """P(event | grid slot), learned on training seasons: the obvious thing to beat.

    `grids` (Year, Round, DriverId, GridPosition) replaces the real grid with predicted slots, so a
    forecast made before qualifying is compared with a prior that learned from equally blurry
    grids; a prior trained on real grids and fed a predicted one would be a strawman.
    """

    def __init__(self, train: pd.DataFrame, grids: pd.DataFrame | None = None):
        if grids is not None:
            train = train.drop(columns=["GridPosition"]).merge(
                grids, on=[*KEY, "DriverId"], how="inner"
            )
        usable = train[train["race_ok"] & train["Position"].notna()]
        grid = usable["GridPosition"].where(usable["GridPosition"] > 0)
        back = grid.groupby([usable["Year"], usable["Round"]]).transform("max") + 1
        x = self._features(grid.fillna(back).fillna(1.0))  # same rule as _slots, all races at once
        self.models = {}
        for event, place in EVENTS.items():
            y = (usable["Position"] <= place).astype(int)
            # A season set with a single outcome (only in toy data) has nothing to fit: use its rate.
            self.models[event] = (
                LogisticRegression(C=10.0, max_iter=500).fit(x, y)
                if y.nunique() > 1
                else float(y.mean())
            )

    @staticmethod
    def _features(slot: pd.Series) -> np.ndarray:
        return np.column_stack([np.log(slot), slot])

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        x = self._features(_slots(race))
        out = {
            f"p_{e}": np.full(len(x), m) if isinstance(m, float) else m.predict_proba(x)[:, 1]
            for e, m in self.models.items()
        }
        return pd.DataFrame(out, index=race.index)


def fit_total(p: np.ndarray, total: float) -> np.ndarray:
    """Scale probabilities so they sum to `total`, keeping each at most 1."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    total = min(total, len(p))
    for _ in range(50):
        p = np.clip(p * (total / p.sum()), 1e-9, 1.0)
        if abs(p.sum() - total) < 1e-9:
            break
    return p


def collect(
    history: pd.DataFrame,
    table: pd.DataFrame,
    mode: str,
    test_seasons: Collection[int],
    allow_locked: bool = False,
) -> pd.DataFrame:
    """Walk-forward raw forecasts for `test_seasons`: one row per driver per race."""
    entry_columns = [c for c in ENTRY_COLUMNS["race"] if c in history.columns]
    scored = history.dropna(subset=["Position"])
    splits = walk_forward_splits(
        sorted(scored["Year"].unique()),
        locked_season=LOCKED_SEASON,
        allow_locked=allow_locked,
        test_seasons=test_seasons,
    )
    rows, grid_cache = [], {}
    for split in splits:
        check_not_locked(split.test_year, LOCKED_SEASON, allow_locked)
        train = history[history["Year"].isin(split.train_years)]
        forecaster = RaceForecaster(table, mode)
        forecaster.fit(train.copy())
        predicted = None
        if forecaster.make_grid_model is not None:
            predicted = out_of_fold_grids(train, forecaster.make_grid_model, grid_cache)
            predicted = None if predicted.empty else predicted  # one training season: real grids
        prior, dnf_base = GridPrior(train, predicted), RateBaseline()
        for (year, rnd), race in history[history["Year"] == split.test_year].groupby(KEY):
            if race["Position"].notna().sum() < 2:
                continue
            earlier = history[
                (history["Year"] < year) | ((history["Year"] == year) & (history["Round"] < rnd))
            ]
            entries = race[entry_columns].copy()
            raw = forecaster.raw_predictions(earlier, entries)
            base = prior.predict(raw)
            raw = raw.join(base.add_prefix("base_"))
            raw["base_p_dnf"] = dnf_base.predict_race(earlier, entries)
            raw = raw.join(race[["Year", "Round", "DriverId", "Position", "dnf"]])
            rows.append(raw)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _simulate_group(group: pd.DataFrame, params: SimParams, noise=None) -> pd.DataFrame:
    n = len(group)
    noise = noise or draw_noise(params.n_sims, n, params.seed)
    positions, retired = simulate_positions(
        group["score"].to_numpy(), group["p_dnf"].to_numpy(), params.sigma, params.gamma, noise
    )
    return summarize(positions, retired).set_index(group.index)


def probabilities(raw: pd.DataFrame, params: SimParams) -> pd.DataFrame:
    """`raw` plus simulated p_win, p_podium, p_top10 and p_dnf (the classifier's own DNF chance)."""
    parts = []
    for (year, rnd), group in raw.groupby(KEY, sort=True):
        seeded = SimParams(
            params.sigma, params.gamma, params.n_sims, params.seed + year * 100 + rnd
        )
        parts.append(_simulate_group(group, seeded)[["p_win", "p_podium", "p_top10"]])
    out = raw.join(pd.concat(parts))
    out["p_dnf"] = raw["p_dnf"]
    return out


def baseline_probabilities(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reference forecasts for every event, on the same rows: grid prior (normalised per race),
    uniform, and (for DNF) the base rate."""
    prior = raw[["Year", "Round"]].copy()
    uniform = raw[["Year", "Round"]].copy()
    for event, place in EVENTS.items():
        prior[f"p_{event}"] = np.nan
        uniform[f"p_{event}"] = np.nan
        for _, group in raw.groupby(KEY):
            n = len(group)
            total = min(place, n)
            prior.loc[group.index, f"p_{event}"] = fit_total(
                group[f"base_p_{event}"].to_numpy(), total if event != "win" else 1.0
            )
            uniform.loc[group.index, f"p_{event}"] = total / n
    prior["p_dnf"] = raw["base_p_dnf"]
    uniform["p_dnf"] = raw["base_p_dnf"]
    return {"grid_prior": prior, "uniform": uniform}


def outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    """Actual 0/1 outcomes per driver per race (NaN where the finishing position is unknown)."""
    known = raw["Position"].notna()
    out = pd.DataFrame(index=raw.index)
    for event, place in EVENTS.items():
        out[event] = (raw["Position"] <= place).astype(float).where(known)
    out["dnf"] = raw["dnf"].astype(float)
    return out


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, CLIP, 1 - CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def per_race_losses(model: str, probs: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Mean log-loss and Brier score of one model per race and event."""
    rows = []
    for (year, rnd), idx in probs.groupby(KEY).groups.items():
        row = {"model": model, "Year": year, "Round": rnd}
        for event in (*EVENTS, "dnf"):
            y = actual.loc[idx, event]
            keep = y.notna().to_numpy()
            p = probs.loc[idx, f"p_{event}"].to_numpy()[keep]
            row[f"ll_{event}"] = float(log_loss(p, y.to_numpy()[keep]).mean())
            row[f"brier_{event}"] = float(((p - y.to_numpy()[keep]) ** 2).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def tune_objective(raw: pd.DataFrame, params: SimParams, noises: dict) -> float:
    """Mean over races of the summed win, podium and top-10 log-loss (lower is better)."""
    total, races = 0.0, 0
    for (year, rnd), group in raw.groupby(KEY, sort=True):
        probs = _simulate_group(group, params, noises[(year, rnd)])
        y = outcomes(group)
        keep = y["win"].notna().to_numpy()
        for event in EVENTS:
            total += float(
                log_loss(probs[f"p_{event}"].to_numpy()[keep], y[event].to_numpy()[keep]).mean()
            )
        races += 1
    return total / races


def tune(raw: pd.DataFrame, seasons: Collection[int]) -> tuple[SimParams, pd.DataFrame]:
    """Pick (sigma, gamma) by log-loss on DEVELOPMENT races only. Returns the best and every trial."""
    check_development_only(seasons)
    check_development_only(raw["Year"].unique())  # the guard must look at the data, not the label
    noises = {
        (year, rnd): draw_noise(TUNE_SIMS, len(group), 1000 + year * 100 + rnd)
        for (year, rnd), group in raw.groupby(KEY)
    }
    trials = []
    for gamma in GAMMAS:
        reference = REFERENCE_SPREAD / REFERENCE_SCORE**gamma
        for step in SIGMA_STEPS:
            params = SimParams(sigma=reference * step, gamma=gamma, n_sims=TUNE_SIMS)
            trials.append(
                {
                    "gamma": gamma,
                    "sigma": params.sigma,
                    "logloss": tune_objective(raw, params, noises),
                }
            )
    log = pd.DataFrame(trials).sort_values("logloss").reset_index(drop=True)
    best = log.iloc[0]
    return SimParams(sigma=float(best["sigma"]), gamma=float(best["gamma"]), n_sims=FINAL_SIMS), log


def write_sim_params(results: dict[str, SimParams], path: Path = PARAMS_PATH) -> None:
    payload = {
        mode: {"sigma": round(p.sigma, 4), "gamma": p.gamma, "n_sims": p.n_sims, "seed": p.seed}
        for mode, p in results.items()
    }
    payload["provenance"] = {
        "dev_seasons": list(DEV_SEASONS),
        "selection": "lowest mean (win + podium + top10) log-loss on the development races",
        "tuned_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Quantile-binned predicted versus observed frequency (equal numbers per bin)."""
    frame = pd.DataFrame({"p": p, "y": y})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    grouped = frame.groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    return grouped.reset_index()


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    table = reliability(p, y, bins)
    weight = table["n"] / table["n"].sum()
    return float((weight * (table["predicted"] - table["observed"]).abs()).sum())


__all__ = [
    "EVENTS",
    "MODES",
    "baseline_probabilities",
    "collect",
    "expected_calibration_error",
    "outcomes",
    "per_race_losses",
    "probabilities",
    "reliability",
    "tune",
    "write_sim_params",
]

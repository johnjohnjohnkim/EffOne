"""CLI: python -m ml.simulation --race YEAR ROUND [--predicted-grid]   |   --tune

--race prints P(win), P(podium), P(top 10) and P(DNF) for every driver of one race in the data,
forecast only from the races before it (models are fitted on the seasons before it). The grid is the
real one unless --predicted-grid, which forecasts as if qualifying had not happened yet.

--tune chooses the simulation noise settings on the development seasons (2019-2021) and writes
ml/simulation/sim_params.json. It refuses to look at any other season.
"""

import argparse

import pandas as pd

from ml.evaluation.data import ENTRY_COLUMNS, KEY, load_history
from ml.evaluation.splits import DEV_SEASONS, LOCKED_SEASON
from ml.features.build import build_feature_table
from ml.simulation.forecast import MODES, PARAMS_PATH, RaceForecaster


def forecast_race(
    history: pd.DataFrame, table: pd.DataFrame, year: int, rnd: int, mode: str
) -> pd.DataFrame:
    """Forecast one race of `history` using only what was known before it."""
    race = history[(history["Year"] == year) & (history["Round"] == rnd)]
    if race.empty:
        raise SystemExit(f"no race {year} round {rnd} in the data")
    before = history[
        (history["Year"] < year) | ((history["Year"] == year) & (history["Round"] < rnd))
    ]
    train = history[history["Year"] < year]
    if train.empty:
        raise SystemExit(f"no earlier seasons to train on for {year}")
    forecaster = RaceForecaster(table, mode)
    forecaster.fit(train.copy())
    entries = race[[c for c in ENTRY_COLUMNS["race"] if c in race.columns]].copy()
    return forecaster.forecast(before, entries)


def tune_command(history: pd.DataFrame, table: pd.DataFrame) -> None:
    from ml.simulation.calibration import collect, tune, write_sim_params

    chosen = {}
    for mode in MODES:
        raw = collect(history, table, mode, DEV_SEASONS)
        params, log = tune(raw, DEV_SEASONS)
        chosen[mode] = params
        print(
            f"\n[{mode}] best of {len(log)} settings on {raw[KEY].drop_duplicates().shape[0]} dev races"
        )
        print(log.head(5).to_string(index=False, float_format="{:.4f}".format))
    write_sim_params(chosen)
    print(f"\nWrote {PARAMS_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--race", nargs=2, type=int, metavar=("YEAR", "ROUND"))
    group.add_argument("--tune", action="store_true")
    parser.add_argument("--predicted-grid", action="store_true")
    args = parser.parse_args()

    history = load_history()
    table = build_feature_table(history)
    if args.tune:
        tune_command(history, table)
        return 0
    year, rnd = args.race
    mode = "predicted_grid" if args.predicted_grid else "real_grid"
    if year == LOCKED_SEASON:
        print(f"note: {year} is the locked final test season; this forecast is not scored.")
    out = forecast_race(history, table, year, rnd, mode)
    shown = out.sort_values("p_win", ascending=False).drop(columns=["DriverId"])
    print(f"{year} round {rnd} ({mode}); probabilities from {len(out)} drivers")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(shown.to_string(index=False))
    print(
        "\nsums: win {:.3f}, podium {:.3f}, top 10 {:.3f}, expected DNFs {:.2f}".format(
            out["p_win"].sum(), out["p_podium"].sum(), out["p_top10"].sum(), out["p_dnf"].sum()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

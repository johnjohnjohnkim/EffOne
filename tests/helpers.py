"""Shared builders for the feature tests: a small random season table and a "scramble" tool."""

import numpy as np
import pandas as pd

from ml.features.tables import WEATHER_COLUMNS

OUTCOME_COLUMNS = ["Position", "Points", "dnf", "QualiPosition", "quali_gap_pct", "pace_gap_pct"]
DRIVERS = [f"d{i}" for i in range(6)]
TEAM_OF = {d: f"t{i // 2}" for i, d in enumerate(DRIVERS)}


def row(year, rnd, driver, team="t0", circuit="c0", **overrides) -> dict:
    """One driver-race row with neutral defaults; override what a test cares about."""
    base = {
        "Year": year,
        "Round": rnd,
        "Abbreviation": driver.upper(),
        "DriverId": driver,
        "TeamId": team,
        "TeamKey": team,
        "TeamName": team,
        "CircuitId": circuit,
        "GridPosition": 5.0,
        "Position": 5.0,
        "Points": 0.0,
        "dnf": False,
        "QualiPosition": 5.0,
        "quali_gap_pct": 1.0,
        "pace_gap_pct": 1.0,
        "race_ok": True,
        **dict.fromkeys(WEATHER_COLUMNS, 10.0),
    }
    return base | overrides


def make_rows(years=(2022, 2023), rounds=5, seed=0) -> pd.DataFrame:
    """A small random but fully populated season table (same columns as build_history)."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for rnd in range(1, rounds + 1):
            finish = rng.permutation(len(DRIVERS)) + 1
            grid = rng.permutation(len(DRIVERS)) + 1
            quali = rng.permutation(len(DRIVERS)) + 1
            for i, d in enumerate(DRIVERS):
                rows.append(
                    row(
                        year,
                        rnd,
                        d,
                        team=TEAM_OF[d],
                        circuit=f"c{(rnd - 1) % 3}",
                        GridPosition=float(grid[i]),
                        Position=float(finish[i]),
                        Points=float(max(0, 11 - finish[i])),
                        dnf=bool(rng.random() < 0.15),
                        QualiPosition=float(quali[i]),
                        quali_gap_pct=float(rng.random()),
                        pace_gap_pct=float(rng.random()),
                        wx_air_temp=20 + float(rng.random() * 10),
                        wx_track_temp=30 + float(rng.random() * 10),
                        wx_humidity=50 + float(rng.random() * 20),
                        wx_wind_speed=float(rng.random() * 5),
                        wx_rain_frac=float(rng.random() * 0.3),
                    )
                )
    return pd.DataFrame(rows)


def _randomise(out: pd.DataFrame, rng, mask: pd.Series, columns: list[str]) -> None:
    for col in columns:
        if col == "dnf":
            out.loc[mask, col] = rng.random(mask.sum()) < 0.5
        else:
            out.loc[mask, col] = rng.random(mask.sum()) * 40 + 1


def scramble(rows: pd.DataFrame, key: tuple[int, int], stage: str, seed: int = 7) -> pd.DataFrame:
    """Randomise everything a builder of `stage` must not be able to see.

    That is every outcome column from the target race onward, every column of later races
    (including who raced, for which team and where), and for early stages the target race's own
    grid and weather.
    """
    rng = np.random.default_rng(seed)
    out = rows.copy()
    year, rnd = key
    target = (out["Year"] == year) & (out["Round"] == rnd)
    later = (out["Year"] > year) | ((out["Year"] == year) & (out["Round"] > rnd))

    _randomise(out, rng, target | later, OUTCOME_COLUMNS)
    _randomise(out, rng, later, ["GridPosition", *WEATHER_COLUMNS])
    # Later races: change the cast and the venue too, not just the numbers.
    n = int(later.sum())
    out.loc[later, "DriverId"] = [f"x{i}" for i in range(n)]  # all-new drivers, none repeated
    out.loc[later, "TeamKey"] = rng.choice(["tx", "ty", "t0"], n)
    later_races = out.loc[later, ["Year", "Round"]].drop_duplicates()
    new_circuit = dict(
        zip(
            map(tuple, later_races.to_numpy().tolist()),
            rng.choice(["c0", "c1", "c2", "cz"], len(later_races)),
            strict=True,
        )
    )  # one venue per race, as in real life
    out.loc[later, "CircuitId"] = [
        new_circuit[(y, r)]
        for y, r in zip(out.loc[later, "Year"], out.loc[later, "Round"], strict=True)
    ]
    if stage in ("pre_weekend", "scenario"):
        _randomise(out, rng, target, ["GridPosition"])
    if stage == "pre_weekend":
        _randomise(out, rng, target, WEATHER_COLUMNS)
    return out

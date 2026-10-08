"""Which past races at a circuit count as history for a target race.

Rule (user-specified): a season at a circuit counts only if at least 25% of the drivers on the
*current grid* raced there that year. This keeps one-off returns and long gaps from feeding noisy,
unrepresentative track history into track-specific features.

Leakage: pass `as_of=(year, round)` for the race being predicted. Only races strictly before it
are used, and the default "current grid" is the grid of the most recent of those races, so a
past race never sees a grid from its own future.

Known limits: a stand-in driver in the reference race shifts the share by 1/grid size; and drivers
are identified by `DriverId`, so a driver who changes team keeps their history.
"""

import math

import pandas as pd

MIN_GRID_SHARE = 0.25
_EPSILON = 1e-9  # guards min_share * n against float noise, e.g. 0.7 * 10 = 7.000000000000001


def before(results: pd.DataFrame, as_of: tuple[int, int]) -> pd.DataFrame:
    """Rows of races strictly before (year, round)."""
    year, rnd = as_of
    mask = (results["Year"] < year) | ((results["Year"] == year) & (results["Round"] < rnd))
    return results[mask]


def latest_grid(results: pd.DataFrame, driver_col: str = "DriverId") -> set[str]:
    """Drivers in the most recent race of `results` (a proxy for the current grid)."""
    if results.empty:
        raise ValueError("no results to take a grid from")
    last = results.sort_values(["Year", "Round"]).iloc[-1]
    race = results[(results["Year"] == last["Year"]) & (results["Round"] == last["Round"])]
    return set(race[driver_col])


def circuit_year_shares(
    results: pd.DataFrame,
    events: pd.DataFrame,
    current_grid: set[str],
    driver_col: str = "DriverId",
) -> pd.DataFrame:
    """Share of the current grid that raced at each circuit in each year.

    `results` needs Year, Round and `driver_col`; `events` needs Year, RoundNumber, CircuitId.
    A circuit hosting two races in one year (2020 Bahrain) pools the drivers of both.
    """
    if not current_grid:
        raise ValueError("current_grid is empty")
    keyed = results.merge(
        events[["Year", "RoundNumber", "CircuitId"]],
        left_on=["Year", "Round"],
        right_on=["Year", "RoundNumber"],
        how="left",
    )
    missing = keyed.loc[keyed["CircuitId"].isna(), ["Year", "Round"]].drop_duplicates()
    if not missing.empty:
        pairs = sorted(map(tuple, missing.to_numpy().tolist()))
        raise ValueError(f"results with no matching event (year, round): {pairs}; ingest events")
    raced = keyed[keyed[driver_col].isin(current_grid)]
    counts = (
        raced.groupby(["CircuitId", "Year"])[driver_col].nunique().rename("raced").reset_index()
    )
    counts["grid_share"] = counts["raced"] / len(current_grid)
    return counts


def eligible_circuit_years(
    results: pd.DataFrame,
    events: pd.DataFrame,
    current_grid: set[str] | None = None,
    min_share: float = MIN_GRID_SHARE,
    as_of: tuple[int, int] | None = None,
    driver_col: str = "DriverId",
) -> pd.DataFrame:
    """CircuitId, Year, raced, grid_share, eligible (True when raced >= ceil(min_share * grid size)).

    With `as_of`, only races strictly before it are considered (leakage-safe).
    """
    history = before(results, as_of) if as_of else results
    grid = current_grid if current_grid is not None else latest_grid(history, driver_col)
    out = circuit_year_shares(history, events, grid, driver_col)
    # Compare whole drivers, not float shares: at least ceil(min_share * grid size) must have raced.
    needed = math.ceil(min_share * len(grid) - _EPSILON)
    out["eligible"] = out["raced"] >= needed
    return out.sort_values(["CircuitId", "Year"]).reset_index(drop=True)

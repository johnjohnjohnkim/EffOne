"""Build the small lookup tables: drivers, teams, circuits.

python -m ml.ingest.dimensions  ->  data/dim/{drivers,teams,circuits}.parquet

Drivers are keyed by `DriverId` (stable across abbreviation changes). Teams are keyed by fastf1's
TeamId, which changes when a team is renamed (Toro Rosso -> AlphaTauri -> RB); linking those
lineages is left to the feature phase. The circuits table is minimal: id, venue, country, years.
"""

import pandas as pd

from ml.ingest.atomic import write_parquet_atomic
from ml.ingest.events import load_events
from ml.ingest.paths import data_dir, raw_dir

RESULT_COLUMNS = ["Year", "Round", "Abbreviation", "DriverId", "FullName", "TeamId", "TeamName"]


def load_named_results() -> pd.DataFrame:
    files = sorted(raw_dir().glob("results/year=*/round=*/race.parquet"))
    if not files:
        raise FileNotFoundError("No race results ingested; run `python -m ml.ingest` first.")
    return pd.concat((pd.read_parquet(f, columns=RESULT_COLUMNS) for f in files), ignore_index=True)


def build_drivers(results: pd.DataFrame) -> pd.DataFrame:
    ordered = results.sort_values(["Year", "Round"])
    return (
        ordered.groupby("DriverId")
        .agg(
            Abbreviation=("Abbreviation", "last"),
            FullName=("FullName", "last"),
            first_year=("Year", "min"),
            last_year=("Year", "max"),
            races=("Round", "size"),
        )
        .reset_index()
    )


def build_teams(results: pd.DataFrame) -> pd.DataFrame:
    ordered = results.sort_values(["Year", "Round"])
    return (
        ordered.groupby("TeamId")
        .agg(
            TeamName=("TeamName", "last"),
            first_year=("Year", "min"),
            last_year=("Year", "max"),
            entries=("Round", "size"),
        )
        .reset_index()
    )


def build_circuits(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.sort_values(["Year", "RoundNumber"])
        .groupby("CircuitId")
        .agg(
            Location=("Location", "last"),
            Country=("Country", "last"),
            first_year=("Year", "min"),
            last_year=("Year", "max"),
            race_weekends=("RoundNumber", "size"),
        )
        .reset_index()
    )


def main() -> int:
    results = load_named_results()
    out = data_dir() / "dim"
    tables = {
        "drivers": build_drivers(results),
        "teams": build_teams(results),
        "circuits": build_circuits(load_events()),
    }
    for name, df in tables.items():
        write_parquet_atomic(df, out / f"{name}.parquet")
        print(f"{name}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

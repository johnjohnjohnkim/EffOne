"""Load ingested race results in the shape the evaluation harness uses."""

import pandas as pd

from ml.ingest.paths import raw_dir

# Columns a model may see before a race starts. Everything else (Position, Status, Points...)
# is an outcome and is hidden from models at prediction time.
PRE_RACE_COLUMNS = ["Year", "Round", "Abbreviation", "DriverId", "TeamName", "GridPosition"]
OUTCOME_COLUMNS = ["Position", "ClassifiedPosition", "Status", "Points", "Time", "Laps"]
KEY = ["Year", "Round"]


def load_race_results() -> pd.DataFrame:
    """All ingested Race results, one row per driver per race, ordered by race then finish."""
    files = sorted(raw_dir().glob("results/year=*/round=*/race.parquet"))
    if not files:
        raise FileNotFoundError("No race results found; run `python -m ml.ingest` first.")
    cols = PRE_RACE_COLUMNS + OUTCOME_COLUMNS
    df = pd.concat((pd.read_parquet(f, columns=cols) for f in files), ignore_index=True)
    return df.sort_values(KEY + ["Position"]).reset_index(drop=True)


def race_keys(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Distinct (year, round) pairs in chronological order."""
    return sorted(set(map(tuple, df[KEY].drop_duplicates().to_numpy().tolist())))

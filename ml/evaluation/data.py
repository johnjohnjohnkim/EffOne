"""Load ingested race results in the shape the evaluation harness uses."""

import pandas as pd

from ml.features.weekend import PQ_COLUMNS
from ml.ingest.paths import raw_dir

# Columns a model may see before a race starts (the old, results-only view). Everything else
# (Position, Status, Points...) is an outcome and is hidden from models at prediction time.
PRE_RACE_COLUMNS = ["Year", "Round", "Abbreviation", "DriverId", "TeamName", "GridPosition"]
OUTCOME_COLUMNS = ["Position", "ClassifiedPosition", "Status", "Points", "Time", "Laps"]
KEY = ["Year", "Round"]

# What the harness scores each target against.
TARGET_COLUMNS = {
    "race": "Position",
    "quali": "QualiPosition",
    "quali_after_practice": "QualiPosition",
}

# What a model is handed for the race it must predict, per target. The modelling table has these
# columns; a plain results table (used in some tests) simply has fewer, and only those are passed.
_KEYS = ["Year", "Round", "DriverId", "Abbreviation", "TeamKey", "TeamName", "CircuitId"]
_WEATHER = ["wx_air_temp", "wx_track_temp", "wx_humidity", "wx_wind_speed", "wx_rain_frac"]
ENTRY_COLUMNS = {
    # Race: the entry list, the weather, and the starting grid (real, or one the user typed in).
    "race": [*_KEYS, *_WEATHER, "GridPosition"],
    # Qualifying is decided before the grid exists and before race-day weather is known.
    "quali": _KEYS,
    # The same, once practice (and any earlier sprint session) has been run.
    "quali_after_practice": [*_KEYS, *PQ_COLUMNS],
}


def load_race_results() -> pd.DataFrame:
    """All ingested Race results, one row per driver per race, ordered by race then driver id."""
    files = sorted(raw_dir().glob("results/year=*/round=*/race.parquet"))
    if not files:
        raise FileNotFoundError("No race results found; run `python -m ml.ingest` first.")
    cols = PRE_RACE_COLUMNS + OUTCOME_COLUMNS
    df = pd.concat((pd.read_parquet(f, columns=cols) for f in files), ignore_index=True)
    # Rows are ordered by a PRE-RACE key (the driver id), never by finishing position: row order must
    # not carry the answer, because ties and positional operations would otherwise read it.
    return df.sort_values(KEY + ["DriverId"]).reset_index(drop=True)


def load_history() -> pd.DataFrame:
    """The modelling table: one row per driver per race with results, qualifying, pace and weather.

    Rows are ordered by race then driver id. This is what the harness and the models use.
    """
    from ml.features.tables import build_history  # imported late: features import this module

    return build_history()


def race_keys(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Distinct (year, round) pairs in chronological order."""
    return sorted(set(map(tuple, df[KEY].drop_duplicates().to_numpy().tolist())))

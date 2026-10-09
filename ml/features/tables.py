"""Turn the raw ingested tables into one tidy table: one row per driver per race.

The pure helpers (`dnf_flag`, `pace_gap_pct`, `quali_gap_pct`, `weather_summary`) work on a single
session and are unit-tested on small synthetic frames. `build_history` reads the real data.

Every column here describes what happened in that row's race, so it is only usable as a feature for
LATER races. The feature builders in `builders.py` are what enforce that.
"""

import numpy as np
import pandas as pd

from ml.features.lineage import team_key
from ml.ingest.audit import audit_session
from ml.ingest.events import load_events
from ml.ingest.paths import raw_dir

MIN_VALID_LAPS = 5  # a driver needs this many clean laps to get a pace figure
MIN_DRIVERS_FOR_PACE = 10  # and the field needs this many to compare against

# Races that happened on paper but carry no performance information, so they must not feed form
# or circuit history. 2021 R12 (Belgian GP): three laps behind the safety car, finishing order equal
# to the grid, half points. The rows stay in the table with race_ok False; remove an entry to
# count a race again. (Default chosen by the loop while the user was away; needs their confirmation.)
NOT_A_RACE: frozenset[tuple[int, int]] = frozenset({(2021, 12)})

RACE_COLUMNS = [
    "Year",
    "Round",
    "Abbreviation",
    "DriverId",
    "TeamId",
    "TeamName",
    "GridPosition",
    "Position",
    "ClassifiedPosition",
    "Points",
]
WEATHER_COLUMNS = ["wx_air_temp", "wx_track_temp", "wx_humidity", "wx_wind_speed", "wx_rain_frac"]
LAP_COLUMNS = [
    "Driver",
    "LapTime",
    "IsAccurate",
    "TrackStatus",
    "PitInTime",
    "PitOutTime",
    "Deleted",
]


def dnf_flag(classified: pd.Series) -> pd.Series:
    """True when the driver was not classified as a finisher ('R', 'D', 'W', missing...)."""
    return ~classified.astype(str).str.fullmatch(r"\d+")


def pace_gap_pct(laps: pd.DataFrame) -> pd.Series:
    """Each driver's median clean-lap time as a percentage slower than the field median.

    Clean = timed, flagged accurate, green flag (track status '1'), not a pit in/out lap, not
    deleted. Fuel load and tyre age are ignored, so this is a rough pace figure, not a true one.
    Returns an empty Series when the lap data cannot support a comparison.
    """
    if laps.empty:
        return pd.Series(dtype=float)
    deleted = laps["Deleted"].astype(str).eq("True")
    clean = (
        laps["LapTime"].notna()
        & laps["IsAccurate"].astype(bool)
        & laps["TrackStatus"].astype(str).eq("1")
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & ~deleted
    )
    seconds = laps.loc[clean, "LapTime"].dt.total_seconds()
    per_driver = seconds.groupby(laps.loc[clean, "Driver"]).agg(["median", "size"])
    medians = per_driver.loc[per_driver["size"] >= MIN_VALID_LAPS, "median"]
    if len(medians) < MIN_DRIVERS_FOR_PACE:
        return pd.Series(dtype=float)
    return (medians / medians.median() - 1) * 100


def quali_gap_pct(qualifying: pd.DataFrame) -> pd.Series:
    """Each driver's best qualifying time as a percentage slower than pole. NaN with no time.

    A driver knocked out in Q1 only has a Q1 time, set on a different track state than pole, so
    gaps for slower drivers include some track evolution.
    """
    best = qualifying[["Q1", "Q2", "Q3"]].min(axis=1).dt.total_seconds()
    if best.notna().sum() == 0:
        return pd.Series(np.nan, index=qualifying.index)
    return (best / best.min() - 1) * 100


def weather_summary(weather: pd.DataFrame) -> dict[str, float]:
    """Race-level weather: means over the session, and the share of readings with rain."""
    if weather.empty:
        return dict.fromkeys(WEATHER_COLUMNS, np.nan)
    return {
        "wx_air_temp": float(weather["AirTemp"].mean()),
        "wx_track_temp": float(weather["TrackTemp"].mean()),
        "wx_humidity": float(weather["Humidity"].mean()),
        "wx_wind_speed": float(weather["WindSpeed"].mean()),
        "wx_rain_frac": float(weather["Rainfall"].astype(float).mean()),
    }


def _files(table: str, slug: str) -> list:
    return sorted(raw_dir().glob(f"{table}/year=*/round=*/{slug}.parquet"))


def _load(table: str, slug: str, columns: list[str]) -> pd.DataFrame:
    files = _files(table, slug)
    if not files:
        raise FileNotFoundError(f"No {slug} {table} ingested; run `python -m ml.ingest` first.")
    return pd.concat((pd.read_parquet(f, columns=columns) for f in files), ignore_index=True)


def _read_if_exists(table: str, year: int, rnd: int, slug: str, columns: list[str]) -> pd.DataFrame:
    path = raw_dir() / table / f"year={year}" / f"round={rnd:02d}" / f"{slug}.parquet"
    return (
        pd.read_parquet(path, columns=columns) if path.exists() else pd.DataFrame(columns=columns)
    )


def build_history() -> pd.DataFrame:
    """One row per driver per race, with that race's results, qualifying, pace and weather."""
    races = _load("results", "race", RACE_COLUMNS)
    races["dnf"] = dnf_flag(races["ClassifiedPosition"])
    races["TeamKey"] = races["TeamId"].map(team_key)

    events = load_events()[["Year", "RoundNumber", "CircuitId"]]
    races = races.merge(
        events, left_on=["Year", "Round"], right_on=["Year", "RoundNumber"], how="left"
    )
    unmatched = races.loc[races["CircuitId"].isna(), ["Year", "Round"]].drop_duplicates()
    if not unmatched.empty:
        raise ValueError(f"races with no calendar event: {unmatched.to_numpy().tolist()}")
    races = races.drop(columns=["RoundNumber", "ClassifiedPosition"])

    quali = _load(
        "results", "qualifying", ["Year", "Round", "Abbreviation", "Position", "Q1", "Q2", "Q3"]
    )
    quali["quali_gap_pct"] = quali.groupby(["Year", "Round"], group_keys=False)[
        ["Q1", "Q2", "Q3"]
    ].apply(quali_gap_pct)
    quali = quali.rename(columns={"Position": "QualiPosition"}).drop(columns=["Q1", "Q2", "Q3"])
    races = races.merge(quali, on=["Year", "Round", "Abbreviation"], how="left")

    pace_rows, weather_rows = [], []
    for year, rnd in races[["Year", "Round"]].drop_duplicates().itertuples(index=False):
        gap = pace_gap_pct(_read_if_exists("laps", year, rnd, "race", LAP_COLUMNS))
        pace_rows.append(
            pd.DataFrame(
                {
                    "Year": year,
                    "Round": rnd,
                    "Abbreviation": gap.index,
                    "pace_gap_pct": gap.to_numpy(),
                }
            )
        )
        weather = _read_if_exists(
            "weather",
            year,
            rnd,
            "race",
            ["AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Rainfall"],
        )
        weather_rows.append({"Year": year, "Round": rnd, **weather_summary(weather)})
    races = races.merge(
        pd.concat(pace_rows, ignore_index=True), on=["Year", "Round", "Abbreviation"], how="left"
    )
    races = races.merge(pd.DataFrame(weather_rows), on=["Year", "Round"], how="left")

    from ml.features.weekend import build_weekend_table  # late: weekend imports this module

    weekend = build_weekend_table(load_events(), races[["Year", "Round", "Abbreviation"]])
    races = races.merge(weekend, on=["Year", "Round", "Abbreviation"], how="left")

    quality = race_quality()
    races = races.merge(quality, on=["Year", "Round"], how="left")
    # Lap-derived figures are only trusted where the audit says the lap data is usable.
    races.loc[~races["laps_ok"].fillna(False).astype(bool), "pace_gap_pct"] = np.nan
    races["race_ok"] = races["results_ok"].fillna(False).astype(bool) & ~pd.Series(
        list(zip(races["Year"], races["Round"], strict=True)), index=races.index
    ).isin(NOT_A_RACE)
    races = races.drop(columns=["results_ok", "laps_ok"])
    # Order rows by a pre-race key, never by finishing position (row order must not carry the answer).
    return races.sort_values(["Year", "Round", "DriverId"]).reset_index(drop=True)


def race_quality() -> pd.DataFrame:
    """Year, Round, results_ok, laps_ok for every ingested race, straight from the data audit."""
    rows = []
    for marker in sorted((raw_dir() / "_done").glob("*_race.json")):
        audited = audit_session(marker)
        rows.append({k: audited[k] for k in ("Year", "Round", "results_ok", "laps_ok")})
    return pd.DataFrame(rows, columns=["Year", "Round", "results_ok", "laps_ok"])

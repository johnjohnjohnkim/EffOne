"""Incremental, idempotent ingestion of fastf1 sessions into Parquet.

Layout: <data>/raw/<table>/year=YYYY/round=RR/<session>.parquet
A session is complete once <data>/raw/_done/<year>_<round>_<session>.json exists; reruns skip it.

A table that the source genuinely lacks has no Parquet file; the done-marker lists it under
"missing". Which (year, round, session) may lack which table is an explicit allow-list
(KNOWN_MISSING). Any other missing table fails the session so it is retried, because fastf1 raises
the same error for a transient load problem as for absent data.
"""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import fastf1
import pandas as pd
from fastf1.exceptions import DataNotLoadedError

from ml.ingest.atomic import write_json_atomic, write_parquet_atomic
from ml.ingest.paths import cache_dir, raw_dir, slugify

log = logging.getLogger(__name__)

# fastf1 can lag behind a finished session, so wait before trying to load it.
SETTLE_TIME = timedelta(hours=6)

# Tables the source is known to lack. 2018 R14 (Italian GP): results and weather load, laps do not.
KNOWN_MISSING: Mapping[tuple[int, int, str], frozenset[str]] = MappingProxyType(
    {(2018, 14, "Race"): frozenset({"laps"})}
)


def _done_path(year: int, rnd: int, session: str) -> Path:
    return raw_dir() / "_done" / f"{year}_{rnd:02d}_{slugify(session)}.json"


def is_done(year: int, rnd: int, session: str) -> bool:
    return _done_path(year, rnd, session).exists()


def _table_path(table: str, year: int, rnd: int, session: str) -> Path:
    return raw_dir() / table / f"year={year}" / f"round={rnd:02d}" / f"{slugify(session)}.parquet"


def _prepare(df: pd.DataFrame, year: int, rnd: int, session: str) -> pd.DataFrame:
    df = pd.DataFrame(df).copy()
    df.insert(0, "Session", session)
    df.insert(0, "Round", rnd)
    df.insert(0, "Year", year)
    # Mixed-type object columns break pyarrow; stringify the ones that are not plain strings.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    return df


def _collect_tables(
    session: object, key: tuple[int, int, str]
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Read every table off a loaded session. Returns (tables, names of allowed-missing tables)."""
    getters = {
        "results": lambda: session.results,
        "laps": lambda: session.laps,
        "weather": lambda: session.weather_data,
        "track_status": lambda: session.track_status,
    }
    allowed = KNOWN_MISSING.get(key, frozenset())
    tables, missing = {}, []
    for name, get in getters.items():
        try:
            frame = get()
        except DataNotLoadedError:
            if name not in allowed:
                raise
            missing.append(name)
            continue
        # An empty table from a partial load looks like success but is not; retry it later.
        if len(frame) == 0 and name not in allowed:
            raise RuntimeError(f"{key}: {name} table is empty, will retry later")
        tables[name] = frame
    return tables, missing


def ingest_session(year: int, rnd: int, session_name: str) -> dict[str, int]:
    """Load one session and write its tables, then (last) its done-marker."""
    session = fastf1.get_session(year, rnd, session_name)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    # Validate every table before writing anything, so a failure leaves no partial session.
    tables, missing = _collect_tables(session, (year, rnd, session_name))

    counts: dict[str, int] = {}
    for name, df in tables.items():
        write_parquet_atomic(
            _prepare(df, year, rnd, session_name), _table_path(name, year, rnd, session_name)
        )
        counts[name] = len(df)
    for name in missing:
        counts[name] = 0
        _table_path(name, year, rnd, session_name).unlink(missing_ok=True)
        log.warning("%s R%s %s: known-missing table %s", year, rnd, session_name, name)

    write_json_atomic(
        {"counts": counts, "missing": missing, "ingested_at": datetime.now(UTC).isoformat()},
        _done_path(year, rnd, session_name),
    )
    return counts


def finished_sessions(
    schedule: pd.DataFrame, now: datetime, only: set[str] | None = None
) -> list[tuple[int, str]]:
    """Sessions whose scheduled START is at least SETTLE_TIME before `now`, as (round, name).

    Start time, not end time, is used, so a long race is tried a few hours after it finishes;
    a session that is not ready yet fails and is retried on the next run."""
    out = []
    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        for i in range(1, 6):
            name, date = event.get(f"Session{i}"), event.get(f"Session{i}DateUtc")
            if not name or pd.isna(date) or (only and name not in only):
                continue
            ts = pd.Timestamp(date)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            if ts + SETTLE_TIME <= now:
                out.append((rnd, name))
    return out


def pending_sessions(year: int, only: set[str] | None = None) -> list[tuple[int, str]]:
    """Finished, not-yet-ingested sessions for a season, as (round, session name)."""
    fastf1.Cache.enable_cache(str(cache_dir()))
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    done = finished_sessions(schedule, datetime.now(UTC), only)
    return [(rnd, name) for rnd, name in done if not is_done(year, rnd, name)]

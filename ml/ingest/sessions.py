"""Incremental, idempotent ingestion of fastf1 sessions into Parquet.

Layout: <data>/raw/<table>/year=YYYY/round=RR/<session>.parquet
A session is complete once <data>/raw/_done/<year>_<round>_<session>.json exists; reruns skip it.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone

import fastf1
import pandas as pd

from ml.ingest.paths import cache_dir, raw_dir

log = logging.getLogger(__name__)

# fastf1 can lag behind a finished session, so wait before trying to load it.
SETTLE_TIME = timedelta(hours=6)
TABLES = ("results", "laps", "weather", "track_status")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _done_path(year: int, rnd: int, session: str):
    path = raw_dir() / "_done"
    path.mkdir(exist_ok=True)
    return path / f"{year}_{rnd:02d}_{_slug(session)}.json"


def is_done(year: int, rnd: int, session: str) -> bool:
    return _done_path(year, rnd, session).exists()


def _write(df: pd.DataFrame, table: str, year: int, rnd: int, session: str) -> int:
    out = raw_dir() / table / f"year={year}" / f"round={rnd:02d}"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(df).copy()
    df.insert(0, "Session", session)
    df.insert(0, "Round", rnd)
    df.insert(0, "Year", year)
    # Mixed-type object columns break pyarrow; stringify the ones that are not plain strings.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    df.to_parquet(out / f"{_slug(session)}.parquet", index=False)
    return len(df)


def ingest_session(year: int, rnd: int, session_name: str) -> dict:
    """Load one session and write its tables. Marks it done only if everything was written."""
    session = fastf1.get_session(year, rnd, session_name)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    tables = {
        "results": session.results,
        "laps": session.laps,
        "weather": session.weather_data,
        "track_status": session.track_status,
    }
    counts = {name: _write(df, name, year, rnd, session_name) for name, df in tables.items()}
    if counts["results"] == 0:
        raise RuntimeError(f"{year} R{rnd} {session_name}: no results yet, will retry later")
    _done_path(year, rnd, session_name).write_text(
        json.dumps({"counts": counts, "ingested_at": datetime.now(timezone.utc).isoformat()})
    )
    return counts


def pending_sessions(year: int, only: set[str] | None = None) -> list[tuple[int, str]]:
    """Finished, not-yet-ingested sessions for a season, as (round, session name)."""
    fastf1.Cache.enable_cache(str(cache_dir()))
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    now = datetime.now(timezone.utc)
    pending = []
    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        for i in range(1, 6):
            name, date = event.get(f"Session{i}"), event.get(f"Session{i}DateUtc")
            if not name or pd.isna(date):
                continue
            if only and name not in only:
                continue
            if pd.Timestamp(date).tz_localize("UTC") + SETTLE_TIME > now:
                continue
            if not is_done(year, rnd, name):
                pending.append((rnd, name))
    return pending

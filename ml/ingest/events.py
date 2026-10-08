"""Ingest the race calendar (one row per Grand Prix weekend) so results can be tied to a circuit.

python -m ml.ingest.events [--years 2018 2019 ...]  ->  data/raw/events/year=YYYY.parquet

The circuit is identified by the schedule's `Location` (e.g. "Silverstone", "Monza"). Different
layouts at the same venue (such as the 2020 Bahrain outer circuit) are therefore treated as one
circuit; that is a deliberate simplification.
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import fastf1
import pandas as pd

from ml.ingest.atomic import write_parquet_atomic
from ml.ingest.circuit_ids import circuit_id
from ml.ingest.paths import cache_dir, raw_dir

log = logging.getLogger("ml.ingest.events")

SESSION_COLUMNS = [f"Session{i}" for i in range(1, 6)] + [f"Session{i}DateUtc" for i in range(1, 6)]
COLUMNS = [
    "RoundNumber",
    "Country",
    "Location",
    "EventName",
    "OfficialEventName",
    "EventDate",
    "EventFormat",
    *SESSION_COLUMNS,
]


def events_path(year: int) -> Path:
    path = raw_dir() / "events"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"year={year}.parquet"


def ingest_events(year: int) -> int:
    fastf1.Cache.enable_cache(str(cache_dir()))
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    df = pd.DataFrame(schedule)[COLUMNS].copy()
    df.insert(0, "Year", year)
    df["CircuitId"] = df["Location"].map(circuit_id)
    write_parquet_atomic(df, events_path(year))
    return len(df)


def load_events() -> pd.DataFrame:
    files = sorted((raw_dir() / "events").glob("year=*.parquet"))
    if not files:
        raise FileNotFoundError("No events ingested; run `python -m ml.ingest.events`.")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years", type=int, nargs="+", default=list(range(2018, datetime.now(UTC).year + 1))
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for year in args.years:
        log.info("%s: %d events", year, ingest_events(year))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI: python -m ml.ingest [--years 2018 2019 ...] [--sessions Race Qualifying] [--limit N] [--dry-run]"""

import argparse
import logging
import time
from collections import Counter
from datetime import UTC, datetime
from functools import partial

import fastf1
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError

from ml.ingest.atomic import sweep_stale_tmp
from ml.ingest.paths import cache_dir, data_dir, raw_dir
from ml.ingest.sessions import ingest_session, pending_sessions

log = logging.getLogger("ml.ingest")

# fastf1 allows 500 API calls per hour; when it trips, wait for the window to roll over.
RATE_LIMIT_WAIT = 300


def with_rate_limit_wait(fn):
    while True:
        try:
            return fn()
        except RateLimitExceededError:
            log.info("fastf1 rate limit reached; sleeping %ds", RATE_LIMIT_WAIT)
            time.sleep(RATE_LIMIT_WAIT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years", type=int, nargs="+", default=list(range(2018, datetime.now(UTC).year + 1))
    )
    parser.add_argument(
        "--sessions", nargs="+", help='Session names, e.g. Race Qualifying "Practice 2"'
    )
    parser.add_argument("--limit", type=int, help="Stop after N sessions (for testing)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List pending sessions and exit (exit code 1 if any are pending). Fetches the "
        "season calendars only; downloads no session data and deletes nothing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fastf1.Cache.enable_cache(str(cache_dir()))
    log.info("data dir: %s", data_dir())

    if not args.dry_run:
        swept = sweep_stale_tmp(raw_dir())
        if swept:
            log.info("removed %d stale temp files", swept)

    only = set(args.sessions) if args.sessions else None
    done = failed = pending_total = 0
    failures: Counter[str] = Counter()
    for year in args.years:
        todo = with_rate_limit_wait(partial(pending_sessions, year, only))
        log.info("%s: %d sessions to ingest", year, len(todo))
        if args.dry_run:
            pending_total += len(todo)
            for rnd, name in todo:
                log.info("  pending: %s R%02d %s", year, rnd, name)
            continue
        for rnd, name in todo:
            if args.limit and done >= args.limit:
                log.info("limit reached")
                return 0
            try:
                counts = with_rate_limit_wait(partial(ingest_session, year, rnd, name))
                done += 1
                log.info("%s R%02d %s ok %s", year, rnd, name, counts)
            except Exception as exc:
                failed += 1
                failures[type(exc).__name__] += 1
                log.warning("%s R%02d %s FAILED: %s: %s", year, rnd, name, type(exc).__name__, exc)
                if not isinstance(exc, DataNotLoadedError | RuntimeError):
                    log.warning("traceback for %s R%02d %s", year, rnd, name, exc_info=True)
    if args.dry_run:
        log.info("dry run: %d sessions pending", pending_total)
        return 1 if pending_total else 0
    log.info(
        "finished: %d ingested, %d failed %s (rerun to retry failures)",
        done,
        failed,
        dict(failures),
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

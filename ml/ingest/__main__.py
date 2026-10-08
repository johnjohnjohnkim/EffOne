"""CLI: python -m ml.ingest --years 2018 2019 ... [--sessions Race Qualifying] [--limit N]"""

import argparse
import logging
import time
from datetime import date

import fastf1
from fastf1.exceptions import RateLimitExceededError

from ml.ingest.paths import cache_dir, data_dir
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
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2018, date.today().year + 1)))
    parser.add_argument("--sessions", nargs="+", help='Session names, e.g. Race Qualifying "Practice 2"')
    parser.add_argument("--limit", type=int, help="Stop after N sessions (for testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fastf1.Cache.enable_cache(str(cache_dir()))
    log.info("data dir: %s", data_dir())

    only = set(args.sessions) if args.sessions else None
    done = failed = 0
    for year in args.years:
        todo = with_rate_limit_wait(lambda: pending_sessions(year, only))
        log.info("%s: %d sessions to ingest", year, len(todo))
        for rnd, name in todo:
            if args.limit and done >= args.limit:
                log.info("limit reached")
                return 0
            try:
                counts = with_rate_limit_wait(lambda: ingest_session(year, rnd, name))
                done += 1
                log.info("%s R%02d %s ok %s", year, rnd, name, counts)
            except Exception as exc:  # keep going; the session is retried on the next run
                failed += 1
                log.warning("%s R%02d %s FAILED: %s", year, rnd, name, exc)
                time.sleep(2)
    log.info("finished: %d ingested, %d failed (rerun to retry failures)", done, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

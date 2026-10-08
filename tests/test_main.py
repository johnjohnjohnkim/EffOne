"""The ingest CLI: dry-run contract, startup sweep, failure accounting, --limit, rate-limit wait."""

import logging
import os
import time

import pytest
from fastf1.exceptions import RateLimitExceededError

from ml.ingest import __main__ as cli
from ml.ingest.atomic import STALE_TMP_SECONDS
from ml.ingest.paths import raw_dir


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("EFFONE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli.fastf1.Cache, "enable_cache", lambda path: None)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    caplog.set_level(logging.INFO, logger="ml.ingest")


def run(monkeypatch, *argv, pending=None, ingest=None):
    pending = pending or {}
    monkeypatch.setattr(cli, "pending_sessions", lambda year, only: pending.get(year, []))
    monkeypatch.setattr(cli, "ingest_session", ingest or (lambda year, rnd, name: {"results": 20}))
    monkeypatch.setattr("sys.argv", ["ml.ingest", *argv])
    return cli.main()


def make_stale_tmp():
    path = raw_dir() / "results" / "year=2020" / "x.parquet.123.tmp"
    path.parent.mkdir(parents=True)
    path.write_text("x")
    old = time.time() - 2 * STALE_TMP_SECONDS
    os.utime(path, (old, old))
    return path


def test_dry_run_with_nothing_pending_logs_zero_and_exits_0(monkeypatch, caplog):
    assert run(monkeypatch, "--years", "2024", "--dry-run") == 0
    assert "2024: 0 sessions to ingest" in caplog.text
    assert "dry run: 0 sessions pending" in caplog.text


def test_dry_run_with_pending_sessions_lists_them_exits_1_and_ingests_nothing(monkeypatch, caplog):
    calls = []
    code = run(
        monkeypatch,
        "--years",
        "2024",
        "--dry-run",
        pending={2024: [(3, "Race"), (3, "Qualifying")]},
        ingest=lambda *a: calls.append(a),
    )
    assert code == 1 and calls == []
    assert "pending: 2024 R03 Race" in caplog.text
    assert "dry run: 2 sessions pending" in caplog.text


def test_dry_run_deletes_nothing_but_a_real_run_sweeps_stale_temp_files(monkeypatch):
    stale = make_stale_tmp()
    run(monkeypatch, "--years", "2024", "--dry-run")
    assert stale.exists()
    run(monkeypatch, "--years", "2024")
    assert not stale.exists()


def test_run_ingests_every_pending_session_and_exits_0(monkeypatch):
    calls = []
    code = run(
        monkeypatch,
        "--years",
        "2023",
        "2024",
        pending={2023: [(1, "Race")], 2024: [(1, "Race"), (2, "Race")]},
        ingest=lambda y, r, n: calls.append((y, r, n)) or {"results": 20},
    )
    assert code == 0 and calls == [(2023, 1, "Race"), (2024, 1, "Race"), (2024, 2, "Race")]


def test_failures_are_counted_by_type_and_do_not_stop_the_run(monkeypatch, caplog):
    def ingest(year, rnd, name):
        if rnd == 1:
            raise RuntimeError("no results yet")
        return {"results": 20}

    code = run(
        monkeypatch, "--years", "2024", pending={2024: [(1, "Race"), (2, "Race")]}, ingest=ingest
    )
    assert code == 1
    assert "1 ingested, 1 failed {'RuntimeError': 1}" in caplog.text
    assert "2024 R02 Race ok" in caplog.text


def test_limit_stops_after_n_successful_sessions(monkeypatch):
    calls = []
    run(
        monkeypatch,
        "--years",
        "2024",
        "--limit",
        "2",
        pending={2024: [(i, "Race") for i in range(1, 6)]},
        ingest=lambda y, r, n: calls.append(r) or {"results": 20},
    )
    assert calls == [1, 2]


def test_rate_limit_error_sleeps_and_retries_the_same_call(caplog):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimitExceededError("any API: 500 calls/h")
        return "done"

    assert cli.with_rate_limit_wait(flaky) == "done" and len(attempts) == 3
    assert caplog.text.count("rate limit reached") == 2

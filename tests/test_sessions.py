"""ingest_session / pending_sessions against a fake fastf1, so no network is needed."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from fastf1.exceptions import DataNotLoadedError

from ml.ingest import atomic, sessions
from ml.ingest.paths import raw_dir


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFONE_DATA_DIR", str(tmp_path))


class FakeSession:
    """Mimics fastf1: every table raises DataNotLoadedError until load() asked for it."""

    def __init__(self, missing=(), empty=(), results_rows=20):
        self._missing = set(missing)
        self._empty = set(empty)
        self._results_rows = results_rows
        self.load_kwargs = None

    def load(self, **kwargs):
        self.load_kwargs = kwargs

    def _get(self, name, frame, needs):
        if self.load_kwargs is None or not self.load_kwargs.get(needs, False):
            raise DataNotLoadedError("not loaded")
        if name in self._missing:
            raise DataNotLoadedError("source has no such table")
        return frame.iloc[0:0] if name in self._empty else frame

    @property
    def results(self):
        names = [f"D{i}" for i in range(self._results_rows)]
        frame = pd.DataFrame({"Abbreviation": names, "Position": range(1, len(names) + 1)})
        return self._get("results", frame, "laps")  # results arrive with the main load

    @property
    def laps(self):
        frame = pd.DataFrame({"Driver": ["D0"], "LapTime": pd.to_timedelta([90.0], unit="s")})
        return self._get("laps", frame, "laps")

    @property
    def weather_data(self):
        return self._get("weather", pd.DataFrame({"AirTemp": [20.0, 21.0]}), "weather")

    @property
    def track_status(self):
        return self._get("track_status", pd.DataFrame({"Status": ["1"]}), "laps")


def use_fake(monkeypatch, fake):
    monkeypatch.setattr(sessions.fastf1, "get_session", lambda year, rnd, name: fake)
    return fake


def files():
    return sorted(str(p.relative_to(raw_dir())) for p in raw_dir().rglob("*") if p.is_file())


def test_ingest_loads_laps_and_weather_then_writes_tables_and_a_marker(monkeypatch):
    fake = use_fake(monkeypatch, FakeSession())
    counts = sessions.ingest_session(2024, 3, "Race")
    assert fake.load_kwargs == {
        "laps": True,
        "telemetry": False,
        "weather": True,
        "messages": False,
    }
    assert counts == {"results": 20, "laps": 1, "weather": 2, "track_status": 1}
    assert sessions.is_done(2024, 3, "Race")
    assert len(files()) == 5  # four tables + marker
    marker = json.loads(sessions._done_path(2024, 3, "Race").read_text())
    assert marker["missing"] == [] and marker["counts"] == counts
    assert not any(f.endswith(".tmp") for f in files())


def test_running_ingest_twice_adds_no_new_files(monkeypatch):
    use_fake(monkeypatch, FakeSession())
    sessions.ingest_session(2024, 3, "Race")
    first = files()
    sessions.ingest_session(2024, 3, "Race")
    assert files() == first


def schedule_for(*sessions_: tuple[str, str]) -> pd.DataFrame:
    row = {"RoundNumber": 1}
    for i, (name, date) in enumerate(sessions_, start=1):
        row[f"Session{i}"] = name
        row[f"Session{i}DateUtc"] = pd.Timestamp(date)
    return pd.DataFrame([row])


NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)


def test_finished_sessions_applies_the_settle_time_and_the_only_filter():
    schedule = schedule_for(
        ("Practice 1", "2026-05-30 10:00"),
        (
            "Qualifying",
            "2026-06-01 08:00",
        ),  # ended 4 hours ago: still inside the 6 hour settle time
        ("Race", "2026-06-02 13:00"),  # in the future
    )
    assert sessions.finished_sessions(schedule, NOW) == [(1, "Practice 1")]
    assert sessions.finished_sessions(schedule, NOW, only={"Race"}) == []
    just_before = datetime(2026, 6, 1, 13, 59, tzinfo=UTC)
    assert sessions.finished_sessions(schedule, just_before) == [(1, "Practice 1")]
    exactly = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)  # start + 6h == now counts as settled
    assert sessions.finished_sessions(schedule, exactly) == [(1, "Practice 1"), (1, "Qualifying")]


def test_finished_sessions_skips_blank_names_and_dates_and_accepts_tz_aware_dates():
    schedule = pd.DataFrame(
        [
            {
                "RoundNumber": 2,
                "Session1": "Practice 1",
                "Session1DateUtc": pd.Timestamp("2026-05-01 10:00", tz="UTC"),
                "Session2": "",
                "Session2DateUtc": pd.Timestamp("2026-05-01 12:00"),
                "Session3": "Race",
                "Session3DateUtc": pd.NaT,
            }
        ]
    )
    assert sessions.finished_sessions(schedule, NOW) == [(2, "Practice 1")]


def test_pending_sessions_skips_done_sessions(monkeypatch):
    schedule = schedule_for(("Qualifying", "2020-03-01"), ("Race", "2020-03-02"))
    monkeypatch.setattr(sessions.fastf1, "get_event_schedule", lambda *a, **k: schedule)
    monkeypatch.setattr(sessions.fastf1.Cache, "enable_cache", lambda path: None)
    assert sessions.pending_sessions(2020) == [(1, "Qualifying"), (1, "Race")]
    use_fake(monkeypatch, FakeSession())
    sessions.ingest_session(2020, 1, "Race")
    assert sessions.pending_sessions(2020) == [(1, "Qualifying")]


def test_known_missing_table_is_recorded_not_fatal(monkeypatch):
    use_fake(monkeypatch, FakeSession(missing={"laps"}))
    counts = sessions.ingest_session(2018, 14, "Race")  # on the KNOWN_MISSING allow-list
    assert counts["laps"] == 0 and counts["results"] == 20
    marker = json.loads(sessions._done_path(2018, 14, "Race").read_text())
    assert marker["missing"] == ["laps"]
    assert not any("laps" in f and f.endswith(".parquet") for f in files())


def test_the_allow_list_is_exact_and_immutable():
    assert dict(sessions.KNOWN_MISSING) == {(2018, 14, "Race"): frozenset({"laps"})}
    with pytest.raises(TypeError):
        sessions.KNOWN_MISSING[(2024, 1, "Race")] = frozenset({"laps"})


def test_unexpected_missing_table_fails_and_writes_nothing(monkeypatch):
    use_fake(monkeypatch, FakeSession(missing={"laps"}))
    with pytest.raises(DataNotLoadedError):
        sessions.ingest_session(2024, 3, "Race")  # not on the allow-list: could be transient
    assert files() == []
    assert not sessions.is_done(2024, 3, "Race")


def test_missing_results_always_fail(monkeypatch):
    use_fake(monkeypatch, FakeSession(missing={"results"}))
    with pytest.raises(DataNotLoadedError):
        sessions.ingest_session(2018, 14, "Race")
    assert files() == []


@pytest.mark.parametrize("table", ["results", "laps", "weather", "track_status"])
def test_an_empty_table_fails_the_session_so_it_is_retried(monkeypatch, table):
    use_fake(monkeypatch, FakeSession(empty={table}))
    with pytest.raises(RuntimeError, match="empty"):
        sessions.ingest_session(2024, 3, "Race")
    assert files() == []


def test_zero_driver_results_fail_without_writing_anything(monkeypatch):
    use_fake(monkeypatch, FakeSession(results_rows=0))
    with pytest.raises(RuntimeError):
        sessions.ingest_session(2024, 3, "Race")
    assert files() == []


def test_crash_while_writing_leaves_no_marker_and_a_rerun_repairs_it(monkeypatch):
    """The marker is the last write, so a crash mid-session leaves the session pending."""
    use_fake(monkeypatch, FakeSession())
    real = atomic.write_parquet_atomic
    calls = []

    def flaky(df, path):
        calls.append(path)
        if len(calls) == 3:
            raise OSError("disk full")
        real(df, path)

    monkeypatch.setattr(sessions, "write_parquet_atomic", flaky)
    with pytest.raises(OSError):
        sessions.ingest_session(2024, 3, "Race")
    assert not sessions.is_done(2024, 3, "Race")
    monkeypatch.setattr(sessions, "write_parquet_atomic", real)
    sessions.ingest_session(2024, 3, "Race")
    assert sessions.is_done(2024, 3, "Race")
    assert len(files()) == 5 and not any(f.endswith(".tmp") for f in files())

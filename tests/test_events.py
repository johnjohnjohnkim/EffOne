import pandas as pd
import pytest

from ml.ingest import events
from ml.ingest.paths import raw_dir


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFONE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events.fastf1.Cache, "enable_cache", lambda path: None)


def fake_schedule(locations):
    rows = []
    for i, loc in enumerate(locations, start=1):
        row = {
            "RoundNumber": i,
            "Country": "X",
            "Location": loc,
            "EventName": f"{loc} Grand Prix",
            "OfficialEventName": f"Official {loc}",
            "EventDate": pd.Timestamp("2024-03-01"),
            "EventFormat": "conventional",
            "extra_column_we_do_not_keep": 1,
        }
        for s in range(1, 6):
            row[f"Session{s}"] = f"S{s}"
            row[f"Session{s}DateUtc"] = pd.Timestamp("2024-03-01")
        rows.append(row)
    return pd.DataFrame(rows)


def test_ingest_events_writes_one_file_per_year_with_circuit_ids(monkeypatch):
    monkeypatch.setattr(
        events.fastf1,
        "get_event_schedule",
        lambda *a, **k: fake_schedule(["Monte Carlo", "São Paulo"]),
    )
    assert events.ingest_events(2024) == 2
    loaded = events.load_events()
    assert loaded["CircuitId"].tolist() == ["monaco", "sao_paulo"]
    assert (loaded["Year"] == 2024).all()
    assert "extra_column_we_do_not_keep" not in loaded.columns
    assert set(events.SESSION_COLUMNS) <= set(loaded.columns)
    assert [p.name for p in (raw_dir() / "events").iterdir()] == ["year=2024.parquet"]


def test_rerunning_replaces_the_year_instead_of_appending(monkeypatch):
    monkeypatch.setattr(
        events.fastf1, "get_event_schedule", lambda *a, **k: fake_schedule(["Monza"])
    )
    events.ingest_events(2024)
    events.ingest_events(2024)
    assert len(events.load_events()) == 1


def test_loading_with_nothing_ingested_is_a_clear_error_and_creates_no_directories(tmp_path):
    with pytest.raises(FileNotFoundError, match="No events"):
        events.load_events()
    assert list(tmp_path.iterdir()) == []

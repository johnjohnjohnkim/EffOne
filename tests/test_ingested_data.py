"""Checks on the REAL ingested data (data/raw): Milestone 1's done-check.

Skipped only when nothing at all is ingested (a fresh clone). A partial backfill FAILS these
tests on purpose, so a green run means the data is complete. They read the live data directory,
so they are marked `realdata`; exclude them with `pytest -m "not realdata"`.
"""

import json
import re
from datetime import UTC, datetime

import pyarrow.parquet as pq
import pytest

from ml.ingest.atomic import stale_tmp_files
from ml.ingest.audit import MAX_RESULT_ROWS, MIN_RESULT_ROWS
from ml.ingest.events import load_events
from ml.ingest.paths import raw_dir
from ml.ingest.sessions import finished_sessions, is_done

pytestmark = pytest.mark.realdata

# Seasons whose results are final (verified against the fastf1 calendar on 2026-10-08).
EXPECTED_RACES = {2018: 21, 2019: 21, 2020: 17, 2021: 22, 2022: 22, 2023: 22, 2024: 24, 2025: 24}
CORE_SESSIONS = {"Race", "Qualifying", "Sprint", "Sprint Qualifying", "Sprint Shootout"}

# Every circuit id in the completed seasons, reviewed by hand. This is frozen history, so a new id
# here means a spelling drifted or a venue was misread, and someone should look.
CIRCUITS_2018_2025 = {
    "austin", "baku", "barcelona", "budapest", "hockenheim", "imola", "istanbul", "jeddah",
    "las_vegas", "le_castellet", "lusail", "marina_bay", "melbourne", "mexico_city", "miami",
    "monaco", "montreal", "monza", "mugello", "nurburgring", "portimao", "sakhir", "sao_paulo",
    "shanghai", "silverstone", "sochi", "spa_francorchamps", "spielberg", "suzuka", "yas_marina",
    "zandvoort",
}  # fmt: skip


def race_files():
    return sorted(raw_dir().glob("results/year=*/round=*/race.parquet"))


@pytest.fixture(autouse=True)
def require_ingested_data():
    if not race_files():
        pytest.skip("no ingested data (fresh clone)")


def ingested_rounds(year):
    return {
        int(p.parent.name.split("=")[1])
        for p in race_files()
        if p.parent.parent.name == f"year={year}"
    }


def test_event_calendar_has_the_expected_number_of_races():
    counts = load_events().groupby("Year").size()
    for year, expected in EXPECTED_RACES.items():
        assert counts[year] == expected, year


@pytest.mark.parametrize("year", sorted(EXPECTED_RACES))
def test_every_completed_season_has_exactly_its_calendar_races(year):
    events = load_events()
    calendar = set(events.loc[events["Year"] == year, "RoundNumber"])
    got = ingested_rounds(year)
    assert got == calendar, (
        f"{year}: missing {sorted(calendar - got)}, extra {sorted(got - calendar)}"
    )


@pytest.mark.parametrize("year", sorted(EXPECTED_RACES))
def test_every_core_session_of_a_completed_season_is_ingested(year):
    """Races, qualifying and sprint sessions: the same sessions `ml.ingest` would call pending."""
    events = load_events()
    schedule = events[events["Year"] == year]
    expected = finished_sessions(schedule, datetime.now(UTC), only=CORE_SESSIONS)
    missing = [(rnd, name) for rnd, name in expected if not is_done(year, rnd, name)]
    assert expected and not missing, f"{year}: core sessions not ingested: {missing}"


def test_completed_seasons_use_exactly_the_frozen_circuit_ids():
    events = load_events()
    ids = set(events.loc[events["Year"] <= 2025, "CircuitId"])
    assert ids == CIRCUITS_2018_2025, (
        f"new: {sorted(ids - CIRCUITS_2018_2025)}, gone: {sorted(CIRCUITS_2018_2025 - ids)}"
    )


def test_every_event_in_every_year_has_a_well_formed_circuit_id():
    """Later or current calendars are still changing, so only the shape is checked."""
    ids = load_events()["CircuitId"]
    assert ids.notna().all() and ids.str.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*").all()
    assert not re.search(r"[^a-z0-9_]", "".join(ids))


def test_2020_bahrain_double_header_shares_one_circuit_id():
    events = load_events()
    sakhir_2020 = events[(events["Year"] == 2020) & (events["CircuitId"] == "sakhir")]
    assert len(sakhir_2020) == 2


def test_each_ingested_race_has_a_sane_unique_driver_list():
    for f in race_files():
        table = pq.read_table(f, columns=["Abbreviation"]).to_pandas()
        where = f"{f.parent.parent.name}/{f.parent.name}"
        assert MIN_RESULT_ROWS <= len(table) <= MAX_RESULT_ROWS, f"{where}: {len(table)} rows"
        assert table["Abbreviation"].is_unique, where


def test_markers_match_the_files_on_disk():
    """Every counted table exists with that many rows; tables marked missing have no file."""
    for marker in sorted((raw_dir() / "_done").glob("*.json")):
        year, rnd, slug = marker.stem.split("_", 2)
        info = json.loads(marker.read_text(encoding="utf-8"))
        for table, expected in info["counts"].items():
            path = raw_dir() / table / f"year={year}" / f"round={rnd}" / f"{slug}.parquet"
            if table in info.get("missing", []):  # early markers predate this field
                assert not path.exists(), marker.name
            else:
                assert path.exists(), f"{marker.name}: {table} file missing"
                assert pq.ParquetFile(path).metadata.num_rows == expected, f"{marker.name}: {table}"


def test_no_stale_temp_files_from_interrupted_writes():
    """Only old temp files count; a fresh one may belong to a backfill that is writing right now."""
    assert stale_tmp_files(raw_dir()) == []

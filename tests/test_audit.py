import json

import pandas as pd
import pytest

from ml.ingest.audit import audit_session, run_audit, session_kind
from ml.ingest.paths import raw_dir


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFONE_DATA_DIR", str(tmp_path))


def write_session(
    slug="race", drivers=20, drivers_with_laps=None, untimed_frac=0.0, no_laps=False, laps_each=10
):
    """Write a synthetic session and its done-marker; return the marker path."""
    names = [f"D{i:02d}" for i in range(drivers)]
    results = pd.DataFrame(
        {
            "Abbreviation": names,
            "Position": [float(i + 1) for i in range(drivers)],
            "GridPosition": [float(i + 1) for i in range(drivers)],
            "Laps": 50.0,
        }
    )
    lap_names = names if drivers_with_laps is None else names[:drivers_with_laps]
    rows = [] if no_laps else [(n, lap) for n in lap_names for lap in range(1, laps_each + 1)]
    laps = pd.DataFrame(rows, columns=["Driver", "LapNumber"])
    laps["LapTime"] = pd.to_timedelta([90.0] * len(laps), unit="s")
    laps.loc[laps.index[: int(len(laps) * untimed_frac)], "LapTime"] = pd.NaT
    for table, df in (("results", results), ("laps", laps)):
        out = raw_dir() / table / "year=2024" / "round=01"
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / f"{slug}.parquet", index=False)
    marker_dir = raw_dir() / "_done"
    marker_dir.mkdir(exist_ok=True)
    marker = marker_dir / f"2024_01_{slug}.json"
    marker.write_text(json.dumps({"counts": {"weather": 50, "track_status": 3}, "missing": []}))
    return marker


def flags(row) -> set[str]:
    return set(filter(None, row["flags"].split(",")))


def rewrite_results(slug, **changes):
    path = raw_dir() / "results" / "year=2024" / "round=01" / f"{slug}.parquet"
    df = pd.read_parquet(path)
    for column, edit in changes.items():
        df[column] = edit(df[column])
    df.to_parquet(path, index=False)


def test_clean_race_has_no_flags():
    row = audit_session(write_session())
    assert flags(row) == set() and row["results_ok"] and row["laps_ok"] and row["weather_ok"]


def test_race_without_laps_is_flagged_but_results_stay_usable():
    row = audit_session(write_session(no_laps=True))
    assert flags(row) == {"no_laps"}
    assert row["results_ok"] and not row["laps_ok"]
    assert pd.isna(row["laptime_null_frac"])


def test_many_untimed_laps_make_a_race_unusable_for_laps_but_not_qualifying():
    race = audit_session(write_session("race", untimed_frac=0.5))
    quali = audit_session(write_session("qualifying", untimed_frac=0.5))
    assert flags(race) == {"many_null_laptimes"} and not race["laps_ok"]
    assert flags(quali) == set() and quali["laps_ok"]


def test_a_few_untimed_laps_are_informational_and_keep_laps_usable():
    row = audit_session(write_session("race", untimed_frac=0.2))  # e.g. red-flag laps
    assert flags(row) == {"some_untimed_laps"} and row["laps_ok"]


def test_an_untimed_first_lap_alone_is_not_flagged():
    marker = write_session("sprint", laps_each=19)  # sprints: lap 1 untimed for every driver
    path = raw_dir() / "laps" / "year=2024" / "round=01" / "sprint.parquet"
    laps = pd.read_parquet(path)
    laps.loc[laps["LapNumber"] == 1, "LapTime"] = pd.NaT
    laps.to_parquet(path, index=False)
    row = audit_session(marker)
    assert flags(row) == set() and row["laptime_null_frac"] == 0.0


def test_unusual_number_of_result_rows_makes_results_unusable():
    row = audit_session(write_session(drivers=10))
    assert "unusual_result_rows" in flags(row) and not row["results_ok"]


def test_duplicate_drivers_are_flagged():
    marker = write_session()
    rewrite_results("race", Abbreviation=lambda s: s.where(s != "D01", "D00"))
    row = audit_session(marker)
    assert "duplicate_drivers" in flags(row) and not row["results_ok"]


def test_race_with_two_winners_or_none_is_flagged():
    marker = write_session()
    rewrite_results("race", Position=lambda s: s.where(s != 2.0, 1.0))
    assert "no_single_winner" in flags(audit_session(marker))
    rewrite_results("race", Position=lambda s: s.where(s > 1.0))
    assert "no_single_winner" in flags(audit_session(marker))


def test_many_missing_positions_are_flagged_for_races():
    marker = write_session()
    rewrite_results("race", Position=lambda s: s.where(s > 6.0))  # 6 of 20 become NaN
    assert "many_null_positions" in flags(audit_session(marker))


def test_laps_missing_for_several_drivers_is_flagged():
    row = audit_session(write_session(drivers_with_laps=12))
    assert flags(row) == {"laps_missing_drivers"} and not row["laps_ok"]


def test_several_lap_problems_are_all_reported():
    row = audit_session(write_session(drivers_with_laps=12, untimed_frac=0.5))
    assert flags(row) == {"many_null_laptimes", "laps_missing_drivers"}


def test_unreadable_marker_is_flagged_instead_of_crashing():
    marker = write_session()
    marker.write_text("{not json")
    row = audit_session(marker)
    assert "unreadable_marker" in flags(row) and not row["results_ok"] and not row["laps_ok"]


def test_session_kind():
    assert session_kind("race") == "race" and session_kind("sprint") == "race"
    assert session_kind("sprint_qualifying") == "qualifying"
    assert session_kind("sprint_shootout") == "qualifying"
    assert session_kind("practice_2") == "practice"


def test_audit_on_empty_data_dir_gives_a_clear_error_and_creates_nothing(tmp_path):
    with pytest.raises(FileNotFoundError, match="No ingested sessions"):
        run_audit()
    assert list(tmp_path.iterdir()) == []


def test_session_without_weather_is_flagged_but_still_usable():
    marker = write_session()
    marker.write_text(json.dumps({"counts": {"weather": 0, "track_status": 3}, "missing": []}))
    row = audit_session(marker)
    assert flags(row) == {"no_weather"} and row["results_ok"] and row["laps_ok"]
    assert not row["weather_ok"]


def test_missing_tables_from_the_marker_are_reported():
    marker = write_session()
    marker.write_text(
        json.dumps({"counts": {"weather": 5, "track_status": 3}, "missing": ["laps"]})
    )
    assert audit_session(marker)["missing_tables"] == "laps"


def test_a_marker_that_is_valid_json_but_not_an_object_is_flagged():
    marker = write_session()
    marker.write_text("[1, 2, 3]")
    assert "unreadable_marker" in flags(audit_session(marker))


def test_main_writes_the_csv_report(tmp_path, capsys):
    from ml.ingest.audit import main

    write_session()
    assert main() == 0
    report = pd.read_csv(tmp_path / "reports" / "data_quality.csv")
    assert report.loc[0, "Session"] == "race" and bool(report.loc[0, "results_ok"])
    assert "1 sessions audited: 0 with problems, 0 informational only" in capsys.readouterr().out


def test_a_race_whose_only_laps_are_lap_one_has_no_usable_timed_laps():
    row = audit_session(write_session("race", laps_each=1))
    assert "many_null_laptimes" in flags(row) and not row["laps_ok"]


def test_the_summary_separates_real_problems_from_informational_flags(capsys):
    from ml.ingest.audit import main

    write_session("race", untimed_frac=0.2)  # informational: some_untimed_laps
    main()
    assert "0 with problems, 1 informational only" in capsys.readouterr().out

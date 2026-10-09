"""Same-weekend features: which sessions count as earlier, and what the pq_ features contain."""

import numpy as np
import pandas as pd
import pytest

from ml.features import weekend
from ml.features.builders import WeekendPace
from ml.features.weekend import (
    PQ_COLUMNS,
    best_lap_gap_pct,
    build_weekend_table,
    earlier_sessions,
    session_kind,
    weekend_features,
)
from ml.ingest.paths import raw_dir


def event(*sessions: tuple[str, str]) -> pd.Series:
    """A calendar row with up to five (name, UTC start) sessions."""
    row = {}
    for i, (name, start) in enumerate(sessions, start=1):
        row[f"Session{i}"] = name
        row[f"Session{i}DateUtc"] = pd.Timestamp(start)
    return pd.Series(row)


# ---- the ordering rule, for every weekend format that has existed --------------------------

STANDARD = event(
    ("Practice 1", "2024-03-01 11:30"),
    ("Practice 2", "2024-03-01 15:00"),
    ("Practice 3", "2024-03-02 11:30"),
    ("Qualifying", "2024-03-02 15:00"),
    ("Race", "2024-03-03 15:00"),
)
SPRINT_2021_22 = event(  # Practice 2 is held AFTER qualifying
    ("Practice 1", "2022-04-22 11:30"),
    ("Qualifying", "2022-04-22 15:00"),
    ("Practice 2", "2022-04-23 11:30"),
    ("Sprint", "2022-04-23 15:30"),
    ("Race", "2022-04-24 15:00"),
)
SPRINT_2023 = event(  # the shootout is after qualifying
    ("Practice 1", "2023-04-28 11:30"),
    ("Qualifying", "2023-04-28 15:00"),
    ("Sprint Shootout", "2023-04-29 11:30"),
    ("Sprint", "2023-04-29 15:30"),
    ("Race", "2023-04-30 15:00"),
)
SPRINT_2024_ON = event(  # Sprint Qualifying and the Sprint are BEFORE qualifying
    ("Practice 1", "2024-04-19 11:30"),
    ("Sprint Qualifying", "2024-04-19 15:30"),
    ("Sprint", "2024-04-20 11:00"),
    ("Qualifying", "2024-04-20 15:00"),
    ("Race", "2024-04-21 15:00"),
)


@pytest.mark.parametrize(
    ("weekend_event", "expected"),
    [
        (STANDARD, ["Practice 1", "Practice 2", "Practice 3"]),
        (SPRINT_2021_22, ["Practice 1"]),
        (SPRINT_2023, ["Practice 1"]),
        (SPRINT_2024_ON, ["Practice 1", "Sprint Qualifying", "Sprint"]),
    ],
    ids=["standard", "sprint 2021-22", "sprint 2023", "sprint 2024 onward"],
)
def test_only_sessions_that_start_before_qualifying_count(weekend_event, expected):
    assert earlier_sessions(weekend_event, "Qualifying") == expected


def test_practice_2_on_a_2021_to_2022_sprint_weekend_is_never_available_before_qualifying():
    assert "Practice 2" not in earlier_sessions(SPRINT_2021_22, "Qualifying")
    assert "Practice 2" in earlier_sessions(
        SPRINT_2021_22, "Race"
    )  # but it IS known before the race


def test_the_target_itself_and_everything_after_it_is_excluded():
    for name in ("Qualifying", "Race"):
        assert name not in earlier_sessions(STANDARD, "Qualifying")
    assert earlier_sessions(STANDARD, "Race") == [
        "Practice 1",
        "Practice 2",
        "Practice 3",
        "Qualifying",
    ]


def test_order_comes_from_start_times_not_from_names_or_list_position():
    shuffled = event(  # the calendar lists the sessions in a misleading order
        ("Qualifying", "2024-03-02 15:00"),
        ("Practice 3", "2024-03-02 11:30"),
        ("Practice 1", "2024-03-01 11:30"),
        ("Race", "2024-03-03 15:00"),
        ("Practice 2", "2024-03-02 16:00"),  # a "Practice 2" that is after qualifying
    )
    assert earlier_sessions(shuffled, "Qualifying") == ["Practice 1", "Practice 3"]


def test_a_session_that_starts_at_the_same_moment_is_not_earlier():
    tie = event(("Practice 3", "2024-03-02 15:00"), ("Qualifying", "2024-03-02 15:00"))
    assert earlier_sessions(tie, "Qualifying") == []


def test_a_target_not_on_the_schedule_gives_nothing():
    assert earlier_sessions(STANDARD, "Sprint") == []


def test_session_kinds():
    assert session_kind("Practice 2") == "practice" and session_kind("Race") == "race"
    assert (
        session_kind("Sprint Qualifying") == session_kind("Sprint Shootout") == "sprint_qualifying"
    )
    assert session_kind("Sprint") == "sprint" and session_kind("Qualifying") == "qualifying"
    assert session_kind("Warm-up") == "other"


# ---- best-lap gaps ---------------------------------------------------------------------------


def laps(**by_driver):
    rows = []
    for driver, times in by_driver.items():
        rows += [{"Driver": driver, "LapTime": t, "Deleted": None} for t in times]
    df = pd.DataFrame(rows)
    df["LapTime"] = pd.to_timedelta(df["LapTime"], unit="s")
    return df


def test_best_lap_gap_is_relative_to_the_fastest_lap_of_the_session():
    gap = best_lap_gap_pct(laps(AAA=[90.0, 95.0], BBB=[91.0, 99.0], CCC=[np.nan, 100.0]))
    assert gap["AAA"] == 0.0
    assert gap["BBB"] == pytest.approx((91 / 90 - 1) * 100)
    assert gap["CCC"] == pytest.approx((100 / 90 - 1) * 100)


def test_deleted_laps_and_missing_laps_do_not_count():
    df = laps(AAA=[80.0, 90.0], BBB=[91.0])
    df.loc[0, "Deleted"] = "True"  # AAA's 80.0 lap was deleted
    gap = best_lap_gap_pct(df)
    assert gap["AAA"] == 0.0 and gap["BBB"] == pytest.approx((91 / 90 - 1) * 100)
    assert best_lap_gap_pct(pd.DataFrame()).empty
    assert best_lap_gap_pct(laps(AAA=[np.nan])).empty


# ---- the features, built from session files on disk --------------------------------------------


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFONE_DATA_DIR", str(tmp_path))


def write_laps(year, rnd, session_slug, **by_driver):
    path = raw_dir() / "laps" / f"year={year}" / f"round={rnd:02d}"
    path.mkdir(parents=True, exist_ok=True)
    laps(**by_driver).to_parquet(path / f"{session_slug}.parquet", index=False)


def write_results(year, rnd, session_slug, positions: dict[str, float]):
    path = raw_dir() / "results" / f"year={year}" / f"round={rnd:02d}"
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"Abbreviation": list(positions), "Position": list(positions.values())})
    df.to_parquet(path / f"{session_slug}.parquet", index=False)


def test_practice_features_take_the_best_gap_and_the_latest_gap():
    write_laps(2024, 1, "practice_1", AAA=[90.0], BBB=[92.0])  # BBB 2.22% off
    write_laps(2024, 1, "practice_2", AAA=[91.0], BBB=[90.0])  # AAA 1.11% off
    write_laps(2024, 1, "practice_3", AAA=[89.0], BBB=[89.5])  # AAA fastest, BBB 0.56% off
    got = weekend_features(2024, 1, STANDARD).loc[["AAA", "BBB"]]
    assert got["pq_fp_n"].tolist() == [3.0, 3.0]
    assert got.loc["AAA", "pq_fp_best_gap"] == 0.0 and got.loc["BBB", "pq_fp_best_gap"] == 0.0
    assert got.loc["AAA", "pq_fp_last_gap"] == 0.0
    assert got.loc["BBB", "pq_fp_last_gap"] == pytest.approx((89.5 / 89 - 1) * 100)


def test_sessions_after_qualifying_are_never_read(monkeypatch):
    """The strongest check: the code must not even open a file for a session that starts later."""
    opened = []
    real = weekend._read

    def spy(table, year, rnd, name, columns):
        opened.append((table, name))
        return real(table, year, rnd, name, columns)

    monkeypatch.setattr(weekend, "_read", spy)
    weekend_features(2022, 5, SPRINT_2021_22)
    names = {name for _, name in opened}
    assert names == {"Practice 1"}  # not Practice 2 (after qualifying), Sprint or Race
    opened.clear()
    weekend_features(2024, 5, SPRINT_2024_ON)
    assert {name for _, name in opened} == {"Practice 1", "Sprint Qualifying", "Sprint"}
    assert not {"Qualifying", "Race"} & {name for _, name in opened}


def test_a_later_practice_session_cannot_change_the_features(monkeypatch):
    write_laps(2022, 5, "practice_1", AAA=[90.0], BBB=[92.0])
    before = weekend_features(2022, 5, SPRINT_2021_22)
    write_laps(2022, 5, "practice_2", AAA=[50.0], BBB=[40.0])  # held AFTER qualifying
    pd.testing.assert_frame_equal(before, weekend_features(2022, 5, SPRINT_2021_22))


def test_sprint_features_for_the_2024_format():
    write_laps(2024, 5, "practice_1", AAA=[90.0], BBB=[90.5])
    write_laps(2024, 5, "sprint_qualifying", AAA=[80.0, 79.0], BBB=[78.0], CCC=[81.0])
    write_results(2024, 5, "sprint", {"AAA": 2.0, "BBB": 1.0, "CCC": 3.0})
    got = weekend_features(2024, 5, SPRINT_2024_ON)
    assert got.loc["BBB", "pq_sprintq_pos"] == 1.0 and got.loc["AAA", "pq_sprintq_pos"] == 2.0
    assert got.loc["BBB", "pq_sprintq_gap"] == 0.0
    assert got.loc["AAA", "pq_sprintq_gap"] == pytest.approx((79 / 78 - 1) * 100)
    assert got["pq_sprint_pos"].to_dict() == {"AAA": 2.0, "BBB": 1.0, "CCC": 3.0}


def test_a_standard_weekend_has_no_sprint_features():
    write_laps(2024, 1, "practice_1", AAA=[90.0])
    got = weekend_features(2024, 1, STANDARD)
    assert got[["pq_sprintq_pos", "pq_sprintq_gap", "pq_sprint_pos"]].isna().all().all()


def test_missing_sessions_give_nan_and_a_count_of_zero_never_an_error():
    got = weekend_features(2024, 1, STANDARD)  # nothing ingested at all
    assert got.empty or got["pq_fp_n"].eq(0).all()
    assert list(got.columns) == list(PQ_COLUMNS)


def test_the_table_is_built_for_every_driver_of_every_race_and_unknown_drivers_get_nan():
    write_laps(2024, 1, "practice_1", AAA=[90.0], BBB=[91.0])
    events = pd.DataFrame([STANDARD.to_dict() | {"Year": 2024, "RoundNumber": 1}])
    races = pd.DataFrame({"Year": 2024, "Round": 1, "Abbreviation": ["AAA", "BBB", "ZZZ"]})
    table = build_weekend_table(events, races)
    assert list(table["Abbreviation"]) == ["AAA", "BBB", "ZZZ"]
    assert table.loc[table["Abbreviation"] == "ZZZ", "pq_fp_best_gap"].isna().all()
    assert (table["pq_fp_n"] == 1.0).all()  # one session had data, for every driver in the race


# ---- the builder -----------------------------------------------------------------------------


def race_entries(gaps, teams):
    df = pd.DataFrame(
        {
            "TeamKey": teams,
            "Year": 2024,
            "Round": 1,
            "DriverId": [f"d{i}" for i in range(len(gaps))],
            "Abbreviation": [f"D{i}" for i in range(len(gaps))],
            "CircuitId": "c",
        }
    )
    for column in PQ_COLUMNS:
        df[column] = 1.0
    df["pq_fp_best_gap"] = gaps
    return df


def test_the_builder_adds_the_rank_in_the_field_and_the_gap_to_the_teammate():
    race = race_entries([0.0, 0.5, 1.0, 2.0], ["a", "a", "b", "b"])
    out = WeekendPace().compute(pd.DataFrame(), race)
    assert out["pq_fp_best_rank"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert out["pq_fp_vs_teammate"].tolist() == [-0.5, 0.5, -1.0, 1.0]
    assert list(out.columns) == list(WeekendPace.columns)


def test_the_builder_copes_with_missing_practice_and_a_missing_teammate():
    race = race_entries([0.0, np.nan, 1.0, 2.0], ["a", "a", "b", "b"])
    out = WeekendPace().compute(pd.DataFrame(), race)
    assert np.isnan(out["pq_fp_best_rank"].iloc[1]) and np.isnan(out["pq_fp_vs_teammate"].iloc[1])
    assert np.isnan(out["pq_fp_vs_teammate"].iloc[0])  # the only teammate has no time
    assert out["pq_fp_vs_teammate"].iloc[2] == -1.0
    alone = WeekendPace().compute(pd.DataFrame(), race_entries([0.0], ["solo"]))
    assert np.isnan(alone["pq_fp_vs_teammate"].iloc[0])  # no teammate at all


# ---- the stage rules ---------------------------------------------------------------------------


def test_practice_features_are_opt_in_and_only_for_targets_known_after_practice():
    from ml.features.build import feature_columns

    base = feature_columns("y_quali")
    assert not set(PQ_COLUMNS) & set(base)
    assert not set(WeekendPace.columns) & set(base)
    with_practice = feature_columns("y_quali", include_practice=True)
    assert set(WeekendPace.columns) <= set(with_practice)
    assert set(base) <= set(with_practice)


def test_pre_weekend_builders_cannot_see_the_practice_columns():
    from ml.features.builders import STAGE_INPUTS

    assert not set(PQ_COLUMNS) & set(STAGE_INPUTS["pre_weekend"])
    assert not set(PQ_COLUMNS) & set(STAGE_INPUTS["scenario"])
    assert set(PQ_COLUMNS) <= set(STAGE_INPUTS["post_practice"])

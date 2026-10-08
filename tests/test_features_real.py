"""Feature pipeline on the REAL ingested data. Skipped on a fresh clone (no data).

Tests pin themselves to completed seasons (2018 to 2025) so they do not break as 2026 grows; the
two 2026 facts used (round 1 entries, race 2 laps) are already history.
"""

import numpy as np
import pandas as pd
import pytest

from ml.features.build import (
    TARGET_STAGES,
    TARGETS,
    build_feature_table,
    feature_columns,
    features_for_entry,
    features_for_race,
)
from ml.features.builders import BUILDERS
from ml.features.tables import NOT_A_RACE, build_history
from ml.ingest.paths import raw_dir
from tests.helpers import OUTCOME_COLUMNS, scramble

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def history():
    if not list(raw_dir().glob("results/year=*/round=*/race.parquet")):
        pytest.skip("no ingested data (fresh clone)")
    return build_history()


@pytest.fixture(scope="module")
def table(history):
    return build_feature_table(history)


def test_history_has_one_unique_driver_row_per_race_for_every_ingested_race(history):
    completed = history[history["Year"] <= 2025]  # 2026 is still growing and being ingested
    n_races = len(list(raw_dir().glob("results/year=201[89]/round=*/race.parquet")))
    n_races += len(list(raw_dir().glob("results/year=202[0-5]/round=*/race.parquet")))
    per_race = completed.groupby(["Year", "Round"]).size()
    assert len(per_race) == n_races
    assert per_race.between(18, 22).all()
    assert not history.duplicated(["Year", "Round", "DriverId"]).any()
    assert history["CircuitId"].notna().all() and history["TeamKey"].notna().all()


def test_the_feature_table_has_the_same_rows_as_the_history_and_a_complete_grid(history, table):
    assert len(table) == len(history)
    assert not table.duplicated(["Year", "Round", "DriverId"]).any()
    completed = table[table["Year"] <= 2025]
    assert completed["grid"].notna().all() and completed["wx_air_temp"].notna().all()


def test_targets_equal_the_real_results(history, table):
    merged = history.merge(table, on=["Year", "Round", "DriverId"], suffixes=("", "_t"))
    pd.testing.assert_series_equal(merged["Position"], merged["y_finish"], check_names=False)
    assert (merged["dnf"] == merged["y_dnf"]).all()


def test_only_the_declared_races_are_marked_not_ok(history):
    completed = history[history["Year"] <= 2025]  # a new 2026 race failing its audit is not a bug
    bad = completed.loc[~completed["race_ok"], ["Year", "Round"]].drop_duplicates()
    assert set(map(tuple, bad.to_numpy().tolist())) == set(NOT_A_RACE) == {(2021, 12)}


def test_unusable_lap_data_gives_no_pace_figure_but_keeps_the_race(history):
    for year, rnd in [(2018, 14), (2021, 12), (2026, 2)]:  # no laps / barely a race / partial laps
        race = history[(history["Year"] == year) & (history["Round"] == rnd)]
        assert len(race) >= 18 and race["pace_gap_pct"].isna().all(), (year, rnd)
    normal = history[(history["Year"] == 2024) & (history["Round"] == 10)]
    assert normal["pace_gap_pct"].notna().sum() >= 15


@pytest.mark.parametrize("key", [(2019, 5), (2022, 12), (2024, 3), (2025, 3)])
def test_scrambling_the_target_session_onward_changes_no_real_feature(history, key):
    for builder in BUILDERS:
        expected = features_for_race(history, key, [builder])[list(builder.columns)]
        scrambled = scramble(history, key, builder.stage)
        got = features_for_race(scrambled, key, [builder])[list(builder.columns)]
        pd.testing.assert_frame_equal(expected, got, obj=f"{builder.name} {key}")


@pytest.mark.parametrize("key", [(2020, 9), (2023, 8), (2025, 12)])
def test_real_features_need_only_the_past_and_the_entry_list(history, table, key):
    """Delete every later race, blank the target's outcomes, rebuild from the entry list alone."""
    year, rnd = key
    prefix = history[
        (history["Year"] < year) | ((history["Year"] == year) & (history["Round"] <= rnd))
    ]
    prefix = prefix.copy()
    at_target = (prefix["Year"] == year) & (prefix["Round"] == rnd)
    prefix[OUTCOME_COLUMNS] = prefix[OUTCOME_COLUMNS].astype(float)
    prefix.loc[at_target, OUTCOME_COLUMNS] = np.nan
    got = features_for_entry(prefix, prefix[at_target], key)
    want = table[(table["Year"] == year) & (table["Round"] == rnd)]
    columns = feature_columns("y_finish", include_scenario=True)
    pd.testing.assert_frame_equal(
        want[columns].reset_index(drop=True), got[columns].reset_index(drop=True)
    )


def test_driver_form_matches_a_recomputation_straight_from_the_raw_result_files(table):
    """Independent of build_history/builders: read the raw parquet, take the driver's last 5 races.

    The 2021 Belgian GP is excluded on purpose (it carries no information); the last case sits
    right after it and would differ if the exclusion were broken.
    """
    files = sorted(raw_dir().glob("results/year=*/round=*/race.parquet"))
    raw = pd.concat(
        pd.read_parquet(f, columns=["Year", "Round", "DriverId", "Position"]) for f in files
    )
    raw = raw[~raw.set_index(["Year", "Round"]).index.isin(list(NOT_A_RACE))]
    for year, rnd, driver in [
        (2024, 10, "max_verstappen"),
        (2022, 7, "leclerc"),
        (2025, 8, "norris"),
        (2021, 14, "max_verstappen"),
    ]:
        past = raw[
            (raw["DriverId"] == driver)
            & ((raw["Year"] < year) | ((raw["Year"] == year) & (raw["Round"] < rnd)))
        ]
        expected = past.sort_values(["Year", "Round"]).tail(5)["Position"].mean()
        got = table[
            (table["Year"] == year) & (table["Round"] == rnd) & (table["DriverId"] == driver)
        ]
        assert len(got) == 1, (year, rnd, driver)
        assert got["drv_finish_l5"].iloc[0] == pytest.approx(expected), (year, rnd, driver)


def test_a_renamed_team_keeps_history_but_a_brand_new_team_has_none(table):
    first_2026 = table[(table["Year"] == 2026) & (table["Round"] == 1)]
    audi = first_2026[first_2026["TeamKey"] == "sauber"]  # Audi took over the Sauber entry
    cadillac = first_2026[first_2026["TeamKey"] == "cadillac"]
    assert len(audi) == 2 and audi["team_finish_l5"].notna().all()
    assert len(cadillac) == 2 and cadillac["team_finish_l5"].isna().all()


def test_a_circuit_new_to_the_calendar_has_no_circuit_history_on_its_first_visit(table):
    first_visits = table[table["CircuitId"] == "las_vegas"].sort_values(["Year", "Round"])
    vegas_2023 = first_visits[first_visits["Year"] == 2023]
    assert (vegas_2023["circ_n_years"] == 0).all() and (vegas_2023["circ_drv_n"] == 0).all()
    assert vegas_2023["circ_overtake"].isna().all()
    # With no counted visit, a driver's circuit figure is just their recent form, never NaN.
    assert vegas_2023.dropna(subset=["drv_finish_l5"])["circ_drv_finish"].notna().all()


def test_the_not_a_race_exclusion_removes_the_belgian_gp_from_spa_history(table):
    spa_2022 = table[(table["CircuitId"] == "spa_francorchamps") & (table["Year"] == 2022)]
    spa_2019 = table[(table["CircuitId"] == "spa_francorchamps") & (table["Year"] == 2019)]
    # Visits before 2022: 2018, 2019, 2020 (2021 excluded). Before 2019: only 2018.
    assert spa_2022["circ_n_years"].iloc[0] == 3
    assert spa_2019["circ_n_years"].iloc[0] == 1


@pytest.mark.parametrize("target", list(TARGETS))
def test_no_allowed_feature_is_an_implausibly_good_predictor_of_its_target(table, target):
    """A hidden leak shows up as a feature that tracks the target almost perfectly."""
    usable = table[table["race_ok"] & table[target].notna()]
    columns = feature_columns(target, include_scenario=target in ("y_finish", "y_dnf"))
    corr = usable[columns].corrwith(usable[target].astype(float), method="spearman")
    assert corr.abs().max() < 0.85, corr.abs().sort_values().tail(3).to_dict()


def test_the_post_qualifying_grid_really_would_leak_into_a_qualifying_target(table):
    """Why TARGET_STAGES exists: the grid is a near copy of the qualifying result."""
    usable = table[table["race_ok"] & table["y_quali"].notna()]
    assert usable["grid"].corr(usable["y_quali"], method="spearman") > 0.9
    assert "post_quali" not in TARGET_STAGES["y_quali"]


def test_rows_are_ordered_by_driver_id_never_by_finishing_position(history, table):
    from ml.evaluation.data import load_race_results
    from ml.evaluation.metrics import race_metrics

    for frame, column in (
        (history, "Position"),
        (table, "y_finish"),
        (load_race_results(), "Position"),
    ):
        known = frame[frame[column].notna()]
        in_finish_order = [
            g[column].is_monotonic_increasing for _, g in known.groupby(["Year", "Round"])
        ]
        assert np.mean(in_finish_order) < 0.05, column
    # A constant score must earn about nothing on the real races, however the rows are sorted.
    results = load_race_results()
    results = results[results["Position"].notna()]
    scores = [
        race_metrics(pd.Series(0.0, index=g.index), g["Position"])
        for _, g in results.groupby(["Year", "Round"])
    ]
    assert all(m["spearman"] == 0.0 for m in scores)
    sizes = results.groupby(["Year", "Round"]).size().to_numpy()
    assert np.mean([m["winner_acc"] for m in scores]) == pytest.approx(np.mean(1 / sizes))

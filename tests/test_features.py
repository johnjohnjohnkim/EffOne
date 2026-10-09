"""Feature builders: leakage safety (the Milestone 3 done-check) and hand-checked values."""

import numpy as np
import pandas as pd
import pytest

from ml.features import build as build_module
from ml.features.build import (
    TARGET_STAGES,
    TARGETS,
    build_feature_table,
    feature_columns,
    features_for_entry,
    features_for_race,
    race_keys,
    training_frame,
)
from ml.features.builders import (
    BUILDERS,
    KEY_COLUMNS,
    STAGE_INPUTS,
    CircuitHistory,
    Context,
    DriverForm,
    Grid,
    TeamForm,
    Weather,
)
from ml.features.lineage import TEAM_LINEAGE, team_key
from ml.features.tables import WEATHER_COLUMNS
from ml.features.weekend import PQ_COLUMNS
from tests.helpers import DRIVERS, OUTCOME_COLUMNS, make_rows, row, scramble

KEYS = [(2022, 1), (2022, 3), (2023, 1), (2023, 3), (2023, 5)]
HISTORY_READERS = {"driver_form", "team_form", "circuit_history"}


def computed(builder, rows, key):
    return features_for_race(rows, key, [builder])[list(builder.columns)]


# ---- leakage: the Milestone 3 done-check ---------------------------------------------------


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.name)
@pytest.mark.parametrize("key", KEYS)
def test_scrambling_the_target_session_onward_leaves_every_builders_features_unchanged(
    builder, key
):
    rows = make_rows()
    expected = computed(builder, rows, key)
    got = computed(builder, scramble(rows, key, builder.stage), key)
    pd.testing.assert_frame_equal(expected, got)


@pytest.mark.parametrize("seed", range(4))
def test_prefix_invariance_features_need_nothing_after_the_target_and_none_of_its_outcomes(seed):
    """Delete every later race and blank the target's outcomes: its features must not change."""
    rows = make_rows(years=(2021, 2022, 2023), rounds=4, seed=seed)
    full = build_feature_table(rows)
    for key in race_keys(rows):
        prefix = rows[
            (rows["Year"] < key[0]) | ((rows["Year"] == key[0]) & (rows["Round"] <= key[1]))
        ].copy()
        at_target = (prefix["Year"] == key[0]) & (prefix["Round"] == key[1])
        prefix[OUTCOME_COLUMNS] = prefix[OUTCOME_COLUMNS].astype(float)
        prefix.loc[at_target, OUTCOME_COLUMNS] = np.nan
        entries = prefix[at_target]
        got = features_for_entry(prefix, entries, key)
        want = full[(full["Year"] == key[0]) & (full["Round"] == key[1])]
        pd.testing.assert_frame_equal(
            want[feature_columns("y_finish", include_scenario=True)].reset_index(drop=True),
            got[feature_columns("y_finish", include_scenario=True)].reset_index(drop=True),
        )


def test_appending_a_different_future_changes_no_earlier_feature():
    rows = make_rows()
    future = make_rows(years=(2024,), rounds=3, seed=99)
    future["DriverId"] = "newcomer_" + future["DriverId"]
    future["TeamKey"] = "other_" + future["TeamKey"]
    future["CircuitId"] = "elsewhere"
    longer = pd.concat([rows, future], ignore_index=True)
    before_ = build_feature_table(rows)
    after = build_feature_table(longer)
    same = after[after["Year"] < 2024].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_, same)


@pytest.mark.parametrize(
    "variant",
    ["includes_target", "whole_year", "everything"],
)
def test_the_leakage_test_can_fail_for_every_history_reader(monkeypatch, variant):
    """Negative control: break the history slice three ways; every history reader must be caught."""

    def leaky_before(frame, key):
        year, rnd = key
        if variant == "includes_target":
            return frame[
                (frame["Year"] < year) | ((frame["Year"] == year) & (frame["Round"] <= rnd))
            ]
        if variant == "whole_year":
            return frame[frame["Year"] <= year]
        return frame

    rows = make_rows()
    monkeypatch.setattr(build_module, "before", leaky_before)
    caught = set()
    for builder in BUILDERS:
        for key in KEYS:
            expected = computed(builder, rows, key)
            try:
                got = computed(builder, scramble(rows, key, builder.stage), key)
                pd.testing.assert_frame_equal(expected, got)
            except (AssertionError, ValueError):  # a changed value, or a loud rejection
                caught.add(builder.name)
    assert HISTORY_READERS <= caught


def test_stage_inputs_are_exactly_the_documented_columns():
    assert STAGE_INPUTS["pre_weekend"] == [
        "Year",
        "Round",
        "DriverId",
        "Abbreviation",
        "TeamKey",
        "CircuitId",
    ]
    assert STAGE_INPUTS["scenario"] == [*STAGE_INPUTS["pre_weekend"], *WEATHER_COLUMNS]
    assert STAGE_INPUTS["post_quali"] == [*STAGE_INPUTS["scenario"], "GridPosition"]


@pytest.mark.parametrize("stage", ["pre_weekend", "scenario"])
def test_a_builder_that_reads_a_column_its_stage_withholds_fails_loudly(stage):
    class Peeker:
        name = "peeker"
        columns = ("peek",)

        def __init__(self, column):
            self.stage = stage
            self.column = column

        def compute(self, history, race):
            return pd.DataFrame({"peek": race[self.column]}, index=race.index)

    rows = make_rows()
    for column in ["Position", "Points", "dnf", "QualiPosition", "pace_gap_pct", "GridPosition"]:
        with pytest.raises(KeyError):
            features_for_race(rows, (2023, 3), [Peeker(column)])
    if stage == "pre_weekend":
        with pytest.raises(KeyError):
            features_for_race(rows, (2023, 3), [Peeker("wx_air_temp")])


def test_each_builder_returns_exactly_its_declared_columns_indexed_like_the_race():
    rows = make_rows()
    for builder in BUILDERS:
        target = rows[(rows["Year"] == 2023) & (rows["Round"] == 4)]
        inputs = target[STAGE_INPUTS[builder.stage]]
        result = builder.compute(rows[rows["Year"] < 2023], inputs)
        assert list(result.columns) == list(builder.columns), builder.name
        assert result.index.equals(inputs.index), builder.name


def test_a_race_marked_not_ok_feeds_nothing_into_later_features():
    rows = make_rows()
    flagged = rows.copy()
    bad = (flagged["Year"] == 2022) & (flagged["Round"] == 3)
    flagged.loc[bad, "race_ok"] = False
    flagged_scrambled = flagged.copy()
    flagged_scrambled[OUTCOME_COLUMNS] = flagged_scrambled[OUTCOME_COLUMNS].astype(float)
    flagged_scrambled.loc[bad, OUTCOME_COLUMNS] = 99.0  # nonsense outcomes (dnf is 99.0 -> True)
    a = features_for_race(flagged, (2023, 2))
    b = features_for_race(flagged_scrambled, (2023, 2))
    columns = feature_columns("y_finish", include_scenario=True)
    pd.testing.assert_frame_equal(a[columns], b[columns])
    unflagged = features_for_race(rows, (2023, 2))
    assert not a["drv_finish_l5"].equals(unflagged["drv_finish_l5"])  # the flag really excluded it
    assert (features_for_race(flagged, (2022, 3))["race_ok"] == False).all()


# ---- prediction-time use -------------------------------------------------------------------


def test_features_for_a_future_race_come_from_the_entry_list_alone():
    rows = make_rows()
    future = (2024, 1)
    entries = pd.DataFrame(
        [
            {
                "Year": 2024,
                "Round": 1,
                "DriverId": d,
                "Abbreviation": d.upper(),
                "TeamKey": rows.loc[rows["DriverId"] == d, "TeamKey"].iloc[0],
                "CircuitId": "c0",
                **dict.fromkeys(WEATHER_COLUMNS, 15.0),
                **dict.fromkeys(PQ_COLUMNS, 1.0),
                "GridPosition": float(i + 1),
            }
            for i, d in enumerate(DRIVERS)
        ]
    )
    got = features_for_entry(rows, entries, future)
    assert len(got) == len(DRIVERS) and (got["drv_n_l5"] == 5).all()
    # Same features as when the race exists in the table with outcomes the builders must not see.
    with_race = pd.concat(
        [
            rows,
            entries.assign(
                **dict.fromkeys(OUTCOME_COLUMNS, 3.0), TeamId="x", TeamName="x", race_ok=True
            ),
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        got[feature_columns("y_finish", include_scenario=True)].reset_index(drop=True),
        features_for_race(with_race, future)[
            feature_columns("y_finish", include_scenario=True)
        ].reset_index(drop=True),
    )


def test_a_pre_qualifying_prediction_needs_no_grid_and_no_weather():
    rows = make_rows()
    entries = rows[(rows["Year"] == 2023) & (rows["Round"] == 5)][KEY_COLUMNS]
    pre = [b for b in BUILDERS if b.stage == "pre_weekend"]
    got = features_for_entry(rows, entries, (2023, 5), pre)
    assert list(got.columns) == [*KEY_COLUMNS, *[c for b in pre for c in b.columns]]


def test_missing_entry_columns_are_reported_by_builder_name():
    rows = make_rows()
    entries = rows[(rows["Year"] == 2023) & (rows["Round"] == 5)][KEY_COLUMNS]
    with pytest.raises(KeyError, match="weekend_pace"):  # the first builder whose inputs are absent
        features_for_entry(rows, entries, (2023, 5), BUILDERS)
    with pytest.raises(KeyError, match="weather"):
        features_for_entry(rows, entries, (2023, 5), [Weather()])


# ---- which features may predict which target -----------------------------------------------


def test_qualifying_and_grid_targets_never_get_grid_features():
    for target in ("y_quali", "y_grid"):
        columns = feature_columns(target)
        assert not set(Grid.columns) & set(columns), target
        assert "drv_finish_l5" in columns
    assert set(Grid.columns) <= set(feature_columns("y_finish", include_scenario=True))
    assert set(Grid.columns) <= set(feature_columns("y_dnf"))


def test_every_target_declares_which_stages_may_predict_it():
    assert set(TARGET_STAGES) == set(TARGETS)
    assert "post_quali" not in TARGET_STAGES["y_quali"] | TARGET_STAGES["y_grid"]
    with pytest.raises(KeyError):
        feature_columns("y_unknown")


# ---- table shape ---------------------------------------------------------------------------


def test_feature_table_has_one_row_per_driver_per_race_with_keys_features_and_targets():
    rows = make_rows()
    table = build_feature_table(rows)
    assert len(table) == len(rows)
    assert not table.duplicated(["Year", "Round", "DriverId"]).any()
    all_features = [c for b in BUILDERS for c in b.columns]
    assert list(table.columns) == [*KEY_COLUMNS, *all_features, "race_ok", *TARGETS]
    assert list(table["y_finish"]) == list(rows["Position"])


def test_the_first_race_has_no_history_features_but_still_has_scenario_and_grid():
    table = build_feature_table(make_rows())
    first = table[(table["Year"] == 2022) & (table["Round"] == 1)]
    assert first["drv_finish_l5"].isna().all() and (first["drv_n_l5"] == 0).all()
    assert first["team_finish_l5"].isna().all() and first["drv_pts_share_ytd"].isna().all()
    assert first["wx_air_temp"].notna().all() and first["grid"].notna().all()


def test_races_are_built_in_chronological_order():
    rows = make_rows().sample(frac=1, random_state=3)  # shuffled input
    table = build_feature_table(rows)
    keys = list(zip(table["Year"], table["Round"], strict=True))
    assert keys == sorted(keys)


# ---- hand-checked values -------------------------------------------------------------------


def test_driver_form_uses_the_last_five_races_and_this_seasons_share_of_points_only():
    rows = [
        row(2022, r, "a", Position=float(r), Points=float(10 - r), GridPosition=float(r))
        for r in (1, 2, 3)
    ]
    rows += [
        row(
            2023,
            r,
            "a",
            Position=float(r + 3),
            Points=float(10 - r),
            GridPosition=0.0 if r == 2 else 3.0,
        )
        for r in (1, 2, 3, 4)
    ]
    rows += [row(2023, r, "b", team="t1", Points=10.0) for r in (1, 2, 3, 4)]
    rows += [row(2023, 5, "a"), row(2023, 5, "b", team="t1")]
    out = computed(DriverForm(), pd.DataFrame(rows), (2023, 5))
    a = out.iloc[0]
    assert a["drv_finish_l5"] == pytest.approx(np.mean([3, 4, 5, 6, 7]))  # last 5, across seasons
    assert a["drv_n_l5"] == 5
    assert a["drv_pts_share_ytd"] == pytest.approx(30 / 70)  # 2023 only: a has 30 of the 70 scored
    assert a["drv_grid_l5"] == pytest.approx(3.0)  # the pit-lane start (0) is ignored
    assert a["drv_rounds_since_last"] == 0


def test_driver_dnf_rate_uses_the_last_ten_races():
    rows = [row(2022, r, "a", dnf=(r <= 2)) for r in range(1, 13)] + [row(2022, 13, "a")]
    out = computed(DriverForm(), pd.DataFrame(rows), (2022, 13)).iloc[0]
    assert out["drv_dnf_l10"] == 0.0  # the two DNFs fell out of the 10-race window


def test_a_returning_driver_is_marked_stale_by_the_races_they_missed():
    rows = [row(2022, 1, "away"), row(2022, 1, "a")]
    rows += [row(2022, r, "a") for r in (2, 3, 4)]
    rows += [row(2022, 5, "a"), row(2022, 5, "away")]
    out = computed(DriverForm(), pd.DataFrame(rows), (2022, 5))
    by_driver = out.set_axis(["a", "away"])
    assert by_driver.loc["away", "drv_rounds_since_last"] == 3  # missed rounds 2, 3 and 4
    assert by_driver.loc["a", "drv_rounds_since_last"] == 0


def test_a_driver_with_no_history_gets_nan_form_and_zero_count():
    rows = [row(2022, 1, "a"), row(2022, 2, "a"), row(2022, 2, "newbie")]
    out = computed(DriverForm(), pd.DataFrame(rows), (2022, 2)).set_axis(["a", "newbie"])
    assert np.isnan(out.loc["newbie", "drv_finish_l5"]) and out.loc["newbie", "drv_n_l5"] == 0
    assert np.isnan(out.loc["newbie", "drv_rounds_since_last"])


def test_the_form_count_matches_the_mean_when_a_finish_is_unknown():
    rows = [row(2022, 1, "a", Position=np.nan), row(2022, 2, "a", Position=4.0), row(2022, 3, "a")]
    out = computed(DriverForm(), pd.DataFrame(rows), (2022, 3)).iloc[0]
    assert out["drv_n_l5"] == 1 and out["drv_finish_l5"] == 4.0


def test_team_form_averages_both_cars_per_race_before_windowing():
    rows = []
    for rnd, (fa, fb) in enumerate([(1, 3), (2, 4), (3, 5)], start=1):
        rows += [
            row(2022, rnd, "a", Position=float(fa), Points=10.0),
            row(2022, rnd, "b", Position=float(fb), Points=5.0),
            row(2022, rnd, "c", team="t1", Points=15.0),
        ]
    rows += [row(2022, 4, "a"), row(2022, 4, "b"), row(2022, 4, "c", team="t1")]
    out = computed(TeamForm(), pd.DataFrame(rows), (2022, 4))
    t0, t1 = out.iloc[0], out.iloc[2]
    assert t0["team_finish_l5"] == pytest.approx(np.mean([2, 3, 4]))
    assert t0["team_pts_share_ytd"] == pytest.approx(45 / 90)  # (10+5)*3 of (15*3 + 45)
    assert t1["team_pts_share_ytd"] == pytest.approx(45 / 90)


def test_a_renamed_team_keeps_its_history_via_the_lineage_table():
    assert team_key("force_india") == team_key("racing_point") == team_key("aston_martin")
    assert team_key("toro_rosso") == team_key("alphatauri") == team_key("rb") == "rb"
    assert team_key("alfa") == team_key("audi") == team_key("sauber")
    assert team_key("cadillac") == "cadillac" and team_key("ferrari") == "ferrari"
    with pytest.raises(TypeError):
        TEAM_LINEAGE["x"] = "y"


def circuit_scenario():
    """Entered in the target race: e0..e7. At circuit X: 2019 only e0 raced (12.5%, ignored);
    2020 e1, e2, e3 raced (37.5%, counted)."""
    entered = [f"e{i}" for i in range(8)]
    rows = [row(2019, 1, "e0", circuit="X", Position=1.0, GridPosition=1.0)]
    rows += [
        row(2019, 1, f"o{i}", circuit="X", Position=float(i + 2), GridPosition=float(i + 2))
        for i in range(7)
    ]
    rows += [
        row(2020, 1, "e1", circuit="X", Position=2.0, GridPosition=4.0),
        row(2020, 1, "e2", circuit="X", Position=6.0, GridPosition=5.0, dnf=True),
        row(2020, 1, "e3", circuit="X", Position=8.0, GridPosition=6.0),
    ]
    rows += [
        row(2020, 1, f"p{i}", circuit="X", Position=float(i + 10), GridPosition=float(i + 10))
        for i in range(5)
    ]
    rows += [row(2021, 1, d, circuit="X") for d in entered]
    return pd.DataFrame(rows), entered


def test_circuit_history_counts_only_seasons_with_enough_of_the_entered_drivers():
    rows, _ = circuit_scenario()
    out = computed(CircuitHistory(), rows, (2021, 1))
    assert (out["circ_n_years"] == 1).all()  # 2019 ignored (1 of 8 entered), 2020 counted (3 of 8)
    by_driver = out.set_index(rows.query("Year == 2021")["DriverId"].to_numpy())
    assert by_driver.loc["e1", "circ_drv_n"] == 1 and by_driver.loc["e0", "circ_drv_n"] == 0


def test_circuit_driver_with_no_counted_visit_gets_their_recent_form_not_nan():
    rows, _ = circuit_scenario()
    history = rows.query("Year < 2021")
    out = computed(CircuitHistory(), rows, (2021, 1))
    by_driver = out.set_index(rows.query("Year == 2021")["DriverId"].to_numpy())
    assert by_driver.loc["e0", "circ_drv_finish"] == pytest.approx(1.0)  # their only race: finish 1
    never_raced = by_driver.loc["e5", "circ_drv_finish"]  # no history at all: field average
    assert never_raced == pytest.approx(history["Position"].mean())


def test_circuit_average_is_shrunk_towards_the_last_twenty_races_not_all_of_them():
    # z raced 30 times: 10 old races finishing 10th, then 20 recent ones finishing 2nd except one
    # visit to circuit Y (race 15) finishing 12th. Prior = mean of the last 20 = 2.5.
    rows = []
    for r in range(1, 31):
        finish = 10.0 if r <= 10 else 2.0
        circuit = "o"
        if r == 15:
            finish, circuit = 12.0, "Y"
        rows.append(row(2022, r, "z", circuit=circuit, Position=finish))
    rows.append(row(2022, 31, "z", circuit="Y"))
    out = computed(CircuitHistory(), pd.DataFrame(rows), (2022, 31)).iloc[0]
    assert out["circ_drv_finish"] == pytest.approx((1 * 12.0 + 3 * 2.5) / 4)  # not 6.75 (all races)
    assert out["circ_drv_n"] == 1


def test_a_season_needs_exactly_the_ceiling_of_a_quarter_of_the_entered_drivers():
    visit = [row(2022, 1, "z", circuit="Y")]  # only z raced at Y in 2022

    def years_counted(extra_entries):
        rows = [*visit, row(2023, 1, "z", circuit="Y")]
        rows += [row(2023, 1, f"n{i}", circuit="Y") for i in range(extra_entries)]
        return computed(CircuitHistory(), pd.DataFrame(rows), (2023, 1))["circ_n_years"].iloc[0]

    assert years_counted(3) == 1  # 1 of 4 entered = 25%, counts
    assert years_counted(4) == 0  # 1 of 5 = 20%, does not


def test_circuit_level_overtaking_and_dnf_rate_use_counted_seasons_only():
    rows, _ = circuit_scenario()
    out = computed(CircuitHistory(), rows, (2021, 1)).iloc[0]
    # 2020 at X: non-DNF starters gain (grid - finish): e1 +2, e3 -2, p0..p4 0 -> mean 0
    assert out["circ_overtake"] == pytest.approx(0.0)
    assert out["circ_dnf_rate"] == pytest.approx(1 / 8)  # 1 DNF among the 8 counted-season rows


def test_a_new_circuit_has_no_circuit_history():
    rows, _ = circuit_scenario()
    rows.loc[rows["Year"] == 2021, "CircuitId"] = "NEW"
    out = computed(CircuitHistory(), rows, (2021, 1))
    assert (out["circ_n_years"] == 0).all() and (out["circ_drv_n"] == 0).all()
    assert out["circ_overtake"].isna().all() and out["circ_dnf_rate"].isna().all()


def test_circuit_features_cope_with_an_object_dtype_dnf_column():
    rows, _ = circuit_scenario()
    rows["dnf"] = rows["dnf"].astype(object)
    out = computed(CircuitHistory(), rows, (2021, 1)).iloc[0]
    assert out["circ_dnf_rate"] == pytest.approx(1 / 8)


def test_context_places_the_race_in_the_season_and_the_regulation_era():
    def ctx(year, rnd):
        race = pd.DataFrame([row(year, rnd, "a")])[KEY_COLUMNS]
        return Context().compute(pd.DataFrame(), race).iloc[0].tolist()

    assert ctx(2018, 4) == [4.0, 1.0]  # 2017 rules, second season of our data
    assert ctx(2022, 1) == [1.0, 0.0]  # ground-effect reset
    assert ctx(2025, 20) == [20.0, 3.0]
    assert ctx(2026, 3) == [3.0, 0.0]  # new rules


def test_weather_passes_through_and_flags_any_rain_as_wet():
    race = pd.DataFrame(
        {**dict.fromkeys(KEY_COLUMNS, 1), **dict.fromkeys(WEATHER_COLUMNS, 10.0)},
        index=[10, 11, 12],
    )
    race["wx_rain_frac"] = [0.0, 0.02, np.nan]
    out = Weather().compute(pd.DataFrame(), race)
    assert out["wx_is_wet"].iloc[:2].tolist() == [0.0, 1.0] and np.isnan(out["wx_is_wet"].iloc[2])
    assert list(out.index) == [10, 11, 12]


def test_grid_puts_pit_lane_and_missing_starts_at_the_back():
    race = pd.DataFrame({"GridPosition": [1.0, 0.0, 2.0, np.nan]}, index=list("abcd"))
    out = Grid().compute(pd.DataFrame(), race)
    assert out["grid"].tolist() == [1.0, 3.0, 2.0, 3.0]
    assert out["grid_pit_lane"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert out["grid_frac"].tolist() == [0.25, 0.75, 0.5, 0.75]


# ---- row order, strictness, validation, weather policy --------------------------------------


def test_rows_are_not_in_finishing_order_and_a_constant_score_earns_nothing():
    from ml.evaluation.metrics import race_metrics

    rows = make_rows(years=(2022, 2023, 2024), rounds=10)
    table = build_feature_table(rows)
    races = list(table.groupby(["Year", "Round"]))
    ordered = [g["y_finish"].is_monotonic_increasing for _, g in races]
    assert np.mean(ordered) < 0.1  # rows are ordered by driver id, not by how they finished
    assert all(g["DriverId"].is_monotonic_increasing for _, g in races)
    # With rows sorted by the answer, ties would have rewarded a model that cannot tell drivers apart.
    scores = [race_metrics(pd.Series(0.0, index=g.index), g["y_finish"]) for _, g in races]
    assert all(m["spearman"] == 0.0 for m in scores)


def test_history_without_race_ok_or_with_gaps_is_rejected_not_defaulted():
    rows = make_rows()
    entries = rows[(rows["Year"] == 2023) & (rows["Round"] == 3)]
    with pytest.raises(KeyError, match="race_ok"):
        features_for_entry(rows.drop(columns="race_ok"), entries, (2023, 3))
    gappy = rows.copy()
    gappy["race_ok"] = gappy["race_ok"].astype(object)
    gappy.loc[gappy.index[0], "race_ok"] = None
    with pytest.raises(ValueError, match="race_ok"):
        features_for_entry(gappy, entries, (2023, 3))
    with pytest.raises(KeyError, match="race_ok"):
        features_for_race(rows.drop(columns="race_ok"), (2023, 3))


def test_entries_are_validated_before_any_feature_is_built():
    rows = make_rows()
    entries = rows[(rows["Year"] == 2023) & (rows["Round"] == 3)]
    key = (2023, 3)
    with pytest.raises(ValueError, match="empty"):
        features_for_entry(rows, entries.iloc[0:0], key)
    with pytest.raises(ValueError, match="unique index"):
        features_for_entry(rows, pd.concat([entries, entries]), key)
    with pytest.raises(ValueError, match="belong to race"):
        features_for_entry(rows, rows[(rows["Year"] == 2023) & (rows["Round"] <= 2)], key)
    with pytest.raises(ValueError, match="belong to race"):
        features_for_entry(rows, entries, (2023, 4))
    twice = pd.concat([entries, entries.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="more than once"):
        features_for_entry(rows, twice, key)


def test_weather_is_opt_in_and_never_offered_for_qualifying_or_grid_targets():
    weather = {c for b in BUILDERS if b.stage == "scenario" for c in b.columns}
    for target in ("y_finish", "y_dnf"):
        assert not weather & set(feature_columns(target))
        assert weather <= set(feature_columns(target, include_scenario=True))
    for target in ("y_quali", "y_grid"):
        assert not weather & set(feature_columns(target))
        with pytest.raises(ValueError, match="scenario"):
            feature_columns(target, include_scenario=True)


def test_race_ok_must_be_real_booleans_for_the_target_race_too():
    rows = make_rows()
    key = (2023, 3)
    in_target = (rows["Year"] == 2023) & (rows["Round"] == 3)
    gap = rows.copy()
    gap["race_ok"] = gap["race_ok"].astype(object)
    gap.loc[in_target, "race_ok"] = None
    with pytest.raises(ValueError, match="race_ok"):
        features_for_race(gap, key)
    text = rows.copy()
    text["race_ok"] = text["race_ok"].astype(object)
    text.loc[~in_target, "race_ok"] = "False"  # a CSV round trip would do this; "False" is truthy
    with pytest.raises(ValueError, match="real booleans"):
        features_for_race(text, key)


def test_entries_must_be_for_one_circuit_and_keys_may_be_lists_or_numpy_ints():
    rows = make_rows()
    entries = rows[(rows["Year"] == 2023) & (rows["Round"] == 3)].copy()
    mixed = entries.copy()
    mixed.iloc[0, mixed.columns.get_loc("CircuitId")] = "elsewhere"
    with pytest.raises(ValueError, match="one circuit"):
        features_for_entry(rows, mixed, (2023, 3))
    expected = features_for_entry(rows, entries, (2023, 3))
    for key in ([2023, 3], (np.int64(2023), np.int64(3))):
        pd.testing.assert_frame_equal(features_for_entry(rows, entries, key), expected)


def test_duplicate_driver_rows_in_history_are_rejected_not_averaged_in():
    rows = make_rows()
    entries = rows[(rows["Year"] == 2023) & (rows["Round"] == 3)]
    duplicated = pd.concat([rows, rows[(rows["Year"] == 2022) & (rows["Round"] == 5)]])
    with pytest.raises(ValueError, match="twice"):
        features_for_entry(duplicated, entries, (2023, 3))


def test_training_frame_returns_only_what_is_safe_to_train_a_target_on():
    rows = make_rows()
    rows.loc[(rows["Year"] == 2022) & (rows["Round"] == 2), "race_ok"] = False
    table = build_feature_table(rows)
    table.loc[table.index[40], "y_quali"] = np.nan
    frame = training_frame(table, "y_quali")
    assert list(frame.columns) == [*KEY_COLUMNS, *feature_columns("y_quali"), "y_quali"]
    assert not set(Grid.columns) & set(frame.columns) and "race_ok" not in frame.columns
    assert not {"y_finish", "y_dnf", "y_grid"} & set(frame.columns)  # no other targets sneak in
    assert len(frame) == len(table) - len(DRIVERS) - 1  # the flagged race and the unknown target
    assert not ((frame["Year"] == 2022) & (frame["Round"] == 2)).any()
    assert (
        frame.groupby(["Year", "Round"])["DriverId"]
        .apply(lambda s: s.is_monotonic_increasing)
        .all()
    )
    race = training_frame(table, "y_finish", include_scenario=True)
    assert set(Grid.columns) <= set(race.columns) and "wx_air_temp" in race.columns
    assert "wx_air_temp" not in training_frame(table, "y_finish").columns

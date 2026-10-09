"""4b.3: recency-weighted form, the gap-to-pole target, and tuning them on development seasons only."""

import json

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.splits import TuningLeakError
from ml.features.build import build_feature_table, feature_columns, training_frame
from ml.features.builders import HALF_LIVES, RecencyForm
from ml.models.base import feature_plan
from ml.models.zoo import build_model, make_feature_variants
from ml.tuning import FEATURE_CHOICES, score_on_development, tune
from tests.helpers import make_rows, row, scramble

TEST_KEY = (2023, 3)


def history_of(*gaps, driver="d0", team="t0"):
    """One driver's qualifying gaps in rounds 1..n of 2022, plus a target race entry list."""
    rows = [
        row(2022, i + 1, driver, team=team, quali_gap_pct=g, QualiPosition=float(i + 1))
        for i, g in enumerate(gaps)
    ]
    history = pd.DataFrame(rows)
    entries = pd.DataFrame([row(2022, len(gaps) + 1, driver, team=team)])
    return history, entries


# ---- the numbers ------------------------------------------------------------------------------


def test_weights_halve_every_half_life():
    history, entries = history_of(2.0, 1.0)  # ages 1 and 0
    got = RecencyForm().compute(history, entries)
    w_old = 0.5 ** (1 / 2)  # half-life 2
    want = (2.0 * w_old + 1.0 * 1.0) / (w_old + 1.0)
    assert got["drv_qgap_ew2"].iloc[0] == pytest.approx(want)
    assert got["drv_quali_ew2"].iloc[0] == pytest.approx((1 * w_old + 2 * 1.0) / (w_old + 1.0))


def test_a_missing_qualifying_time_is_skipped_not_counted_as_zero():
    history, entries = history_of(3.0, np.nan)
    got = RecencyForm().compute(history, entries)
    assert got["drv_qgap_ew4"].iloc[0] == pytest.approx(3.0)


def test_no_history_gives_nan_not_an_error():
    history, entries = history_of(1.0)
    got = RecencyForm().compute(history.iloc[0:0], entries)
    assert got.isna().all().all()
    assert list(got.columns) == list(RecencyForm.columns)


def test_a_driver_with_no_earlier_races_is_nan_but_teammates_are_not():
    history, _ = history_of(1.0, 2.0)
    entries = pd.DataFrame([row(2022, 3, "d0"), row(2022, 3, "rookie", team="t0")])
    got = RecencyForm().compute(history, entries)
    assert got["drv_qgap_ew4"].iloc[0] == pytest.approx((1.0 * 0.5**0.25 + 2.0) / (0.5**0.25 + 1.0))
    assert np.isnan(got["drv_qgap_ew4"].iloc[1])
    assert np.isfinite(got["team_qgap_ew4"]).all()  # the rookie inherits the team's form


def test_a_shorter_half_life_follows_the_latest_race_more_closely():
    history, entries = history_of(5.0, 5.0, 5.0, 1.0)
    got = RecencyForm().compute(history, entries).iloc[0]
    assert got["drv_qgap_ew2"] < got["drv_qgap_ew4"] < got["drv_qgap_ew8"] < 5.0
    assert got["drv_qgap_ew2"] > 1.0


def test_age_is_counted_in_races_so_a_missed_race_makes_form_older():
    # d0 raced rounds 1-2 only; d1 raced rounds 1-4. Same gaps in the rounds both raced.
    rows = [row(2022, r, "d0", quali_gap_pct=g) for r, g in ((1, 1.0), (3, 3.0))]  # skips round 2
    rows += [
        row(2022, r, "d1", team="t1", quali_gap_pct=2.0 if r < 4 else 0.0) for r in (1, 2, 3, 4)
    ]
    entries = pd.DataFrame([row(2022, 5, "d0"), row(2022, 5, "d1", team="t1")])
    got = RecencyForm().compute(pd.DataFrame(rows), entries)
    # Calendar ages before round 5: round 4 -> 0, round 3 -> 1, round 1 -> 3, so d0's two races are
    # 2 apart. A per-driver age would put them 1 apart (and only the gap between them matters for a
    # single driver's mean), which gives a different value.
    w3, w1 = 0.5 ** (1 / 2), 0.5 ** (3 / 2)
    assert got["drv_qgap_ew2"].iloc[0] == pytest.approx((3.0 * w3 + 1.0 * w1) / (w3 + w1))


# ---- opt-in: the default boards must not change ------------------------------------------------


def test_recency_columns_are_not_in_any_default_feature_set():
    for target in ("y_finish", "y_quali", "y_qgap"):
        assert not any("_ew" in c for c in feature_columns(target))
    assert not any("_ew" in c for c in feature_plan("y_quali")[1])


def test_asking_for_one_half_life_gives_only_that_half_lifes_columns():
    _, columns = feature_plan("y_quali", recency=4)
    ew = [c for c in columns if "_ew" in c]
    assert ew and all(c.endswith("_ew4") for c in ew)
    assert len(ew) == len(RecencyForm.columns) // len(HALF_LIVES)
    with pytest.raises(ValueError, match="half-life"):
        feature_plan("y_quali", recency=3)


def test_the_training_frame_carries_recency_columns_only_when_asked():
    table = build_feature_table(make_rows(years=(2022, 2023), rounds=4))
    assert not any("_ew" in c for c in training_frame(table, "y_quali").columns)
    assert any("_ew" in c for c in training_frame(table, "y_quali", include_recency=True).columns)


# ---- the gap-to-pole target ---------------------------------------------------------------------


def test_the_gap_target_is_the_races_own_gap_and_drops_rows_without_one():
    rows = make_rows(years=(2022, 2023), rounds=4)
    rows.loc[(rows["Year"] == 2023) & (rows["DriverId"] == "d0"), "quali_gap_pct"] = np.nan
    table = build_feature_table(rows)
    frame = training_frame(table, "y_qgap")
    assert frame["y_qgap"].notna().all()
    assert len(frame) == len(rows) - ((rows["Year"] == 2023) & (rows["DriverId"] == "d0")).sum()
    assert not {"y_quali", "y_finish", "y_grid"} & set(frame.columns)  # no other outcome columns
    assert not {"grid", "grid_frac"} & set(frame.columns)  # the gap is decided before the grid


def test_the_gap_target_is_refused_where_it_makes_no_sense():
    table = build_feature_table(make_rows())
    for name in ("elo", "plackett_luce"):
        with pytest.raises(ValueError, match="gap-to-pole"):
            build_model(name, "quali", table, gap_target=True)
    with pytest.raises(ValueError, match="gap-to-pole"):
        build_model("ridge", "race", table, gap_target=True)


@pytest.mark.parametrize("recency", [None, 2, 8])
@pytest.mark.parametrize("gap", [False, True])
def test_scrambling_the_target_race_onward_cannot_change_a_variants_prediction(gap, recency):
    rows = make_rows(years=(2021, 2022, 2023), rounds=6, seed=3)
    scrambled = scramble(rows, TEST_KEY, "pre_weekend")
    outputs = []
    for data in (rows, scrambled):
        table = build_feature_table(data)
        model = build_model("ridge", "quali", table, {}, gap_target=gap, recency=recency)
        model.fit(data[data["Year"] < TEST_KEY[0]])
        race = data[(data["Year"] == TEST_KEY[0]) & (data["Round"] == TEST_KEY[1])]
        entries = race[["Year", "Round", "DriverId", "Abbreviation", "TeamKey", "CircuitId"]]
        before = data[(data["Year"] < 2023) | ((data["Year"] == 2023) & (data["Round"] < 3))]
        outputs.append(model.predict_race(before, entries.copy()))
    pd.testing.assert_series_equal(outputs[0], outputs[1])
    # The model path cuts history itself, so also compare the feature rows of the target race as
    # built from the FULL scrambled table (this is where a leak in a builder would show up).
    columns = ["DriverId", *feature_plan("y_quali", recency=recency)[1]]
    pair = [
        build_feature_table(d).query("Year == 2023 and Round == 3")[columns].reset_index(drop=True)
        for d in (rows, scrambled)
    ]
    pd.testing.assert_frame_equal(*pair)


def test_variants_are_named_after_what_they_change_and_plain_choices_are_left_out():
    table = build_feature_table(make_rows())
    tuned = {
        "quali_features": {
            "ridge": {"gap_target": True, "recency": 4},
            "lightgbm": {"gap_target": False, "recency": None},
            "xgboost": {"gap_target": False, "recency": 8},
        }
    }
    names = [m.name for m in make_feature_variants(table, tuned)]
    assert names == ["ridge+gap+ew4", "xgboost+ew8"]
    assert make_feature_variants(table, {}) == []


# ---- tuning touches development seasons only -----------------------------------------------------


def test_scoring_a_feature_choice_refuses_report_seasons():
    rows = make_rows()
    model = build_model("ridge", "quali", build_feature_table(rows), {})
    with pytest.raises(TuningLeakError):
        score_on_development(model, rows, "quali", seasons=(2022,))


def test_the_plain_choice_is_listed_first_so_a_tie_keeps_it():
    assert FEATURE_CHOICES[0] == {"gap_target": False, "recency": None}
    assert len(FEATURE_CHOICES) == 2 * (1 + len(HALF_LIVES))


def test_retuning_the_settings_keeps_the_feature_choices(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"quali_features": {"ridge": {"gap_target": True}}}))
    tune([], [], pd.DataFrame(), pd.DataFrame(), tmp_path / "log.csv", params)
    assert json.loads(params.read_text())["quali_features"] == {"ridge": {"gap_target": True}}

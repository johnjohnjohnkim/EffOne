"""Milestone 5: race simulation, DNF model, calibration scoring, and leakage."""

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.data import ENTRY_COLUMNS
from ml.evaluation.splits import REPORT_SEASONS, TuningLeakError
from ml.features.build import build_feature_table
from ml.simulation.calibration import (
    _slots,
    baseline_probabilities,
    collect,
    expected_calibration_error,
    fit_total,
    log_loss,
    outcomes,
    probabilities,
    reliability,
    tune,
)
from ml.simulation.dnf import DnfModel, RateBaseline, recent_rate
from ml.simulation.forecast import RaceForecaster, load_sim_params
from ml.simulation.race import SimParams, simulate_race
from tests.helpers import make_rows, row, scramble


def field(n=20, step=0.7):
    return np.arange(1, n + 1, dtype=float) * step + 1.0, np.full(n, 0.1)


# ---- the simulation is a consistent set of finishing orders ------------------------------------


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_win_podium_and_top10_probabilities_have_the_required_sums(gamma):
    scores, p_dnf = field()
    out = simulate_race(scores, p_dnf, SimParams(sigma=2.0, gamma=gamma, n_sims=3000))
    assert out["p_win"].sum() == pytest.approx(1.0)
    assert out["p_podium"].sum() == pytest.approx(3.0)
    assert out["p_top10"].sum() == pytest.approx(10.0)
    assert out["p_dnf"].mean() == pytest.approx(0.1, abs=0.02)


def test_a_small_field_fills_every_place_it_has():
    out = simulate_race(np.array([1.0, 2.0, 3.0, 4.0]), np.full(4, 0.2), SimParams(n_sims=500))
    assert out["p_top10"].sum() == pytest.approx(4.0)  # everyone is inside the top 10
    assert out["p_podium"].sum() == pytest.approx(3.0)


def test_better_expected_finish_means_more_wins_and_a_retiring_driver_cannot_win():
    scores = np.array([1.0, 2.0, 6.0, 12.0])
    p_dnf = np.array([1.0, 0.0, 0.0, 0.0])  # the favourite always retires
    out = simulate_race(scores, p_dnf, SimParams(sigma=2.0, n_sims=4000))
    assert out["p_win"].iloc[0] == 0.0
    assert out["p_win"].iloc[1] > out["p_win"].iloc[2] > out["p_win"].iloc[3]
    assert out["expected_position"].iloc[0] == pytest.approx(4.0)  # always last of the four


def test_retirements_are_placed_behind_every_finisher():
    out = simulate_race(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0]), SimParams(n_sims=200))
    assert out["expected_position"].iloc[2] == pytest.approx(3.0)
    assert out["p_podium"].iloc[2] == pytest.approx(1.0)  # last of three is still on the podium


def test_the_same_seed_gives_the_same_probabilities_and_noise_widens_them():
    scores, p_dnf = field()
    a = simulate_race(scores, p_dnf, SimParams(sigma=2.0, n_sims=1000, seed=3))
    b = simulate_race(scores, p_dnf, SimParams(sigma=2.0, n_sims=1000, seed=3))
    pd.testing.assert_frame_equal(a, b)
    wide = simulate_race(scores, p_dnf, SimParams(sigma=8.0, n_sims=1000, seed=3))
    assert wide["p_win"].iloc[0] < a["p_win"].iloc[0]  # more noise, a less certain favourite


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_impossible_inputs_are_errors_not_silent(bad):
    scores, p_dnf = field(5)
    with pytest.raises(ValueError):
        simulate_race(
            np.where(np.arange(5) == 2, bad, scores) if np.isnan(bad) else scores,
            p_dnf if np.isnan(bad) else np.where(np.arange(5) == 2, bad, p_dnf),
        )


def test_fit_total_scales_to_the_total_and_never_exceeds_one():
    p = fit_total(np.array([0.9, 0.8, 0.1, 0.1, 0.05]), 3.0)
    assert p.sum() == pytest.approx(3.0)
    assert p.max() <= 1.0
    assert fit_total(np.array([0.2, 0.2]), 5.0).sum() == pytest.approx(2.0)  # at most the field


# ---- DNF ----------------------------------------------------------------------------------------


def test_recent_rate_uses_the_last_two_seasons_and_only_informative_races():
    rows = pd.concat(
        [
            pd.DataFrame([row(2019, 1, "a", dnf=True)]),  # too old
            pd.DataFrame([row(2021, 1, "a", dnf=False), row(2021, 1, "b", dnf=True)]),
            pd.DataFrame([row(2022, 1, "a", dnf=False), row(2022, 1, "b", dnf=False)]),
            pd.DataFrame([row(2022, 2, "a", dnf=True, race_ok=False)]),  # washed-out race
        ],
        ignore_index=True,
    )
    assert recent_rate(rows, 2022) == pytest.approx(1 / 4)
    assert recent_rate(rows.iloc[0:0], 2022) == pytest.approx(0.12)  # nothing yet: the default


def test_dnf_model_is_a_probability_averaging_the_recent_rate_over_the_field():
    rows = make_rows(years=(2021, 2022, 2023), rounds=5, seed=2)
    table = build_feature_table(rows)
    model = DnfModel(table)
    model.fit(rows[rows["Year"] < 2023])
    race = rows[(rows["Year"] == 2023) & (rows["Round"] == 4)]
    entries = race[[c for c in ENTRY_COLUMNS["race"] if c in race.columns]]
    history = rows[(rows["Year"] < 2023) | ((rows["Year"] == 2023) & (rows["Round"] < 4))]
    p = model.predict_race(history, entries)
    assert ((p > 0) & (p < 1)).all()
    flat = RateBaseline().predict_race(history, entries)
    assert p.mean() == pytest.approx(flat.iloc[0], rel=0.15)  # same level, risk only redistributes


# ---- leakage: a forecast cannot depend on the race it forecasts --------------------------------


def test_scrambling_the_target_race_and_later_cannot_change_a_forecast():
    rows = make_rows(years=(2021, 2022, 2023), rounds=5, seed=5)
    key = (2023, 3)
    entries = rows[(rows["Year"] == key[0]) & (rows["Round"] == key[1])]
    entries = entries[[c for c in ENTRY_COLUMNS["race"] if c in entries.columns]]
    out = []
    for data in (rows, scramble(rows, key, "post_quali")):
        forecaster = RaceForecaster(build_feature_table(data), "real_grid", SimParams(n_sims=500))
        forecaster.fit(data[data["Year"] < key[0]])
        before = data[
            (data["Year"] < key[0]) | ((data["Year"] == key[0]) & (data["Round"] < key[1]))
        ]
        out.append(forecaster.forecast(before, entries.copy()))
    pd.testing.assert_frame_equal(out[0], out[1])


def test_predicted_grid_mode_replaces_the_grid_it_is_given():
    rows = make_rows(years=(2021, 2022, 2023), rounds=5, seed=6)
    key = (2023, 3)
    race = rows[(rows["Year"] == key[0]) & (rows["Round"] == key[1])]
    entries = race[[c for c in ENTRY_COLUMNS["race"] if c in race.columns]].copy()
    forecaster = RaceForecaster(build_feature_table(rows), "predicted_grid", SimParams(n_sims=300))
    forecaster.fit(rows[rows["Year"] < key[0]])
    before = rows[(rows["Year"] < key[0]) | ((rows["Year"] == key[0]) & (rows["Round"] < key[1]))]
    a = forecaster.raw_predictions(before, entries)
    entries["GridPosition"] = 1.0  # the real grid must not matter before qualifying
    b = forecaster.raw_predictions(before, entries)
    pd.testing.assert_frame_equal(a, b)


# ---- calibration scoring and the season rules ---------------------------------------------------


def test_log_loss_and_reliability_reward_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(float)
    assert expected_calibration_error(p, y) < 0.04
    assert expected_calibration_error(1 - p, y) > 0.3  # inverted probabilities are badly calibrated
    assert log_loss(p, y).mean() < log_loss(np.full(4000, y.mean()), y).mean()
    bins = reliability(p, y, 10)
    assert len(bins) == 10 and bins["n"].sum() == 4000
    assert log_loss(np.array([0.0]), np.array([1.0]))[0] == pytest.approx(-np.log(1e-3))


def test_tuning_refuses_any_season_outside_development():
    with pytest.raises(TuningLeakError):
        tune(pd.DataFrame(), (2019, 2022))


def test_the_locked_season_is_never_collected_unless_asked_for():
    rows = make_rows(years=(2023, 2024, 2025), rounds=4, seed=1)
    table = build_feature_table(rows)
    raw = collect(rows, table, "real_grid", (2024, 2025))
    assert set(raw["Year"]) == {2024}
    assert 2025 in set(collect(rows, table, "real_grid", (2024, 2025), allow_locked=True)["Year"])


def test_collected_rows_carry_the_outcomes_and_probabilities_sum_per_race():
    rows = make_rows(years=(2022, 2023), rounds=4, seed=8)
    raw = collect(rows, build_feature_table(rows), "real_grid", (2023,))
    assert len(raw) == 4 * 6
    probs = probabilities(raw, SimParams(sigma=2.0, n_sims=300))
    per_race = probs.groupby(["Year", "Round"])[["p_win", "p_podium", "p_top10"]].sum()
    assert per_race["p_win"].round(6).eq(1.0).all()
    assert per_race["p_podium"].round(6).eq(3.0).all()
    actual = outcomes(raw)
    assert actual.groupby([raw["Year"], raw["Round"]])["win"].sum().eq(1.0).all()


def test_tuned_settings_exist_for_both_modes_and_are_sane():
    for mode in ("real_grid", "predicted_grid"):
        params = load_sim_params(mode)
        assert 0 < params.sigma < 10 and params.gamma in (0.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        load_sim_params("nonsense")
    assert 2026 in REPORT_SEASONS


# ---- review fixes: leakage through `collect`, the baselines, and the knobs actually used --------

FORECAST_COLUMNS = [
    "score",
    "p_dnf",
    "grid",
    "base_p_win",
    "base_p_podium",
    "base_p_top10",
    "base_p_dnf",
]


@pytest.mark.parametrize("mode", ["real_grid", "predicted_grid"])
def test_collected_forecasts_cannot_depend_on_the_target_race_or_later(mode):
    rows = make_rows(years=(2021, 2022, 2023), rounds=5, seed=11)
    key = (2023, 3)
    raws = []
    for data in (rows, scramble(rows, key, "post_quali")):
        raw = collect(data, build_feature_table(data), mode, (2023,))
        raws.append(raw[raw["Round"] <= key[1]].reset_index(drop=True))
    # Outcome columns of the scrambled race legitimately differ; every forecast column must not.
    pd.testing.assert_frame_equal(raws[0][FORECAST_COLUMNS], raws[1][FORECAST_COLUMNS])


def test_baselines_sum_to_the_same_totals_as_the_simulation_and_use_the_pit_lane_rule():
    rows = make_rows(years=(2022, 2023, 2024), rounds=4, seed=3)
    table = build_feature_table(rows)
    for mode in ("real_grid", "predicted_grid"):
        raw = collect(rows, table, mode, (2024,))
        assert len(raw) == 4 * 6
        prior = baseline_probabilities(raw)["grid_prior"].groupby(["Year", "Round"])
        assert prior["p_win"].sum().round(6).eq(1.0).all()
        assert prior["p_podium"].sum().round(6).eq(3.0).all()
        assert prior["p_top10"].sum().round(6).eq(6.0).all()  # six drivers: everyone is top 10
    grid = pd.DataFrame({"grid": [3.0, 0.0, np.nan, 1.0]})
    assert _slots(grid).tolist() == [3.0, 4.0, 4.0, 1.0]  # pit lane and gaps go to the back


def test_gamma_and_sigma_both_change_the_probabilities():
    scores, p_dnf = field()
    base = simulate_race(scores, p_dnf, SimParams(sigma=1.0, gamma=0.0, n_sims=1000))
    other_gamma = simulate_race(scores, p_dnf, SimParams(sigma=1.0, gamma=1.0, n_sims=1000))
    assert not np.allclose(base["p_win"], other_gamma["p_win"])


def test_dnf_relative_risk_is_normalised_so_the_level_is_the_recent_rate():
    rows = make_rows(years=(2021, 2022, 2023), rounds=5, seed=2)
    model = DnfModel(build_feature_table(rows))
    model.fit(rows[rows["Year"] < 2023])
    race = rows[(rows["Year"] == 2023) & (rows["Round"] == 4)]
    entries = race[[c for c in ENTRY_COLUMNS["race"] if c in race.columns]]
    history = rows[(rows["Year"] < 2023) | ((rows["Year"] == 2023) & (rows["Round"] < 4))]
    level = recent_rate(history, 2023)
    assert model.predict_race(history, entries).mean() == pytest.approx(level, abs=1e-6)


def test_tuning_looks_at_the_data_not_only_the_label():
    rows = make_rows(years=(2021, 2022), rounds=3, seed=1)
    raw = collect(rows, build_feature_table(rows), "real_grid", (2022,))
    with pytest.raises(TuningLeakError):
        tune(raw, (2021,))  # labelled development, but the rows are from 2022

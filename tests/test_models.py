"""The model zoo: interface, leakage through the model path, grid handling, Elo and Plackett-Luce."""

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.data import ENTRY_COLUMNS, TARGET_COLUMNS
from ml.features.build import build_feature_table
from ml.models.base import FeatureModel, feature_plan
from ml.models.chain import DropGrid, PredictedGridRace
from ml.models.elo import EloModel
from ml.models.factories import ridge
from ml.models.plackett_luce import PlackettLuceModel, race_log_likelihood
from ml.models.zoo import make_models, make_no_grid_models, make_predicted_grid_models
from tests.helpers import make_rows, row, scramble

TEST_KEY = (2023, 3)


@pytest.fixture(scope="module")
def rows():
    return make_rows(years=(2021, 2022, 2023), rounds=6, seed=3)


@pytest.fixture(scope="module")
def table(rows):
    return build_feature_table(rows)


def entries_for(rows, key, target):
    """The entry columns the harness hands a model for `key`."""
    race = rows[(rows["Year"] == key[0]) & (rows["Round"] == key[1])]
    return race[[c for c in ENTRY_COLUMNS[target] if c in rows.columns]].copy()


def history_before(rows, key):
    return rows[(rows["Year"] < key[0]) | ((rows["Year"] == key[0]) & (rows["Round"] < key[1]))]


def train_rows(rows):
    return rows[rows["Year"] < TEST_KEY[0]]


ALL_MODELS = [(t, m.name) for t in ("race", "quali") for m in make_models(t, pd.DataFrame())]


def fresh_models(target, table):
    return {m.name: m for m in make_models(target, table)}


# ---- interface -------------------------------------------------------------------------------


@pytest.mark.parametrize(("target", "name"), ALL_MODELS)
def test_every_model_fits_and_scores_every_driver(rows, table, target, name):
    model = fresh_models(target, table)[name]
    model.fit(train_rows(rows))
    entries = entries_for(rows, TEST_KEY, target)
    scores = model.predict_race(history_before(rows, TEST_KEY), entries)
    assert scores.index.equals(entries.index)
    assert np.isfinite(scores).all()
    assert scores.nunique() > 1  # a trained model tells drivers apart


@pytest.mark.parametrize(("target", "name"), ALL_MODELS)
def test_models_are_deterministic(rows, table, target, name):
    first = fresh_models(target, table)[name]
    second = fresh_models(target, table)[name]
    entries = entries_for(rows, TEST_KEY, target)
    out = []
    for model in (first, second):
        model.fit(train_rows(rows))
        out.append(model.predict_race(history_before(rows, TEST_KEY), entries))
    pd.testing.assert_series_equal(out[0], out[1])


def test_predicting_before_fitting_is_an_error(rows, table):
    entries = entries_for(rows, TEST_KEY, "race")
    for model in (fresh_models("race", table)["ridge"], PlackettLuceModel("pl", table)):
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_race(history_before(rows, TEST_KEY), entries)


def test_refitting_on_the_same_races_is_skipped_and_on_new_races_retrains(rows, table):
    model = fresh_models("race", table)["ridge"]
    model.fit(train_rows(rows))
    first = model._estimator
    model.fit(train_rows(rows).copy())
    assert model._estimator is first  # identical training races: no work
    model.fit(rows[rows["Year"] < 2022])
    assert model._estimator is not first


# ---- leakage through the model path ---------------------------------------------------------


@pytest.mark.parametrize(("target", "name"), ALL_MODELS)
def test_scrambling_the_target_race_onward_cannot_change_a_models_prediction(rows, target, name):
    """Fit on earlier seasons, predict one race; destroy everything from that race on; same scores."""
    stage = "post_quali" if target == "race" else "pre_weekend"
    scrambled = scramble(rows, TEST_KEY, stage)
    outputs = []
    for data in (rows, scrambled):
        table = build_feature_table(data)
        model = fresh_models(target, table)[name]
        model.fit(train_rows(data))
        entries = entries_for(data, TEST_KEY, target)
        outputs.append(model.predict_race(history_before(data, TEST_KEY), entries))
    pd.testing.assert_series_equal(outputs[0], outputs[1])


def test_the_harness_never_hands_a_model_outcomes_or_a_grid_for_qualifying(rows):
    quali = entries_for(rows, TEST_KEY, "quali")
    race = entries_for(rows, TEST_KEY, "race")
    outcomes = {
        "Position",
        "QualiPosition",
        "Points",
        "dnf",
        "pace_gap_pct",
        "quali_gap_pct",
        "race_ok",
    }
    assert not outcomes & set(quali.columns) and not outcomes & set(race.columns)
    assert "GridPosition" not in quali.columns and not {"wx_air_temp"} & set(quali.columns)
    assert {"GridPosition", "wx_air_temp"} <= set(race.columns)
    assert TARGET_COLUMNS == {"race": "Position", "quali": "QualiPosition"}


# ---- grids: user-supplied and predicted -------------------------------------------------------


def test_the_race_model_accepts_a_user_supplied_grid_and_uses_it(rows, table):
    model = fresh_models("race", table)["ridge"]
    model.fit(train_rows(rows))
    entries = entries_for(rows, TEST_KEY, "race")
    history = history_before(rows, TEST_KEY)
    forward = entries.assign(GridPosition=np.arange(1.0, len(entries) + 1))
    backward = entries.assign(GridPosition=np.arange(float(len(entries)), 0, -1))
    a, b = model.predict_race(history, forward), model.predict_race(history, backward)
    assert not np.allclose(a, b)  # the grid the user typed in changes the forecast
    pd.testing.assert_series_equal(a, model.predict_race(history, forward))
    # With a positive weight on grid, starting up front means a better (lower) predicted score.
    best_first = a.sort_values().index[0]
    assert forward.loc[best_first, "GridPosition"] <= forward["GridPosition"].median()


class Spy:
    """Records the entry list it is given, returns a fixed score."""

    def __init__(self, name, scores=None):
        self.name = name
        self.seen = []
        self.scores = scores

    def fit(self, train):
        pass

    def predict_race(self, history, race):
        self.seen.append(race.copy())
        if self.scores is not None:
            return pd.Series(self.scores, index=race.index)
        return pd.Series(np.arange(len(race), dtype=float), index=race.index)


def test_a_predicted_grid_is_the_rank_of_the_qualifying_models_scores(rows):
    entries = entries_for(rows, TEST_KEY, "race")
    quali_scores = [3.0, 1.0, 2.0, 6.0, 5.0, 4.0]  # driver order is the entry order
    grid_model, race_model = Spy("quali", quali_scores), Spy("race")
    PredictedGridRace(race_model, grid_model).predict_race(history_before(rows, TEST_KEY), entries)
    assert "GridPosition" not in grid_model.seen[0].columns  # the grid is what is being predicted
    assert race_model.seen[0]["GridPosition"].tolist() == [3.0, 1.0, 2.0, 6.0, 5.0, 4.0]


def test_the_predicted_grid_ignores_the_real_grid_in_the_entry_list(rows, table):
    entries = entries_for(rows, TEST_KEY, "race")
    other_real_grid = entries.assign(GridPosition=entries["GridPosition"].iloc[::-1].to_numpy())
    grid_model = FeatureModel("g", ridge, table, "y_quali")
    chain = PredictedGridRace(fresh_models("race", table)["ridge"], grid_model)
    chain.fit(train_rows(rows))
    history = history_before(rows, TEST_KEY)
    pd.testing.assert_series_equal(
        chain.predict_race(history, entries), chain.predict_race(history, other_real_grid)
    )


def test_drop_grid_hides_the_grid_from_the_model_it_wraps(rows):
    entries = entries_for(rows, TEST_KEY, "race")
    spy = Spy("inner")
    DropGrid(spy).predict_race(history_before(rows, TEST_KEY), entries)
    assert "GridPosition" not in spy.seen[0].columns and DropGrid(spy).name == "inner [no grid]"


def test_no_grid_models_use_no_grid_feature_and_work_with_the_grid_hidden(rows, table):
    builders, columns = feature_plan("y_finish", with_grid=False)
    assert not {"grid", "grid_pit_lane", "grid_frac"} & set(columns)
    assert all(b.stage != "post_quali" for b in builders)
    for model in make_no_grid_models(table):
        model.fit(train_rows(rows))
        scores = model.predict_race(
            history_before(rows, TEST_KEY), entries_for(rows, TEST_KEY, "race")
        )
        assert np.isfinite(scores).all() and model.name.endswith("[no grid]")


def test_the_predicted_grid_zoo_has_independent_baselines_chained_and_no_grid_models(table):
    models, baseline_names = make_predicted_grid_models(table)
    names = [m.name for m in models]
    assert len(names) == len(set(names))
    assert baseline_names == [
        "grid_equals_finish [predicted grid]",
        "previous_race_equals_finish [no grid]",
        "mean_last5_finish [no grid]",
    ]
    assert set(baseline_names) <= set(names)
    assert "ridge [predicted grid]" in names and "ridge [no grid]" in names
    assert "elo [no grid]" in names and "elo [predicted grid]" not in names  # Elo never uses a grid


def test_tied_qualifying_scores_share_a_grid_slot_instead_of_an_arbitrary_order(rows):
    entries = entries_for(rows, TEST_KEY, "race")
    grid_model, race_model = Spy("quali", [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]), Spy("race")
    PredictedGridRace(race_model, grid_model).predict_race(history_before(rows, TEST_KEY), entries)
    assert race_model.seen[0]["GridPosition"].tolist() == [1.5, 1.5, 3.0, 5.0, 5.0, 5.0]


# ---- Elo -------------------------------------------------------------------------------------


def elo_history():
    rows = []
    for rnd in range(1, 6):
        rows += [
            row(2023, rnd, "ace", team="fast", Position=1.0, QualiPosition=1.0),
            row(2023, rnd, "mid", team="mid", Position=2.0, QualiPosition=2.0),
            row(2023, rnd, "back", team="slow", Position=3.0, QualiPosition=3.0),
        ]
    return pd.DataFrame(rows)


def test_elo_ranks_consistent_winners_above_consistent_losers():
    history = elo_history()
    race = pd.DataFrame(
        [row(2023, 6, d, team=t) for d, t in (("back", "slow"), ("ace", "fast"), ("mid", "mid"))]
    )
    scores = EloModel("elo").predict_race(history, race)
    assert scores["ace" == race["DriverId"]].iloc[0] < scores["mid" == race["DriverId"]].iloc[0]
    assert scores["mid" == race["DriverId"]].iloc[0] < scores["back" == race["DriverId"]].iloc[0]


def test_elo_gives_unseen_drivers_and_teams_a_neutral_rating():
    race = pd.DataFrame([row(2023, 6, "ace", team="fast"), row(2023, 6, "rookie", team="newteam")])
    scores = EloModel("elo").predict_race(elo_history(), race)
    assert scores.iloc[1] == 0.0 and scores.iloc[0] < 0  # the proven winner beats the unknown


def test_elo_ratings_are_replayed_from_history_only_and_skip_races_marked_not_ok():
    history = elo_history()
    flipped = history.copy()
    flipped.loc[flipped["Round"] == 5, ["Position", "QualiPosition"]] = 3.0
    flipped.loc[flipped["Round"] == 5, "race_ok"] = False
    race = pd.DataFrame([row(2023, 6, "ace", team="fast"), row(2023, 6, "back", team="slow")])
    base = EloModel("elo").predict_race(history[history["Round"] <= 4], race)
    # Round 5 is flagged not ok, so adding it (even scrambled) changes nothing.
    pd.testing.assert_series_equal(base, EloModel("elo").predict_race(flipped, race))


def test_elo_shrinks_ratings_at_a_new_season_and_resets_teams_at_a_regulation_change():
    model = EloModel("elo", season_shrink=0.5, regulation_team_shrink=0.1)
    drivers, teams = {"a": 100.0}, {"t": 100.0}
    model._new_season(2024, drivers, teams)
    assert drivers["a"] == 50.0 and teams["t"] == 50.0
    drivers, teams = {"a": 100.0}, {"t": 100.0}
    model._new_season(2026, drivers, teams)  # new rules: the car is reset, the driver much less
    assert drivers["a"] == 50.0 and teams["t"] == pytest.approx(10.0)


def test_elo_target_selects_the_qualifying_column():
    history = elo_history()
    history["QualiPosition"] = history["QualiPosition"].iloc[::-1].to_numpy()  # quali order differs
    race = pd.DataFrame([row(2023, 6, "ace", team="fast"), row(2023, 6, "back", team="slow")])
    race_scores = EloModel("e", target="race").predict_race(history, race)
    quali_scores = EloModel("e", target="quali").predict_race(history, race)
    assert race_scores.iloc[0] < race_scores.iloc[1]
    assert quali_scores.iloc[0] != race_scores.iloc[0]


# ---- Plackett-Luce -------------------------------------------------------------------------


def brute_force_log_likelihood(u):
    total = 0.0
    for k in range(len(u)):
        total += u[k] - np.log(np.exp(u[k:]).sum())
    return total


def test_plackett_luce_log_likelihood_matches_a_brute_force_calculation():
    u = np.array([1.2, 0.3, -0.5, 0.9])
    ll, _ = race_log_likelihood(u)
    assert ll == pytest.approx(brute_force_log_likelihood(u - u.max()))
    # a tiny hand case: two drivers, probability the first one wins = e^1 / (e^1 + e^0)
    assert race_log_likelihood(np.array([1.0, 0.0]))[0] == pytest.approx(np.log(np.e / (np.e + 1)))


def test_plackett_luce_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    u = rng.normal(size=7)
    _, grad = race_log_likelihood(u)
    for i in range(len(u)):
        bump = np.zeros_like(u)
        bump[i] = 1e-6
        numeric = (race_log_likelihood(u + bump)[0] - race_log_likelihood(u - bump)[0]) / 2e-6
        assert grad[i] == pytest.approx(numeric, abs=1e-5)


def test_plackett_luce_learns_the_direction_of_the_signal(rows):
    """Make finishing order follow the grid exactly: the fitted weight on grid must favour a low grid."""
    ordered = rows.copy()
    ordered["Position"] = ordered["GridPosition"]
    ordered["QualiPosition"] = ordered["GridPosition"]
    table = build_feature_table(ordered)
    model = PlackettLuceModel("pl", table)
    model.fit(ordered[ordered["Year"] < 2023])
    weights = pd.Series(model.weights, index=model.columns)
    assert weights["grid"] < 0  # a bigger grid number means a lower utility
    scores = model.predict_race(
        history_before(ordered, TEST_KEY), entries_for(ordered, TEST_KEY, "race")
    )
    grid = entries_for(ordered, TEST_KEY, "race")["GridPosition"]
    assert scores.corr(grid, method="spearman") > 0.9


def test_elo_ignores_any_race_at_or_after_the_one_being_predicted():
    history = elo_history()
    race = pd.DataFrame([row(2023, 3, "ace", team="fast"), row(2023, 3, "back", team="slow")])
    clean = EloModel("elo").predict_race(history[history["Round"] < 3], race)
    polluted = history.copy()  # includes round 3 (the target) and later, with the order reversed
    polluted.loc[polluted["Round"] >= 3, "Position"] = polluted.loc[
        polluted["Round"] >= 3, "Position"
    ].map({1.0: 3.0, 2.0: 2.0, 3.0: 1.0})
    pd.testing.assert_series_equal(clean, EloModel("elo").predict_race(polluted, race))


@pytest.mark.parametrize(("year", "team_factor"), [(2024, 0.75), (2026, 0.3)])
def test_elo_applies_the_new_season_shrink_to_the_first_race_of_a_season(year, team_factor):
    history = elo_history()  # five 2023 races
    model = EloModel("elo", season_shrink=0.75, regulation_team_shrink=0.3)
    drivers, teams, season = model._replay(history)
    assert season == 2023
    same_season = pd.DataFrame([row(2023, 6, "ace", team="fast")])
    new_season = pd.DataFrame([row(year, 1, "ace", team="fast")])
    before_score = model.predict_race(history, same_season).iloc[0]
    after_score = model.predict_race(history, new_season).iloc[0]
    expected = -(drivers["ace"] * 0.75 + teams["fast"] * team_factor)
    assert before_score == pytest.approx(-(drivers["ace"] + teams["fast"]))
    assert after_score == pytest.approx(expected)


def test_elo_refuses_a_history_with_gaps_in_race_ok():
    history = elo_history()
    history["race_ok"] = history["race_ok"].astype(object)
    history.loc[0, "race_ok"] = None
    race = pd.DataFrame([row(2023, 6, "ace", team="fast")])
    with pytest.raises(ValueError, match="race_ok"):
        EloModel("elo").predict_race(history, race)


# ---- training data selection and convergence ------------------------------------------------


def test_training_on_races_missing_from_the_feature_table_is_an_error(rows, table):
    from ml.models.base import training_rows

    foreign = pd.DataFrame({"Year": [1999], "Round": [1]})
    with pytest.raises(ValueError, match="missing from the feature table"):
        training_rows(table, foreign, "y_finish", False)


def test_training_with_no_usable_rows_is_an_error_not_a_crash_deep_in_the_estimator(rows, table):
    from ml.models.base import training_rows

    nothing_to_learn_from = table[(table["Year"] == 2021) & (table["Round"] == 1)].copy()
    nothing_to_learn_from["race_ok"] = False
    with pytest.raises(ValueError, match="no usable training rows"):
        training_rows(nothing_to_learn_from, nothing_to_learn_from, "y_finish", False)


def test_the_first_season_has_nothing_to_train_on_and_says_so(rows, table):
    model = fresh_models("race", table)["ridge"]
    first_season_only = rows[rows["Year"] == 2021]
    # training_frame drops nothing here, but a model asked to learn from no races at all cannot.
    with pytest.raises(ValueError, match="no usable training rows|missing from"):
        model.fit(first_season_only.iloc[0:0])


def test_plackett_luce_reports_whether_the_optimiser_converged(rows, table):
    model = PlackettLuceModel("pl", table)
    assert model.converged is None
    model.fit(train_rows(rows))
    assert model.converged is True


def test_a_driver_with_no_grid_and_no_result_in_the_entry_list_is_handled_end_to_end(rows):
    """A late non-starter: entered, but with a missing grid slot and (in the table) no outcome."""
    data = rows.copy()
    target = (data["Year"] == 2023) & (data["Round"] == 3)
    ghost = data[target].index[0]
    data.loc[ghost, ["GridPosition", "Position", "QualiPosition"]] = np.nan
    table = build_feature_table(data)
    entries = entries_for(data, TEST_KEY, "race")
    assert len(entries) == 6 and entries["GridPosition"].isna().sum() == 1
    history = history_before(data, TEST_KEY)
    models = fresh_models("race", table)
    for name in ("ridge", "lightgbm", "plackett_luce", "elo"):
        models[name].fit(data[data["Year"] < 2023])
        scores = models[name].predict_race(history, entries)
        assert len(scores) == 6 and np.isfinite(scores).all(), name
    grid_model = FeatureModel("g", ridge, table, "y_quali")
    chain = PredictedGridRace(models["ridge"], grid_model)
    chain.fit(data[data["Year"] < 2023])
    chained = chain.predict_race(history, entries)
    assert len(chained) == 6 and np.isfinite(chained).all()


def test_the_pre_qualifying_board_has_no_model_that_uses_realised_weather(table):
    models, _ = make_predicted_grid_models(table)
    assert not any("weather" in m.name for m in models)
    assert all(
        m.model.include_scenario is False
        for m in models
        if hasattr(m, "model") and hasattr(m.model, "include_scenario")
    )

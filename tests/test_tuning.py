"""Validation scheme and tuning: season roles, the development-only guard, the trial log."""

import json

import pandas as pd
import pytest

from ml import tuning
from ml.evaluation import __main__ as cli
from ml.evaluation.splits import (
    DEV_SEASONS,
    LOCKED_SEASON,
    REPORT_SEASONS,
    SEASON_SETS,
    TuningLeakError,
    check_development_only,
    walk_forward_splits,
)
from ml.features.build import build_feature_table
from ml.models.tuned import load_tuned, params_for
from ml.models.zoo import MODEL_NAMES, build_model, make_models
from tests.helpers import make_rows


@pytest.fixture(scope="module")
def history():
    return make_rows(years=(2018, 2019, 2020, 2021, 2022), rounds=3, seed=21)


@pytest.fixture(scope="module")
def table(history):
    return build_feature_table(history)


# ---- season roles ----------------------------------------------------------------------------


def test_the_season_roles_are_disjoint_and_the_locked_season_belongs_to_neither():
    assert DEV_SEASONS == (2019, 2020, 2021) and REPORT_SEASONS == (2022, 2023, 2024, 2026)
    assert not set(DEV_SEASONS) & set(REPORT_SEASONS)
    assert LOCKED_SEASON not in DEV_SEASONS + REPORT_SEASONS
    assert SEASON_SETS["all"] is None and SEASON_SETS["dev"] == DEV_SEASONS


def test_test_seasons_limit_what_is_scored_but_training_still_uses_every_earlier_season():
    splits = walk_forward_splits([2018, 2019, 2020, 2021, 2022], test_seasons=(2021, 2022))
    assert [s.test_year for s in splits] == [2021, 2022]
    assert splits[0].train_years == (2018, 2019, 2020)  # earlier dev seasons still train the model
    assert splits[1].train_years == (2018, 2019, 2020, 2021)


def test_the_locked_season_is_not_scored_even_if_it_is_listed_in_test_seasons():
    splits = walk_forward_splits([2023, 2024, 2025], test_seasons=(2024, 2025))
    assert [s.test_year for s in splits] == [2024]
    allowed = walk_forward_splits([2023, 2024, 2025], test_seasons=(2024, 2025), allow_locked=True)
    assert [s.test_year for s in allowed] == [2024, 2025]


def test_development_only_guard_rejects_report_and_locked_seasons():
    check_development_only(DEV_SEASONS)
    check_development_only((2020,))
    for bad in [(2022,), (2025,), (2021, 2024), REPORT_SEASONS]:
        with pytest.raises(TuningLeakError, match="development seasons"):
            check_development_only(bad)


# ---- the tuner can only score development seasons ---------------------------------------------


def test_scoring_a_candidate_asks_the_harness_for_development_seasons_only(
    history, table, monkeypatch
):
    seen = []
    real = tuning.evaluate

    def spy(models, results, target, **kwargs):
        seen.append(kwargs["test_seasons"])
        return real(models, results, target, **kwargs)

    monkeypatch.setattr(tuning, "evaluate", spy)
    model = build_model("ridge", "quali", table)
    spearman, races = tuning.score_on_development(model, history, "quali")
    assert seen == [DEV_SEASONS] and races > 0 and -1 <= spearman <= 1


def test_the_tuner_refuses_to_score_a_report_or_locked_season(history, table):
    model = build_model("ridge", "quali", table)
    for seasons in [(2022,), (2025,), (2021, 2022)]:
        with pytest.raises(TuningLeakError):
            tuning.score_on_development(model, history, "quali", seasons=seasons)


def test_nothing_outside_the_development_seasons_appears_in_a_tuning_score(
    history, table, monkeypatch
):
    """Even if the harness were asked for more, the tuner checks what was actually scored."""
    real = tuning.evaluate
    monkeypatch.setattr(
        tuning, "evaluate", lambda models, results, target, **kw: real(models, results, target)
    )  # ignores the restriction, so 2022 gets scored
    with pytest.raises(TuningLeakError):
        tuning.score_on_development(build_model("ridge", "quali", table), history, "quali")


# ---- the search, the log and the chosen settings ----------------------------------------------


@pytest.fixture
def tiny_grids(monkeypatch):
    monkeypatch.setitem(tuning.GRIDS, "ridge", {"alpha": [100, 10]})
    monkeypatch.setitem(tuning.GRIDS, "elo", {"k_driver": [15, 60]})


def test_candidates_follow_the_grid_in_a_fixed_order_with_the_simplest_first():
    assert tuning.candidates("ridge")[0] == {"alpha": 300}
    assert len(tuning.candidates("elo")) == 27 and len(tuning.candidates("random_forest")) == 9
    assert tuning.candidates("random_forest")[0] == {"max_depth": 4, "min_samples_leaf": 40}
    assert set(tuning.GRIDS) == set(MODEL_NAMES)


def test_every_trial_is_logged_and_exactly_one_setting_is_chosen_per_model(
    history, table, tiny_grids, tmp_path
):
    log, params = tmp_path / "log.csv", tmp_path / "params.json"
    chosen = tuning.tune(["quali"], ["ridge", "elo"], history, table, log, params)
    trials = pd.read_csv(log)
    assert len(trials) == 4 and set(trials["model"]) == {"ridge", "elo"}
    assert (trials.groupby("model")["chosen"].sum() == 1).all()
    assert trials["dev_seasons"].eq("2019 2020 2021").all() and (trials["dev_races"] > 0).all()
    best = trials[trials["model"] == "ridge"].sort_values("dev_spearman", ascending=False).iloc[0]
    assert json.loads(best["params"]) == chosen["quali"]["ridge"]
    assert json.loads(params.read_text())["quali"]["ridge"] == chosen["quali"]["ridge"]


def test_the_chosen_settings_carry_their_provenance(history, table, tiny_grids, tmp_path):
    tuning.tune(["quali"], ["ridge"], history, table, tmp_path / "log.csv", tmp_path / "p.json")
    provenance = json.loads((tmp_path / "p.json").read_text())["provenance"]
    assert provenance["dev_seasons"] == [2019, 2020, 2021] and provenance["trials"] == 2
    assert "development" in provenance["selection"] and provenance["log"].endswith("tuning_log.csv")


def test_ties_go_to_the_setting_listed_first(history, table, tmp_path, monkeypatch):
    monkeypatch.setitem(tuning.GRIDS, "elo", {"k_driver": [30, 30.0001], "k_team": [50]})
    monkeypatch.setattr(
        tuning, "score_on_development", lambda *a, **k: (0.5, 10)
    )  # identical scores
    chosen = tuning.tune(
        ["quali"], ["elo"], history, table, tmp_path / "l.csv", tmp_path / "p.json"
    )
    assert chosen["quali"]["elo"]["k_driver"] == 30


def test_the_log_is_a_record_across_runs_not_a_cache(history, table, tiny_grids, tmp_path):
    log, params = tmp_path / "log.csv", tmp_path / "params.json"
    tuning.tune(["quali"], ["ridge"], history, table, log, params)
    tuning.tune(["quali"], ["ridge"], history, table, log, params)
    assert len(pd.read_csv(log)) == 4  # both runs are kept


def test_dry_run_lists_trials_and_scores_nothing(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["ml.tuning", "--dry-run", "--models", "ridge", "--targets", "quali"]
    )
    monkeypatch.setattr(tuning, "load_history", lambda: pytest.fail("a dry run must not load data"))
    assert tuning.main() == 0
    assert "5 trials on development seasons" in capsys.readouterr().out


# ---- the zoo reads the tuned settings ---------------------------------------------------------


def test_missing_tuned_file_means_defaults(tmp_path):
    assert load_tuned(tmp_path / "nope.json") == {}
    assert params_for({}, "race", "ridge") == {}
    assert params_for({"race": {"ridge": {"alpha": 7}}}, "race", "ridge") == {"alpha": 7}
    assert params_for({"race": {"ridge": {"alpha": 7}}}, "quali", "ridge") == {}


def test_the_zoo_builds_models_with_the_tuned_settings(history, table):
    tuned = {"quali": {"ridge": {"alpha": 7.0}, "elo": {"k_driver": 99.0}}}
    models = {m.name: m for m in make_models("quali", table, tuned)}
    models["ridge"].fit(history[history["Year"] < 2022])
    assert models["ridge"]._estimator.named_steps["model"].alpha == 7.0
    assert models["elo"].k_driver == 99.0
    defaults = {m.name: m for m in make_models("quali", table, {})}
    defaults["ridge"].fit(history[history["Year"] < 2022])
    assert defaults["ridge"]._estimator.named_steps["model"].alpha == 30.0
    assert defaults["elo"].k_driver == 30.0


def test_tuned_settings_for_one_target_do_not_leak_into_the_other(table):
    tuned = {"race": {"elo": {"k_driver": 99.0}}}
    assert {m.name: m for m in make_models("quali", table, tuned)}["elo"].k_driver == 30.0
    assert {m.name: m for m in make_models("race", table, tuned)}["elo"].k_driver == 99.0


def test_an_unknown_setting_is_an_error_not_silently_ignored(table):
    model = build_model("ridge", "quali", table, {"alphaa": 3})
    with pytest.raises(TypeError):
        model._make()  # the estimator is only built when fitting; a typo must fail loudly then


# ---- the evaluation CLI understands season roles ----------------------------------------------


def test_season_choices_map_to_the_right_sets():
    assert cli.test_seasons("dev", False) == DEV_SEASONS
    assert cli.test_seasons("report", False) == REPORT_SEASONS
    assert cli.test_seasons("all", False) is None
    assert cli.test_seasons("report", True) == (*REPORT_SEASONS, LOCKED_SEASON)
    assert cli.test_seasons("dev", True) == DEV_SEASONS  # the locked season never joins dev

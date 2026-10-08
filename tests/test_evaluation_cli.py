"""`python -m ml.evaluation`: leaderboards per target, the locked season, and the exit codes."""

import pandas as pd
import pytest

from ml.evaluation import __main__ as cli
from ml.evaluation.baselines import PreviousQualiBaseline, PreviousRaceBaseline
from tests.helpers import make_rows


class Constant:
    name = "constant"

    def fit(self, train):
        pass

    def predict_race(self, history, race):
        return pd.Series(0.0, index=race.index)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFONE_DATA_DIR", str(tmp_path))
    table = make_rows(years=(2022, 2023, 2024, 2025), rounds=4, seed=5)
    monkeypatch.setattr(cli, "load_history", lambda: table)
    return tmp_path / "reports", table


def run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["ml.evaluation", *argv])
    return cli.main()


def test_the_full_zoo_runs_for_the_qualifying_target_and_writes_a_leaderboard(
    setup, monkeypatch, capsys
):
    reports, _ = setup
    status = run(monkeypatch, "--models", "all", "--target", "quali", "--n-boot", "50")
    board = pd.read_csv(reports / "leaderboard_quali.csv")
    expected = {
        "previous_quali_equals_quali",
        "previous_race_equals_quali",
        "mean_last5_quali",
        "ridge",
        "random_forest",
        "lightgbm",
        "xgboost",
        "lightgbm_lambdarank",
        "xgboost_rank",
        "elo",
        "plackett_luce",
    }
    assert set(board["model"]) == expected
    assert board["is_baseline"].sum() == 3
    printed = capsys.readouterr().out
    assert "=== quali" in printed
    # The exit code must agree with the saved leaderboard: 3 exactly when nobody clearly beats.
    nobody_clearly_beats = not board["clearly_beats"].any()
    assert status == (cli.NO_MODEL_BEATS_BASELINES if nobody_clearly_beats else 0)
    assert ("NO MODEL BEATS THE BEST BASELINE" in printed) == nobody_clearly_beats


def test_the_race_and_predicted_grid_leaderboards_include_every_model(setup, monkeypatch):
    reports, _ = setup
    run(monkeypatch, "--models", "all", "--target", "race", "--n-boot", "50")
    run(monkeypatch, "--models", "all", "--target", "race_predicted_grid", "--n-boot", "50")
    race = set(pd.read_csv(reports / "leaderboard_race.csv")["model"])
    chained = set(pd.read_csv(reports / "leaderboard_race_predicted_grid.csv")["model"])
    assert {"grid_equals_finish", "previous_race_equals_finish", "mean_last5_finish"} <= race
    assert "lightgbm_with_weather" in race
    assert "grid_equals_finish [predicted grid]" in chained and "ridge [no grid]" in chained
    assert {"previous_race_equals_finish [no grid]", "mean_last5_finish [no grid]"} <= chained


def test_the_locked_season_is_never_scored_unless_asked(setup, monkeypatch):
    reports, _ = setup
    run(monkeypatch, "--target", "quali", "--models", "baselines", "--n-boot", "50")
    assert 2025 not in set(pd.read_csv(reports / "per_race_quali.csv")["Year"])
    run(
        monkeypatch,
        "--target",
        "quali",
        "--models",
        "baselines",
        "--n-boot",
        "50",
        "--allow-locked",
    )
    assert 2025 in set(pd.read_csv(reports / "per_race_quali.csv")["Year"])


def test_it_exits_with_a_distinct_code_when_no_model_beats_the_best_baseline(
    setup, monkeypatch, capsys
):
    _, table = setup
    order = {d: i + 1.0 for i, d in enumerate(sorted(table["DriverId"].unique()))}
    table["QualiPosition"] = table["DriverId"].map(
        order
    )  # last qualifying is now a perfect predictor
    monkeypatch.setattr(
        cli,
        "build_models",
        lambda target, which, history: (
            "quali",
            [PreviousQualiBaseline(), Constant()],
            ["previous_quali_equals_quali"],
        ),
    )
    assert (
        run(monkeypatch, "--target", "quali", "--models", "all", "--n-boot", "50")
        == cli.NO_MODEL_BEATS_BASELINES
    )
    assert "NO MODEL BEATS THE BEST BASELINE" in capsys.readouterr().out


def test_the_predicted_grid_board_is_informational_and_never_sets_the_failing_exit_code(
    setup, monkeypatch
):
    _, table = setup
    order = {d: i + 1.0 for i, d in enumerate(sorted(table["DriverId"].unique()))}
    table["Position"] = table["DriverId"].map(
        order
    )  # the previous race predicts the next one perfectly
    monkeypatch.setattr(
        cli,
        "build_models",
        lambda target, which, history: (
            "race",
            [PreviousRaceBaseline(), Constant()],
            ["previous_race_equals_finish"],
        ),
    )
    assert (
        run(monkeypatch, "--target", "race_predicted_grid", "--models", "all", "--n-boot", "50")
        == 0
    )
    monkeypatch.setattr(
        "sys.argv", ["ml.evaluation", "--target", "race", "--models", "all", "--n-boot", "50"]
    )
    assert (
        cli.main() == cli.NO_MODEL_BEATS_BASELINES
    )  # the same situation on a primary target fails

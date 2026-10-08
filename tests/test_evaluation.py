import pandas as pd
import pytest

from ml.evaluation.baselines import GridBaseline, PreviousRaceBaseline
from ml.evaluation.data import OUTCOME_COLUMNS
from ml.evaluation.harness import evaluate
from ml.evaluation.metrics import race_metrics, summarize
from ml.evaluation.splits import LockedSeasonError, check_not_locked, walk_forward_splits

DRIVERS = ["AAA", "BBB", "CCC", "DDD"]


def make_results(years=(2019, 2020, 2021), rounds=3) -> pd.DataFrame:
    """Tiny synthetic season: grid is the driver order, finish is the reverse for odd rounds."""
    rows = []
    for year in years:
        for rnd in range(1, rounds + 1):
            finish = DRIVERS if rnd % 2 == 0 else DRIVERS[::-1]
            for grid, driver in enumerate(DRIVERS, start=1):
                rows.append(
                    {
                        "Year": year,
                        "Round": rnd,
                        "Abbreviation": driver,
                        "DriverId": driver.lower(),
                        "TeamName": "T",
                        "GridPosition": float(grid),
                        "Position": float(finish.index(driver) + 1),
                        "ClassifiedPosition": "1",
                        "Status": "Finished",
                        "Points": 0.0,
                        "Time": pd.NaT,
                        "Laps": 50.0,
                    }
                )
    return pd.DataFrame(rows)


def test_splits_never_train_on_the_future():
    splits = walk_forward_splits([2018, 2019, 2020, 2021, 2022], locked_season=None)
    assert [s.test_year for s in splits] == [2019, 2020, 2021, 2022]
    for s in splits:
        assert max(s.train_years) < s.test_year


def test_locked_season_is_skipped_and_refused():
    years = [2022, 2023, 2024, 2025]
    assert 2025 not in [s.test_year for s in walk_forward_splits(years, locked_season=2025)]
    assert 2025 in [
        s.test_year for s in walk_forward_splits(years, locked_season=2025, allow_locked=True)
    ]
    with pytest.raises(LockedSeasonError):
        check_not_locked(2025, 2025, allow_locked=False)
    check_not_locked(2025, 2025, allow_locked=True)


def test_race_metrics_perfect_and_reversed():
    actual = pd.Series([1, 2, 3, 4, 5], index=list("abcde"), dtype=float)
    perfect = race_metrics(actual, actual)
    assert perfect["spearman"] == pytest.approx(1.0) and perfect["winner_acc"] == 1.0
    assert perfect["podium_overlap"] == 1.0 and perfect["mae_position"] == 0.0
    reversed_ = race_metrics(-actual, actual)
    assert reversed_["spearman"] == pytest.approx(-1.0) and reversed_["winner_acc"] == 0.0


def test_grid_baseline_puts_pit_lane_starts_last():
    race = pd.DataFrame({"GridPosition": [1.0, 0.0, 2.0, None]})
    scores = GridBaseline().predict_race(pd.DataFrame(), race)
    order = scores.rank(method="first").astype(int).tolist()
    assert order[0] == 1 and order[2] == 2 and set(order[1::2]) == {3, 4}


def test_previous_race_baseline_uses_only_history():
    results = make_results(years=(2020,), rounds=2)
    history = results[results["Round"] == 1]
    race = results[results["Round"] == 2]
    scores = PreviousRaceBaseline().predict_race(history, race[["Abbreviation", "GridPosition"]])
    last_finish = history.set_index("Abbreviation")["Position"]
    assert scores.tolist() == [last_finish[d] for d in race["Abbreviation"]]


class SpyModel:
    """Fails the test if the harness leaks outcomes or future races to a model."""

    name = "spy"

    def __init__(self):
        self.calls = 0

    def fit(self, train):
        self.train_max_year = train["Year"].max()

    def predict_race(self, history, race):
        self.calls += 1
        assert not set(OUTCOME_COLUMNS) & set(race.columns), "outcomes leaked into the race frame"
        year, rnd = race["Year"].iloc[0], race["Round"].iloc[0]
        assert self.train_max_year < year, "model was fit on the test season or later"
        later = (history["Year"] > year) | ((history["Year"] == year) & (history["Round"] >= rnd))
        assert not later.any(), "history contains the current or a later race"
        return race["GridPosition"]


def test_harness_hides_outcomes_and_future_from_models():
    spy = SpyModel()
    per_race = evaluate([spy], make_results(), locked_season=None)
    assert spy.calls == len(per_race) == 6  # test years 2020 and 2021, 3 rounds each


def test_evaluation_is_deterministic_and_skips_locked_season():
    results = make_results(years=(2023, 2024, 2025))
    models = lambda: [GridBaseline(), PreviousRaceBaseline()]
    a = summarize(evaluate(models(), results, locked_season=2025))
    b = summarize(evaluate(models(), results, locked_season=2025))
    pd.testing.assert_frame_equal(a, b)
    assert set(evaluate(models(), results, locked_season=2025)["Year"]) == {2024}
    assert 2025 in set(evaluate(models(), results, locked_season=2025, allow_locked=True)["Year"])

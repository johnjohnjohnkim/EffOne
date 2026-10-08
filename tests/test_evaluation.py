import numpy as np
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


def finish_ordered(n):
    """Actual positions where the rows are in exact finishing order, the worst case for ties."""
    return pd.Series(range(1, n + 1), index=range(n), dtype=float)


@pytest.mark.parametrize("n", [8, 10, 20, 22])
def test_a_constant_score_earns_exactly_chance_even_when_rows_are_sorted_by_the_answer(n):
    got = race_metrics(pd.Series(0.0, index=range(n)), finish_ordered(n))
    assert got["spearman"] == 0.0
    assert got["winner_acc"] == pytest.approx(1 / n)
    assert got["podium_overlap"] == pytest.approx(3 / n)
    assert got["top10_overlap"] == pytest.approx(min(10, n) / n)
    assert got["mae_position"] == pytest.approx((n * n - 1) / (3 * n))  # E|i - j| for random order


def test_partial_ties_are_scored_by_expectation_not_by_a_tie_break():
    scores = pd.Series([1.0, 1.0, 2.0, 3.0], index=list("abcd"))  # a and b tied for 1st/2nd
    actual = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
    got = race_metrics(scores, actual)
    assert got["winner_acc"] == pytest.approx(
        0.5
    )  # a is the real winner; 50% chance it leads the tie
    assert got["podium_overlap"] == pytest.approx(
        1.0
    )  # a, b, c fill the top 3 whatever the tie does
    assert got["mae_position"] == pytest.approx(0.25)  # a, b each 0.5 off on average; c, d exact
    average_ranks = pd.Series([1.5, 1.5, 3.0, 4.0], index=list("abcd"))
    assert got["spearman"] == pytest.approx(average_ranks.corr(actual))


def test_metrics_do_not_depend_on_row_order_or_the_order_scores_are_returned_in():
    rng = np.random.default_rng(5)
    scores = pd.Series(rng.integers(0, 4, size=20).astype(float), index=range(20))  # many ties
    actual = pd.Series(rng.permutation(20) + 1.0, index=range(20))
    base = race_metrics(scores, actual)
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(20)
        assert race_metrics(scores.iloc[order], actual) == pytest.approx(base)
        assert race_metrics(scores, actual.iloc[order]) == pytest.approx(base)


def test_untied_scores_behave_exactly_as_a_plain_ranking():
    scores = pd.Series([3.0, 1.0, 2.0, 4.0], index=list("abcd"))
    actual = pd.Series([2.0, 1.0, 3.0, 4.0], index=list("abcd"))
    got = race_metrics(scores, actual)
    # predicted order b, c, a, d against actual b, a, c, d: same top 3 as a set, two swapped places
    assert got["winner_acc"] == 1.0 and got["podium_overlap"] == pytest.approx(1.0)
    assert got["mae_position"] == pytest.approx(0.5)
    assert got["spearman"] == pytest.approx(
        pd.Series([3.0, 1.0, 2.0, 4.0]).corr(pd.Series([2.0, 1.0, 3.0, 4.0]))
    )


def test_a_nan_score_is_an_error_not_a_free_pass():
    with pytest.raises(ValueError, match="NaN"):
        race_metrics(
            pd.Series([1.0, np.nan, 3.0], index=list("abc")),
            pd.Series([1.0, 2.0, 3.0], index=list("abc")),
        )

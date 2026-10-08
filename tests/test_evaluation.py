import numpy as np
import pandas as pd
import pytest

from ml.evaluation.baselines import GridBaseline, PreviousRaceBaseline
from ml.evaluation.data import OUTCOME_COLUMNS
from ml.evaluation.harness import evaluate
from ml.evaluation.metrics import race_metrics, summarize
from ml.evaluation.splits import LockedSeasonError, check_not_locked, walk_forward_splits
from tests.helpers import row

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


# ---- the qualifying target and the entry lists the harness hands out ------------------------


class RecordingModel:
    name = "recorder"

    def __init__(self):
        self.entries = []

    def fit(self, train):
        pass

    def predict_race(self, history, race):
        self.entries.append(race.copy())
        return pd.Series(0.0, index=race.index)


def modelling_table():
    from tests.helpers import make_rows

    return make_rows(years=(2021, 2022, 2023), rounds=4, seed=11)


def test_the_race_target_gets_the_grid_and_weather_the_quali_target_gets_neither():
    table = modelling_table()
    race_model, quali_model = RecordingModel(), RecordingModel()
    evaluate([race_model], table, "race", locked_season=None)
    evaluate([quali_model], table, "quali", locked_season=None)
    race_cols, quali_cols = set(race_model.entries[0].columns), set(quali_model.entries[0].columns)
    assert {"GridPosition", "wx_air_temp", "TeamKey", "CircuitId"} <= race_cols
    assert {"TeamKey", "CircuitId", "DriverId"} <= quali_cols
    assert not {"GridPosition", "wx_air_temp"} & quali_cols
    outcomes = {"Position", "QualiPosition", "Points", "dnf", "pace_gap_pct", "race_ok"}
    assert not outcomes & race_cols and not outcomes & quali_cols


def test_each_target_is_scored_against_its_own_outcome():
    from ml.evaluation.baselines import PreviousQualiBaseline

    table = modelling_table()
    order = {d: i + 1.0 for i, d in enumerate(sorted(table["DriverId"].unique()))}
    table["QualiPosition"] = table["DriverId"].map(order)  # qualifying order never changes
    quali = evaluate([PreviousQualiBaseline()], table, "quali", locked_season=None)
    race = evaluate([PreviousQualiBaseline()], table, "race", locked_season=None)
    # Repeating last qualifying is perfect for qualifying, and nothing special for the race.
    assert quali["spearman"].iloc[1:].eq(1.0).all()
    assert race["spearman"].abs().mean() < 0.6


def test_models_see_the_full_entry_list_but_only_drivers_with_an_outcome_are_scored(monkeypatch):
    from ml.evaluation import harness

    table = modelling_table()
    first_test_race = table[(table["Year"] == 2022) & (table["Round"] == 1)]
    table.loc[first_test_race.index[:2], "QualiPosition"] = np.nan  # 2021 is training-only
    scored_sizes = []
    real_metrics = harness.race_metrics

    def capture(scores, actual):
        scored_sizes.append(len(actual))
        return real_metrics(scores, actual)

    monkeypatch.setattr(harness, "race_metrics", capture)
    model = RecordingModel()
    evaluate([model], table, "quali", locked_season=None)
    assert {len(e) for e in model.entries} == {6}  # nobody is hidden from the model by the result
    assert min(scored_sizes) == 4 and max(scored_sizes) == 6  # but the unknown ones are not scored


def test_races_with_fewer_than_two_known_outcomes_are_skipped():
    table = modelling_table()
    one_race = (table["Year"] == 2022) & (table["Round"] == 1)
    table.loc[one_race, "QualiPosition"] = np.nan
    model = RecordingModel()
    out = evaluate([model], table, "quali", locked_season=None)
    assert not ((out["Year"] == 2022) & (out["Round"] == 1)).any()


def _history_for_form():
    rows = [
        row(2022, rnd, driver, Position=finish, QualiPosition=finish, Abbreviation=driver.upper())
        for rnd, (a, b) in enumerate(
            [(1.0, 5.0), (2.0, 5.0), (3.0, 5.0), (4.0, 5.0), (5.0, 5.0), (6.0, 1.0)], 1
        )
        for driver, finish in (("a", a), ("b", b))
    ]
    return pd.DataFrame(rows)


def test_mean_of_the_last_five_baseline_averages_the_last_five_results_only():
    from ml.evaluation.baselines import MeanLast5FinishBaseline

    history = _history_for_form()
    race = pd.DataFrame([row(2022, 7, "a", Abbreviation="A"), row(2022, 7, "b", Abbreviation="B")])
    scores = MeanLast5FinishBaseline().predict_race(history, race)
    # driver a: last five finishes are 2, 3, 4, 5, 6 -> 4.0; driver b: 5, 5, 5, 5, 1 -> 4.2
    assert scores.tolist() == pytest.approx([4.0, 4.2])


def test_mean_of_the_last_five_ignores_races_marked_not_ok_and_puts_debutants_last():
    from ml.evaluation.baselines import MeanLast5FinishBaseline

    history = _history_for_form()
    history.loc[history["Round"] == 6, "race_ok"] = False  # the 6th race carries no information
    race = pd.DataFrame(
        [row(2022, 7, "a", Abbreviation="A"), row(2022, 7, "rookie", Abbreviation="R")]
    )
    scores = MeanLast5FinishBaseline().predict_race(history, race)
    assert scores.iloc[0] == pytest.approx(3.0)  # finishes 1 to 5
    assert scores.iloc[1] > scores.iloc[0]  # a driver with no results goes behind


def test_every_target_has_a_mean_of_last_five_baseline():
    from ml.evaluation.baselines import BASELINES_BY_TARGET

    names = {t: [cls().name for cls in classes] for t, classes in BASELINES_BY_TARGET.items()}
    assert "mean_last5_finish" in names["race"] and "mean_last5_quali" in names["quali"]
    assert len(names["race"]) == len(names["quali"]) == 3


def test_a_tie_in_the_actual_result_does_not_penalise_a_perfect_predictor():
    perfect = pd.Series(range(1, 13), index=range(12), dtype=float)
    tied_pole = pd.Series([1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=range(12), dtype=float)
    # Either of the two co-leaders may be called the winner: a coin flip, not a miss.
    assert race_metrics(perfect, tied_pole)["winner_acc"] == pytest.approx(0.5)
    tied_at_the_cut = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 12], index=range(12), dtype=float
    )
    got = race_metrics(perfect, tied_at_the_cut)
    assert got["top10_overlap"] == pytest.approx(
        0.95
    )  # one of the two tied for 10th is in, one out
    assert got["podium_overlap"] == 1.0 and got["spearman"] > 0.99


def test_swapping_which_driver_holds_a_tied_actual_position_changes_nothing():
    scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=list("abcde"))
    actual = pd.Series([1.0, 2.0, 2.0, 4.0, 5.0], index=list("abcde"))
    mirrored = pd.Series([1.0, 2.0, 2.0, 4.0, 5.0], index=list("abcde"))
    mirrored.loc[["b", "c"]] = mirrored.loc[["c", "b"]].to_numpy()  # same values, same drivers
    assert race_metrics(scores, actual) == pytest.approx(race_metrics(scores, mirrored))
    # ... and swapping the two PREDICTED scores of the tied drivers is also a no-op on the metrics
    swapped = scores.copy()
    swapped.loc[["b", "c"]] = scores.loc[["c", "b"]].to_numpy()
    flipped = race_metrics(swapped, actual)
    original = race_metrics(scores, actual)
    for key in ("winner_acc", "podium_overlap", "top10_overlap", "mae_position"):
        assert flipped[key] == pytest.approx(original[key])


def test_previous_result_baselines_ignore_races_marked_not_ok_and_reject_bad_flags():
    from ml.evaluation.baselines import PreviousRaceBaseline

    history = _history_for_form()
    history.loc[history["Round"] == 6, "race_ok"] = False
    race = pd.DataFrame([row(2022, 7, "a", Abbreviation="A"), row(2022, 7, "b", Abbreviation="B")])
    scores = PreviousRaceBaseline().predict_race(history, race)
    assert scores.tolist() == [5.0, 5.0]  # round 5 is the last race that counts, not round 6
    gappy = history.copy()
    gappy["race_ok"] = gappy["race_ok"].astype(object)
    gappy.loc[0, "race_ok"] = None
    with pytest.raises(ValueError, match="race_ok"):
        PreviousRaceBaseline().predict_race(gappy, race)
    text = history.copy()
    text["race_ok"] = text["race_ok"].astype(object)
    text.loc[:, "race_ok"] = "False"
    with pytest.raises(ValueError, match="real booleans"):
        PreviousRaceBaseline().predict_race(text, race)

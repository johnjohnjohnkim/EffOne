"""4b.4: race models trained on walk-forward (out-of-fold) predicted grids."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from ml.features.build import build_feature_table
from ml.models.chain import OutOfFoldGridRace, out_of_fold_grids, with_grids
from ml.models.zoo import build_model, make_predicted_grid_models
from tests.helpers import make_rows, scramble


@pytest.fixture(scope="module")
def rows():
    return make_rows(years=(2021, 2022, 2023), rounds=5, seed=4)


class Spy:
    """A qualifying 'model' that records which seasons it was fitted on and ranks by driver id."""

    name = "spy"
    fitted_years: ClassVar[list] = []

    def fit(self, train):
        Spy.fitted_years.append(sorted(train["Year"].unique()))

    def predict_race(self, history, race):
        assert "GridPosition" not in race.columns  # the grid is what is being predicted
        return pd.Series(np.arange(len(race), dtype=float), index=race.index)


def test_each_season_is_predicted_by_a_model_fitted_on_earlier_seasons_only(rows):
    Spy.fitted_years = []
    grids = out_of_fold_grids(rows, Spy)
    assert Spy.fitted_years == [[2021], [2021, 2022]]
    assert sorted(grids["Year"].unique()) == [2022, 2023]  # the first season has no earlier one


def test_a_single_season_has_no_out_of_fold_grids(rows):
    assert out_of_fold_grids(rows[rows["Year"] == 2021], Spy).empty


def test_out_of_fold_grids_ignore_the_real_grid_and_the_seasons_own_outcomes(rows):
    real = build_model("ridge", "quali", build_feature_table(rows), {})
    first = out_of_fold_grids(rows, lambda: real)
    # Destroy season 2023 from its first race on: its first race's predicted grid cannot move.
    key = (2023, 1)
    scrambled = scramble(rows, key, "pre_weekend")
    again = out_of_fold_grids(
        scrambled, lambda: build_model("ridge", "quali", build_feature_table(scrambled), {})
    )

    def pick(g):
        return g[(g["Year"] == 2023) & (g["Round"] == 1)].sort_values("DriverId")

    pd.testing.assert_series_equal(
        pick(first)["GridPosition"].reset_index(drop=True),
        pick(again)["GridPosition"].reset_index(drop=True),
    )


def test_with_grids_replaces_the_grid_features_and_keeps_only_covered_races(rows):
    table = build_feature_table(rows)
    grids = out_of_fold_grids(rows, Spy)
    out = with_grids(table, grids)
    assert set(out["Year"]) == {2022, 2023}
    race = out[(out["Year"] == 2023) & (out["Round"] == 2)]
    assert sorted(race["grid"]) == list(range(1, len(race) + 1))  # a permutation of 1..n slots
    assert (race["grid_pit_lane"] == 0).all()
    assert "GridPosition" not in out.columns


def test_the_chained_model_fits_predicts_and_is_deterministic(rows):
    table = build_feature_table(rows)
    train = rows[rows["Year"] < 2023]
    race = rows[(rows["Year"] == 2023) & (rows["Round"] == 3)]
    history = rows[(rows["Year"] < 2023) | ((rows["Year"] == 2023) & (rows["Round"] < 3))]
    outputs = []
    for _ in range(2):
        model = OutOfFoldGridRace(
            lambda t: build_model("ridge", "race", t, {}),
            lambda: build_model("ridge", "quali", table, {}),
            table,
            "ridge [oof grid]",
        )
        model.fit(train)
        outputs.append(model.predict_race(history, race.drop(columns=["Position"])))
    pd.testing.assert_series_equal(outputs[0], outputs[1])
    assert outputs[0].nunique() > 1


class SpyRace:
    """A race 'model' that keeps the table it was built on and the races it was fitted on."""

    name = "spy_race"

    def __init__(self, table):
        self.table, self.fitted = table, None

    def fit(self, train):
        self.fitted = sorted(train["Year"].unique())

    def predict_race(self, history, race):
        return pd.Series(0.0, index=race.index)


def test_the_race_model_is_trained_on_predicted_grids_not_the_real_ones(rows):
    table = build_feature_table(rows)
    built = []

    def make_race(t):
        built.append(SpyRace(t))
        return built[-1]

    model = OutOfFoldGridRace(make_race, Spy, table, "spy [oof grid]")
    model.fit(rows[rows["Year"] < 2023])
    spy = built[-1]
    assert spy.fitted == [2022]  # 2021 has no earlier season, so it is not trained on
    assert set(spy.table["Year"]) == {2022}
    for _, race in spy.table.groupby("Round"):
        # Spy ranks drivers by their position in the entry list, so the predicted slots are 1..n,
        # which is not (for these random grids) the real grid.
        assert list(race["grid"]) == list(range(1, len(race) + 1))
    real = table[table["Year"] == 2022]
    assert not (spy.table["grid"].to_numpy() == real["grid"].to_numpy()).all()


def test_a_longer_partial_season_is_not_served_stale_grids_from_the_cache(rows):
    cache: dict = {}
    short = rows[(rows["Year"] < 2023) | ((rows["Year"] == 2023) & (rows["Round"] <= 3))]
    longer = rows[(rows["Year"] < 2023) | ((rows["Year"] == 2023) & (rows["Round"] <= 5))]
    first = out_of_fold_grids(short, Spy, cache)
    second = out_of_fold_grids(longer, Spy, cache)
    assert sorted(first[first["Year"] == 2023]["Round"].unique()) == [1, 2, 3]
    assert sorted(second[second["Year"] == 2023]["Round"].unique()) == [1, 2, 3, 4, 5]


def test_one_training_season_falls_back_to_real_grids_and_says_so(rows):
    table = build_feature_table(rows)
    model = OutOfFoldGridRace(
        lambda t: build_model("ridge", "race", t, {}),
        lambda: build_model("ridge", "quali", table, {}),
        table,
        "ridge [oof grid]",
    )
    model.fit(rows[rows["Year"] == 2021])
    assert model.fell_back
    model.fit(rows[rows["Year"] < 2023])
    assert not model.fell_back


def test_predicting_before_fitting_is_an_error(rows):
    model = OutOfFoldGridRace(lambda t: None, lambda: None, build_feature_table(rows), "x")
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_race(rows, rows.iloc[:6])


def test_the_predicted_grid_board_includes_the_out_of_fold_models(rows):
    models, baselines = make_predicted_grid_models(build_feature_table(rows), {})
    names = [m.name for m in models]
    assert any(n.endswith("[oof grid]") for n in names)
    assert not any(n.endswith("[oof grid]") for n in baselines)

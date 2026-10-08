import numpy as np
import pandas as pd
import pytest

from ml.evaluation.compare import leaderboard, paired_bootstrap


def per_race(**models):
    """models: name -> list of per-race Spearman values (same races for every model)."""
    rows = []
    for name, values in models.items():
        for i, value in enumerate(values):
            rows.append(
                {
                    "model": name,
                    "Year": 2023 + i // 5,
                    "Round": i % 5 + 1,
                    "spearman": value,
                    "mae_position": 3.0,
                    "winner_acc": 0.5,
                    "podium_overlap": 0.5,
                    "top10_overlap": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_a_constant_improvement_has_a_zero_width_interval():
    data = per_race(base=[0.5] * 20, better=[0.6] * 20)
    delta, low, high = paired_bootstrap(data, "better", "base")
    assert delta == pytest.approx(0.1) and low == pytest.approx(0.1) and high == pytest.approx(0.1)


def test_noise_around_zero_gives_an_interval_that_includes_zero():
    rng = np.random.default_rng(1)
    base = list(0.5 + rng.normal(0, 0.1, 40))
    other = [b + d for b, d in zip(base, rng.normal(0, 0.1, 40), strict=True)]
    delta, low, high = paired_bootstrap(per_race(base=base, other=other), "other", "base")
    assert low < 0 < high and abs(delta) < 0.05


def test_the_bootstrap_is_paired_by_race_and_reproducible():
    base = [0.2, 0.9, 0.2, 0.9] * 5
    model = [
        0.3,
        1.0,
        0.3,
        1.0,
    ] * 5  # always +0.1 on the same race, despite huge race-to-race swings
    data = per_race(base=base, model=model)
    first, second = paired_bootstrap(data, "model", "base"), paired_bootstrap(data, "model", "base")
    assert first == second
    assert first[1] == pytest.approx(0.1) and first[2] == pytest.approx(
        0.1
    )  # pairing removes the swings


def test_leaderboard_picks_the_best_baseline_and_flags_who_clearly_beats_it():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.05, 30)
    data = per_race(
        weak_baseline=list(0.3 + noise),
        strong_baseline=list(0.5 + noise),
        clearly_better=list(0.6 + noise),
        same=list(0.5 + noise),
        worse=list(0.4 + noise),
    )
    board = leaderboard(data, ["weak_baseline", "strong_baseline"]).set_index("model")
    assert board.attrs["best_baseline"] == "strong_baseline"
    assert (
        board.loc["clearly_better", "beats_best_baseline"]
        and board.loc["clearly_better", "clearly_beats"]
    )
    assert board.loc["clearly_better", "vs_best_baseline"] == pytest.approx(0.1)
    assert (
        not board.loc["worse", "beats_best_baseline"] and board.loc["worse", "vs_best_baseline"] < 0
    )
    assert not board.loc["strong_baseline", "beats_best_baseline"]
    assert board.loc[["weak_baseline", "strong_baseline"], "is_baseline"].all()
    assert board["is_baseline"].sum() == 2
    assert list(board.index) == sorted(board.index, key=lambda m: -board.loc[m, "spearman"])


def test_a_small_average_gain_is_not_called_clear_when_the_interval_spans_zero():
    rng = np.random.default_rng(3)
    base = list(0.5 + rng.normal(0, 0.1, 25))
    lucky = [b + 0.01 + d for b, d in zip(base, rng.normal(0, 0.15, 25), strict=True)]
    board = leaderboard(per_race(base=base, lucky=lucky), ["base"]).set_index("model")
    assert board.loc["lucky", "ci_low"] < 0
    assert not board.loc["lucky", "clearly_beats"]


def test_leaderboard_counts_the_seasons_a_model_beats_the_best_baseline():
    # per_race() puts 5 races in each season (Year 2023, 2024, 2025, 2026). The model wins the
    # first two seasons and loses the last two.
    base = [0.5] * 20
    model = [0.6] * 10 + [0.4] * 10
    board = leaderboard(per_race(base=base, model=model), ["base"]).set_index("model")
    assert board.loc["model", "seasons"] == 4 and board.loc["model", "seasons_better"] == 2
    assert board.loc["base", "seasons_better"] == 0


def per_race_by_season(effects):
    """One model and one baseline over seasons; the model's gain differs by season."""
    rows = []
    for year, (gain, n) in enumerate(effects, start=2020):
        for rnd in range(1, n + 1):
            base = 0.5 + 0.001 * rnd
            rows.append(
                {
                    "model": "base",
                    "Year": year,
                    "Round": rnd,
                    **dict.fromkeys(
                        [
                            "spearman",
                            "mae_position",
                            "winner_acc",
                            "podium_overlap",
                            "top10_overlap",
                        ],
                        base,
                    ),
                }
            )
            rows.append(
                {
                    "model": "m",
                    "Year": year,
                    "Round": rnd,
                    **dict.fromkeys(
                        [
                            "spearman",
                            "mae_position",
                            "winner_acc",
                            "podium_overlap",
                            "top10_overlap",
                        ],
                        base + gain,
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_the_season_clustered_interval_is_wider_than_the_race_level_one():
    # The gain is +0.10 in some seasons and -0.05 in others: races within a season agree, so
    # treating them as independent makes the interval look much tighter than it should.
    data = per_race_by_season([(0.10, 8), (-0.05, 8), (0.10, 8), (-0.05, 8), (0.10, 8), (0.10, 8)])
    _, clustered_low, clustered_high = paired_bootstrap(data, "m", "base", clustered=True)
    _, iid_low, iid_high = paired_bootstrap(data, "m", "base", clustered=False)
    assert (clustered_high - clustered_low) > 1.5 * (iid_high - iid_low)


def test_a_gain_that_holds_in_every_season_is_clearly_better_under_the_clustered_interval():
    data = per_race_by_season([(0.05, 8)] * 6)
    board = leaderboard(data, ["base"]).set_index("model")
    assert board.loc["m", "clearly_beats"] and board.loc["m", "seasons_better"] == 6


def test_a_gain_carried_by_a_single_season_is_not_clearly_better():
    data = per_race_by_season(
        [(0.30, 8), (-0.01, 8), (-0.01, 8), (-0.01, 8), (-0.01, 8), (-0.01, 8)]
    )
    board = leaderboard(data, ["base"]).set_index("model")
    assert board.loc["m", "vs_best_baseline"] > 0 and board.loc["m", "beats_best_baseline"]
    assert not board.loc["m", "clearly_beats"] and board.loc["m", "seasons_better"] == 1


def test_the_leaderboard_keeps_the_race_level_interval_for_reference():
    board = leaderboard(per_race_by_season([(0.05, 8)] * 4), ["base"]).set_index("model")
    assert {"iid_ci_low", "iid_ci_high", "ci_low", "ci_high"} <= set(board.columns)


def test_uneven_coverage_or_duplicate_scores_are_errors_not_silent_drops():
    data = per_race_by_season([(0.05, 8)] * 3)
    with pytest.raises(ValueError, match="same races"):
        paired_bootstrap(data[~((data["model"] == "m") & (data["Year"] == 2020))], "m", "base")
    with pytest.raises(ValueError, match="more than one"):
        paired_bootstrap(pd.concat([data, data.iloc[[0]]]), "m", "base")

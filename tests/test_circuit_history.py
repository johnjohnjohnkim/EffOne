import pandas as pd
import pytest

from ml.ingest.circuit_history import before, eligible_circuit_years, latest_grid
from ml.ingest.circuit_ids import circuit_id

GRID = {f"d{i:02d}" for i in range(20)}


def results(rows):
    return pd.DataFrame(rows, columns=["Year", "Round", "DriverId"])


def events(*rows):
    """rows: (year, round, circuit)"""
    return pd.DataFrame(rows, columns=["Year", "RoundNumber", "CircuitId"])


def race(year, rnd, drivers):
    return [(year, rnd, d) for d in drivers]


def share_table(res, evs, grid=GRID, **kwargs):
    return eligible_circuit_years(res, evs, grid, **kwargs).set_index(["CircuitId", "Year"])


def test_exactly_a_quarter_of_the_grid_counts_but_less_does_not():
    five = results(race(2019, 1, sorted(GRID)[:5]))
    four = results(race(2019, 1, sorted(GRID)[:4]))
    evs = events((2019, 1, "monza"))
    assert share_table(five, evs).loc[("monza", 2019), "eligible"]
    assert not share_table(four, evs).loc[("monza", 2019), "eligible"]


@pytest.mark.parametrize(
    ("min_share", "grid_size", "raced", "eligible"),
    [
        (0.25, 12, 3, True),  # 3 of 12
        (0.25, 12, 2, False),
        (1 / 3, 9, 3, True),  # 1/3 has no exact float form
        (1 / 3, 9, 2, False),
        (0.7, 10, 7, True),  # 0.7 * 10 = 7.000000000000001 in floats; 7 drivers must still do
        (0.7, 10, 6, False),
        (0.25, 21, 6, True),  # ceil(5.25) = 6 whole drivers
        (0.25, 21, 5, False),
    ],
)
def test_threshold_is_whole_drivers_at_ceil_of_min_share_times_grid(
    min_share, grid_size, raced, eligible
):
    grid = {f"g{i}" for i in range(grid_size)}
    res = results(race(2019, 1, sorted(grid)[:raced]))
    out = share_table(res, events((2019, 1, "x")), grid, min_share=min_share)
    assert bool(out.loc[("x", 2019), "eligible"]) is eligible


def test_drivers_not_on_the_current_grid_do_not_count():
    res = results(race(2019, 1, ["old1", "old2", "old3"] + sorted(GRID)[:4]))
    out = share_table(res, events((2019, 1, "monza")))
    assert out.loc[("monza", 2019), "raced"] == 4 and out.loc[("monza", 2019), "grid_share"] == 0.2


def test_two_races_at_one_circuit_in_a_year_pool_their_drivers():
    res = results(race(2020, 1, sorted(GRID)[:3]) + race(2020, 2, sorted(GRID)[3:6]))
    out = share_table(res, events((2020, 1, "sakhir"), (2020, 2, "sakhir")))
    assert out.loc[("sakhir", 2020), "raced"] == 6


def test_before_keeps_earlier_rounds_of_the_target_season_only():
    res = results(
        race(2023, 5, ["a"])
        + race(2024, 1, ["a"])
        + race(2024, 2, ["a"])
        + race(2024, 3, ["a"])
        + race(2024, 4, ["a"])
    )
    kept = before(res, (2024, 3))
    assert set(zip(kept["Year"], kept["Round"], strict=True)) == {(2023, 5), (2024, 1), (2024, 2)}


def test_as_of_uses_earlier_rounds_of_the_same_season_but_not_the_target_or_later_ones():
    res = results(
        race(2024, 1, sorted(GRID)) + race(2024, 2, sorted(GRID)) + race(2024, 3, sorted(GRID))
    )
    evs = events((2024, 1, "a"), (2024, 2, "b"), (2024, 3, "c"))
    out = share_table(res, evs, as_of=(2024, 3))
    assert sorted(out.index) == [("a", 2024), ("b", 2024)]


def test_as_of_ignores_the_target_race_and_everything_after_it():
    res = results(race(2023, 1, sorted(GRID)) + race(2024, 1, sorted(GRID)))
    evs = events((2023, 1, "monza"), (2024, 1, "monza"))
    out = share_table(res, evs, as_of=(2024, 1))
    assert list(out.index) == [("monza", 2023)]


def test_as_of_takes_the_reference_grid_from_before_the_target_race():
    # A driver who only appears in the target race must not shape the grid used for its features.
    res = results(race(2023, 1, ["a", "b", "c", "d"]) + race(2024, 1, ["a", "b", "c", "z"]))
    evs = events((2023, 1, "monza"), (2024, 1, "monza"))
    out = eligible_circuit_years(res, evs, as_of=(2024, 1))
    assert out.loc[0, "grid_share"] == 1.0  # grid = a, b, c, d (the 2023 race), all raced


def test_unmatched_results_error_names_the_missing_races():
    res = results(race(2030, 9, ["a"]) + race(2019, 1, ["a"]))
    with pytest.raises(ValueError, match=r"\(2030, 9\)"):
        eligible_circuit_years(res, events((2019, 1, "monza")), {"a"})


def test_empty_grid_and_empty_history_are_errors():
    with pytest.raises(ValueError, match="empty"):
        eligible_circuit_years(results(race(2019, 1, ["a"])), events((2019, 1, "x")), set())
    with pytest.raises(ValueError, match="no results"):
        latest_grid(results([]))


def test_latest_grid_uses_the_most_recent_race():
    res = results(race(2023, 5, ["old"]) + race(2024, 1, ["a", "b"]))
    assert latest_grid(res) == {"a", "b"}


def test_circuit_ids_strip_accents_and_merge_spellings():
    assert circuit_id("Nürburgring") == "nurburgring"
    assert circuit_id("São Paulo") == "sao_paulo"
    assert circuit_id("Spa-Francorchamps") == "spa_francorchamps"
    assert circuit_id("Monte Carlo") == circuit_id("Monaco") == "monaco"
    assert circuit_id("Yas Island") == circuit_id("Yas Marina")
    assert circuit_id("Singapore") == circuit_id("Marina Bay")

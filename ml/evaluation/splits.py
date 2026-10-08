"""Walk-forward splits: train on seasons up to N, test on season N+1, then roll forward."""

from dataclasses import dataclass

# The latest complete season is held back as a final test set. Model selection must never look
# at it; it is evaluated once, after the model choice is frozen (pass allow_locked=True then).
LOCKED_SEASON = 2025


@dataclass(frozen=True)
class Split:
    train_years: tuple[int, ...]
    test_year: int


class LockedSeasonError(RuntimeError):
    pass


def walk_forward_splits(
    years: list[int],
    min_train_years: int = 1,
    locked_season: int | None = LOCKED_SEASON,
    allow_locked: bool = False,
) -> list[Split]:
    """One split per test season; every training year is strictly before the test year."""
    years = sorted(set(years))
    splits = []
    for i in range(min_train_years, len(years)):
        test_year = years[i]
        if test_year == locked_season and not allow_locked:
            continue
        splits.append(Split(tuple(years[:i]), test_year))
    return splits


def check_not_locked(test_year: int, locked_season: int | None, allow_locked: bool) -> None:
    if test_year == locked_season and not allow_locked:
        raise LockedSeasonError(
            f"Season {test_year} is the locked final test set. Pass allow_locked=True "
            "(--allow-locked) only once the model choice is frozen."
        )

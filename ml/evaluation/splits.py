"""Walk-forward splits: train on seasons up to N, test on season N+1, then roll forward.

Seasons have roles, so that looking and reporting are kept apart:
- DEVELOPMENT (2019 to 2021): tuning and feature decisions may use these freely.
- REPORT (2022 to 2024, and 2026 so far): scored for the leaderboards. Once a configuration is
  frozen they are not used to choose anything. They were looked at during Milestone 4, so they are
  not fresh; the only truly unseen season is the locked one.
- LOCKED (2025): scored once, at the end, when the user says so (allow_locked=True).
Training for a test season always uses every earlier season, whatever its role.
"""

from collections.abc import Collection
from dataclasses import dataclass

# The latest complete season is held back as a final test set. Model selection must never look
# at it; it is evaluated once, after the model choice is frozen (pass allow_locked=True then).
LOCKED_SEASON = 2025
DEV_SEASONS: tuple[int, ...] = (2019, 2020, 2021)
REPORT_SEASONS: tuple[int, ...] = (2022, 2023, 2024, 2026)
SEASON_SETS: dict[str, tuple[int, ...] | None] = {
    "dev": DEV_SEASONS,
    "report": REPORT_SEASONS,
    "all": None,  # every season except the locked one
}


@dataclass(frozen=True)
class Split:
    train_years: tuple[int, ...]
    test_year: int


class LockedSeasonError(RuntimeError):
    pass


class TuningLeakError(RuntimeError):
    """Raised when something that tunes or chooses tries to score a non-development season."""


def walk_forward_splits(
    years: list[int],
    min_train_years: int = 1,
    locked_season: int | None = LOCKED_SEASON,
    allow_locked: bool = False,
    test_seasons: Collection[int] | None = None,
) -> list[Split]:
    """One split per test season; every training year is strictly before the test year.

    `test_seasons` limits which seasons are SCORED; training still uses all earlier seasons.
    """
    years = sorted(set(years))
    splits = []
    for i in range(min_train_years, len(years)):
        test_year = years[i]
        if test_year == locked_season and not allow_locked:
            continue
        if test_seasons is not None and test_year not in test_seasons:
            continue
        splits.append(Split(tuple(years[:i]), test_year))
    return splits


def check_not_locked(test_year: int, locked_season: int | None, allow_locked: bool) -> None:
    if test_year == locked_season and not allow_locked:
        raise LockedSeasonError(
            f"Season {test_year} is the locked final test set. Pass allow_locked=True "
            "(--allow-locked) only once the model choice is frozen."
        )


def check_development_only(seasons: Collection[int]) -> None:
    """Tuning may only score development seasons: report and locked seasons are off limits."""
    outside = sorted(set(seasons) - set(DEV_SEASONS))
    if outside:
        raise TuningLeakError(
            f"tuning may only score the development seasons {DEV_SEASONS}; got {outside}"
        )

"""An Elo-style rating model: a rating for each driver and each team entry, updated race by race.

There is nothing to fit. Every prediction replays the history (earlier races only) in order, so
the ratings it uses cannot contain the race being predicted. A driver's strength in a race is their
own rating plus their team's; both move after each race according to how many rivals they beat
versus how many their ratings said they should. At each new season ratings shrink towards zero, and
team ratings shrink much harder when the rules change (the car is reset, the driver is not).

The constants were set by hand and never searched, but they were chosen with these seasons in view
(see ml/models/factories.py), so treat the results as mildly optimistic.
"""

import numpy as np
import pandas as pd

from ml.features.build import as_flags
from ml.features.builders import REGULATION_ERA_STARTS
from ml.ingest.circuit_history import before

ACTUAL_COLUMN = {"race": "Position", "quali": "QualiPosition"}


class EloModel:
    def __init__(
        self,
        name: str = "elo",
        target: str = "race",
        k_driver: float = 30.0,
        k_team: float = 50.0,
        scale: float = 400.0,
        season_shrink: float = 0.75,
        regulation_team_shrink: float = 0.3,
    ):
        self.name = name
        self.column = ACTUAL_COLUMN[target]
        self.k_driver, self.k_team, self.scale = k_driver, k_team, scale
        self.season_shrink, self.regulation_team_shrink = season_shrink, regulation_team_shrink

    def fit(self, train: pd.DataFrame) -> None:
        pass  # ratings are replayed from `history` at prediction time

    def predict_race(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.Series:
        year, rnd = int(race["Year"].iloc[0]), int(race["Round"].iloc[0])
        # Never trust the caller: only races strictly before this one may shape the ratings.
        drivers, teams, last_season = self._replay(before(history, (year, rnd)))
        if last_season is not None and year > last_season:
            self._new_season(year, drivers, teams)  # the first race of a season starts shrunk
        strength = race["DriverId"].map(drivers).fillna(0.0) + race["TeamKey"].map(teams).fillna(
            0.0
        )
        return -strength  # lower score = better finish

    def _replay(
        self, history: pd.DataFrame
    ) -> tuple[dict[str, float], dict[str, float], int | None]:
        """Ratings after every race in `history`, and the season of the last race replayed."""
        drivers: dict[str, float] = {}
        teams: dict[str, float] = {}
        usable = history
        if "race_ok" in usable.columns:
            usable = usable[as_flags(usable["race_ok"], "history race_ok")]
        usable = usable.dropna(subset=[self.column]).sort_values(["Year", "Round", "DriverId"])
        season = None
        for (year, _), race in usable.groupby(["Year", "Round"], sort=True):
            if season is not None and year != season:
                self._new_season(year, drivers, teams)
            season = year
            self._update(race, drivers, teams)
        return drivers, teams, season

    def _new_season(self, year: int, drivers: dict[str, float], teams: dict[str, float]) -> None:
        for key in drivers:
            drivers[key] *= self.season_shrink
        factor = (
            self.regulation_team_shrink if year in REGULATION_ERA_STARTS else self.season_shrink
        )
        for key in teams:
            teams[key] *= factor

    def _update(
        self, race: pd.DataFrame, drivers: dict[str, float], teams: dict[str, float]
    ) -> None:
        n = len(race)
        if n < 2:
            return
        d_ids, t_ids = race["DriverId"].to_numpy(), race["TeamKey"].to_numpy()
        strength = np.array(
            [drivers.get(d, 0.0) + teams.get(t, 0.0) for d, t in zip(d_ids, t_ids, strict=True)]
        )
        position = race[self.column].to_numpy(dtype=float)

        # P[i, j]: the chance our ratings give i of finishing ahead of j.
        p = 1.0 / (1.0 + 10.0 ** ((strength[None, :] - strength[:, None]) / self.scale))
        expected = (p.sum(axis=1) - 0.5) / (n - 1)  # drop the i-versus-i term (0.5)
        actual = (position[:, None] < position[None, :]).sum(axis=1) / (n - 1)
        surprise = actual - expected  # positive: beat more rivals than expected

        team_surprise: dict[str, list[float]] = {}
        for driver, team, delta in zip(d_ids, t_ids, surprise, strict=True):
            drivers[driver] = drivers.get(driver, 0.0) + self.k_driver * delta
            team_surprise.setdefault(team, []).append(delta)
        for team, deltas in team_surprise.items():
            teams[team] = teams.get(team, 0.0) + self.k_team * float(np.mean(deltas))

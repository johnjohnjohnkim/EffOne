"""Feature builders. Leakage is prevented structurally, not by discipline.

A builder is called as `compute(history, race)`:

- `history` is the results of every race strictly BEFORE the target race (outcomes included),
  already limited to races that carry real information (`race_ok`).
- `race` has one row per driver entered in the target race, with ONLY the columns its stage allows
  (STAGE_INPUTS). Outcome columns (finish, points, qualifying, pace...) of the target race are not
  in it, so a builder cannot read them.

Stages say what is known when the prediction is made:
- pre_weekend: before the weekend starts. The entry list, the circuit, the calendar position.
- scenario:    inputs a user can set for a what-if (weather).
- post_quali:  after qualifying. Adds the starting grid.

Known train/serve gaps (documented, not hidden):
- Weather: in the history it is the realised whole-race average; when predicting a future race it
  will be a forecast or a user's choice. Absolute weather has almost no signal on its own, so treat
  it with care (or leave it out) until a forecast source and a driver wet-weather skill exist.
- Entry list: for past races it comes from the results, so it includes late substitutes. For a
  future race it will be the published entry list, which can differ slightly.
- Points are race points only; sprint points are not included.
"""

from typing import Literal, Protocol

import numpy as np
import pandas as pd

from ml.features.tables import WEATHER_COLUMNS
from ml.features.weekend import PQ_COLUMNS
from ml.ingest.circuit_history import eligible_circuit_years

Stage = Literal["pre_weekend", "recency", "post_practice", "scenario", "post_quali"]

KEY_COLUMNS = ["Year", "Round", "DriverId", "Abbreviation", "TeamKey", "CircuitId"]
STAGE_INPUTS: dict[str, list[str]] = {
    "pre_weekend": KEY_COLUMNS,
    "recency": KEY_COLUMNS,
    "post_practice": [*KEY_COLUMNS, *PQ_COLUMNS],
    "scenario": [*KEY_COLUMNS, *WEATHER_COLUMNS],
    "post_quali": [*KEY_COLUMNS, *WEATHER_COLUMNS, "GridPosition"],
}

CIRCUIT_SHRINKAGE = 3  # a driver's circuit record counts as this many extra races of their form
FORM_WINDOW = 5
DNF_WINDOW = 10
PRIOR_WINDOW = 20  # races of recent form used as the prior for circuit shrinkage
DEFAULT_FINISH = 10.5  # mean finish of a 20-car field; used only when a driver has no history
WET_RAIN_FRACTION = 0.0  # a race counts as wet if more than this share of readings show rain

# First season of each regulation era. 2017 begins the era our data (from 2018) sits in; 2022 is the
# ground-effect cars; 2026 is the new power-unit and aero rules. Car performance resets at these.
REGULATION_ERA_STARTS = (2017, 2022, 2026)


class FeatureBuilder(Protocol):
    name: str
    stage: Stage
    columns: tuple[str, ...]

    def compute(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        """Features for `race`, indexed like `race`, with exactly `columns`."""


def _aligned(frame: pd.DataFrame, keys: pd.Series, race: pd.DataFrame) -> pd.DataFrame:
    """Look up `frame` (indexed by key) for each row of `race`, keeping race's index."""
    return frame.reindex(keys.to_numpy()).set_axis(race.index)


def _share_of_season_points(points: pd.Series, season_history: pd.DataFrame) -> pd.Series:
    """Points as a share of ALL points scored so far this season (NaN before any are scored)."""
    total = season_history["Points"].sum()
    return points / total if total > 0 else points * np.nan


class DriverForm:
    name = "driver_form"
    stage: Stage = "pre_weekend"
    columns = (
        "drv_finish_l5",
        "drv_grid_l5",
        "drv_quali_l5",
        "drv_qgap_l5",
        "drv_pace_l5",
        "drv_n_l5",
        "drv_dnf_l10",
        "drv_pts_share_ytd",
        "drv_rounds_since_last",
    )

    def compute(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        drivers = race["DriverId"]
        year = race["Year"].iloc[0]
        h = history[history["DriverId"].isin(drivers)].sort_values(["Year", "Round"])
        recent = h.groupby("DriverId").tail(FORM_WINDOW)
        recent = recent.assign(grid=recent["GridPosition"].where(recent["GridPosition"] > 0))
        agg = recent.groupby("DriverId").agg(
            drv_finish_l5=("Position", "mean"),
            drv_grid_l5=("grid", "mean"),
            drv_quali_l5=("QualiPosition", "mean"),
            drv_qgap_l5=("quali_gap_pct", "mean"),
            drv_pace_l5=("pace_gap_pct", "mean"),
            drv_n_l5=("Position", "count"),  # races with a known finish, so it matches the mean
        )
        last_ten = h.groupby("DriverId").tail(DNF_WINDOW)
        agg["drv_dnf_l10"] = last_ten["dnf"].astype(float).groupby(last_ten["DriverId"]).mean()

        season = history[history["Year"] == year]
        driver_points = season.groupby("DriverId")["Points"].sum().reindex(agg.index).fillna(0)
        agg["drv_pts_share_ytd"] = _share_of_season_points(driver_points, season)

        # Races elapsed since the driver's last start: a returning driver's "last 5" is stale.
        races = history[["Year", "Round"]].drop_duplicates()
        last = h.groupby("DriverId")[["Year", "Round"]].last()
        since = pd.Series(
            [
                int(((races["Year"] > y) | ((races["Year"] == y) & (races["Round"] > r))).sum())
                for y, r in zip(last["Year"], last["Round"], strict=True)
            ],
            index=last.index,
        )
        agg["drv_rounds_since_last"] = since.reindex(agg.index)

        out = _aligned(agg, drivers, race)
        out["drv_n_l5"] = out["drv_n_l5"].fillna(0).astype(float)  # one dtype whatever the data
        return out[list(self.columns)]


class TeamForm:
    name = "team_form"
    stage: Stage = "pre_weekend"
    columns = (
        "team_finish_l5",
        "team_qgap_l5",
        "team_pace_l3",
        "team_dnf_l10",
        "team_pts_share_ytd",
    )

    def compute(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        teams = race["TeamKey"]
        year = race["Year"].iloc[0]
        h = history[history["TeamKey"].isin(teams)]
        h = h.assign(dnf=h["dnf"].astype(float))
        per_race = (
            h.groupby(["TeamKey", "Year", "Round"], as_index=False)
            .agg(
                finish=("Position", "mean"),
                qgap=("quali_gap_pct", "mean"),
                pace=("pace_gap_pct", "mean"),
                dnf=("dnf", "mean"),
                pts=("Points", "sum"),
            )
            .sort_values(["TeamKey", "Year", "Round"])
        )
        by_team = per_race.groupby("TeamKey")
        agg = (
            by_team.tail(FORM_WINDOW)
            .groupby("TeamKey")
            .agg(team_finish_l5=("finish", "mean"), team_qgap_l5=("qgap", "mean"))
        )
        agg["team_pace_l3"] = by_team.tail(3).groupby("TeamKey")["pace"].mean()
        agg["team_dnf_l10"] = by_team.tail(DNF_WINDOW).groupby("TeamKey")["dnf"].mean()

        season = history[history["Year"] == year]
        team_points = season.groupby("TeamKey")["Points"].sum().reindex(agg.index).fillna(0)
        agg["team_pts_share_ytd"] = _share_of_season_points(team_points, season)
        out = _aligned(agg, teams, race)
        return out[list(self.columns)]


HALF_LIVES = (2, 4, 8)  # races; a result this many races old counts half as much as the latest


def _recency_weighted_mean(values: pd.Series, age: pd.Series, groups: pd.Series, half_life: float):
    """Mean of `values` per group with weight 0.5 ** (age / half_life); NaN values are skipped."""
    weight = (0.5 ** (age / half_life)).where(values.notna())
    total = (values * weight).groupby(groups).sum(min_count=1)
    return total / weight.groupby(groups).sum(min_count=1)


class RecencyForm:
    """Qualifying form where recent races count more, at several half-lives (in races).

    The plain `*_l5` features weight the last five races equally and ignore everything older, so a
    car upgrade or a regulation reset shows up late. Here every earlier race counts, newest most.
    Age is the number of races since, counted over the whole calendar in `history`, so a driver
    who missed races is not made to look recent. Half-lives are separate columns so a model can
    pick one (`feature_plan(recency=...)`); the choice is tuned on development seasons only.
    """

    name = "recency_form"
    stage: Stage = "recency"
    columns = tuple(
        f"{stem}_ew{h}" for h in HALF_LIVES for stem in ("drv_qgap", "drv_quali", "team_qgap")
    )

    def compute(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=race.index)
        if history.empty:
            for column in self.columns:
                out[column] = np.nan
            return out[list(self.columns)]
        order = history[["Year", "Round"]].drop_duplicates().sort_values(["Year", "Round"])
        position = pd.MultiIndex.from_frame(order)
        index = position.get_indexer(pd.MultiIndex.from_frame(history[["Year", "Round"]]))
        h = history.assign(age=len(order) - 1 - index)
        drivers = h[h["DriverId"].isin(race["DriverId"])]
        teams = (
            h[h["TeamKey"].isin(race["TeamKey"])]
            .groupby(["TeamKey", "Year", "Round"], as_index=False)
            .agg(qgap=("quali_gap_pct", "mean"), age=("age", "first"))
        )
        for half_life in HALF_LIVES:
            by_driver = {
                "drv_qgap": _recency_weighted_mean(
                    drivers["quali_gap_pct"], drivers["age"], drivers["DriverId"], half_life
                ),
                "drv_quali": _recency_weighted_mean(
                    drivers["QualiPosition"].astype(float),
                    drivers["age"],
                    drivers["DriverId"],
                    half_life,
                ),
            }
            for stem, series in by_driver.items():
                out[f"{stem}_ew{half_life}"] = race["DriverId"].map(series).to_numpy()
            by_team = _recency_weighted_mean(
                teams["qgap"], teams["age"], teams["TeamKey"], half_life
            )
            out[f"team_qgap_ew{half_life}"] = race["TeamKey"].map(by_team).to_numpy()
        return out[list(self.columns)]


class CircuitHistory:
    """How drivers and cars have fared at this circuit, using only seasons that count.

    A past season counts if at least 25% of the drivers ENTERED in the target race also raced there
    that year (the entry list is known before the weekend). A driver's circuit average is shrunk
    towards their recent overall form, so one lucky or unlucky visit does not dominate; with no
    counted visit it is simply that recent form. Raw finishes mix cars from different years, so
    treat `circ_drv_finish` as a weak signal.
    """

    name = "circuit_history"
    stage: Stage = "pre_weekend"
    columns = (
        "circ_drv_finish",
        "circ_drv_n",
        "circ_overtake",
        "circ_dnf_rate",
        "circ_n_years",
    )

    def compute(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        year, rnd = int(race["Year"].iloc[0]), int(race["Round"].iloc[0])
        circuit = race["CircuitId"].iloc[0]
        entered = set(race["DriverId"])

        events = history[["Year", "Round", "CircuitId"]].drop_duplicates()
        events = events.rename(columns={"Round": "RoundNumber"})
        shares = eligible_circuit_years(
            history[["Year", "Round", "DriverId"]], events, current_grid=entered, as_of=(year, rnd)
        )
        counted = set(shares.loc[(shares["CircuitId"] == circuit) & shares["eligible"], "Year"])
        there = history[(history["CircuitId"] == circuit) & history["Year"].isin(counted)]

        keys = race["DriverId"].to_numpy()
        visits = there.groupby("DriverId")["Position"].agg(["mean", "count"])
        recent = history.sort_values(["Year", "Round"]).groupby("DriverId").tail(PRIOR_WINDOW)
        prior = recent.groupby("DriverId")["Position"].mean()
        fallback = history["Position"].mean() if len(history) else DEFAULT_FINISH
        n = visits["count"].reindex(keys).fillna(0).to_numpy()
        mean = visits["mean"].reindex(keys).to_numpy()
        base = prior.reindex(keys).fillna(fallback).to_numpy()
        shrunk = np.where(
            n > 0,
            (n * np.nan_to_num(mean) + CIRCUIT_SHRINKAGE * base) / (n + CIRCUIT_SHRINKAGE),
            base,
        )

        started = there[there["GridPosition"] > 0]
        finishers = started[~started["dnf"].astype(bool)]
        out = pd.DataFrame(index=race.index)
        out["circ_drv_finish"] = shrunk
        out["circ_drv_n"] = n
        out["circ_overtake"] = (
            float((finishers["GridPosition"] - finishers["Position"]).mean())
            if len(finishers)
            else np.nan
        )
        out["circ_dnf_rate"] = float(there["dnf"].astype(bool).mean()) if len(there) else np.nan
        out["circ_n_years"] = float(len(counted))
        return out[list(self.columns)]


class Context:
    """Where the race sits in the season and in the rules cycle (both known from the calendar)."""

    name = "context"
    stage: Stage = "pre_weekend"
    columns = ("season_round", "seasons_since_reg_change")

    def compute(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        year = race["Year"].iloc[0]
        era_start = max(start for start in REGULATION_ERA_STARTS if start <= year)
        out = pd.DataFrame(index=race.index)
        out["season_round"] = race["Round"].astype(float)
        out["seasons_since_reg_change"] = float(year - era_start)
        return out[list(self.columns)]


class WeekendPace:
    """Same-weekend pace, known once practice (and any earlier sprint session) has been run.

    The raw `pq_` columns come from the earlier sessions of this weekend (see weekend.py) and are
    inputs, like the weather; this builder adds within-field views of them: the rank of each
    driver's best practice gap, and the gap to their teammate(s), which removes the car and leaves
    the driver. Missing sessions leave NaN.
    """

    name = "weekend_pace"
    stage: Stage = "post_practice"
    columns = (
        "pq_fp_best_gap",
        "pq_fp_last_gap",
        "pq_fp_best_rank",
        "pq_fp_vs_teammate",
        "pq_fp_n",
        "pq_sprintq_pos",
        "pq_sprintq_gap",
        "pq_sprint_pos",
    )

    def compute(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        out = race[list(PQ_COLUMNS)].copy()
        gap = race["pq_fp_best_gap"]
        out["pq_fp_best_rank"] = gap.rank(method="average")
        team_sum = gap.groupby(race["TeamKey"]).transform("sum")  # NaN counts as 0 here
        team_n = gap.notna().astype(float).groupby(race["TeamKey"]).transform("sum")
        others_n = team_n - gap.notna().astype(float)
        others_mean = (team_sum - gap.fillna(0.0)) / others_n.where(others_n > 0)
        out["pq_fp_vs_teammate"] = gap - others_mean
        return out[list(self.columns)]


class Weather:
    """Scenario inputs: the race-day weather (a user-chosen or forecast value when predicting)."""

    name = "weather"
    stage: Stage = "scenario"
    columns = (*WEATHER_COLUMNS, "wx_is_wet")

    def compute(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        out = race[WEATHER_COLUMNS].copy()
        wet = race["wx_rain_frac"] > WET_RAIN_FRACTION
        out["wx_is_wet"] = wet.astype(float).where(race["wx_rain_frac"].notna())
        return out[list(self.columns)]


class Grid:
    """Starting grid, known after qualifying (or entered by the user). Pit-lane starts go last.

    `grid` is a copy of the target `y_grid` and is almost the same as `y_quali`: use it only for
    targets that happen after qualifying (see `feature_columns` in build.py).
    """

    name = "grid"
    stage: Stage = "post_quali"
    columns = ("grid", "grid_pit_lane", "grid_frac")

    def compute(self, history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
        slot = race["GridPosition"].where(race["GridPosition"] > 0)
        last = slot.max() + 1 if slot.notna().any() else 1
        filled = slot.fillna(last)
        out = pd.DataFrame(index=race.index)
        out["grid"] = filled
        out["grid_pit_lane"] = slot.isna().astype(float)
        out["grid_frac"] = filled / len(race)
        return out[list(self.columns)]


BUILDERS: list[FeatureBuilder] = [
    DriverForm(),
    TeamForm(),
    RecencyForm(),
    CircuitHistory(),
    Context(),
    WeekendPace(),
    Weather(),
    Grid(),
]

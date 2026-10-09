"""Build the feature table by running every builder race by race, in order.

For each target race the builders see only: the entry list (and stage-allowed inputs) of that race,
and the results of earlier races that carried real information (`race_ok`). See `builders.py`.

Two entry points:
- `features_for_entry`: features for a race that has NOT happened (or whose outcome we hide). It
  needs only the history and the entry list; this is what prediction uses.
- `features_for_race`: the same features for a race that is in the table, plus its keys and
  targets. Training and evaluation use it.

Rules for whoever trains on the table:
- Rows of a race are ordered by driver id, never by finishing position. Keep it that way: row order
  must not carry the answer.
- `race_ok` says whether a race carried real performance information. It is derived from that
  race's own results and laps, so it is known only afterwards: use it to drop TRAINING rows, never
  as a feature and never to filter test races.
- `feature_columns(target)` returns only features that are legitimately known when predicting that
  target. Weather is excluded unless asked for (see `include_scenario`).
"""

import warnings

import numpy as np
import pandas as pd

from ml.evaluation.data import race_keys
from ml.features.builders import BUILDERS, KEY_COLUMNS, STAGE_INPUTS, FeatureBuilder
from ml.ingest.circuit_history import before

TARGETS = {
    "y_finish": "Position",
    "y_dnf": "dnf",
    "y_quali": "QualiPosition",
    "y_qgap": "quali_gap_pct",  # continuous: % slower than pole; NaN with no qualifying time
    "y_grid": "GridPosition",
}

# Which feature stages may be used to predict each target. A target decided at qualifying must not
# be predicted from the grid it produces (`grid` correlates 0.96 with `y_quali` and is the same
# thing as `y_grid`), and race-day weather is not known when qualifying is decided.
TARGET_STAGES: dict[str, frozenset[str]] = {
    "y_finish": frozenset({"pre_weekend", "recency", "post_practice", "scenario", "post_quali"}),
    "y_dnf": frozenset({"pre_weekend", "recency", "post_practice", "scenario", "post_quali"}),
    "y_quali": frozenset({"pre_weekend", "recency", "post_practice"}),
    "y_qgap": frozenset({"pre_weekend", "recency", "post_practice"}),
    "y_grid": frozenset({"pre_weekend", "recency", "post_practice"}),
}
# Stages a model has to ask for: weather because history holds the realised value but a forecast is
# all that exists when predicting; practice because it only exists once practice has been run;
# recency because it is an experiment (4b.3) and the default boards must stay reproducible.
OPT_IN_STAGES = {
    "scenario": "include_scenario",
    "post_practice": "include_practice",
    "recency": "include_recency",
}


def as_flags(series: pd.Series, name: str) -> pd.Series:
    """`series` as real booleans; refuse gaps and look-alikes (strings, numbers) rather than guess."""
    if series.isna().any():
        raise ValueError(f"{name} has missing values")
    if series.dtype != bool and not series.map(lambda v: isinstance(v, bool | np.bool_)).all():
        raise ValueError(
            f"{name} must hold real booleans (a string like 'False' would read as True)"
        )
    return series.astype(bool)


def usable_history(history: pd.DataFrame) -> pd.DataFrame:
    """Only the races that carried real performance information (`race_ok` True).

    Raises if `race_ok` is absent, has gaps or is not boolean, or if a driver appears twice in a
    race: silently using every race would let a washed-out race (the 2021 Belgian GP) back into
    form and circuit features, and duplicated rows would quietly change every average.
    """
    if "race_ok" not in history.columns:
        raise KeyError("history needs a 'race_ok' column (build it with ml.features.tables)")
    flags = as_flags(history["race_ok"], "history race_ok")
    if history.duplicated(["Year", "Round", "DriverId"]).any():
        raise ValueError("history has the same driver twice in one race")
    return history[flags]


def _check_entries(entries: pd.DataFrame, key: tuple[int, int]) -> None:
    if entries.empty:
        raise ValueError("entries is empty")
    if not entries.index.is_unique:
        raise ValueError("entries must have a unique index (one row per driver)")
    races = entries[["Year", "Round"]].drop_duplicates()
    if len(races) != 1 or (int(races.iloc[0]["Year"]), int(races.iloc[0]["Round"])) != key:
        raise ValueError(f"entries must all belong to race {key}")
    if entries["DriverId"].duplicated().any():
        raise ValueError("entries has the same driver more than once")
    if entries["CircuitId"].nunique() != 1:
        raise ValueError("entries must all be for one circuit")


def features_for_entry(
    history: pd.DataFrame,
    entries: pd.DataFrame,
    key: tuple[int, int],
    builders: list[FeatureBuilder] = BUILDERS,
) -> pd.DataFrame:
    """Features for the race `key`, from `history` (earlier races only) and its `entries`.

    `entries` has one row per driver and must contain the stage inputs of every requested builder
    (KEY_COLUMNS always; weather columns for the scenario stage; GridPosition for post_quali).
    Pass only the builders whose inputs are known; for a pre-qualifying prediction that is
    `[b for b in BUILDERS if b.stage == "pre_weekend"]`. Every feature is returned as float64.
    """
    key = (int(key[0]), int(key[1]))
    _check_entries(entries, key)
    history = usable_history(before(history, key))
    out = entries[KEY_COLUMNS].copy()
    for builder in builders:
        needed = STAGE_INPUTS[builder.stage]
        missing = [c for c in needed if c not in entries.columns]
        if missing:
            raise KeyError(f"builder {builder.name!r} needs entry columns {missing}")
        inputs = entries[needed].copy()
        features = builder.compute(history, inputs).astype(float)  # one dtype for every feature
        if list(features.columns) != list(builder.columns) or not features.index.equals(
            inputs.index
        ):
            raise ValueError(f"builder {builder.name!r} returned the wrong shape or columns")
        out = out.join(features)
    return out


def features_for_race(
    rows: pd.DataFrame, key: tuple[int, int], builders: list[FeatureBuilder] = BUILDERS
) -> pd.DataFrame:
    """Keys, features, `race_ok` and targets for a race that is in `rows`."""
    year, rnd = key = (int(key[0]), int(key[1]))
    target = rows[(rows["Year"] == year) & (rows["Round"] == rnd)]
    if target.empty:
        raise KeyError(f"no race {key}")
    out = features_for_entry(rows, target, key, builders)
    out["race_ok"] = as_flags(target["race_ok"], "race_ok")
    for name, source in TARGETS.items():
        out[name] = target[source]
    return out


def build_feature_table(
    rows: pd.DataFrame, builders: list[FeatureBuilder] = BUILDERS
) -> pd.DataFrame:
    """One row per driver per race, in chronological order."""
    parts = [features_for_race(rows, key, builders) for key in race_keys(rows)]
    with warnings.catch_warnings():
        # Early races have all-NaN history features; pandas warns that their dtype is ignored
        # when combining. They are floats either way, so the result is the same.
        warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation")
        return pd.concat(parts, ignore_index=True)


def feature_columns(
    target: str = "y_finish",
    builders: list[FeatureBuilder] = BUILDERS,
    include_scenario: bool = False,
    include_practice: bool = False,
    include_recency: bool = False,
) -> list[str]:
    """Feature columns that are legitimately known when predicting `target`.

    Three stages are opt-in. Scenario features (weather): in history they are the realised race-day
    average but at prediction time they are a forecast or a user's guess, so a model trained on them
    looks better than it will be; qualifying and grid targets never get them. Practice features
    (`post_practice`): they only exist once practice has been run, so a forecast made earlier must
    not use them. Recency-weighted form (`recency`): known before the weekend like the rest of the
    form, but opt-in so the earlier boards do not change. Report results with and without.
    """
    allowed = TARGET_STAGES[target]
    if include_scenario and "scenario" not in allowed:
        raise ValueError(f"{target} cannot use scenario (weather) features")
    stages = set(allowed)
    if not include_scenario:
        stages.discard("scenario")
    if not include_practice:
        stages.discard("post_practice")
    if not include_recency:
        stages.discard("recency")
    return [column for b in builders if b.stage in stages for column in b.columns]


def training_frame(
    table: pd.DataFrame,
    target: str,
    include_scenario: bool = False,
    include_practice: bool = False,
    include_recency: bool = False,
) -> pd.DataFrame:
    """Rows and columns that are safe to train `target` on: the rule book as a function.

    Keeps only races that carried real information (`race_ok`) and rows where the target is known,
    and only the keys, the features legitimately known for that target, and the target itself (no
    other `y_*` columns, no `race_ok`). Row order (by driver id within a race) is preserved.
    """
    features = feature_columns(
        target,
        include_scenario=include_scenario,
        include_practice=include_practice,
        include_recency=include_recency,
    )
    columns = [*KEY_COLUMNS, *features, target]
    keep = as_flags(table["race_ok"], "race_ok") & table[target].notna()
    return table.loc[keep, columns].reset_index(drop=True)

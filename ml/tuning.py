"""Hyperparameter search on the DEVELOPMENT seasons only, with every trial logged.

python -m uv run python -m ml.tuning [--targets quali race] [--models ridge elo ...] [--dry-run]

Why this exists: Milestone 4's settings were set by hand with the reported seasons in view, and no
record was kept. Here a candidate is scored by walk-forward evaluation on the development seasons
(2019 to 2021) and nothing else; the harness is called with those seasons only and
`check_development_only` refuses anything else. All trials go to `ml/models/tuning_log.csv`, and the
best setting per model and target is written to `ml/models/tuned_params.json`.

Selection rule, fixed in advance: the highest mean Spearman over the development races; a tie goes
to the setting listed first, and each grid lists the simpler or more regularised values first.
The grids are deliberately small. A bigger search would only find more noise to fit.
"""

import argparse
import itertools
import json
import time
from datetime import UTC, datetime

import pandas as pd

from ml.evaluation.data import load_history
from ml.evaluation.harness import evaluate
from ml.evaluation.splits import DEV_SEASONS, check_development_only
from ml.features.build import build_feature_table
from ml.models.tuned import LOG_PATH, PARAMS_PATH, load_tuned, params_for
from ml.models.zoo import FEATURE_EXPERIMENT_MODELS, MODEL_NAMES, build_model

# Simpler / more regularised values first, so a tie goes to the simpler setting.
GRIDS: dict[str, dict[str, list]] = {
    "ridge": {"alpha": [300, 100, 30, 10, 3]},
    "random_forest": {"max_depth": [4, 6, 8], "min_samples_leaf": [40, 20, 10]},
    "lightgbm": {
        "num_leaves": [4, 8],
        "n_estimators": [150, 300],
        "min_child_samples": [40, 20],
    },
    "xgboost": {"max_depth": [2, 3, 4], "n_estimators": [150, 300], "min_child_weight": [15, 5]},
    "lightgbm_lambdarank": {"num_leaves": [4, 8], "n_estimators": [100, 250]},
    "xgboost_rank": {"max_depth": [2, 3], "n_estimators": [100, 250]},
    "elo": {
        "k_driver": [15, 30, 60],
        "k_team": [25, 50, 100],
        "season_shrink": [0.9, 0.75, 0.5],
    },
    "plackett_luce": {"l2": [10, 3, 1, 0.3]},
}
LOG_COLUMNS = [
    "timestamp",
    "target",
    "model",
    "params",
    "dev_spearman",
    "dev_races",
    "dev_seasons",
    "seconds",
    "chosen",
]


def candidates(model: str) -> list[dict]:
    """Every setting in the model's grid, in a fixed order."""
    grid = GRIDS[model]
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def score_on_development(
    model, history: pd.DataFrame, target: str, seasons: tuple[int, ...] = DEV_SEASONS
) -> tuple[float, int]:
    """(mean Spearman, races) of one model over the development seasons. Refuses other seasons."""
    check_development_only(seasons)
    per_race = evaluate([model], history, target, test_seasons=seasons)
    scored_seasons = set(per_race["Year"])
    check_development_only(scored_seasons)  # belt and braces: what was actually scored
    return float(per_race["spearman"].mean()), len(per_race)


def tune(
    targets: list[str],
    models: list[str],
    history: pd.DataFrame,
    table: pd.DataFrame,
    log_path=LOG_PATH,
    params_path=PARAMS_PATH,
) -> dict:
    """Run the grids, append every trial to the log, write the chosen settings."""
    previous = load_tuned(params_path)
    trials, chosen = [], {"provenance": {}}
    if "quali_features" in previous:  # chosen by tune_features; a rerun here must not drop it
        chosen["quali_features"] = previous["quali_features"]
    for target in targets:
        chosen.setdefault(target, {})
        for name in models:
            best = None
            for params in candidates(name):
                started = time.time()
                model = build_model(name, target, table, params)
                spearman, races = score_on_development(model, history, target)
                trial = {
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    "target": target,
                    "model": name,
                    "params": json.dumps(params, sort_keys=True),
                    "dev_spearman": round(spearman, 6),
                    "dev_races": races,
                    "dev_seasons": " ".join(map(str, DEV_SEASONS)),
                    "seconds": round(time.time() - started, 1),
                    "chosen": False,
                }
                trials.append(trial)
                if (
                    best is None or spearman > best[0] + 1e-12
                ):  # strictly better: ties keep the first
                    best = (spearman, params, len(trials) - 1)
            trials[best[2]]["chosen"] = True
            chosen[target][name] = best[1]
            print(f"[{target}] {name}: dev Spearman {best[0]:.4f} with {best[1]}", flush=True)
    chosen["provenance"] = {
        "tuned_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dev_seasons": list(DEV_SEASONS),
        "selection": "highest mean Spearman on the development races; ties go to the first listed",
        "log": "ml/models/tuning_log.csv",
        "trials": len(trials),
    }
    log = pd.DataFrame(trials, columns=LOG_COLUMNS)
    if log_path.exists():  # keep earlier runs: the log is a record, not a cache
        log = pd.concat([pd.read_csv(log_path), log], ignore_index=True)
    log.to_csv(log_path, index=False)
    params_path.write_text(json.dumps(chosen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return chosen


FEATURE_CHOICES = [
    {"gap_target": gap, "recency": recency} for gap in (False, True) for recency in (None, 2, 4, 8)
]  # plain first, so a tie keeps the plain setup


def tune_features(
    history: pd.DataFrame,
    table: pd.DataFrame,
    models: tuple[str, ...] = FEATURE_EXPERIMENT_MODELS,
    log_path=LOG_PATH,
    params_path=PARAMS_PATH,
) -> dict:
    """4b.3: per model, pick the target (rank or gap to pole) and the recency half-life.

    Each model keeps its already-tuned settings from `tuned_params.json`; only the target and the
    half-life vary. Scored on the development seasons only, like everything in this module. The
    result is stored under "quali_features" without touching the other settings.
    """
    tuned = load_tuned(params_path)
    trials, chosen = [], {}
    for name in models:
        best = None
        for choice in FEATURE_CHOICES:
            started = time.time()
            model = build_model(name, "quali", table, params_for(tuned, "quali", name), **choice)
            spearman, races = score_on_development(model, history, "quali")
            trials.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    "target": "quali_features",
                    "model": name,
                    "params": json.dumps(choice, sort_keys=True),
                    "dev_spearman": round(spearman, 6),
                    "dev_races": races,
                    "dev_seasons": " ".join(map(str, DEV_SEASONS)),
                    "seconds": round(time.time() - started, 1),
                    "chosen": False,
                }
            )
            if best is None or spearman > best[0] + 1e-12:
                best = (spearman, choice, len(trials) - 1)
        trials[best[2]]["chosen"] = True
        chosen[name] = best[1]
        print(f"[quali_features] {name}: dev Spearman {best[0]:.4f} with {best[1]}", flush=True)
    log = pd.DataFrame(trials, columns=LOG_COLUMNS)
    if log_path.exists():
        log = pd.concat([pd.read_csv(log_path), log], ignore_index=True)
    log.to_csv(log_path, index=False)
    tuned["quali_features"] = chosen
    params_path.write_text(json.dumps(tuned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        action="store_true",
        help="4b.3: tune the qualifying target (rank or gap) and recency half-life instead.",
    )
    parser.add_argument(
        "--targets", nargs="+", choices=["race", "quali"], default=["quali", "race"]
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--dry-run", action="store_true", help="List the trials and exit.")
    args = parser.parse_args()

    if args.features:
        trials = len(FEATURE_CHOICES) * len(FEATURE_EXPERIMENT_MODELS)
        print(f"{trials} trials on development seasons {DEV_SEASONS} (nothing else is scored)")
        if args.dry_run:
            return 0
        history = load_history()
        tune_features(history, build_feature_table(history))
        return 0
    total = sum(len(candidates(m)) for m in args.models) * len(args.targets)
    print(f"{total} trials on development seasons {DEV_SEASONS} (nothing else is scored)")
    if args.dry_run:
        for m in args.models:
            print(f"  {m}: {len(candidates(m))} settings {GRIDS[m]}")
        return 0
    history = load_history()
    table = build_feature_table(history)
    tune(args.targets, args.models, history, table)
    print(f"\nlog:    {LOG_PATH}\nchosen: {PARAMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

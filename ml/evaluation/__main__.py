"""CLI: python -m ml.evaluation [--models baselines|all] [--target ...] [--allow-locked].

Prints and saves one leaderboard per target:
- race:                  predict the finishing order given the (real or user-supplied) grid
- quali:                 predict the qualifying order
- race_predicted_grid:   predict the finishing order BEFORE qualifying: from a grid predicted by a
                         qualifying model, or by models that never see a grid

"Clearly beats" means the season-clustered 95% interval for the gain over the best baseline is
above zero. The exit code is 3 when no model clearly beats the best baseline on one of the two
primary targets (race, quali). The predicted-grid board is informational: its main baseline is
itself the output of a model.
"""

import argparse

import pandas as pd

from ml.evaluation.compare import leaderboard
from ml.evaluation.data import load_history
from ml.evaluation.harness import evaluate
from ml.evaluation.metrics import summarize
from ml.evaluation.splits import LOCKED_SEASON
from ml.ingest.atomic import write_csv_atomic
from ml.ingest.paths import data_dir

TARGETS = ("race", "quali", "race_predicted_grid")
PRIMARY_TARGETS = ("race", "quali")
NO_MODEL_BEATS_BASELINES = 3  # exit code: a result to discuss, not to hide


def build_models(target: str, which: str, history: pd.DataFrame) -> tuple[str, list, list[str]]:
    """(harness target, models, baseline names) for one leaderboard."""
    from ml.features.build import build_feature_table
    from ml.models.zoo import baselines, make_models, make_predicted_grid_models

    harness_target = "quali" if target == "quali" else "race"
    if target == "race_predicted_grid":
        if which == "baselines":
            return harness_target, [], []
        models, baseline_names = make_predicted_grid_models(build_feature_table(history))
        return harness_target, models, baseline_names
    base = baselines(harness_target)
    names = [b.name for b in base]
    if which == "baselines":
        return harness_target, base, names
    return (
        harness_target,
        [*base, *make_models(harness_target, build_feature_table(history))],
        names,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=["baselines", "all"], default="baselines")
    parser.add_argument("--target", choices=[*TARGETS, "all"], default="all")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--allow-locked",
        action="store_true",
        help=f"Also evaluate the locked final test season ({LOCKED_SEASON}). Use once, after the "
        "model choice is frozen.",
    )
    args = parser.parse_args()

    history = load_history()
    out = data_dir() / "reports"
    targets = TARGETS if args.target == "all" else (args.target,)
    status = 0
    for target in targets:
        harness_target, models, baseline_names = build_models(target, args.models, history)
        if not models:
            continue
        per_race = evaluate(models, history, harness_target, allow_locked=args.allow_locked)
        if per_race.empty:
            print(f"[{target}] not enough seasons ingested to evaluate yet.")
            status = status or 1
            continue
        board = leaderboard(per_race, baseline_names, n_boot=args.n_boot)
        write_csv_atomic(board, out / f"leaderboard_{target}.csv", index=False)
        write_csv_atomic(per_race, out / f"per_race_{target}.csv", index=False)
        write_csv_atomic(
            summarize(per_race), out / f"leaderboard_by_year_{target}.csv", index=False
        )

        shown = board.drop(
            columns=["is_baseline", "beats_best_baseline", "iid_ci_low", "iid_ci_high"]
        )
        print(f"\n=== {target}  (best baseline: {board.attrs['best_baseline']}) ===")
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
            print(shown.to_string(index=False))
        candidates = board[~board["is_baseline"]]
        if len(candidates) and not candidates["clearly_beats"].any():
            note = f"[{target}] no model clearly beats the best baseline (95% season-clustered interval)"
            if target in PRIMARY_TARGETS:
                print(f"\n*** {note}: NO MODEL BEATS THE BEST BASELINE ***")
                status = NO_MODEL_BEATS_BASELINES
            else:
                print(f"\n{note}")
    print(f"\nSaved to {out}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

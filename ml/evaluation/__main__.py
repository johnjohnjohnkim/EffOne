"""CLI: python -m ml.evaluation [--models baselines|all] [--target ...] [--seasons ...] [--allow-locked].

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
from ml.evaluation.splits import LOCKED_SEASON, SEASON_SETS
from ml.ingest.atomic import write_csv_atomic
from ml.ingest.paths import data_dir

TARGETS = ("race", "quali", "quali_after_practice", "race_predicted_grid", "quali_features")
PRIMARY_TARGETS = ("race", "quali", "quali_after_practice")
NO_MODEL_BEATS_BASELINES = 3  # exit code: a result to discuss, not to hide


def test_seasons(which: str, allow_locked: bool) -> tuple[int, ...] | None:
    """The seasons to score for a --seasons choice; the locked season joins only when asked for."""
    chosen = SEASON_SETS[which]
    if chosen is None:
        return None
    return (*chosen, LOCKED_SEASON) if allow_locked and which != "dev" else chosen


def build_models(target: str, which: str, history: pd.DataFrame) -> tuple[str, list, list[str]]:
    """(harness target, models, baseline names) for one leaderboard."""
    from ml.features.build import build_feature_table
    from ml.models.zoo import (
        baselines,
        make_feature_variants,
        make_models,
        make_predicted_grid_models,
    )

    if target == "quali_features":  # 4b.3: the qualifying board plus the gap/recency variants
        base = baselines("quali")
        names = [b.name for b in base]
        if which == "baselines":
            return "quali", base, names
        table = build_feature_table(history)
        variants = make_feature_variants(table)
        return "quali", [*base, *make_models("quali", table), *variants], names
    harness_target = target if target in ("quali", "quali_after_practice") else "race"
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


def calibration_command(history: pd.DataFrame, seasons, out, args) -> int:
    from ml.features.build import build_feature_table
    from ml.simulation.report import calibration_report

    if seasons is None:  # "all": every season except the locked one
        seasons = tuple(s for s in sorted(history["Year"].unique()) if s != LOCKED_SEASON)
    summary = calibration_report(
        history, build_feature_table(history), seasons, out, args.allow_locked, args.n_boot
    )
    shown = summary.drop(columns=["best_baseline", "races"])
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(shown.to_string(index=False))
    print(f"\nSaved to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=["baselines", "all"], default="baselines")
    parser.add_argument("--target", choices=[*TARGETS, "all"], default="all")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--seasons",
        choices=list(SEASON_SETS),
        default="report",
        help="Which test seasons to score: dev (2019-2021, where tuning happens), report "
        "(2022-2024 and 2026, the default) or all (both). The locked season needs --allow-locked.",
    )
    parser.add_argument(
        "--allow-locked",
        action="store_true",
        help=f"Also evaluate the locked final test season ({LOCKED_SEASON}). Use once, after the "
        "model choice is frozen.",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Score the simulated win/podium/top-10/DNF probabilities instead of the leaderboards: "
        "writes log-loss tables and reliability plots to data/reports.",
    )
    args = parser.parse_args()

    history = load_history()
    out = data_dir() / "reports"
    if args.calibration:
        return calibration_command(
            history, test_seasons(args.seasons, args.allow_locked), out, args
        )
    targets = TARGETS if args.target == "all" else (args.target,)
    status = 0
    for target in targets:
        harness_target, models, baseline_names = build_models(target, args.models, history)
        if not models:
            continue
        per_race = evaluate(
            models,
            history,
            harness_target,
            allow_locked=args.allow_locked,
            test_seasons=test_seasons(args.seasons, args.allow_locked),
        )
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

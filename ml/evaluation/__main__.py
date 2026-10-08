"""CLI: python -m ml.evaluation [--allow-locked] -> prints and saves the leaderboard."""

import argparse

import pandas as pd

from ml.evaluation.baselines import BASELINES
from ml.evaluation.data import load_race_results
from ml.evaluation.harness import evaluate
from ml.evaluation.metrics import summarize
from ml.evaluation.splits import LOCKED_SEASON
from ml.ingest.paths import data_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-locked",
        action="store_true",
        help=f"Also evaluate the locked final test season ({LOCKED_SEASON}). Use once, after the "
        "model choice is frozen.",
    )
    args = parser.parse_args()

    results = load_race_results()
    per_race = evaluate([cls() for cls in BASELINES], results, allow_locked=args.allow_locked)
    if per_race.empty:
        print("Not enough seasons ingested to evaluate yet (need at least two).")
        return 1
    board = summarize(per_race)

    out = data_dir() / "reports"
    out.mkdir(exist_ok=True)
    board.to_csv(out / "leaderboard.csv", index=False)
    per_race.to_csv(out / "per_race.csv", index=False)

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(board[board["Year"] == "all"].drop(columns="Year").to_string(index=False))
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

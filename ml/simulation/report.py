"""The calibration report: log-loss tables, bootstrap intervals and reliability plots.

For each mode (real grid, predicted grid) the simulated probabilities are scored on the chosen test
seasons against two references: a grid-slot prior learned on the training seasons, and the uniform
guess. The interval is the same season-clustered bootstrap the leaderboards use, applied to the
per-race log-loss difference (negative means the simulation is better).
"""

from collections.abc import Collection
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ml.evaluation.compare import paired_bootstrap
from ml.simulation.calibration import (
    EVENTS,
    MODES,
    baseline_probabilities,
    collect,
    expected_calibration_error,
    outcomes,
    per_race_losses,
    probabilities,
    reliability,
)
from ml.simulation.forecast import load_sim_params

ALL_EVENTS = (*EVENTS, "dnf")


def score_mode(
    history: pd.DataFrame,
    table: pd.DataFrame,
    mode: str,
    seasons: Collection[int],
    allow_locked: bool,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """(summary, by-season summary, reliability bins, predictions per model) for one mode."""
    raw = collect(history, table, mode, seasons, allow_locked)
    if raw.empty:
        raise ValueError(f"no races to score for seasons {sorted(seasons)}")
    actual = outcomes(raw)
    models = {
        "simulation": probabilities(raw, load_sim_params(mode)),
        **baseline_probabilities(raw),
    }
    per_race = pd.concat(
        [per_race_losses(name, probs, actual) for name, probs in models.items()], ignore_index=True
    )

    summary, bins = [], []
    for event in ALL_EVENTS:
        column = f"ll_{event}"
        means = per_race.groupby("model")[column].mean()
        best_baseline = means.drop("simulation").idxmin()
        diff, low, high = paired_bootstrap(per_race, "simulation", best_baseline, column, n_boot)
        for name, probs in models.items():
            known = actual[event].notna()
            p, y = probs.loc[known, f"p_{event}"].to_numpy(), actual.loc[known, event].to_numpy()
            summary.append(
                {
                    "mode": mode,
                    "event": event,
                    "model": name,
                    "logloss": means[name],
                    "brier": per_race.groupby("model")[f"brier_{event}"].mean()[name],
                    "ece": expected_calibration_error(p, y),
                    "races": per_race["model"].eq(name).sum(),
                    "best_baseline": best_baseline if name == "simulation" else "",
                    "vs_best_baseline": diff if name == "simulation" else float("nan"),
                    "ci_low": low if name == "simulation" else float("nan"),
                    "ci_high": high if name == "simulation" else float("nan"),
                    "clearly_better": bool(high < 0) if name == "simulation" else False,
                }
            )
            table_ = reliability(p, y)
            bins.append(table_.assign(mode=mode, event=event, model=name))
    summary, bins = pd.DataFrame(summary), pd.concat(bins, ignore_index=True)
    # For DNF the two references are the same flat recent rate: show it once, under its real name.
    for frame in (summary, bins):
        frame.drop(
            frame[(frame["event"] == "dnf") & (frame["model"] == "uniform")].index, inplace=True
        )
        frame.loc[(frame["event"] == "dnf") & (frame["model"] == "grid_prior"), "model"] = (
            "recent_rate"
        )
    summary.loc[summary["event"] == "dnf", "best_baseline"] = summary["best_baseline"].where(
        summary["best_baseline"] == "", "recent_rate"
    )
    by_season = (
        per_race.groupby(["model", "Year"])[[f"ll_{e}" for e in ALL_EVENTS]].mean().reset_index()
    )
    by_season.insert(0, "mode", mode)
    return summary, by_season, bins, models


def plot_reliability(bins: pd.DataFrame, mode: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, event in zip(axes.ravel(), ALL_EVENTS, strict=True):
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, label="perfect")
        reference = "recent_rate" if event == "dnf" else "grid_prior"
        for model, style in (("simulation", "o-"), (reference, "s--")):
            part = bins[(bins["event"] == event) & (bins["model"] == model)]
            ax.plot(part["predicted"], part["observed"], style, ms=4, label=model)
        ax.set_title(f"{event} ({mode})")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def calibration_report(
    history: pd.DataFrame,
    table: pd.DataFrame,
    seasons: Collection[int],
    out: Path,
    allow_locked: bool = False,
    n_boot: int = 2000,
) -> pd.DataFrame:
    summaries, seasons_tables, bin_tables = [], [], []
    for mode in MODES:
        summary, by_season, bins, _ = score_mode(
            history, table, mode, seasons, allow_locked, n_boot
        )
        summaries.append(summary)
        seasons_tables.append(by_season)
        bin_tables.append(bins)
        plot_reliability(bins, mode, out / f"calibration_{mode}.png")
    summary = pd.concat(summaries, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "calibration_logloss.csv", index=False)
    pd.concat(seasons_tables, ignore_index=True).to_csv(
        out / "calibration_by_season.csv", index=False
    )
    pd.concat(bin_tables, ignore_index=True).to_csv(
        out / "calibration_reliability.csv", index=False
    )
    return summary

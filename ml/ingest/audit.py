"""Read-only data-quality audit of ingested sessions.

python -m ml.ingest.audit  ->  data/reports/data_quality.csv plus a printed list of flagged sessions.

Nothing in data/raw is changed or deleted. The report says which sessions are fit for
results-based features (`results_ok`) and lap-based features (`laps_ok`); later phases filter on it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.ingest.atomic import write_csv_atomic
from ml.ingest.paths import data_dir, raw_dir

# Grids have been 20 cars (2018-2025) and 22 from 2026. The range is wide enough for stand-ins but
# tight enough to catch duplicated or truncated results.
MIN_RESULT_ROWS, MAX_RESULT_ROWS = 18, 22
# Only races and sprints are expected to have a time for nearly every lap. Qualifying and
# practice legitimately contain many untimed out-, in- and aborted laps, so they are not
# checked for this. Lap 1 is ignored: in sprints it is untimed for every driver (1 of ~19 laps).
# Above SOFT, some laps are untimed (red-flag laps, for example); that is informational, since
# lap-based features can drop those laps one by one. Above HARD the lap data is unusable.
SOFT_NULL_LAPTIME_FRAC = 0.05
HARD_NULL_LAPTIME_FRAC = 0.25
MAX_NULL_POSITION_FRAC = 0.2

RESULT_FLAGS = {
    "unusual_result_rows",
    "duplicate_drivers",
    "many_null_positions",
    "no_single_winner",
    "unreadable_marker",
}
# Informational only: the session is still usable, so it never changes results_ok / laps_ok.
INFO_FLAGS = {"some_untimed_laps"}
LAP_FLAGS = {"no_laps", "many_null_laptimes", "laps_missing_drivers", "unreadable_marker"}


def _untimed_fraction(laps: pd.DataFrame) -> float:
    """Share of laps without a lap time, ignoring lap 1. NaN when there are no laps to judge."""
    judged = laps[laps["LapNumber"] != 1]
    return float(judged["LapTime"].isna().mean()) if len(judged) else np.nan


def session_kind(slug: str) -> str:
    if slug in ("race", "sprint"):
        return "race"
    if "qualifying" in slug or "shootout" in slug:
        return "qualifying"
    return "practice"


def _read(table: str, year: int, rnd: int, slug: str, columns: list[str]) -> pd.DataFrame:
    path = raw_dir() / table / f"year={year}" / f"round={rnd:02d}" / f"{slug}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path, columns=columns)


def parse_marker_name(marker: Path) -> tuple[int, int, str]:
    year, rnd, slug = marker.stem.split("_", 2)
    return int(year), int(rnd), slug


def audit_session(marker: Path) -> dict[str, object]:
    """Audit one session from its done-marker (year, round and session come from its name)."""
    year, rnd, slug = parse_marker_name(marker)
    kind = session_kind(slug)
    row = {"Year": year, "Round": rnd, "Session": slug, "kind": kind}
    flags: list[str] = []
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(info, dict) or not isinstance(info.get("counts"), dict):
            raise TypeError("marker is not a counts object")
    except (OSError, TypeError, ValueError):  # JSONDecodeError is a ValueError
        info = {"counts": {}, "missing": []}
        flags.append("unreadable_marker")

    res = _read("results", year, rnd, slug, ["Abbreviation", "Position"])
    laps = _read("laps", year, rnd, slug, ["Driver", "LapNumber", "LapTime"])
    row.update(
        results_rows=len(res),
        results_null_position=int(res["Position"].isna().sum()),
        results_dup_drivers=int(res["Abbreviation"].duplicated().sum()),
        laps_rows=len(laps),
        laps_drivers=int(laps["Driver"].nunique()),
        laptime_null_frac=_untimed_fraction(laps),
        weather_rows=info["counts"].get("weather", 0),
        track_status_rows=info["counts"].get("track_status", 0),
        missing_tables=",".join(info.get("missing", [])),
    )

    if not MIN_RESULT_ROWS <= row["results_rows"] <= MAX_RESULT_ROWS:
        flags.append("unusual_result_rows")
    if row["results_dup_drivers"]:
        flags.append("duplicate_drivers")
    if kind == "race" and row["results_rows"]:
        if row["results_null_position"] / row["results_rows"] > MAX_NULL_POSITION_FRAC:
            flags.append("many_null_positions")
        if (res["Position"] == 1).sum() != 1:
            flags.append("no_single_winner")
    # Independent checks: a session can have several lap problems and all are reported.
    if row["laps_rows"] == 0:
        flags.append("no_laps")
    else:
        # NaN means every lap present is lap 1, so there is nothing timed to use.
        if kind == "race" and not row["laptime_null_frac"] <= HARD_NULL_LAPTIME_FRAC:
            flags.append("many_null_laptimes")
        elif kind == "race" and row["laptime_null_frac"] > SOFT_NULL_LAPTIME_FRAC:
            flags.append("some_untimed_laps")
        if row["laps_drivers"] < row["results_rows"] - 2:
            flags.append("laps_missing_drivers")
    if row["weather_rows"] == 0:
        flags.append("no_weather")

    row["flags"] = ",".join(flags)
    row["results_ok"] = not RESULT_FLAGS & set(flags)
    row["laps_ok"] = not LAP_FLAGS & set(flags)
    row["weather_ok"] = "no_weather" not in flags
    return row


def run_audit() -> pd.DataFrame:
    markers = sorted((raw_dir() / "_done").glob("*.json"))
    if not markers:
        raise FileNotFoundError("No ingested sessions found; run `python -m ml.ingest` first.")
    rows = [audit_session(m) for m in markers]
    return pd.DataFrame(rows).sort_values(["Year", "Round", "Session"]).reset_index(drop=True)


def main() -> int:
    report = run_audit()
    out = data_dir() / "reports"
    write_csv_atomic(report, out / "data_quality.csv", index=False)

    flagged = report[report["flags"] != ""]
    informational = flagged["flags"].map(lambda f: set(f.split(",")) <= INFO_FLAGS)
    print(
        f"{len(report)} sessions audited: {int((~informational).sum())} with problems, "
        f"{int(informational.sum())} informational only\n"
    )
    summary = report.groupby("kind").agg(
        sessions=("Year", "size"),
        results_ok=("results_ok", "sum"),
        laps_ok=("laps_ok", "sum"),
        weather_ok=("weather_ok", "sum"),
    )
    print(summary.to_string())

    cols = ["Year", "Round", "Session", "results_rows", "laps_rows", "laptime_null_frac", "flags"]
    core = flagged[flagged["kind"] != "practice"]
    print("\nFlagged races, sprints and qualifying:")
    print(core[cols].to_string(index=False) if len(core) else "  none")
    practice = flagged[flagged["kind"] == "practice"]
    print(
        f"\nFlagged practice sessions: {len(practice)}", practice["flags"].value_counts().to_dict()
    )
    print(f"\nSaved to {out / 'data_quality.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

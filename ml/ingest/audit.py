"""Read-only data-quality audit of ingested sessions.

python -m ml.ingest.audit  ->  data/reports/data_quality.csv plus a printed list of flagged sessions.

Nothing is deleted. Raw files stay as downloaded; the report says which sessions are fit for
results-based features and which for lap-based features, and later phases filter on it.
"""

import json

import pandas as pd

from ml.ingest.paths import data_dir, raw_dir

# Thresholds are deliberately loose; the goal is to surface real gaps, not normal variation.
MIN_RESULT_ROWS, MAX_RESULT_ROWS = 18, 24
# Only races and sprints are expected to have a time for nearly every lap. Qualifying and
# practice legitimately contain many untimed out-, in- and aborted laps, so they are not
# checked for this.
MAX_NULL_LAPTIME_FRAC = 0.05
MAX_NULL_POSITION_FRAC = 0.2

RESULT_FLAGS = {
    "unusual_result_rows",
    "duplicate_drivers",
    "many_null_positions",
    "no_single_winner",
}
LAP_FLAGS = {"no_laps", "many_null_laptimes", "laps_missing_drivers"}


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


def audit_session(marker, year: int, rnd: int, slug: str) -> dict:
    info = json.loads(marker.read_text(encoding="utf-8"))
    kind = session_kind(slug)
    res = _read("results", year, rnd, slug, ["Abbreviation", "Position"])
    laps = _read("laps", year, rnd, slug, ["Driver", "LapTime"])
    row = {
        "Year": year,
        "Round": rnd,
        "Session": slug,
        "kind": kind,
        "results_rows": len(res),
        "results_null_position": int(res["Position"].isna().sum()),
        "results_dup_drivers": int(res["Abbreviation"].duplicated().sum()),
        "laps_rows": len(laps),
        "laps_drivers": int(laps["Driver"].nunique()) if len(laps) else 0,
        "laptime_null_frac": float(laps["LapTime"].isna().mean()) if len(laps) else 1.0,
        "weather_rows": info["counts"].get("weather", 0),
        "track_status_rows": info["counts"].get("track_status", 0),
        "missing_tables": ",".join(info.get("missing", [])),
    }

    flags = []
    if not MIN_RESULT_ROWS <= row["results_rows"] <= MAX_RESULT_ROWS:
        flags.append("unusual_result_rows")
    if row["results_dup_drivers"]:
        flags.append("duplicate_drivers")
    if kind == "race" and row["results_rows"]:
        if row["results_null_position"] / row["results_rows"] > MAX_NULL_POSITION_FRAC:
            flags.append("many_null_positions")
        if (res["Position"] == 1).sum() != 1:
            flags.append("no_single_winner")
    if row["laps_rows"] == 0:
        flags.append("no_laps")
    elif kind == "race" and row["laptime_null_frac"] > MAX_NULL_LAPTIME_FRAC:
        flags.append("many_null_laptimes")
    elif row["laps_drivers"] < row["results_rows"] - 2:
        flags.append("laps_missing_drivers")
    if row["weather_rows"] == 0:
        flags.append("no_weather")

    row["flags"] = ",".join(flags)
    row["results_ok"] = bool(row["results_rows"]) and not RESULT_FLAGS & set(flags)
    row["laps_ok"] = not LAP_FLAGS & set(flags)
    return row


def run_audit() -> pd.DataFrame:
    rows = []
    for marker in sorted((raw_dir() / "_done").glob("*.json")):
        year, rnd, slug = marker.stem.split("_", 2)
        rows.append(audit_session(marker, int(year), int(rnd), slug))
    return pd.DataFrame(rows).sort_values(["Year", "Round", "Session"]).reset_index(drop=True)


def main() -> int:
    report = run_audit()
    out = data_dir() / "reports"
    out.mkdir(exist_ok=True)
    report.to_csv(out / "data_quality.csv", index=False)

    flagged = report[report["flags"] != ""]
    print(f"{len(report)} sessions audited, {len(flagged)} flagged\n")
    summary = report.groupby("kind").agg(
        sessions=("Year", "size"), results_ok=("results_ok", "sum"), laps_ok=("laps_ok", "sum")
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

"""Same-weekend information: what earlier sessions of a race weekend say about qualifying.

A prediction made after practice can use practice. Which sessions have happened by the time
qualifying starts depends on the weekend's format, and the format changed over the years:

- usual weekend:           Practice 1, 2, 3 > Qualifying > Race
- 2021 and 2022 sprints:   Practice 1 > Qualifying > Practice 2 > Sprint > Race
                           (Practice 2 is AFTER qualifying, so it must not be used)
- 2023 sprints:            Practice 1 > Qualifying > Sprint Shootout > Sprint > Race
- 2024 onward sprints:     Practice 1 > Sprint Qualifying > Sprint > Qualifying > Race
                           (Sprint Qualifying and the Sprint are BEFORE qualifying, so they can be used)

So the rule is never by session name: a session counts only if its scheduled START is before the
start of the target session. `earlier_sessions` implements exactly that.

Features (all "pq_", for "before qualifying") describe each driver:
- pq_fp_best_gap: best practice lap, as % slower than the fastest lap of that session, best across
  the earlier practice sessions
- pq_fp_last_gap: the same for the latest earlier practice session only
- pq_fp_n:        how many earlier practice sessions had lap data (the same for every driver)
- pq_sprintq_pos / pq_sprintq_gap: Sprint Qualifying (or Shootout) gap to the fastest lap and the
  order of best laps, if it came before the target session (from laps: fastf1 has no results there)
- pq_sprint_pos:  Sprint result position, if it came before the target session
A missing session (not run, not yet ingested) gives NaN, never an error.
"""

import re

import numpy as np
import pandas as pd

from ml.ingest.paths import raw_dir, slugify

PQ_COLUMNS = (
    "pq_fp_best_gap",
    "pq_fp_last_gap",
    "pq_fp_n",
    "pq_sprintq_pos",
    "pq_sprintq_gap",
    "pq_sprint_pos",
)
_PRACTICE = re.compile(r"^Practice \d$")


def session_kind(name: str) -> str:
    """'practice', 'sprint_qualifying', 'sprint', 'qualifying' or 'race'."""
    if _PRACTICE.match(name):
        return "practice"
    if name in ("Sprint Qualifying", "Sprint Shootout"):
        return "sprint_qualifying"
    return {"Sprint": "sprint", "Qualifying": "qualifying", "Race": "race"}.get(name, "other")


def session_schedule(event: pd.Series) -> list[tuple[str, pd.Timestamp]]:
    """(name, scheduled start) of every session of a weekend, in the order the calendar lists them."""
    out = []
    for i in range(1, 6):
        name, start = event.get(f"Session{i}"), event.get(f"Session{i}DateUtc")
        if name and not pd.isna(start):
            out.append((str(name), pd.Timestamp(start)))
    return out


def earlier_sessions(event: pd.Series, target: str = "Qualifying") -> list[str]:
    """Sessions of this weekend that START strictly before `target` starts, earliest first.

    Returns [] if the target is not on the schedule. Ordering is by scheduled start time, never by
    name or by the position in the calendar list.
    """
    schedule = dict(session_schedule(event))
    if target not in schedule:
        return []
    cutoff = schedule[target]
    earlier = [(start, name) for name, start in schedule.items() if start < cutoff]
    return [name for _, name in sorted(earlier)]


def best_lap_gap_pct(laps: pd.DataFrame) -> pd.Series:
    """Each driver's best lap as % slower than the session's fastest lap (NaN if no valid lap).

    Practice laps carry no reliable 'accurate' flag, so any timed, not-deleted lap counts; the gap
    is within one session, so track evolution and weather are shared by everyone in it.
    """
    if laps.empty:
        return pd.Series(dtype=float)
    deleted = laps["Deleted"].astype(str).eq("True") if "Deleted" in laps else False
    valid = laps[laps["LapTime"].notna() & ~deleted]
    if valid.empty:
        return pd.Series(dtype=float)
    best = valid.groupby("Driver")["LapTime"].min().dt.total_seconds()
    return (best / best.min() - 1) * 100


def _read(table: str, year: int, rnd: int, name: str, columns: list[str]) -> pd.DataFrame:
    path = raw_dir() / table / f"year={year}" / f"round={rnd:02d}" / f"{slugify(name)}.parquet"
    return (
        pd.read_parquet(path, columns=columns) if path.exists() else pd.DataFrame(columns=columns)
    )


def weekend_features(
    year: int, rnd: int, event: pd.Series, target: str = "Qualifying"
) -> pd.DataFrame:
    """The pq_ features for every driver seen in the weekend's earlier sessions (index: Abbreviation)."""
    sessions = earlier_sessions(event, target)
    practice_gaps: list[pd.Series] = []
    out: dict[str, pd.Series] = {}
    for name in sessions:
        kind = session_kind(name)
        if kind == "practice":
            gap = best_lap_gap_pct(_read("laps", year, rnd, name, ["Driver", "LapTime", "Deleted"]))
            if not gap.empty:
                practice_gaps.append(gap)
        elif kind == "sprint_qualifying":
            # fastf1's result table for these sessions is empty (no positions, no times), so the
            # pace comes from the laps, as for practice. The order is therefore the order of each
            # driver's best lap in the whole session, not the official Q1/Q2/Q3 classification.
            gap = best_lap_gap_pct(_read("laps", year, rnd, name, ["Driver", "LapTime", "Deleted"]))
            if not gap.empty:
                out["pq_sprintq_gap"] = gap
                out["pq_sprintq_pos"] = gap.rank(method="min")
        elif kind == "sprint":
            res = _read("results", year, rnd, name, ["Abbreviation", "Position"])
            if not res.empty:
                out["pq_sprint_pos"] = res.set_index("Abbreviation")["Position"]
    if practice_gaps:
        stacked = pd.concat(practice_gaps, axis=1)
        out["pq_fp_best_gap"] = stacked.min(axis=1)
        out["pq_fp_last_gap"] = practice_gaps[-1]
    frame = pd.DataFrame(out)
    frame.index.name = "Abbreviation"
    frame["pq_fp_n"] = float(len(practice_gaps))
    return frame.reindex(columns=list(PQ_COLUMNS))


def build_weekend_table(events: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """pq_ features for every driver of every race in `races` (Year, Round, Abbreviation).

    Drivers or weekends with no earlier-session data get NaN (and pq_fp_n of 0 when the weekend had
    no practice data at all).
    """
    parts = []
    keyed = events.set_index(["Year", "RoundNumber"])
    for (year, rnd), drivers in races.groupby(["Year", "Round"]):
        event = keyed.loc[(year, rnd)]
        features = weekend_features(year, rnd, event)
        table = features.reindex(drivers["Abbreviation"].to_numpy()).reset_index()
        table["pq_fp_n"] = float(features["pq_fp_n"].iloc[0]) if len(features) else 0.0
        table.insert(0, "Round", rnd)
        table.insert(0, "Year", year)
        parts.append(table)
    out = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["Year", "Round", "Abbreviation", *PQ_COLUMNS])
    )
    return out.replace({np.inf: np.nan, -np.inf: np.nan})

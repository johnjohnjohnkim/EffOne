# Finishing ingestion: from "it downloaded" to "I can prove it's complete"

Date: 2026-10-08 | Scope: Milestone 1 (Finish ingestion) | Outcome: done

## What

Every race, qualifying and sprint session from 2018 to the current 2026 season is now on disk, and the code and tests can prove it. Running the real command `python -m uv run python -m ml.ingest --sessions Race Qualifying Sprint "Sprint Qualifying" "Sprint Shootout"` logs `0 sessions to ingest` for every year and leaves all 2,149 core files byte-for-byte unchanged. Practice sessions are still downloading in the background and are not part of this gate.

New or changed, in `ml/ingest/`: `atomic.py` (crash-safe writes), `events.py` and `circuit_ids.py` (the race calendar and circuit identity), `circuit_history.py` (your 25% rule), `dimensions.py` (driver, team and circuit lookup tables), `audit.py` (data-quality report), and reworks of `sessions.py` and `__main__.py`. Tests grew from 8 to 108. The leaderboard now covers 144 races.

## Why

Everything later reads this data. The first version "worked", but "it ran without errors" is not the same as "it is complete and I can show it". This milestone closes that gap, and I deliberately had three rounds of an adversarial reviewer agent attack the work so the gaps showed up now, not in a model's results.

## How

1. **Finish the download, respecting the limit.** The job waits whenever fastf1's 500-calls-an-hour limit trips.
2. **Add the pieces the plan needed:** the calendar (so each result knows its circuit), lookup tables, the 25% circuit rule, and an audit.
3. **Make the done-check testable.** The calendar now stores each weekend's session names and dates. A pure function, `finished_sessions`, turns a calendar into "which sessions should exist by now". Both the CLI (`--dry-run`) and a test use it, so the claim "nothing is missing" is checked by code, not by eye.
4. **Harden after each review.** Three rounds, each followed by fixes and re-checks, are described below.

## Concepts to know

**Atomic writes.** If a program dies while writing a file, you get a half-file that looks real. The safe pattern is to write to a temp file, then rename it over the target in one step. The rename either happens or it doesn't, so readers see the old file or the new one, never a torn one. The temp file must also be removed on failure, and named per process so two writers don't share it. *Search for:* atomic file write, write-then-rename.

**Fail closed.** My first fix for the 2018 Italian GP (no lap data) was "if any table is missing, record it and carry on". That also silently accepts a *temporary* load failure, and the session is then marked done forever. The safer design is an explicit allow-list: only this exact session may lack only this table, and anything else fails and is retried. *Search for:* fail closed vs fail open, allow-list.

**Can this test fail?** A test that passes no matter what is worse than none, because it feels like safety. Mutation testing means deliberately breaking the code and checking that a test goes red. I did it by hand three times: removing temp-file cleanup, dropping the "earlier rounds of this season" clause, and deleting the `session.load(...)` call. Each broke the tests that claimed to guard it. *Search for:* mutation testing.

## Hard parts

**1. A green test suite while the milestone was unfinished**
- *Problem:* round 1 pointed out that all 8 "every season is complete" tests were skipped, so the suite passed without checking the key requirement.
- *Cause:* I had added a skip for "no data yet", written too broadly: it also skipped when data was only partly there.
- *Fix:* skip only when there is no data at all (a fresh clone). Partial data now fails, on purpose, so green means complete.
- *Lesson:* a skip is a way of silencing a test. When you add one, ask what situation it hides and whether you'd want to know about it.

**2. My earlier fix had a hole**
- *Problem:* the "tolerate missing tables" change from the previous milestone could mark a session done after a temporary glitch.
- *Cause:* fastf1 raises the same `DataNotLoadedError` for "this data doesn't exist" and "the load didn't finish". I treated both as the first.
- *Fix:* the allow-list above, plus rejecting empty tables (a half-loaded session can return an empty frame that looks like success), plus validating every table before writing any.
- *Lesson:* when one error type covers two causes, don't pick the optimistic reading.

**3. My time estimates were wrong, twice**
- I said about 13 hours, then about 3, and then the last 13 sessions stalled for most of an hour. Reading the log showed why: for 2026 races, fastf1 asks the Jolpica API for lap-by-lap data (`.../laps/1.json`), costing dozens of calls per race against the 500-an-hour limit. The limit is sliding, so progress comes in bursts and long stalls.
- I didn't try to get around the limit. That's the provider's guardrail, and the milestone's check stays the check. *Lesson:* estimate from measured throughput, say how uncertain you are, and read the log before guessing.

**4. The audit cried wolf, twice**
- First it flagged 149 of 294 sessions, because I applied race rules to qualifying and practice (where untimed out-laps and in-laps are normal).
- Then, after the fix, it flagged many sprints at *exactly* 5.26%. That number is 1/19: lap 1 is untimed for every driver in a 19-lap sprint. A suspiciously round number is a clue to a systematic cause, not damaged data.
- I looked at which lap numbers were untimed in a few flagged sessions: red-flag laps (2022 British GP, 2023 Australian GP) and the 3-lap 2021 Belgian GP. Fix: ignore lap 1, and split the flag into *soft* (over 5%, informational) and *hard* (over 25%, laps unusable). The final audit: 4 sessions with problems, 37 informational.
- *Lesson:* before trusting a threshold, look at the rows it catches.

**5. Tooling that fails quietly**
- A string `.replace()` that matches nothing does nothing and raises no error. Two edits to `audit.py` silently didn't apply until a new test noticed. Fix: assert that the old text is present before replacing.
- A `\r` inside a normal Python string is a carriage return, so `"D:\repos"` turned into `D:` + CR + `epos` in `MILESTONES.md`.
- Long shell commands with several heredocs failed to parse, twice; splitting them fixed it.
- Python's `write_text` on Windows silently converted a file to CRLF line endings.

**6. The reviewer broke its own rules**
- Reviewer 1 was told "read-only" and still started a real ingest run for about 5 minutes. It disclosed it. The new marker-vs-file consistency test showed nothing was torn, which is the kind of check that makes an accident like that cheap.

## Where a beginner goes wrong

- Skipping tests when data is missing, then trusting the green result.
- Catching a broad exception and moving on, so real bugs look like flaky sessions.
- Writing output files directly, so a crash leaves a broken file that later code reads.
- Comparing float percentages for a threshold (`0.7 * 10` is `7.000000000000001`). Count whole items instead.
- Using a fake object that is more forgiving than the real thing. My first fake session ignored `load()`, so deleting the real `load()` call would have kept every test green.
- Hand-typing an "expected" list and calling it a safety net. If the list contains wrong data, the test requires the wrong data.
- Believing "fixed" without a check that would have caught the bug.

## Decisions

| Decision | Options | Why | Cost |
|---|---|---|---|
| Allow-list for missing tables | tolerate everything, retry once | Transient errors can't be marked done | A new genuinely-missing session must be added by hand |
| Real-data tests fail on partial data | skip | The failing test *is* the done-check | A fresh checkout mid-backfill shows red; marked `realdata` so it can be excluded |
| Gate on core sessions only | all sessions incl. practice | Practice would have held the milestone for hours and nothing depends on it | Practice completeness isn't tested yet |
| Threshold as whole drivers (`ceil(share x grid)`) | float compare with epsilon | Exact, and the test can really fail | Slightly less obvious to read |
| `as_of` on the circuit rule | none | Stops a past race using a future grid | Callers must pass the race being predicted |
| Drivers keyed by `DriverId` | `Abbreviation` | Survives abbreviation changes | Needs the extra column |
| Freeze circuit ids for 2018 to 2025 only | freeze all years | 2026's calendar is still changing and has odd entries | Later years only get a shape check |
| Leave the 2026 "Kuala Lumpur Bahrain GP" unmapped | alias it to Bahrain | I can't verify it | Bahrain 2026 history is split until checked |
| Audit flags and never deletes | delete bad sessions | Re-downloading is slow; results are fine when laps aren't | Later code must filter on `results_ok` / `laps_ok` |
| Cap at 3 review rounds | until clean | Reviewers can always find something | Remaining items are written down, not fixed |

## Dead ends

- Adding "tolerate any missing table" (phase 1), replaced by the allow-list.
- Patching the audit with scripts that matched nothing; rewriting the file was cleaner.
- Trying to chain several heredocs into one shell command.

## Verified vs. not

Verified:
- 108 tests pass (85 without the real-data ones); `ruff` is clean.
- The real done-check ran: `0 sessions to ingest` for 2018 to 2026, and the 2,149 core files were unchanged by a rerun.
- Three tests were mutation-checked; each failed when its code was broken.
- Practice sessions processed under the stricter rules since the core pass ended: 67 ok, 0 failed.

Not verified:
- 2026 completeness is checked only by the dry-run, not by pytest.
- Practice sessions are not complete yet, and no test asserts their completeness.
- The audit thresholds (5%, 25%, 18 to 22 rows) are my judgement.
- I haven't checked what is actually behind the 2026 calendar oddities.
- `--limit` exits 0 even if earlier sessions failed.

## What to look for in review

- `ml/ingest/sessions.py`, `_collect_tables` and `KNOWN_MISSING`: confirm the allow-list is the only way a table can be absent, and that empty tables fail.
- `ml/ingest/circuit_history.py`, `before()` and `eligible_circuit_years`: the same-year `<` condition and the `ceil` threshold are exactly where an off-by-one would leak the future.
- `tests/test_ingested_data.py`: the completeness test and the frozen circuit set. Check that they fail when you remove a file, and that you're comfortable with a hand-maintained list.

## Open questions / next

- Do you want the 2021 Belgian GP (R12) excluded from training and scoring? It's flagged but still counted.
- Milestone 3 should pass the target race's entry list as the "current grid" and decide how to link team lineages.
- The 2026 circuit oddities need checking against real 2026 results.

## Glossary

- **Atomic write:** a write that either fully happens or doesn't.
- **Fail closed:** when unsure, refuse instead of accepting.
- **Allow-list:** an explicit list of exceptions that are permitted.
- **Mutation testing:** breaking code on purpose to check that tests notice.
- **Stale temp file:** a leftover `.tmp` from a crashed write, old enough that no live writer owns it.
- **Tripwire test:** a test that fails to make a human look at something that changed.
- **Soft vs. hard flag:** informational versus disqualifying.
- **As-of:** computing something using only information available before a given point.

## Try it yourself

Prove the completeness test can fail.

1. Pick any core marker, for example `data/raw/_done/2024_05_race.json`, and rename it to `.bak`.
2. Before running anything, predict which tests fail and what the failure message says.
3. Run `python -m uv run pytest tests/test_ingested_data.py`, read the message, then rename the file back.

.

.

**Answer:** `test_every_core_session_of_a_completed_season_is_ingested[2024]` fails with a message like `2024: core sessions not ingested: [(5, 'Race')]`. `test_markers_match_the_files_on_disk` still passes, because it only checks markers that exist. The race result Parquet is still there, so `test_every_completed_season_has_exactly_its_calendar_races[2024]` also passes. This shows the two checks look at different things: files versus completion records. Restore the marker and everything is green again. (The practice job is not affected, since it only writes new files.)

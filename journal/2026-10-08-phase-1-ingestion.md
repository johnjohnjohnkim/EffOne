# Phase 1: pulling F1 data into Parquet, and what slowed it down

Date: 2026-10-08 | Scope: Phase 1 (data ingestion) | Outcome: partial

## What

An ingestion command now exists: `python -m uv run python -m ml.ingest`. It finds every finished F1 session from 2018 onward (practice, qualifying, sprint, race), loads it with fastf1, and saves four tables per session as Parquet files under `data/raw/`: `results`, `laps`, `weather`, `track_status`.

At the time of writing the backfill is still running (5 matching processes were alive). Sessions finished so far: 2018: 80, 2019: 42, 2020: 34, 2021: 47, 2022: 47, 2023: 43, about 0.9 GB on disk. 2024 and 2025 have not started. Key files: `ml/ingest/sessions.py`, `ml/ingest/__main__.py`, `ml/ingest/paths.py`.

Phase 1 is not finished. Still to do: check row counts against known race and driver numbers, write real tests, add the `circuits`/`drivers`/`teams` tables, and implement the 25% circuit-history rule.

## Why

Every later phase (baselines, features, models, simulation) reads from these files. Pulling the data once into a local, well-organised store means later work never waits on the network, and the scheduled job can later add only what's new.

## How

1. **Paths** (`paths.py`): one function decides where data lives (`<repo>/data`, or `EFFONE_DATA_DIR`). The fastf1 cache goes there too, because its default is on C:, which had about 8 GB free.
2. **Finding work** (`pending_sessions`): ask fastf1 for the season schedule, list each event's sessions, drop any that finished less than 6 hours ago (fastf1 can lag), and drop any that already have a "done" marker.
3. **Loading one session** (`ingest_session`): call `session.load(telemetry=False, ...)`. Telemetry is skipped because it is huge and the plan doesn't need it. Write each table to `data/raw/<table>/year=YYYY/round=RR/<session>.parquet`, then write a small JSON marker to `data/raw/_done/`. The marker is written last, so a session that crashes halfway is not marked finished.
4. **The loop** (`__main__.py`): for each year, for each pending session, ingest it. A failure is logged and the loop moves on. If fastf1 raises `RateLimitExceededError`, `with_rate_limit_wait` sleeps 5 minutes and retries the same call.
5. **Order:** the second launch runs races, qualifying and sprints first, then practice, so the data the models depend on arrives first.

## Concepts to know

**Idempotency.** An operation is idempotent if running it twice gives the same result as running it once. Here, the done-marker means a rerun skips finished work, so a crash or a closed laptop costs nothing but time. Any job that runs on a schedule needs this. *Search for:* idempotent jobs, checkpointing.

**Rate limiting and backoff.** Many APIs cap calls per window; fastf1 enforces 500 per hour itself. When you hit a cap, retrying immediately makes it worse. You wait, ideally for about as long as the window needs to reset. *Search for:* exponential backoff, rate limit.

**Partitioned Parquet.** Parquet is a compressed, column-based file format. Putting `year=2019/round=05` in the folder names lets tools read only the partitions they need. *Search for:* Hive-style partitioning, Parquet.

## Hard parts

**1. The rate limit I did not plan for**
- *Problem:* the first full run ingested 65 sessions in about 5 minutes, then logged dozens of failures and crashed fetching the 2019 schedule.
- *Cause:* fastf1 allows 500 API calls an hour, at roughly 8 calls per session. My loop treated the error like any other failure and immediately tried the next session, burning through the whole list.
- *Fix:* catch `RateLimitExceededError` specifically and sleep and retry. Other errors still just log and continue. The log now shows 44 rate-limit sleeps, so the wait is doing its job.
- *Lesson:* read the library's limits before writing a loop that calls it thousands of times. A catch-all `except` hid a problem that needed a different response.
- *Consequence:* about 60 sessions an hour means the full backfill takes many hours. This is slow by design, not a bug.

**2. A command ran in the wrong folder**
- *Problem:* `uv init` created `pyproject.toml`, `.python-version` and `src/effone/` in `D:\repos` instead of `D:\repos\EffOne`.
- *Cause:* the shell's working directory had reset between tool calls, and I assumed it hadn't.
- *Fix:* I checked the timestamps (3:13:53, matching my command) before deleting, removed exactly those three, and left alone the `EffOne` setup that already existed from a minute earlier (made by something other than me).
- *Lesson:* put `Set-Location` in the same command as anything that creates files, and check timestamps before deleting something you didn't write.

**3. A race that will not load (unresolved)**
- 2018 round 14 Race fails every run with fastf1's "data … has not been loaded yet" message (the log line is cut off, so I haven't seen the full text). It is the only repeat failure (2 in the current log). I don't know the cause. It could be missing timing data for that session. It needs investigation, not a retry loop.

## Where a beginner goes wrong

- Writing the "done" marker before the data, or at the start. A crash then leaves a session marked finished with no data.
- Treating every exception the same way. Rate limits need waiting, bad data needs logging, bugs need stopping.
- Leaving default cache folders alone. Libraries often write gigabytes under `AppData` on the system drive.
- Declaring success when the command exits. A job that exited 0 can still have written empty tables. Check row counts.
- Running a multi-hour job with no way to resume it.

## Decisions

| Decision | Options | Why | Cost |
|---|---|---|---|
| Skip telemetry | include it | Plan doesn't need car-data streams; much smaller and faster | Can't build telemetry features without re-ingesting |
| Parquet files, no database | SQLite, Postgres | Simple, fast to read in pandas, maps to S3 later | No ad-hoc SQL until we add DuckDB or similar |
| Marker file per session | one manifest file | Each marker is written independently, so no shared file to corrupt | Many tiny files |
| Wait 6 h after a session ends | load immediately | fastf1 can return incomplete data right after a session | A fresh race isn't available for several hours. The 6 h is a guess, untested. |
| Core sessions before practice | chronological | The race and qualifying data matter most for the models | Practice data for early years arrives late |
| Sleep a fixed 5 min on rate limit | wait for the exact reset | Simple | Probably wastes some minutes per hour |

## Dead ends

Restarting the same loop and hoping the limit had reset. The first rerun just failed faster. The real fix was the wait logic.

## Verified vs. not

Verified:
- One 2018 race ingested end to end (20 results, 940 laps, 111 weather rows, 14 track-status rows) with files on D:.
- The pytest smoke test and `ruff` passed during phase 0.
- The done-marker count and the log both show the backfill progressing past the rate limit.

Not verified:
- Row counts against known numbers, the lap and results tables beyond that one race, and 2018 R14.
- There are no tests for the ingestion code, so the idempotency claim rests on one manual rerun pattern, not a test.
- The 6-hour wait and the 5-minute sleep were not tuned.

## What to look for in review

- `ml/ingest/sessions.py`, `_write`: it converts mixed-type object columns to strings so pyarrow doesn't fail. Check this doesn't turn real numbers or timestamps into text that later phases must parse back.
- `ml/ingest/sessions.py`, `ingest_session`: the marker is written only after all four tables are saved and results are non-empty. Check the ordering and that empty `laps` or `weather` tables are acceptable.
- `ml/ingest/__main__.py`, `with_rate_limit_wait`: it loops forever on the rate-limit error. Check that nothing else can raise it forever, and think about whether a maximum wait is needed.

## Open questions / next

- Why does 2018 R14 Race fail?
- Are the `Session` names consistent across years (sprint sessions were renamed over time)?
- Finish phase 1 properly (counts, tests, remaining tables, 25% rule) before phase 2 leans on this data.

## Glossary

- **fastf1:** Python library that downloads F1 timing and results data.
- **Parquet:** compressed, column-oriented file format for tables.
- **Partition:** a folder level (e.g. `year=2019`) that splits a dataset.
- **Idempotent:** safe to run repeatedly with the same outcome.
- **Backoff:** waiting before retrying after an error.
- **Telemetry:** per-car sensor data (speed, throttle, etc.), very large.

## Try it yourself

Delete one done-marker and rerun a single session, to see how the skip logic works.

1. Pick any file in `data/raw/_done/`, for example one from 2018, and delete it. (Wait for the background job to finish first, or pause it, so two runs don't write at once.)
2. Before running anything, predict: will the rerun touch that session only, or others too? Will it call the network?
3. From `D:\repos\EffOne`, run `python -m uv run python -m ml.ingest --years 2018 --limit 1` and read the log.

.

.

**Answer:** it re-ingests only the session whose marker you deleted, since every other session still has its marker, then stops at `--limit 1`. It should load mostly from the local fastf1 cache (look for "cached data" lines in the log), so it is much faster than the first time, and it recreates the marker. The Parquet file is overwritten with identical content. If it picks a different session than the one you deleted, check that 2018 R14 Race hasn't failed again, since that one is always pending.

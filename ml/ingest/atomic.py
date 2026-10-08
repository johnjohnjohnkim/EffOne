"""Crash-safe file writes: write to a temp file, then atomically rename over the target.

A failure at any point leaves the previous file (if any) intact and no temp file behind. Temp
names include the process id, so two processes writing the same target do not share a temp file.
"""

import contextlib
import json
import os
import time
from pathlib import Path

import pandas as pd

TMP_SUFFIX = ".tmp"
# A temp file this old cannot belong to a write in progress; it is left over from a crash.
STALE_TMP_SECONDS = 3600


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}{TMP_SUFFIX}")


def _replace(tmp: Path, path: Path, retries: int = 5) -> None:
    """os.replace, retrying briefly: on Windows it fails if a reader has the target open."""
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.2)


def _cleanup(tmp: Path) -> None:
    # Never let a failed cleanup hide the error that caused it.
    with contextlib.suppress(OSError):
        tmp.unlink(missing_ok=True)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        df.to_parquet(tmp, index=False)
        _replace(tmp, path)
    finally:
        _cleanup(tmp)


def write_csv_atomic(df: pd.DataFrame, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        df.to_csv(tmp, **kwargs)
        _replace(tmp, path)
    finally:
        _cleanup(tmp)


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        _replace(tmp, path)
    finally:
        _cleanup(tmp)


def stale_tmp_files(root: Path, max_age_seconds: float = STALE_TMP_SECONDS) -> list[Path]:
    """Temp files under `root` old enough to be crash leftovers (not a write in progress)."""
    cutoff = time.time() - max_age_seconds
    stale = []
    for path in root.rglob(f"*{TMP_SUFFIX}"):
        try:
            if path.stat().st_mtime < cutoff:
                stale.append(path)
        except FileNotFoundError:  # a live writer renamed or removed it while we were looking
            continue
    return stale


def sweep_stale_tmp(root: Path) -> int:
    stale = stale_tmp_files(root)
    for path in stale:
        path.unlink(missing_ok=True)
    return len(stale)

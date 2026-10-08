"""Where ingested data lives. Defaults to <repo>/data (on D:); override with EFFONE_DATA_DIR."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    path = Path(os.environ.get("EFFONE_DATA_DIR", REPO_ROOT / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = data_dir() / "fastf1_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    path = data_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path

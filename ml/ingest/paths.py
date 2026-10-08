"""Where ingested data lives. Defaults to <repo>/data (on D:); override with EFFONE_DATA_DIR.

Reading never creates directories; writers call `mkdir(parents=True, exist_ok=True)` themselves,
except `cache_dir()`, which fastf1 requires to exist.
"""

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("EFFONE_DATA_DIR", REPO_ROOT / "data"))


def cache_dir() -> Path:
    path = data_dir() / "fastf1_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    return data_dir() / "raw"


def slugify(name: str) -> str:
    """Lowercase ascii-ish slug: 'Sprint Qualifying' -> 'sprint_qualifying'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

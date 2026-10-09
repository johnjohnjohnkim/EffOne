"""Load the tuned model settings.

`ml/models/tuned_params.json` is written by `python -m ml.tuning` and holds, per target and model,
the settings that scored best on the DEVELOPMENT seasons only, plus a provenance block. Every trial
behind it is in `ml/models/tuning_log.csv`. With no file, every model uses its factory defaults.
"""

import json
from collections.abc import Mapping
from pathlib import Path

PARAMS_PATH = Path(__file__).with_name("tuned_params.json")
LOG_PATH = Path(__file__).with_name("tuning_log.csv")


def load_tuned(path: Path = PARAMS_PATH) -> dict:
    """The tuned settings, or an empty dict when none have been written yet."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def params_for(tuned: Mapping, target: str, model: str) -> dict:
    """Settings for one model on one target ("race" or "quali"); empty means use the defaults."""
    found = tuned.get(target, {}).get(model)
    if found is None and target == "quali_after_practice":
        found = tuned.get("quali", {}).get(model)  # not tuned separately yet: use qualifying's
    return dict(found or {})

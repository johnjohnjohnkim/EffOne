"""`python -m ml.features`: writes the feature table and a per-column missing-value count."""

import pandas as pd
import pytest

from ml.features import __main__ as cli
from ml.features.build import TARGET_STAGES, TARGETS, feature_columns
from ml.features.builders import KEY_COLUMNS
from tests.helpers import make_rows


@pytest.fixture
def run_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EFFONE_DATA_DIR", str(tmp_path))
    rows = make_rows()
    monkeypatch.setattr(cli, "build_history", lambda: rows)
    assert cli.main() == 0
    return tmp_path / "features", rows, capsys.readouterr().out


def test_it_writes_one_row_per_driver_per_race(run_cli):
    out, rows, _ = run_cli
    table = pd.read_parquet(out / "race_features.parquet")
    assert len(table) == len(rows)
    assert not table.duplicated(["Year", "Round", "DriverId"]).any()


def test_it_writes_a_missing_value_count_for_every_column(run_cli):
    out, _, _ = run_cli
    table = pd.read_parquet(out / "race_features.parquet")
    missing = pd.read_csv(out / "feature_missing.csv", index_col="column")
    assert list(missing.index) == list(table.columns)  # every column, in order
    assert missing["missing"].tolist() == table.isna().sum().tolist()
    assert missing.loc["drv_finish_l5", "missing"] > 0  # first-race drivers have no history
    assert missing.loc["grid", "missing"] == 0


def test_roles_distinguish_keys_stages_flag_and_targets(run_cli):
    out, _, _ = run_cli
    roles = pd.read_csv(out / "feature_missing.csv", index_col="column")["role"]
    assert set(roles[KEY_COLUMNS]) == {"key"}
    assert roles["race_ok"] == "flag"
    assert set(roles[list(TARGETS)]) == {"target"}
    assert roles["grid"] == "post_quali" and roles["wx_air_temp"] == "scenario"
    assert roles["drv_finish_l5"] == "pre_weekend"
    assert set(roles) == {
        "key",
        "flag",
        "target",
        "pre_weekend",
        "recency",
        "post_practice",
        "scenario",
        "post_quali",
    }


def test_the_printout_names_the_usable_features_per_target(run_cli):
    _, _, printed = run_cli
    for target in TARGETS:
        count = len(feature_columns(target))
        assert f"{target}: {count} features from stages {sorted(TARGET_STAGES[target])}" in printed
    assert "rows (" in printed and "race_ok" in printed

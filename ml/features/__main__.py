"""CLI: python -m ml.features -> data/features/race_features.parquet and a missing-value summary."""

import pandas as pd

from ml.features.build import TARGET_STAGES, TARGETS, build_feature_table, feature_columns
from ml.features.builders import BUILDERS, KEY_COLUMNS
from ml.features.tables import build_history
from ml.ingest.atomic import write_csv_atomic, write_parquet_atomic
from ml.ingest.paths import data_dir


def main() -> int:
    history = build_history()
    table = build_feature_table(history)

    out = data_dir() / "features"
    write_parquet_atomic(table, out / "race_features.parquet")

    stage = {c: b.stage for b in BUILDERS for c in b.columns}

    def role(column: str) -> str:
        if column in stage:
            return stage[column]
        return "target" if column in TARGETS else ("flag" if column == "race_ok" else "key")

    missing = pd.DataFrame(
        {
            "role": [role(c) for c in table.columns],
            "missing": table.isna().sum().to_numpy(),
            "missing_pct": (table.isna().mean() * 100).round(1).to_numpy(),
        },
        index=table.columns,
    )
    write_csv_atomic(missing, out / "feature_missing.csv", index_label="column")

    races = table[["Year", "Round"]].drop_duplicates().shape[0]
    print(f"{len(table)} rows ({races} races, {table['DriverId'].nunique()} drivers)")
    print(f"keys: {KEY_COLUMNS}  | flag: race_ok (False = race carries no performance info)")
    for target in TARGETS:
        usable = feature_columns(target)
        print(f"{target}: {len(usable)} features from stages {sorted(TARGET_STAGES[target])}")
    print(f"default features for y_finish (no weather): {feature_columns('y_finish')}")
    print("weather is opt-in for finish/DNF only: feature_columns(target, include_scenario=True)\n")
    with pd.option_context("display.max_rows", 100, "display.width", 120):
        print(missing.to_string())
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

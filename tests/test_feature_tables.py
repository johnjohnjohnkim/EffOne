"""The per-session helpers in ml.features.tables, on small synthetic frames."""

import numpy as np
import pandas as pd
import pytest

from ml.features.tables import (
    WEATHER_COLUMNS,
    dnf_flag,
    pace_gap_pct,
    quali_gap_pct,
    weather_summary,
)


def lap_rows(driver, seconds, n=6, **overrides):
    base = {
        "Driver": driver,
        "LapTime": seconds,
        "IsAccurate": True,
        "TrackStatus": "1",
        "PitInTime": None,
        "PitOutTime": None,
        "Deleted": None,
    }
    return [base | overrides for _ in range(n)]


def frame(rows):
    """Lap rows use plain seconds; convert the time columns to real timedeltas, as in the data."""
    df = pd.DataFrame(rows)
    for col in ("LapTime", "PitInTime", "PitOutTime"):
        df[col] = pd.to_timedelta(df[col], unit="s")
    return df


def field(extra=()):
    """12 drivers: D00 fastest at 90.0s, each next one 0.1s slower; plus any extra lap rows."""
    rows = []
    for i in range(12):
        rows += lap_rows(f"D{i:02d}", 90.0 + 0.1 * i)
    return frame([*rows, *extra])


def test_dnf_flag_marks_everything_that_is_not_a_classified_position():
    flags = dnf_flag(pd.Series(["1", "12", "R", "D", "W", None, "N"]))
    assert flags.tolist() == [False, False, True, True, True, True, True]


def test_pace_gap_is_relative_to_the_field_median_in_percent():
    gap = pace_gap_pct(field())
    median = np.median([90.0 + 0.1 * i for i in range(12)])
    assert gap["D00"] == pytest.approx((90.0 / median - 1) * 100)
    assert gap["D00"] < 0 < gap["D11"]
    assert gap.median() == pytest.approx(0.0)


def test_pace_ignores_pit_laps_inaccurate_laps_deleted_laps_and_non_green_laps():
    junk = (
        lap_rows("D00", 200.0, n=20, PitInTime=1.0)
        + lap_rows("D00", 200.0, n=20, PitOutTime=1.0)
        + lap_rows("D00", 200.0, n=20, IsAccurate=False)
        + lap_rows("D00", 200.0, n=20, Deleted="True")
        + lap_rows("D00", 200.0, n=20, TrackStatus="4")
    )
    clean, dirty = pace_gap_pct(field()), pace_gap_pct(field(junk))
    assert dirty["D00"] == pytest.approx(clean["D00"])  # 100 junk laps changed nothing


def test_pace_needs_enough_laps_per_driver_and_enough_drivers():
    assert pace_gap_pct(field()).notna().all() and len(pace_gap_pct(field())) == 12
    few_laps = frame([r for i in range(12) for r in lap_rows(f"D{i:02d}", 90.0, n=3)])
    assert pace_gap_pct(few_laps).empty  # under 5 clean laps each
    few_drivers = frame([r for i in range(5) for r in lap_rows(f"D{i:02d}", 90.0)])
    assert pace_gap_pct(few_drivers).empty  # under 10 drivers to compare
    assert pace_gap_pct(pd.DataFrame()).empty


def test_quali_gap_uses_each_drivers_best_time_relative_to_pole():
    q = pd.DataFrame(
        {
            "Q1": pd.to_timedelta([91.0, 92.0, 93.0, None], unit="s"),
            "Q2": pd.to_timedelta([90.5, 91.5, None, None], unit="s"),
            "Q3": pd.to_timedelta([90.0, None, None, None], unit="s"),
        }
    )
    gap = quali_gap_pct(q)
    assert gap.tolist()[:3] == pytest.approx([0.0, (91.5 / 90 - 1) * 100, (93 / 90 - 1) * 100])
    assert np.isnan(gap.iloc[3])


def test_quali_gap_is_all_nan_when_nobody_set_a_time():
    q = pd.DataFrame({c: pd.to_timedelta([None, None], unit="s") for c in ("Q1", "Q2", "Q3")})
    assert quali_gap_pct(q).isna().all()


def test_weather_summary_averages_and_reports_the_share_of_rainy_readings():
    w = pd.DataFrame(
        {
            "AirTemp": [20.0, 22.0],
            "TrackTemp": [30.0, 34.0],
            "Humidity": [50.0, 60.0],
            "WindSpeed": [1.0, 3.0],
            "Rainfall": [False, True],
        }
    )
    s = weather_summary(w)
    assert s == {
        "wx_air_temp": 21.0,
        "wx_track_temp": 32.0,
        "wx_humidity": 55.0,
        "wx_wind_speed": 2.0,
        "wx_rain_frac": 0.5,
    }


def test_weather_summary_of_no_readings_is_all_nan():
    s = weather_summary(pd.DataFrame())
    assert set(s) == set(WEATHER_COLUMNS) and all(np.isnan(v) for v in s.values())

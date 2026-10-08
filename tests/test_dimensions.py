import pandas as pd

from ml.ingest.dimensions import build_circuits, build_drivers, build_teams


def results():
    return pd.DataFrame(
        {
            "Year": [2023, 2024, 2024],
            "Round": [1, 1, 2],
            "Abbreviation": ["VER", "VER", "HAM"],
            "DriverId": ["max_verstappen", "max_verstappen", "hamilton"],
            "FullName": ["Max Verstappen", "Max Verstappen", "Lewis Hamilton"],
            "TeamId": ["red_bull", "red_bull", "mercedes"],
            "TeamName": ["Red Bull Racing", "Red Bull Racing", "Mercedes"],
        }
    )


def test_drivers_are_keyed_by_driver_id_with_first_and_last_year():
    drivers = build_drivers(results()).set_index("DriverId")
    assert drivers.loc["max_verstappen", ["first_year", "last_year", "races"]].tolist() == [
        2023,
        2024,
        2,
    ]
    assert drivers.loc["hamilton", "Abbreviation"] == "HAM"


def test_a_driver_who_changes_abbreviation_is_still_one_driver():
    r = results()
    r.loc[1, "Abbreviation"] = "MAX"
    drivers = build_drivers(r)
    assert drivers["DriverId"].is_unique and len(drivers) == 2
    assert drivers.set_index("DriverId").loc["max_verstappen", "Abbreviation"] == "MAX"


def test_teams_count_entries_and_keep_the_latest_name():
    r = results()
    r.loc[1, "TeamName"] = "Oracle Red Bull Racing"
    teams = build_teams(r).set_index("TeamId")
    assert teams.loc["red_bull", "entries"] == 2
    assert teams.loc["red_bull", "TeamName"] == "Oracle Red Bull Racing"


def test_circuits_aggregate_race_weekends_across_years():
    events = pd.DataFrame(
        {
            "Year": [2019, 2020, 2020],
            "RoundNumber": [1, 1, 2],
            "CircuitId": ["sakhir", "sakhir", "sakhir"],
            "Location": ["Sakhir"] * 3,
            "Country": ["Bahrain"] * 3,
        }
    )
    circuits = build_circuits(events)
    assert len(circuits) == 1
    assert circuits.loc[0, ["first_year", "last_year", "race_weekends"]].tolist() == [2019, 2020, 3]

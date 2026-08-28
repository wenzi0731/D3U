import numpy as np
import pandas as pd

from baseline1.data import HEEWConditionDataset


def make_csvs(tmp_path):
    energy_rows = []
    weather_rows = []
    for year in range(2014, 2023):
        for hour in range(24):
            stamp = {"Year": year, "Month": 1, "Day": 2, "Hour": hour}
            value = year - 2014 + hour / 24
            energy_rows.append(
                stamp
                | {
                    "Electricity": value,
                    "Heat": value + 1,
                    "Cooling": value + 2,
                    "PV": value + max(np.sin(np.pi * hour / 24), 0),
                }
            )
            weather_rows.append(
                stamp
                | {
                    "Temperature": value,
                    "Dew Point": value + 1,
                    "Humidity": value + 2,
                    "Wind Speed": value + 3,
                    "Pressure": value + 4,
                    "Precip": value + 5,
                    "ALLSKY_SFC_SW_DWN": value + 6,
                    "CLRSKY_SFC_SW_DWN": value + 7,
                    "PV_CLEARNESS_RATIO": value + 8,
                    "PV_IS_DAYLIGHT": float(6 <= hour <= 18),
                }
            )
    energy = tmp_path / "energy.csv"
    weather = tmp_path / "weather.csv"
    pd.DataFrame(energy_rows).to_csv(energy, index=False)
    pd.DataFrame(weather_rows).to_csv(weather, index=False)
    return energy, weather


def test_split_normalization_and_no_history(tmp_path):
    energy, weather = make_csvs(tmp_path)
    train = HEEWConditionDataset(energy, weather, "train")
    val = HEEWConditionDataset(energy, weather, "val")
    test = HEEWConditionDataset(energy, weather, "test")
    assert len(train) == 7
    assert val.dates == ["2021-01-02"]
    assert test.dates == ["2022-01-02"]
    np.testing.assert_allclose(train.target_mean, test.target_mean)
    conditions, pv_year, target, date = test[0]
    assert conditions.shape == (18, 24)
    assert pv_year.shape == (1, 24)
    assert target.shape == (4, 24)
    assert date == "2022-01-02"
    assert len(test[0]) == 4  # no historical-energy field


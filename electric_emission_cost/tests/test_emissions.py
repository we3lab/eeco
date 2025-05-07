import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime as dt

from electric_emission_cost import emissions

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
skip_all_tests = False

input_dir = "tests/data/input/"
output_dir = "tests/data/output/"

@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize("start_dt, end_dt, emissions_path, resolution, expected_path",
    [
        (
            np.datetime64("2024-07-10T00:00"), # Summer weekday
            np.datetime64("2024-07-11T00:00"), # Summer weekday
            "data/emissions.csv",
            "1h",
            output_dir + "july_len1d_res1h.csv"           
        ),
    ]
)
def test_get_carbon_intensity(start_dt, end_dt, emissions_path, resolution, expected_path):
    emissions_df = pd.read_csv(emissions_path)
    expected = pd.read_csv(expected_path)
    result = emissions.get_carbon_intensity(start_dt, end_dt, emissions_df, resolution=resolution)
    assert np.allclose(result.magnitude, expected["co2_eq_kg_per_kWh"].values)

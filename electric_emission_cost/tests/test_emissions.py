import os
import pytest
import numpy as np
import pandas as pd
from datetime import timedelta

from electric_emission_cost.units import u
from electric_emission_cost import emissions
from electric_emission_cost import utils as ut



os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
skip_all_tests = False

input_dir = "tests/data/input/"
output_dir = "tests/data/output/"


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "emissions_path, consumption_path, net_demand_varname, resolution, expected",
    [
        (
            "data/emissions.csv",
            input_dir + "flat_load.csv",
            "VirtualDemand_Electricity_InFlow",
            "15m",
            276375.87356000004 * u.kg,
        ),
    ],
)
def test_calculate_grid_emissions_pd(
    emissions_path, consumption_path, net_demand_varname, resolution, expected
):
    emissions_data = pd.read_csv(emissions_path)
    consumption_df = pd.read_csv(consumption_path, parse_dates=[emissions.DT_VARNAME])
    emissions_factors = emissions.get_carbon_intensity(
        consumption_df[emissions.DT_VARNAME].iloc[0], 
        consumption_df[emissions.DT_VARNAME].iloc[-1] 
        + timedelta(minutes=ut.get_freq_binsize_minutes(resolution)), 
        emissions_data, 
        resolution=resolution
    )
    result, _ = emissions.calculate_grid_emissions(
        emissions_factors, consumption_df[net_demand_varname].values
    )
    assert result == expected


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "start_dt, end_dt, emissions_path, resolution, expected_path",
    [
        (
            np.datetime64("2024-07-10T00:00"),  # Summer weekday
            np.datetime64("2024-07-11T00:00"),  # Summer weekday
            "data/emissions.csv",
            "1h",
            output_dir + "july_len1d_res1h.csv",
        ),
        (
            np.datetime64("2024-07-31T00:00"),  # Summer weekday
            np.datetime64("2024-08-02T00:00"),  # Summer weekday
            "data/emissions.csv",
            "1h",
            output_dir + "july_aug_len2d_res1h.csv",
        ),
    ],
)
def test_get_carbon_intensity(
    start_dt, end_dt, emissions_path, resolution, expected_path
):
    emissions_df = pd.read_csv(emissions_path)
    expected = pd.read_csv(expected_path)
    result = emissions.get_carbon_intensity(
        start_dt, end_dt, emissions_df, resolution=resolution
    )
    assert np.allclose(result.magnitude, expected["co2_eq_kg_per_kWh"].values)

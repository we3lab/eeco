import os
import pytest
import cvxpy as cp
import numpy as np
import pandas as pd
import pyomo.environ as pyo
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
    "emissions_path, consumption_path, net_demand_varname, "
    "emissions_units, resolution, expected",
    [
        (
            "data/emissions.csv",
            input_dir + "flat_load.csv",
            "VirtualDemand_Electricity_InFlow",
            u.kg / u.kWh,
            "15m",
            276375.87356000004 * u.kg,
        ),
        (
            "data/emissions.csv",
            input_dir + "flat_load.csv",
            "VirtualDemand_Electricity_InFlow",
            u.kg / u.MWh,
            "15m",
            276375.8735600001 * u.kg,
        ),
    ],
)
def test_calculate_grid_emissions_np(
    emissions_path,
    consumption_path,
    net_demand_varname,
    emissions_units,
    resolution,
    expected,
):
    emissions_data = pd.read_csv(emissions_path)
    consumption_df = pd.read_csv(consumption_path, parse_dates=[emissions.DT_VARNAME])
    emissions_factors = emissions.get_carbon_intensity(
        consumption_df[emissions.DT_VARNAME].iloc[0],
        consumption_df[emissions.DT_VARNAME].iloc[-1]
        + timedelta(minutes=ut.get_freq_binsize_minutes(resolution)),
        emissions_data,
        resolution=resolution,
    )
    emissions_factors = emissions_factors.to(emissions_units)
    result, model = emissions.calculate_grid_emissions(
        emissions_factors.magnitude,
        consumption_df[net_demand_varname].values,
        emissions_units=emissions_units,
    )
    assert result.units == expected.units
    assert result.magnitude == pytest.approx(expected.magnitude)
    assert model is None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "emissions_path, consumption_path, net_demand_varname, "
    "emissions_units, resolution, expected",
    [
        (
            "data/emissions.csv",
            input_dir + "flat_load.csv",
            "VirtualDemand_Electricity_InFlow",
            u.kg / u.kWh,
            "15m",
            276375.8735600001,
        ),
    ],
)
def test_calculate_grid_emissions_cvx(
    emissions_path,
    consumption_path,
    net_demand_varname,
    emissions_units,
    resolution,
    expected,
):
    emissions_data = pd.read_csv(emissions_path)
    consumption_df = pd.read_csv(consumption_path, parse_dates=[emissions.DT_VARNAME])
    emissions_factors = emissions.get_carbon_intensity(
        consumption_df[emissions.DT_VARNAME].iloc[0],
        consumption_df[emissions.DT_VARNAME].iloc[-1]
        + timedelta(minutes=ut.get_freq_binsize_minutes(resolution)),
        emissions_data,
        resolution=resolution,
    )
    emissions_factors = emissions_factors.to(emissions_units)

    constraints = []
    electric_consumption = cp.Variable(len(consumption_df[net_demand_varname].values))
    constraints.append(
        electric_consumption == consumption_df[net_demand_varname].values
    )
    result, model = emissions.calculate_grid_emissions(
        emissions_factors.magnitude,
        electric_consumption,
        emissions_units=emissions_units,
    )
    prob = cp.Problem(cp.Minimize(result), constraints)
    prob.solve()
    assert result.value == expected
    assert model is None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "emissions_path, consumption_path, net_demand_varname, "
    "emissions_units, resolution, expected",
    [
        (
            "data/emissions.csv",
            input_dir + "flat_load.csv",
            "VirtualDemand_Electricity_InFlow",
            u.kg / u.kWh,
            "15m",
            276375.8735600004,
        ),
    ],
)
def test_calculate_grid_emissions_pyo(
    emissions_path,
    consumption_path,
    net_demand_varname,
    emissions_units,
    resolution,
    expected,
):
    emissions_data = pd.read_csv(emissions_path)
    consumption_df = pd.read_csv(consumption_path, parse_dates=[emissions.DT_VARNAME])
    emissions_factors = emissions.get_carbon_intensity(
        consumption_df[emissions.DT_VARNAME].iloc[0],
        consumption_df[emissions.DT_VARNAME].iloc[-1]
        + timedelta(minutes=ut.get_freq_binsize_minutes(resolution)),
        emissions_data,
        resolution=resolution,
    )
    emissions_factors = emissions_factors.to(emissions_units)

    model = pyo.ConcreteModel()
    model.T = len(consumption_df[net_demand_varname])
    model.t = range(model.T)
    pyo_vars = {}
    for col_name in consumption_df.columns:
        var = pyo.Var(
            range(len(consumption_df[col_name])),
            initialize=np.zeros(len(consumption_df[col_name])),
            bounds=(0, None),
        )
        model.add_component(col_name, var)
        pyo_vars[col_name] = var

    @model.Constraint(model.t)
    def electric_constraint(m, t):
        return (
            consumption_df[net_demand_varname][t] == getattr(m, net_demand_varname)[t]
        )

    result, model = emissions.calculate_grid_emissions(
        emissions_factors.magnitude,
        getattr(model, net_demand_varname),
        emissions_units=emissions_units,
        model=model,
    )
    model.obj = pyo.Objective(expr=result)
    solver = pyo.SolverFactory("ipopt")
    solver.solve(model)
    assert pyo.value(result) == expected
    assert model is not None


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

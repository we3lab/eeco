import os
import pytest
import numpy as np
import pandas as pd

from electric_emission_cost import metrics

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
skip_all_tests = False

input_dir = "tests/data/input/"
output_dir = "tests/data/output/"

np.random.seed(0)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "metric, power_capacity_type, energy_capacity_type",
    [
        # test RTE
        (
            "rte",
            None,
            None,
        ),
        # test Power Capacity with options
        (
            "pc",
            "average",
            None,
        ),
        (
            "pc",
            "charging",
            None,
        ),
        (
            "pc",
            "discharging",
            None,
        ),
        (
            "pc",
            "maximum",
            None,
        ),
        (
            "ec",
            None,
            "average",
        ),
        (
            "ec",
            None,
            "charging",
        ),
        (
            "ec",
            None,
            "discharging",
        ),
    ],
)
def test_metrics(
    metric, power_capacity_type, energy_capacity_type, baseline_datafile="flat_load.csv"
):
    flat_baseline_df = pd.read_csv(input_dir + baseline_datafile)
    baseline_kW = flat_baseline_df["VirtualDemand_Electricity_InFlow"].values

    # create a random load profile with the same length
    flexible_kW = np.random.normal(loc=1200, scale=300, size=len(baseline_kW))
    if metric == "rte":
        rte = metrics.roundtrip_efficiency(baseline_kW, flexible_kW)
    elif metric == "pc":
        pc = metrics.power_capacity(
            baseline_kW,
            flexible_kW,
            timestep=0.25,
            pc_type=power_capacity_type,
            relative=True,
        )
    elif metric == "ec":
        ec = metrics.energy_capacity(
            baseline_kW,
            flexible_kW,
            timestep=0.25,
            ec_type=energy_capacity_type,
            relative=True,
        )

    return


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_npv():
    npv = metrics.net_present_value(
        capital_cost=0, electricity_savings=0, simulation_years=1, upgrade_lifetime=30
    )  # default value is 0
    assert npv == 0, "Net present value should be 0 by default"

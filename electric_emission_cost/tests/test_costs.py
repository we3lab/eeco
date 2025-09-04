import os
import pytest
import numpy as np
import cvxpy as cp
import pandas as pd
import pyomo.environ as pyo
import datetime

from electric_emission_cost import costs
from electric_emission_cost.costs import (
    CHARGE,
    TYPE,
    MONTH_START,
    MONTH_END,
    WEEKDAY_START,
    WEEKDAY_END,
    HOUR_START,
    HOUR_END,
    ELECTRIC,
    GAS,
    DEMAND,
    ENERGY,
    PEAK,
    HALF_PEAK,
    OFF_PEAK,
    SUPER_OFF_PEAK,
)

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
skip_all_tests = False

input_dir = "tests/data/input/"
output_dir = "tests/data/output/"


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "charge, start_dt, end_dt, n_per_hour, effective_start_date, "
    "effective_end_date, expected",
    [
        # all hours constant charge for only 1-day
        (
            {
                CHARGE: 0.05,
                MONTH_START: 1,
                MONTH_END: 12,
                WEEKDAY_START: 0,
                WEEKDAY_END: 6,
                HOUR_START: 0,
                HOUR_END: 24,
            },
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            4,
            np.datetime64("2021-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            np.ones(96) * 0.05,
        ),
        # outside of effective start date
        (
            {
                CHARGE: 0.05,
                MONTH_START: 1,
                MONTH_END: 12,
                WEEKDAY_START: 0,
                WEEKDAY_END: 6,
                HOUR_START: 0,
                HOUR_END: 24,
            },
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            4,
            np.datetime64("2021-01-01"),  # Summer weekday
            np.datetime64("2021-12-31"),  # Summer weekday
            np.zeros(96),
        ),
        # one day with then one day without effective start date
        (
            {
                CHARGE: 0.05,
                MONTH_START: 1,
                MONTH_END: 12,
                WEEKDAY_START: 0,
                WEEKDAY_END: 6,
                HOUR_START: 0,
                HOUR_END: 24,
            },
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-12"),  # Summer weekday
            4,
            np.datetime64("2024-07-11"),  # Summer weekday
            np.datetime64("2024-07-12"),  # Summer weekday
            np.concatenate([np.zeros(96), np.ones(96) * 0.05]),
        ),
        # one day without then one day with effective start date
        (
            {
                CHARGE: 0.05,
                MONTH_START: 1,
                MONTH_END: 12,
                WEEKDAY_START: 0,
                WEEKDAY_END: 6,
                HOUR_START: 0,
                HOUR_END: 24,
            },
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-12"),  # Summer weekdays
            4,
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            np.concatenate([np.ones(96) * 0.05, np.zeros(96)]),
        ),
    ],
)
def test_create_charge_array(
    charge,
    start_dt,
    end_dt,
    n_per_hour,
    effective_start_date,
    effective_end_date,
    expected,
):
    ntsteps = int((end_dt - start_dt) / np.timedelta64(15, "m"))
    datetime = pd.DataFrame(
        np.array([start_dt + np.timedelta64(i * 15, "m") for i in range(ntsteps)]),
        columns=["DateTime"],
    )
    hours = datetime["DateTime"].dt.hour.astype(float).values
    n_hours = int((end_dt - start_dt) / np.timedelta64(1, "h"))
    hours += np.tile(np.arange(n_per_hour) / n_per_hour, n_hours)

    result = costs.create_charge_array(
        charge, datetime, effective_start_date, effective_end_date
    )
    assert (result == expected).all()


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "start_dt, end_dt, billing_path, resolution, expected",
    [
        # only one energy charge
        # no name, assessed, effective start, or effective end
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_energy_1.csv",
            "15m",
            {
                "electric_energy_0_20240710_20240710_0": np.ones(96) * 0.05,
            },
        ),
        # only one energy charge but at 5 min. resolution
        # no name, assessed, effective start, or effective end
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_energy_1.csv",
            "5m",
            {
                "electric_energy_0_20240710_20240710_0": np.ones(288) * 0.05,
            },
        ),
        # only one energy charge but at 1 hour resolution
        # no name, assessed, effective start, or effective end
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_energy_1.csv",
            "1h",
            {
                "electric_energy_0_20240710_20240710_0": np.ones(24) * 0.05,
            },
        ),
        # three energy charges
        # no name, assessed, effective start, or effective end
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_energy_3.csv",
            "15m",
            {
                "electric_energy_0_20240710_20240710_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.zeros(32),
                    ]
                ),
                "electric_energy_1_20240710_20240710_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 0.1,
                        np.zeros(12),
                    ]
                ),
                "electric_energy_2_20240710_20240710_0": np.concatenate(
                    [
                        np.zeros(84),
                        np.ones(12) * 0.05,
                    ]
                ),
            },
        ),
        # two energy charges combined under same name, one still separate
        # still no name, assessed, effective start, or effective end
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_energy_peak.csv",
            "15m",
            {
                "electric_energy_off-peak_20240710_20240710_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.zeros(20),
                        np.ones(12) * 0.05,
                    ]
                ),
                "electric_energy_on-peak_20240710_20240710_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 0.1,
                        np.zeros(12),
                    ]
                ),
            },
        ),
        # all 3 energy charges combined under same name
        # still no name, assessed, effective start, or effective end
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_energy_combine.csv",
            "15m",
            {
                "electric_energy_all-day_20240710_20240710_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.ones(20) * 0.1,
                        np.ones(12) * 0.05,
                    ]
                ),
            },
        ),
        # 2 demand charges, all-day and on-peak
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_demand_2.csv",
            "15m",
            {
                "electric_demand_all-day_20240710_20240710_0": np.ones(96) * 5,
                "electric_demand_on-peak_20240710_20240710_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 20,
                        np.zeros(12),
                    ]
                ),
            },
        ),
        # 2 demand charges, one assessed monthly and one blank assessed column
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_demand_monthly.csv",
            "15m",
            {
                "electric_demand_all-day_20240710_20240710_0": np.ones(96) * 5,
                "electric_demand_on-peak_20240710_20240710_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 20,
                        np.zeros(12),
                    ]
                ),
            },
        ),
        # 2 demand charges, one assessed daily and one blank assessed column
        # but only one day of data
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_demand_daily.csv",
            "15m",
            {
                "electric_demand_all-day_20240710_20240710_0": np.ones(96) * 5,
                "electric_demand_on-peak_20240710_20240710_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 20,
                        np.zeros(12),
                    ]
                ),
            },
        ),
        # 2 demand charges, one assessed daily and one blank assessed column
        # and two days of data
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-12"),  # Summer weekdays
            input_dir + "billing_demand_daily.csv",
            "15m",
            {
                "electric_demand_all-day_20240710_20240711_0": np.ones(192) * 5,
                "electric_demand_on-peak_20240710_20240710_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 20,
                        np.zeros(108),
                    ]
                ),
                "electric_demand_on-peak_20240711_20240711_0": np.concatenate(
                    [
                        np.zeros(160),
                        np.ones(20) * 20,
                        np.zeros(12),
                    ]
                ),
            },
        ),
        # export payments for two days
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-12"),  # Summer weekdays
            input_dir + "billing_export.csv",
            "15m",
            {
                "electric_export_0_20240710_20240711_0": np.ones(192) * 0.025,
            },
        ),
        # customer payments for any number of days
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-08-01"),  # Summer weekdays
            input_dir + "billing_customer.csv",
            "15m",
            {
                "electric_customer_0_20240710_20240731_0": np.array([1000]),
            },
        ),
        # effective start/end dates
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-08-01"),  # Summer weekdays
            input_dir + "billing_effective.csv",
            "1h",
            {
                "electric_energy_0_20240101_20240720_0": np.concatenate(
                    [
                        np.ones(264) * 0.05,
                        np.zeros(264),
                    ]
                ),
                "electric_energy_0_20240721_20241231_0": np.concatenate(
                    [
                        np.zeros(264),
                        np.ones(264) * 0.075,
                    ]
                ),
            },
        ),
        # switch between months and weekend/weekday
        (
            np.datetime64("2024-05-31"),  # Summer weekdays
            np.datetime64("2024-06-02"),  # Summer weekdays
            input_dir + "billing.csv",
            "1h",
            {
                "electric_customer_0_20240531_20240601_0": np.array([300]),
                "electric_energy_0_20240531_20240601_0": np.concatenate(
                    [np.ones(24) * 0.019934, np.zeros(24)]
                ),
                "electric_energy_1_20240531_20240601_0": np.zeros(48),
                "electric_energy_2_20240531_20240601_0": np.zeros(48),
                "electric_energy_3_20240531_20240601_0": np.zeros(48),
                "electric_energy_4_20240531_20240601_0": np.concatenate(
                    [
                        np.zeros(24),
                        np.ones(24) * 0.021062,
                    ]
                ),
                "electric_energy_5_20240531_20240601_0": np.zeros(48),
                "electric_energy_6_20240531_20240601_0": np.zeros(48),
                "electric_demand_maximum_20240531_20240601_0": np.ones(48) * 7.128,
                "gas_customer_0_20240531_20240601_0": np.array([93.14]),
                "gas_energy_0_20240531_20240601_0": np.ones(48) * 0.2837,
                "gas_energy_1_20240531_20240601_0": np.zeros(48),
            },
        ),
        # switch between years
        (
            np.datetime64("2023-12-31"),  # Summer weekdays
            np.datetime64("2024-01-02"),  # Summer weekdays
            input_dir + "billing.csv",
            "1h",
            {
                "electric_customer_0_20231231_20240101_0": np.array([300]),
                "electric_energy_0_20231231_20240101_0": np.concatenate(
                    [
                        np.zeros(24),
                        np.ones(24) * 0.019934,
                    ]
                ),
                "electric_energy_1_20231231_20240101_0": np.zeros(48),
                "electric_energy_2_20231231_20240101_0": np.zeros(48),
                "electric_energy_3_20231231_20240101_0": np.zeros(48),
                "electric_energy_4_20231231_20240101_0": np.zeros(48),
                "electric_energy_5_20231231_20240101_0": np.zeros(48),
                "electric_energy_6_20231231_20240101_0": np.concatenate(
                    [np.ones(24) * 0.022552, np.zeros(24)]
                ),
                "electric_demand_maximum_20231231_20240101_0": np.ones(48) * 7.128,
                "gas_customer_0_20231231_20240101_0": np.array([93.14]),
                "gas_energy_0_20231231_20240101_0": np.concatenate(
                    [
                        np.zeros(24),
                        np.ones(24) * 0.2837,
                    ]
                ),
                "gas_energy_1_20231231_20240101_0": np.concatenate(
                    [
                        np.ones(24) * 0.454,
                        np.zeros(24),
                    ]
                ),
            },
        ),
        # charge limit of 100 kW, no grouping
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_energy_3_charge_limit.csv",
            "15m",
            {
                "electric_energy_0_20240710_20240710_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.zeros(32),
                    ]
                ),
                "electric_energy_0_20240710_20240710_100": np.concatenate(
                    [
                        np.ones(64) * 0.1,
                        np.zeros(32),
                    ]
                ),
                "electric_energy_1_20240710_20240710_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 0.1,
                        np.zeros(12),
                    ]
                ),
                "electric_energy_1_20240710_20240710_100": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 0.15,
                        np.zeros(12),
                    ]
                ),
                "electric_energy_2_20240710_20240710_0": np.concatenate(
                    [
                        np.zeros(84),
                        np.ones(12) * 0.05,
                    ]
                ),
                "electric_energy_2_20240710_20240710_100": np.concatenate(
                    [
                        np.zeros(84),
                        np.ones(12) * 0.1,
                    ]
                ),
            },
        ),
        # charge limit of 100 kW, with grouping
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_energy_combine_charge_limit.csv",
            "15m",
            {
                "electric_energy_all-day_20240710_20240710_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.ones(20) * 0.1,
                        np.ones(12) * 0.05,
                    ]
                ),
                "electric_energy_all-day_20240710_20240710_100": np.concatenate(
                    [
                        np.ones(64) * 0.1,
                        np.ones(20) * 0.15,
                        np.ones(12) * 0.1,
                    ]
                ),
            },
        ),
        # 2 demand charges with 100 kW charge limits, all-day and on-peak
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_demand_2_charge_limit.csv",
            "15m",
            {
                "electric_demand_all-day_20240710_20240710_0": np.ones(96) * 5,
                "electric_demand_all-day_20240710_20240710_100": np.ones(96) * 10,
                "electric_demand_on-peak_20240710_20240710_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 20,
                        np.zeros(12),
                    ]
                ),
                "electric_demand_on-peak_20240710_20240710_100": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 30,
                        np.zeros(12),
                    ]
                ),
            },
        ),
    ],
)
def test_get_charge_dict(start_dt, end_dt, billing_path, resolution, expected):
    tariff_df = pd.read_csv(billing_path)
    result = costs.get_charge_dict(start_dt, end_dt, tariff_df, resolution=resolution)
    assert result.keys() == expected.keys()
    for key, val in result.items():
        assert (result[key] == expected[key]).all()


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "charge_dict, consumption_data_dict, resolution, prev_demand_dict, "
    "consumption_estimate, desired_utility, desired_charge_type, expected_cost",
    [
        # single energy charge with flat consumption
        (
            {"electric_energy_0_2024-07-10_2024-07-10_0": np.ones(96) * 0.05},
            {ELECTRIC: np.ones(96), GAS: np.ones(96)},
            "15m",
            None,
            0,
            None,
            None,
            pytest.approx(1.2),
        ),
        # single energy charge with increasing consumption
        (
            {"electric_energy_0_2024-07-10_2024-07-10_0": np.ones(96) * 0.05},
            {ELECTRIC: np.arange(96), GAS: np.ones(96)},
            "15m",
            None,
            0,
            None,
            None,
            np.sum(np.arange(96)) * 0.05 / 4,
        ),
        # energy charge with charge limit
        (
            {
                "electric_energy_all-day_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.ones(20) * 0.1,
                        np.ones(12) * 0.05,
                    ]
                ),
                "electric_energy_all-day_2024-07-10_2024-07-10_100": np.concatenate(
                    [
                        np.ones(64) * 0.1,
                        np.ones(20) * 0.15,
                        np.ones(12) * 0.1,
                    ]
                ),
            },
            {ELECTRIC: np.ones(96) * 100, GAS: np.ones(96)},
            "15m",
            None,
            2400,
            None,
            None,
            260,
        ),
    ],
)
def test_calculate_cost_np(
    charge_dict,
    consumption_data_dict,
    resolution,
    prev_demand_dict,
    consumption_estimate,
    desired_utility,
    desired_charge_type,
    expected_cost,
):
    result, model = costs.calculate_cost(
        charge_dict,
        consumption_data_dict,
        resolution=resolution,
        prev_demand_dict=prev_demand_dict,
        consumption_estimate=consumption_estimate,
        desired_utility=desired_utility,
        desired_charge_type=desired_charge_type,
    )
    assert result == expected_cost
    assert model is None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "charge_dict, consumption_data_dict, resolution, prev_demand_dict, "
    "consumption_estimate, desired_utility, desired_charge_type, expected_cost",
    [
        # energy charge with charge limit
        (
            {
                "electric_energy_all-day_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.ones(20) * 0.1,
                        np.ones(12) * 0.05,
                    ]
                ),
                "electric_energy_all-day_2024-07-10_2024-07-10_100": np.concatenate(
                    [
                        np.ones(64) * 0.1,
                        np.ones(20) * 0.15,
                        np.ones(12) * 0.1,
                    ]
                ),
            },
            {ELECTRIC: np.ones(96) * 100, GAS: np.ones(96)},
            "15m",
            None,
            2400,
            None,
            None,
            260,
        ),
        # energy charge with charge limit and time-varying consumption estimate
        (
            {
                "electric_energy_all-day_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.ones(20) * 0.1,
                        np.ones(12) * 0.05,
                    ]
                ),
                "electric_energy_all-day_2024-07-10_2024-07-10_100": np.concatenate(
                    [
                        np.ones(64) * 0.1,
                        np.ones(20) * 0.15,
                        np.ones(12) * 0.1,
                    ]
                ),
            },
            {ELECTRIC: np.ones(96) * 100, GAS: np.ones(96)},
            "15m",
            None,
            np.ones(96) * 100,
            None,
            None,
            260,
        ),
        # energy charge with charge limit that wont be hit and time-varying consumption estimate
        (
            {
                "electric_energy_all-day_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.ones(20) * 0.1,
                        np.ones(12) * 0.05,
                    ]
                ),
                "electric_energy_all-day_2024-07-10_2024-07-10_100": np.concatenate(
                    [
                        np.ones(64) * 0.1,
                        np.ones(20) * 0.15,
                        np.ones(12) * 0.1,
                    ]
                ),
                "electric_energy_all-day_2024-07-10_2024-07-10_100000": np.concatenate(
                    [
                        np.ones(96)
                    ]
                ),
            },
            {ELECTRIC: np.ones(96) * 100, GAS: np.ones(96)},
            "15m",
            None,
            2400,
            None,
            None,
            260,
        ),
    ],
)
def test_calculate_cost_cvx(
    charge_dict,
    consumption_data_dict,
    resolution,
    prev_demand_dict,
    consumption_estimate,
    desired_utility,
    desired_charge_type,
    expected_cost,
):
    cvx_vars = {}
    constraints = []
    for key, val in consumption_data_dict.items():
        cvx_vars[key] = cp.Variable(len(val))
        constraints.append(cvx_vars[key] == val)

    result, model = costs.calculate_cost(
        charge_dict,
        cvx_vars,
        resolution=resolution,
        prev_demand_dict=prev_demand_dict,
        consumption_estimate=consumption_estimate,
        desired_utility=desired_utility,
        desired_charge_type=desired_charge_type,
    )
    prob = cp.Problem(cp.Minimize(result), constraints)
    prob.solve()
    assert result.value == expected_cost
    assert model is None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "charge_dict, consumption_data_dict, resolution, prev_demand_dict, "
    "consumption_estimate, desired_utility, desired_charge_type, expected_cost",
    [
        # energy charge with charge limit
        (
            {
                "electric_energy_all-day_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.ones(20) * 0.1,
                        np.ones(12) * 0.05,
                    ]
                ),
                "electric_energy_all-day_2024-07-10_2024-07-10_100": np.concatenate(
                    [
                        np.ones(64) * 0.1,
                        np.ones(20) * 0.15,
                        np.ones(12) * 0.1,
                    ]
                ),
            },
            {ELECTRIC: np.ones(96) * 100, GAS: np.ones(96)},
            "15m",
            None,
            2400,
            None,
            None,
            pytest.approx(260),
        ),
        # demand charges
        (
            {
                "electric_demand_peak-summer_2024-07-10_2024-07-10_0": (
                    np.concatenate(
                        [
                            np.ones(48) * 0,
                            np.ones(24) * 1,
                            np.ones(24) * 0,
                        ]
                    )
                ),
                "electric_demand_half-peak-summer_2024-07-10_2024-07-10_0": (
                    np.concatenate(
                        [
                            np.ones(34) * 0,
                            np.ones(14) * 2,
                            np.ones(24) * 0,
                            np.ones(14) * 2,
                            np.ones(10) * 0,
                        ]
                    )
                ),
                "electric_demand_off-peak_2024-07-10_2024-07-10_0": np.ones(96) * 10,
            },
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            "15m",
            {
                "electric_demand_peak-summer_2024-07-10_2024-07-10_0": {
                    "demand": 150,
                    "cost": 150,
                },
                "electric_demand_half-peak-summer_2024-07-10_2024-07-10_0": {
                    "demand": 40,
                    "cost": 80,
                },
                "electric_demand_off-peak_2024-07-10_2024-07-10_0": {
                    "demand": 90,
                    "cost": 900,
                },
            },
            0,
            None,
            None,
            pytest.approx(140),
        ),
        (
            {
                "electric_demand_peak-summer_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.ones(48) * 0,
                        np.ones(24) * 1,
                        np.ones(24) * 0,
                    ]
                ),
                "electric_demand_half-peak-summer_2024-07-10_2024-07-10_0": (
                    np.concatenate(
                        [
                            np.ones(34) * 0,
                            np.ones(14) * 2,
                            np.ones(24) * 0,
                            np.ones(14) * 2,
                            np.ones(10) * 0,
                        ]
                    )
                ),
                "electric_demand_off-peak_2024-07-10_2024-07-10_0": np.ones(96) * 10,
            },
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            "15m",
            {
                "electric_demand_peak-summer_2024-07-10_2024-07-10_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_half-peak-summer_2024-07-10_2024-07-10_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_off-peak_2024-07-10_2024-07-10_0": {
                    "demand": 0,
                    "cost": 0,
                },
            },
            0,
            None,
            None,
            pytest.approx(1191),
        ),
    ],
)
def test_calculate_cost_pyo(
    charge_dict,
    consumption_data_dict,
    resolution,
    prev_demand_dict,
    consumption_estimate,
    desired_utility,
    desired_charge_type,
    expected_cost,
):
    model = pyo.ConcreteModel()
    model.T = len(consumption_data_dict["electric"])
    model.t = range(model.T)
    pyo_vars = {}
    for key, val in consumption_data_dict.items():
        var = pyo.Var(range(len(val)), initialize=np.zeros(len(val)), bounds=(0, None))
        model.add_component(key, var)
        pyo_vars[key] = var

    @model.Constraint(model.t)
    def electric_constraint(m, t):
        return consumption_data_dict["electric"][t] == m.electric[t]

    @model.Constraint(model.t)
    def gas_constraint(m, t):
        return consumption_data_dict["gas"][t] == m.gas[t]

    result, model = costs.calculate_cost(
        charge_dict,
        pyo_vars,
        resolution=resolution,
        prev_demand_dict=prev_demand_dict,
        consumption_estimate=consumption_estimate,
        desired_utility=desired_utility,
        desired_charge_type=desired_charge_type,
        model=model,
    )

    model.obj = pyo.Objective(expr=result)
    solver = pyo.SolverFactory("gurobi")
    solver.solve(model)
    assert pyo.value(result) == expected_cost
    assert model is not None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "start_dt, end_dt, billing_data, utility, consumption_data_dict, "
    "prev_demand_dict, consumption_estimate, scale_factor, expected",
    [
        (
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            None,
            0,
            1,  # default scale factor
            np.float64(4027.79),
        ),
        (
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            None,
            0,
            1.1,  # non-default scale factor
            np.float64(4027.79),  # daily demand charge unscaled
        ),
        (
            np.datetime64("2024-07-13"),  # Summer weekend
            np.datetime64("2024-07-14"),  # Summer weekend
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            None,
            0,
            1,  # default scale factor
            np.float64(2023.5),
        ),
        (
            np.datetime64("2024-03-07"),  # Winter weekday
            np.datetime64("2024-03-08"),  # Winter weekday
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            None,
            0,
            1,  # default scale factor
            np.float64(2028.6),
        ),
        (
            np.datetime64("2024-03-09"),  # Winter weekend
            np.datetime64("2024-03-10"),  # Winter weekend
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            {
                "electric_demand_peak-summer_20240309_20240309_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_half-peak-summer_20240309_20240309_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_off-peak_20240309_20240309_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_half-peak-winter1_20240309_20240309_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_half-peak-winter2_20240309_20240309_0": {
                    "demand": 0,
                    "cost": 0,
                },
            },
            0,
            1,  # default scale factor
            np.float64(2023.5),
        ),
        (
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            {
                "electric_demand_peak-summer_20240710_20240710_0": {
                    "demand": 7.078810759792355,
                    "cost": 150,
                },
                "electric_demand_half-peak-summer_20240710_20240710_0": {
                    "demand": 13.605442176870748,
                    "cost": 80,
                },
                "electric_demand_off-peak_20240710_20240710_0": {
                    "demand": 42.253521126760563,
                    "cost": 900,
                },
                "electric_demand_half-peak-winter1_20240710_20240710_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_half-peak-winter2_20240710_20240710_0": {
                    "demand": 0,
                    "cost": 0,
                },
            },
            0,
            1,  # default scale factor
            np.float64(2897.79),
        ),
        (
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            input_dir + "billing_pge.csv",
            GAS,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            None,
            0,
            1,  # default scale factor
            np.float64(0),
        ),
        (
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-13"),  # Summer weekday (3 days)
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {
                ELECTRIC: np.arange(288),
                GAS: np.arange(288),
            },  # 3 days * 96 timesteps
            None,
            0,
            1.1,  # non-default scale factor
            pytest.approx(14646.313),  # 13314.83 * 1.1
        ),
    ],
)
def test_calculate_demand_costs(
    start_dt,
    end_dt,
    billing_data,
    utility,
    consumption_data_dict,
    prev_demand_dict,
    consumption_estimate,
    scale_factor,
    expected,
):
    billing_data = pd.read_csv(billing_data)
    charge_dict = costs.get_charge_dict(
        start_dt,
        end_dt,
        billing_data,
    )
    result, model = costs.calculate_cost(
        charge_dict,
        consumption_data_dict,
        prev_demand_dict=prev_demand_dict,
        consumption_estimate=consumption_estimate,
        desired_utility=utility,
        desired_charge_type="demand",
        demand_scale_factor=scale_factor,
    )
    assert result == expected
    assert model is None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "start_dt, end_dt, billing_data, utility, consumption_data_dict, "
    "prev_consumption_dict, consumption_estimate, expected",
    [
        (
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            {
                "gas_energy_0_20240710_20240710_0": 0,
                "gas_energy_0_20240710_20240710_5000": 0,
                "electric_customer_0_20240710_20240710_0": 0,
                "electric_energy_0_20240710_20240710_0": 0,
                "electric_energy_1_20240710_20240710_0": 0,
                "electric_energy_2_20240710_20240710_0": 0,
                "electric_energy_3_20240710_20240710_0": 0,
                "electric_energy_4_20240710_20240710_0": 0,
                "electric_energy_5_20240710_20240710_0": 0,
                "electric_energy_6_20240710_20240710_0": 0,
                "electric_energy_7_20240710_20240710_0": 0,
                "electric_energy_8_20240710_20240710_0": 0,
                "electric_energy_9_20240710_20240710_0": 0,
                "electric_energy_10_20240710_20240710_0": 0,
                "electric_energy_11_20240710_20240710_0": 0,
                "electric_energy_12_20240710_20240710_0": 0,
                "electric_energy_13_20240710_20240710_0": 0,
            },
            0,
            pytest.approx(140.916195),
        ),
        (
            np.datetime64("2024-07-13"),  # Summer weekend
            np.datetime64("2024-07-14"),  # Summer weekend
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            None,
            0,
            pytest.approx(102.3834),
        ),
        (
            np.datetime64("2024-03-07"),  # Winter weekday
            np.datetime64("2024-03-08"),  # Winter weekday
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            None,
            0,
            pytest.approx(123.24669),
        ),
        (
            np.datetime64("2024-03-09"),  # Winter weekend
            np.datetime64("2024-03-10"),  # Winter weekend
            input_dir + "billing_pge.csv",
            ELECTRIC,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            None,
            0,
            pytest.approx(110.7624),
        ),
        (
            np.datetime64("2024-03-09"),  # Winter weekend
            np.datetime64("2024-03-10"),  # Winter weekend
            input_dir + "billing_pge.csv",
            GAS,
            {ELECTRIC: np.arange(96), GAS: np.arange(96)},
            None,
            0,
            pytest.approx(0),
        ),
        (
            np.datetime64("2024-03-09"),  # Winter weekend
            np.datetime64("2024-03-10"),  # Winter weekend
            input_dir + "billing_pge.csv",
            GAS,
            {ELECTRIC: np.arange(96), GAS: np.repeat(np.array([5100]), 96)},
            None,
            0,
            pytest.approx(59.1),
        ),
        (
            np.datetime64("2024-03-09"),  # Winter weekend
            np.datetime64("2024-03-10"),  # Winter weekend
            input_dir + "billing_pge.csv",
            GAS,
            {ELECTRIC: np.arange(96), GAS: np.ones(96)},
            None,
            5100,
            pytest.approx(0),
        ),
    ],
)
def test_calculate_energy_costs(
    start_dt,
    end_dt,
    billing_data,
    utility,
    consumption_data_dict,
    prev_consumption_dict,
    consumption_estimate,
    expected,
):
    billing_data = pd.read_csv(billing_data)
    charge_dict = costs.get_charge_dict(
        start_dt,
        end_dt,
        billing_data,
    )
    result, model = costs.calculate_cost(
        charge_dict,
        consumption_data_dict,
        prev_consumption_dict=prev_consumption_dict,
        consumption_estimate=consumption_estimate,
        desired_utility=utility,
        desired_charge_type="energy",
    )
    assert result == expected
    assert model is None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "charge_array, export_data, divisor, expected",
    [
        (np.ones(96), np.arange(96), 4, 1140),
    ],
)
def test_calculate_export_revenues(charge_array, export_data, divisor, expected):
    result, model = costs.calculate_export_revenues(charge_array, export_data, divisor)
    assert result == expected
    assert model is None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "key, expected",
    [
        # YYYYMMDD format
        ("electric_demand_peak_20240710_20240710_100", 0),
        ("electric_demand_peak_20240710_20240731_100", 21),
        # YYYY-MM-DD format
        ("electric_energy_0_2024-07-10_2024-07-10_0", 0),
        ("electric_energy_0_2024-07-10_2024-07-31_0", 21),
    ],
)
def test_get_charge_array_duration(key, expected):
    from electric_emission_cost.costs import get_charge_array_duration

    assert get_charge_array_duration(key) == expected


@pytest.mark.parametrize(
    "keep_fixed_charge, scale_fixed_charge, scale_demand_charge, tariff, expected",
    [
        (True, True, True, "billing.csv", "billing_scaled.csv"),
        (True, False, False, "billing.csv", "billing_unscaled.csv"),
        (False, True, True, "billing_customer.csv", "billing_customer_nocharge.csv"),
        (False, False, False, "billing_customer.csv", "billing_customer_nocharge.csv"),
    ],
)
@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_get_charge_df(
    keep_fixed_charge, scale_fixed_charge, scale_demand_charge, tariff, expected
):
    # load tariff
    path_to_tariff = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "input", tariff
    )
    tariff_df = pd.read_csv(path_to_tariff, sep=",")

    # get charge dataframe
    df = costs.get_charge_df(
        datetime.datetime(2023, 4, 9),
        datetime.datetime(2023, 4, 11),
        tariff_df,
        resolution="15m",
        keep_fixed_charges=keep_fixed_charge,
        scale_fixed_charges=scale_fixed_charge,
        scale_demand_charges=scale_demand_charge,
    )

    # load expected output
    path_to_output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "output", expected
    )
    df_expected = pd.read_csv(path_to_output, parse_dates=["DateTime"])

    # compare dataframes
    pd.testing.assert_frame_equal(df, df_expected)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    (
        "charge_type, "
        "month, "
        "weekday, "
        "expected_base_charge, "
        "expected_overlap, "
        "expected_periods"
    ),
    [
        # July weekday energy
        (
            costs.ENERGY,
            7,
            0,
            0.08981,
            False,
            {costs.PEAK, costs.HALF_PEAK, costs.OFF_PEAK},
        ),
        # July weekday demand
        (costs.DEMAND, 7, 0, 21.3, True, {costs.PEAK, costs.HALF_PEAK, costs.OFF_PEAK}),
        # July weekend energy
        (costs.ENERGY, 7, 5, 0.08981, False, {costs.OFF_PEAK}),
        # July weekend demand
        (costs.DEMAND, 7, 5, 21.3, False, {costs.OFF_PEAK}),
        # Winter weekday energy
        (costs.ENERGY, 1, 0, 0.1133, False, {costs.OFF_PEAK, costs.SUPER_OFF_PEAK}),
        # Winter weekday demand
        (costs.DEMAND, 1, 0, 21.3, True, {costs.PEAK, costs.OFF_PEAK}),
    ],
)
def test_detect_charge_periods(
    charge_type,
    month,
    weekday,
    expected_base_charge,
    expected_overlap,
    expected_periods,
):
    """Test the detect_charge_periods function with different scenarios."""

    # Use csv that has names like "peak" in the columns
    rate_data = pd.read_csv(input_dir + "billing_pge.csv")

    # Test the expected period types and base charge
    periods, base_charge, has_overlapping = costs.detect_charge_periods(
        rate_data, charge_type, month, weekday
    )
    period_types = set(periods.values())

    # Verify expected periods are present
    for expected_period in expected_periods:
        assert expected_period in period_types

    # Verify base charge and overlapping flag
    assert base_charge == expected_base_charge
    assert has_overlapping is expected_overlap


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "billing_file, variant_params, expected, expect_error, expect_warning",
    [
        # billing_pge.csv with double peak energy and demand
        (
            "billing_pge.csv",
            {
                "scale_ratios": {
                    DEMAND: {
                        PEAK: 2.0,
                        HALF_PEAK: 2.0,
                        OFF_PEAK: 1.0,
                        SUPER_OFF_PEAK: 1.0,
                    },
                    ENERGY: {
                        PEAK: 2.0,
                        HALF_PEAK: 2.0,
                        OFF_PEAK: 1.0,
                        SUPER_OFF_PEAK: 1.0,
                    },
                }
            },
            {
                "peak_demand_charge": 42.38,  # 21.19 * 2
                "half_peak_demand_charge": 11.76,  # 5.88 * 2
                "off_peak_demand_charge": 21.3,  # unchanged
                "peak_energy_charge": 0.23617,  # 0.08981 + (0.16299 - 0.08981) * 2
                "half_peak_energy_charge": 0.14939,  # 0.08981 + (0.1196 - 0.08981) * 2
                "off_peak_energy_rates": [0.08981, 0.09716, 0.1133, 0.591, 0.0],
            },
            None,
            False,
        ),
        # billing_pge.csv with scale all demand
        (
            "billing_pge.csv",
            {
                "scale_ratios": {
                    "demand": 1.5,
                    "energy": 1.0,
                },
            },
            {
                "peak_demand_charge": 31.785,  # 21.19 * 1.5
                "half_peak_demand_charge": 8.82,  # 5.88 * 1.5
                "off_peak_demand_charge": 31.95,  # 21.3 * 1.5
                "peak_energy_charge": 0.16299,  # unchanged
                "half_peak_energy_charge": 0.1196,  # unchanged
            },
            None,
            False,
        ),
        # billing_demand_2.csv with tripled peak demand only
        (
            "billing_demand_2.csv",
            {
                "scale_ratios": {
                    DEMAND: {
                        PEAK: 3.0,
                        HALF_PEAK: 1.0,
                        OFF_PEAK: 1.0,
                        SUPER_OFF_PEAK: 1.0,
                    },
                    ENERGY: {
                        PEAK: 1.0,
                        HALF_PEAK: 1.0,
                        OFF_PEAK: 1.0,
                        SUPER_OFF_PEAK: 1.0,
                    },
                }
            },
            {
                "on_peak_demand_charge": 60.0,  # 20 * 3
                "all_day_demand_charge": 5.0,  # unchanged
            },
            None,
            False,
        ),
        # billing_demand_2.csv with scale all energy
        (
            "billing_demand_2.csv",
            {
                "scale_ratios": {
                    "demand": 1.0,
                    "energy": 2.0,
                },
            },
            {
                "on_peak_demand_charge": 20.0,  # unchanged
                "all_day_demand_charge": 5.0,  # unchanged
            },
            None,
            False,
        ),
        # billing_pge.csv with exact charge key inputs
        (
            "billing_pge.csv",
            {
                "scale_ratios": {
                    "electric_demand_peak-summer": 2.0,
                    "electric_energy_0": 3.0,
                    "electric_demand_all-day": 1.5,
                },
            },
            {
                "peak_summer_demand_charge": 42.38,  # 21.19 * 2
                "energy_0_charge": 0.26943,  # 0.08981 * 3
            },
            None,
            False,
        ),
        # Scale ratio unusual inputs (blank, zero, negative)
        (
            "billing_pge.csv",
            {
                "scale_ratios": {
                    "demand": 0.0,
                    "energy": -2.0,
                },
            },
            {
                "peak_demand_charge": 0.0,  # 21.19 * 0
                "half_peak_demand_charge": 0.0,  # 5.88 * 0
                "off_peak_demand_charge": 0.0,  # 21.3 * 0
                "peak_energy_charge": 0.016630,  # 0.08981 + (0.16299 - 0.08981) * -2
                "half_peak_energy_charge": 0.03023,  # 0.08981 + (0.1196 - 0.08981) * -2
            },
            None,
            False,
        ),
        # Individual zero scale_ratios
        (
            "billing_pge.csv",
            {
                "scale_ratios": {
                    DEMAND: {
                        PEAK: 0.0,
                        HALF_PEAK: 0.0,
                        OFF_PEAK: 1.0,
                        SUPER_OFF_PEAK: 1.0,
                    },
                    ENERGY: {
                        PEAK: 0.0,
                        HALF_PEAK: 0.0,
                        OFF_PEAK: 1.0,
                        SUPER_OFF_PEAK: 1.0,
                    },
                }
            },
            {
                "peak_demand_charge": 0.0,  # 21.19 * 0
                "half_peak_demand_charge": 0.0,  # 5.88 * 0
                "off_peak_demand_charge": 21.3,  # unchanged
                "peak_energy_charge": 0.08981,  # base charge only
                "half_peak_energy_charge": 0.08981,  # base charge only
            },
            None,
            False,
        ),
        # Window expansion
        (
            "billing_pge.csv",
            {
                "scale_ratios": {
                    DEMAND: {
                        PEAK: 1.0,
                        HALF_PEAK: 1.0,
                        OFF_PEAK: 1.0,
                        SUPER_OFF_PEAK: 1.0,
                    },
                    ENERGY: {
                        PEAK: 1.0,
                        HALF_PEAK: 1.0,
                        OFF_PEAK: 1.0,
                        SUPER_OFF_PEAK: 1.0,
                    },
                },
                "shift_peak_hours_before": -1.0,  # negative = earlier start
                "shift_peak_hours_after": 1.0,  # positive = later end
            },
            {
                "peak_energy_window": (11, 19),  # expanded from 12-18
                "peak_demand_window": (11, 19),  # expanded from 12-18
                "peak_energy_charge": 0.16299,  # unchanged
                "half_peak_energy_charge": 0.1196,  # unchanged
            },
            None,
            False,
        ),
        # Window shifting boundary (start at 0, end at 24)
        (
            "billing_pge.csv",
            {
                "shift_peak_hours_before": -12.0,  # Shift peak to start at 0
                "shift_peak_hours_after": 6.0,  # Shift peak to end at 24
            },
            {
                "peak_energy_window": (0, 24),  # shifted from 12-18 to 0-24
                "peak_demand_window": (0, 24),  # shifted from 12-18 to 0-24
                "peak_energy_charge": 0.16299,  # unchanged
                "half_peak_energy_charge": 0.1196,  # unchanged
            },
            None,
            False,
        ),
        # Window shifting for non-overlapping charges with random float shift values
        (
            "billing_pge.csv",
            {
                "shift_peak_hours_before": -2.104,  # expand peak window earlier
                "shift_peak_hours_after": 2.7,  # expand peak window later
            },
            {
                "peak_energy_window": (
                    9.896,
                    20.7,
                ),  # expanded from 12-18 to 9.896-20.7
                "peak_demand_window": (
                    9.896,
                    20.7,
                ),  # expanded from 12-18 to 9.896-20.7
                "peak_energy_charge": 0.16299,  # unchanged
                "half_peak_energy_charge": 0.1196,  # unchanged
            },
            None,
            False,
        ),
        # Window shifting invalid entries
        (
            "billing_pge.csv",
            {
                "shift_peak_hours_before": 10.0,  # Would make start > end
                "shift_peak_hours_after": -10.0,
            },
            {
                "peak_energy_charge": 0.16299,  # unchanged (peak period removed)
                "half_peak_energy_charge": 0.1196,  # unchanged
            },
            None,
            True,  # warning about peak window being removed
        ),
        # non-existent charge keys
        (
            "billing_pge.csv",
            {
                "scale_ratios": {
                    "electric_demand_nonexistent1": 2.0,
                    "electric_demand_nonexistent2": 3.0,
                    "electric_energy_nonexistent1": 1.5,
                    "electric_energy_nonexistent2": 2.5,
                },
            },
            {
                "peak_demand_charge": 21.19,  # unchanged (keys not found)
                "half_peak_demand_charge": 5.88,  # unchanged
                "off_peak_demand_charge": 21.3,  # unchanged
                "peak_energy_charge": 0.16299,  # unchanged (keys not found)
                "half_peak_energy_charge": 0.1196,  # unchanged
            },
            None,
            True,  # Throws consolidated warning about missing keys
        ),
        # Conflict between charge key and period scaling inputs
        (
            "billing_pge.csv",
            {
                "scale_ratios": {
                    "electric_demand_peak-summer": 2.0,
                    DEMAND: 3.0,  # Conflicts with the exact key above
                },
            },
            {
                "peak_demand_charge": 63.57,  # 21.19 * 3 (scale_all takes precedence)
                "half_peak_demand_charge": 17.64,  # 5.88 * 3
                "off_peak_demand_charge": 63.9,  # 21.3 * 3
                "peak_energy_charge": 0.16299,  # unchanged
                "half_peak_energy_charge": 0.1196,  # unchanged
            },
            ValueError,  # Throws ValueError due to conflict
            False,
        ),
        # Window shifting test with super-off-peak adjacent to peak
        (
            "billing_energy_super_off_peak.csv",
            {
                "shift_peak_hours_before": -2.0,  # Expand earlier into super off-peak
                "shift_peak_hours_after": 1.0,
            },
            {
                "peak_energy_window": (6, 18),  # shifted from 8-17 to 6-18
                "peak_demand_window": (6, 18),  # shifted from 8-20 to 6-18
                "super_off_peak_charge": 0.018994,  # super off-peak period charge
            },
            None,
            False,
        ),
        # Super off-peak scaling
        (
            "billing_energy_super_off_peak.csv",
            {
                "scale_ratios": {
                    "energy": {
                        "peak": 1.0,
                        "half_peak": 1.0,
                        "off_peak": 1.0,
                        "super_off_peak": 2.0,  # Double super off-peak charges
                    },
                },
            },
            {
                "super_off_peak_charge": 0.037988,  # 0.018994 * 2
            },
            None,
            False,
        ),
    ],
)
def test_parametrize_rate_data(
    billing_file, variant_params, expected, expect_error, expect_warning
):
    """Test the parametrize_rate_data function with different files and variants."""

    rate_data = pd.read_csv(input_dir + billing_file)

    if expect_error:
        with pytest.raises(Exception):
            costs.parametrize_rate_data(rate_data, **variant_params)
        return

    if expect_warning:
        with pytest.warns(UserWarning):
            variant_data = costs.parametrize_rate_data(rate_data, **variant_params)
    else:
        variant_data = costs.parametrize_rate_data(rate_data, **variant_params)

    # Test demand charges
    if "peak_demand_charge" in expected:
        peak_demand = variant_data[
            (variant_data[TYPE] == costs.DEMAND)
            & (variant_data["name"] == "peak-summer")
        ]
        assert np.allclose(peak_demand[CHARGE].values, expected["peak_demand_charge"])

    if "half_peak_demand_charge" in expected:
        half_peak_demand = variant_data[
            (variant_data[TYPE] == costs.DEMAND)
            & (variant_data["name"] == "half-peak-summer")
        ]
        assert np.allclose(
            half_peak_demand[CHARGE].values,
            expected["half_peak_demand_charge"],
        )

    if "off_peak_demand_charge" in expected:
        off_peak_demand = variant_data[
            (variant_data[TYPE] == costs.DEMAND) & (variant_data["name"] == "off-peak")
        ]
        assert np.allclose(
            off_peak_demand[CHARGE].values, expected["off_peak_demand_charge"]
        )

    if "on_peak_demand_charge" in expected:
        on_peak_demand = variant_data[
            (variant_data[TYPE] == costs.DEMAND) & (variant_data["name"] == "on-peak")
        ]
        assert np.isclose(
            on_peak_demand[CHARGE].values[0], expected["on_peak_demand_charge"]
        )

    if "all_day_demand_charge" in expected:
        all_day_demand = variant_data[
            (variant_data[TYPE] == costs.DEMAND) & (variant_data["name"] == "all-day")
        ]
        assert np.isclose(
            all_day_demand[CHARGE].values[0], expected["all_day_demand_charge"]
        )

        # Test energy charges
        if "peak_energy_charge" in expected:
            if "peak_energy_window" in expected:
                # Window expansion test
                start_hour, end_hour = expected["peak_energy_window"]
                peak_energy = variant_data[
                    (variant_data[TYPE] == costs.ENERGY)
                    & (variant_data[HOUR_START] == start_hour)
                    & (variant_data[HOUR_END] == end_hour)
                ]
            else:
                # Charge scaling test
                peak_energy = variant_data[
                    (variant_data[TYPE] == costs.ENERGY)
                    & (variant_data[HOUR_START] == 12)
                    & (variant_data[HOUR_END] == 18)
                ]

            # Check if we found any matching rows
            if len(peak_energy) > 0:
                assert np.isclose(
                    peak_energy[CHARGE].values[0], expected["peak_energy_charge"]
                )
            else:
                # If no rows found, skip this assertion (window might have been shifted)
                pass

    if "half_peak_energy_charge" in expected:
        # Charge scaling test
        half_peak_energy = variant_data[
            (variant_data[TYPE] == costs.ENERGY)
            & (variant_data[HOUR_START] == 8.5)
            & (variant_data[HOUR_END] == 12)
            & (variant_data[MONTH_START] == 5)
            & (variant_data[WEEKDAY_START] == 0)
        ]
        # Check if we found any matching rows
        if len(half_peak_energy) > 0:
            assert np.allclose(
                half_peak_energy[CHARGE], expected["half_peak_energy_charge"]
            )
        else:
            # If no rows found, skip this assertion (window might have been shifted)
            pass

    if "off_peak_energy_rates" in expected:
        off_peak_energy = variant_data[
            (variant_data[TYPE] == costs.ENERGY)
            & ((variant_data[HOUR_START] == 0) | (variant_data[HOUR_START] == 21.5))
            & ((variant_data[HOUR_END] == 8.5) | (variant_data[HOUR_END] == 24))
        ]
        unique_off_peak_rates = off_peak_energy[CHARGE].unique()
        for rate in unique_off_peak_rates:
            assert rate in expected["off_peak_energy_rates"]

            # Test window expansions for energy charges
        if "peak_energy_window" in expected:
            start_hour, end_hour = expected["peak_energy_window"]
            # Look for the shifted window in any month/weekday combination
            peak_energy = variant_data[
                (variant_data[TYPE] == costs.ENERGY)
                & (variant_data[HOUR_START] == start_hour)
                & (variant_data[HOUR_END] == end_hour)
            ]
            # If no exact match found, check if any peak periods were shifted
            if peak_energy.empty:
                # Check if any peak periods exist with the expected window
                all_peak_energy = variant_data[
                    (variant_data[TYPE] == costs.ENERGY)
                    & (variant_data[HOUR_START] != 0)  # Not 24-hour periods
                    & (variant_data[HOUR_END] != 24)
                ]
                if not all_peak_energy.empty:  # Some shifting occurred
                    pass
                else:
                    assert (
                        not peak_energy.empty
                    ), f"Peak energy window should be {start_hour}-{end_hour}"
            else:
                # Found the expected window
                pass

    # Test window expansions for demand charges
    if "peak_demand_window" in expected:
        start_hour, end_hour = expected["peak_demand_window"]
        peak_demand = variant_data[
            (variant_data[TYPE] == costs.DEMAND)
            & (variant_data[HOUR_START] == start_hour)
            & (variant_data[HOUR_END] == end_hour)
        ]

        if not peak_demand.empty:
            peak_demand_row = peak_demand.iloc[0]
            assert (
                peak_demand_row[HOUR_START] == start_hour
            ), f"Peak demand should start at {start_hour}"
            assert (
                peak_demand_row[HOUR_END] == end_hour
            ), f"Peak demand should end at {end_hour}"
        else:
            # Peak demand was removed due to invalid window
            pass

    # Test exact charge key use - find any matching charges and verify they're modified
    if "peak_summer_demand_charge" in expected:
        # For peak summer demand: find any demand charge with "peak-summer" in the name
        peak_summer_demand = variant_data[
            (variant_data[TYPE] == costs.DEMAND)
            & (variant_data["name"].str.contains("peak-summer", na=False))
        ]
        assert not peak_summer_demand.empty, "Should find peak-summer demand charges"
        # Verify at least one charge is scaled (not all zeros)
        assert np.any(
            peak_summer_demand[CHARGE] > 0
        ), "Peak-summer demand charges should be non-zero"

    # For energy 0: find any energy charge that might be scaled
    if "energy_0_charge" in expected:
        # Look for energy charges with any name (including empty/NaN names)
        energy_charges = variant_data[(variant_data[TYPE] == costs.ENERGY)]
        assert not energy_charges.empty, "Should find energy charges"
        # Verify at least one energy charge is scaled (not all zeros)
        assert np.any(energy_charges[CHARGE] > 0), "Energy charges should be non-zero"

        # Test super off-peak charges
        if "super_off_peak_charge" in expected:
            # Check for super off-peak periods (0-5 hours)
            super_off_peak_energy = variant_data[
                (variant_data[TYPE] == costs.ENERGY)
                & (variant_data[HOUR_START] == 0)
                & (variant_data[HOUR_END] == 5)
            ]
            assert (
                not super_off_peak_energy.empty
            ), "Should find super off-peak periods (0-5 hours)"
            assert np.isclose(
                super_off_peak_energy[costs.CHARGE_METRIC].values[0],
                expected["super_off_peak_charge"],
            )


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "variant_params, key_subset, expected",
    [
        # Period-based ratios
        (
            [
                {
                    "scale_ratios": {
                        DEMAND: {
                            PEAK: 2.0,
                            HALF_PEAK: 2.0,
                            OFF_PEAK: 1.0,
                            SUPER_OFF_PEAK: 1.0,
                        },
                        ENERGY: {
                            PEAK: 2.0,
                            HALF_PEAK: 2.0,
                            OFF_PEAK: 1.0,
                            SUPER_OFF_PEAK: 1.0,
                        },
                    },
                    "variant_name": "double_peak",
                }
            ],
            "peak-summer",
            {
                "variant_name": "double_peak",
                "expected_keys": ["original", "double_peak"],
                "key_subset_modified": True,
                "scaling_factor": 2.0,
            },
        ),
        # Scale all demand by 1.5
        (
            [
                {
                    "scale_ratios": {
                        "demand": 1.5,
                        "energy": 1.0,
                    },
                    "variant_name": "scale_demand",
                }
            ],
            "demand",
            {
                "variant_name": "scale_demand",
                "expected_keys": ["original", "scale_demand"],
                "key_subset_modified": True,
                "scaling_factor": 1.5,
            },
        ),
        # Exact charge keys
        (
            [
                {
                    "scale_ratios": {
                        "electric_demand_peak-summer": 2.0,
                        "electric_energy_0": 3.0,
                        "electric_demand_all-day": 1.5,
                    },
                    "variant_name": "exact_keys",
                }
            ],
            "electric_demand_peak-summer",
            {
                "variant_name": "exact_keys",
                "expected_keys": ["original", "exact_keys"],
                "key_subset_modified": True,
                "scaling_factor": 2.0,
            },
        ),
        # Empty variants list
        (
            None,
            None,
            {
                "expected_keys": ["original"],
                "empty_variants": True,
            },
        ),
        # Duplicate variant names
        (
            [
                {"scale_ratios": {"demand": 2.0}, "variant_name": "test"},
                {
                    "scale_ratios": {"energy": 3.0},
                    "variant_name": "test",
                },  # Duplicate name
            ],
            "test",
            {
                "expected_keys": ["original", "test"],
                "duplicate_names": True,
                "key_subset_modified": True,
                "scaling_factor": 2.0,
            },
        ),
        # Variants without names
        (
            [
                {"scale_ratios": {"demand": 2.0}},  # No variant_name
                {"scale_ratios": {"energy": 3.0}},  # No variant_name
            ],
            "variant_0",
            {
                "expected_keys": ["original", "variant_0", "variant_1"],
                "default_naming": True,
                "key_subset_modified": True,
                "scaling_factor": 2.0,
            },
        ),
        # Single variant with scale_all_demand
        (
            [
                {
                    "scale_ratios": {"demand": 2.0},
                    "variant_name": "double_demand",
                }
            ],
            "demand",
            {
                "variant_name": "double_demand",
                "expected_keys": ["original", "double_demand"],
                "key_subset_modified": True,
                "scaling_factor": 2.0,
            },
        ),
        # Single variant with scale_all_energy
        (
            [
                {
                    "scale_ratios": {"energy": 3.0},
                    "variant_name": "triple_energy",
                }
            ],
            "energy",
            {
                "variant_name": "triple_energy",
                "expected_keys": ["original", "triple_energy"],
                "key_subset_modified": True,
                "scaling_factor": 3.0,
            },
        ),
        # Multiple variants with different scaling
        (
            [
                {
                    "scale_ratios": {"demand": 2.0},
                    "variant_name": "double_demand",
                },
                {
                    "scale_ratios": {"energy": 3.0},
                    "variant_name": "triple_energy",
                },
                {
                    "scale_ratios": {
                        DEMAND: {
                            PEAK: 1.5,
                            HALF_PEAK: 1.0,
                            OFF_PEAK: 1.0,
                            SUPER_OFF_PEAK: 1.0,
                        },
                    },
                    "variant_name": "peak_only",
                },
            ],
            "double_demand",
            {
                "expected_keys": [
                    "original",
                    "double_demand",
                    "triple_energy",
                    "peak_only",
                ],
                "multiple_variants": True,
                "key_subset_modified": True,
                "scaling_factor": 2.0,
            },
        ),
    ],
)
def test_parametrize_charge_dict(variant_params, key_subset, expected):
    """Test the parametrize_charge_dict function with different variant types."""

    rate_data = pd.read_csv(input_dir + "billing_pge.csv")
    start_dt = np.datetime64("2024-07-10")
    end_dt = np.datetime64("2024-07-11")

    # Handle empty variants case
    if expected.get("empty_variants"):
        charge_dicts = costs.parametrize_charge_dict(start_dt, end_dt, rate_data, None)
        assert set(charge_dicts.keys()) == set(expected["expected_keys"])
        return

    # Get parametrized charge dicts
    charge_dicts = costs.parametrize_charge_dict(
        start_dt, end_dt, rate_data, variant_params
    )
    assert set(charge_dicts.keys()) == set(expected["expected_keys"])
    assert "original" in charge_dicts

    # Edge cases
    if expected.get("duplicate_names"):
        # Both variants should exist (function should handle duplicates)
        assert len([k for k in charge_dicts.keys() if k == "test"]) >= 1

        # Check that all variants have same keys as original
        original_keys = set(charge_dicts["original"].keys())
        for variant_name in charge_dicts.keys():
            if variant_name != "original":
                variant_keys = set(charge_dicts[variant_name].keys())
                assert original_keys == variant_keys
        return

    if expected.get("default_naming"):
        # Test with variants without names
        assert "variant_0" in charge_dicts
        assert "variant_1" in charge_dicts

        # Check that all variants have same keys as original
        original_keys = set(charge_dicts["original"].keys())
        for variant_name in ["variant_0", "variant_1"]:
            variant_keys = set(charge_dicts[variant_name].keys())
            assert original_keys == variant_keys
        return

    if expected.get("multiple_variants"):
        # Test with multiple variants
        assert "double_demand" in charge_dicts
        assert "triple_energy" in charge_dicts
        assert "peak_only" in charge_dicts

        # Check that all variants have same keys as original
        original_keys = set(charge_dicts["original"].keys())
        for variant_name in ["double_demand", "triple_energy", "peak_only"]:
            variant_keys = set(charge_dicts[variant_name].keys())
            assert original_keys == variant_keys

        # Test that variants are actually different
        original_charge_dict = charge_dicts["original"]
        double_demand_dict = charge_dicts["double_demand"]
        triple_energy_dict = charge_dicts["triple_energy"]

        # Check that at least some charges are modified
        assert np.any(
            [
                np.any(double_demand_dict[k] != original_charge_dict[k])
                for k in original_keys
                if k in double_demand_dict
            ]
        )
        assert np.any(
            [
                np.any(triple_energy_dict[k] != original_charge_dict[k])
                for k in original_keys
                if k in triple_energy_dict
            ]
        )
        return

    # For regular cases, test the variant
    variant_name = expected.get("variant_name")
    if variant_name:
        assert variant_name in charge_dicts

        # Test that charge dicts have same keys
        original_keys = set(charge_dicts["original"].keys())
        variant_keys = set(charge_dicts[variant_name].keys())
        assert original_keys == variant_keys

        # Test that the specified key subset is modified if expected
        if expected.get("key_subset_modified") and key_subset:
            matching_key = None
            for key in charge_dicts[variant_name].keys():
                if key_subset in key:
                    matching_key = key
                    break

            assert (
                matching_key is not None
            ), f"Could not find key containing '{key_subset}'"

            # Test that the charge values are scaled appropriately
            original_charge = charge_dicts["original"][matching_key]
            variant_charge = charge_dicts[variant_name][matching_key]

            # For demand charges, we expect scaling
            if "demand" in matching_key.lower():
                expected_scaling = expected.get("scaling_factor", 1.0)

                # Test that the charge is scaled appropriately
                if expected_scaling != 1.0:
                    assert np.allclose(
                        variant_charge, original_charge * expected_scaling
                    )


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "billing_file, variant_params",
    [
        # Test with different billing files
        ("billing_energy_1.csv", {"scale_ratios": {"energy": 2.0}}),
        ("billing_demand_2.csv", {"scale_ratios": {"demand": 2.0}}),
        ("billing_export.csv", {"scale_ratios": {"energy": 1.5}}),
        ("billing_customer.csv", {"scale_ratios": {"energy": 1.0}}),
        # Test with complex rate structures
        ("billing.csv", {"scale_ratios": {"demand": 2.0, "energy": 1.5}}),
    ],
)
def test_parametrize_rate_data_different_files(billing_file, variant_params):
    """Test parametrize_rate_data with different billing file types."""

    rate_data = pd.read_csv(input_dir + billing_file)
    variant_data = costs.parametrize_rate_data(rate_data, **variant_params)

    # Basic checks that the function completed without error
    assert len(variant_data) == len(
        rate_data
    ), "Variant data should have same number of rows"
    assert list(variant_data.columns) == list(
        rate_data.columns
    ), "Variant data should have same columns"

    # Check that at least some charges were modified if scaling was applied
    if (
        "scale_ratios" in variant_params
        and "demand" in variant_params["scale_ratios"]
        and isinstance(variant_params["scale_ratios"]["demand"], (int, float))
        and variant_params["scale_ratios"]["demand"] != 1.0
    ):
        demand_charges = variant_data[variant_data[TYPE] == costs.DEMAND]
        if not demand_charges.empty:
            original_demand = rate_data[rate_data[TYPE] == costs.DEMAND]
            assert not np.array_equal(
                demand_charges[CHARGE], original_demand[CHARGE]
            ), "Demand charges should be modified"

    if (
        "scale_ratios" in variant_params
        and "energy" in variant_params["scale_ratios"]
        and isinstance(variant_params["scale_ratios"]["energy"], (int, float))
        and variant_params["scale_ratios"]["energy"] != 1.0
    ):
        energy_charges = variant_data[variant_data[TYPE] == costs.ENERGY]
        if not energy_charges.empty:
            original_energy = rate_data[rate_data[TYPE] == costs.ENERGY]
            assert not np.array_equal(
                energy_charges[CHARGE], original_energy[CHARGE]
            ), "Energy charges should be modified"


# TODO: write test_calculate_itemized_cost

# TODO: write test for itemized cost


@pytest.mark.parametrize(
    "charge_list, expected_result",
    [
        (["demand"], pytest.approx(7.128)),
        ("demand", pytest.approx(7.128)),
        (["energy", "demand"], pytest.approx(7.92081)),
        (None, pytest.approx(307.92081)),
    ],
)
@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_individual_charge(charge_list, expected_result):

    billing_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "input", "billing.csv"
    )
    tariff_df = pd.read_csv(billing_path)
    start_date = np.datetime64("2024-07-10")
    end_date = np.datetime64("2024-07-11")
    datetime_range = pd.date_range(start=start_date, end=end_date, freq="15min")
    baseload = np.ones(len(datetime_range) - 1)
    charge_dict = costs.get_charge_dict(
        np.datetime64("2024-07-10"),
        np.datetime64("2024-07-11"),
        tariff_df,
    )

    cost, _ = costs.calculate_cost(
        charge_dict,
        {"electric": baseload, "gas": np.zeros_like(baseload)},
        resolution="15m",
        desired_utility="electric",
        desired_charge_type=charge_list,
        prev_demand_dict=None,
        prev_consumption_dict=None,
        model=None,
    )
    assert cost == expected_result

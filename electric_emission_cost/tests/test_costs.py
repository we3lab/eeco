import os
import pytest
import numpy as np
import cvxpy as cp
import pandas as pd
import pyomo.environ as pyo

from electric_emission_cost import costs
from electric_emission_cost.costs import (
    CHARGE,
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
        )
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
    model.T = len(consumption_data_dict[ELECTRIC])
    model.t = range(model.T)
    pyo_vars = {}
    for key, val in consumption_data_dict.items():
        var = pyo.Var(range(len(val)), initialize=np.zeros(len(val)), bounds=(0, None))
        model.add_component(key, var)
        pyo_vars[key] = var

    @model.Constraint(model.t)
    def electric_constraint(m, t):
        return consumption_data_dict[ELECTRIC][t] == m.electric[t]

    @model.Constraint(model.t)
    def gas_constraint(m, t):
        return consumption_data_dict[GAS][t] == m.gas[t]

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


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_detect_charge_periods():
    """Test the detect_charge_periods function."""

    # Use csv that has names like "peak" in the columns
    rate_data = pd.read_csv(input_dir + "billing_pge.csv")

    # Test the expected energy period types for July weekday
    energy_periods = costs.detect_charge_periods(rate_data, costs.ENERGY, 7, 0)
    period_types = set(energy_periods.values())
    assert costs.PEAK in period_types
    assert costs.HALF_PEAK in period_types
    assert costs.OFF_PEAK in period_types

    # Test the expected demand period types for July weekday
    demand_periods = costs.detect_charge_periods(rate_data, costs.DEMAND, 7, 0)
    period_types = set(demand_periods.values())
    assert costs.PEAK in period_types
    assert costs.HALF_PEAK in period_types
    assert costs.OFF_PEAK in period_types


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "billing_file, variant_params, expected_checks",
    [
        # Test 1: billing_pge.csv with double peak charges
        (
            "billing_pge.csv",
            {
                "individual_ratios": {
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
        ),
        # Test 2: billing_pge.csv with scale all demand
        (
            "billing_pge.csv",
            {
                "scale_all_demand": 1.5,
                "scale_all_energy": 1.0,
            },
            {
                "peak_demand_charge": 31.785,  # 21.19 * 1.5
                "half_peak_demand_charge": 8.82,  # 5.88 * 1.5
                "off_peak_demand_charge": 31.95,  # 21.3 * 1.5
                "peak_energy_charge": 0.16299,  # unchanged
                "half_peak_energy_charge": 0.1196,  # unchanged
            },
        ),
        # Test 3: billing_demand_2.csv with tripled peak demand
        (
            "billing_demand_2.csv",
            {
                "individual_ratios": {
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
        ),
        # Test 4: billing_demand_2.csv with scale all energy
        (
            "billing_demand_2.csv",
            {
                "scale_all_demand": 1.0,
                "scale_all_energy": 2.0,
            },
            {
                "on_peak_demand_charge": 20.0,  # unchanged
                "all_day_demand_charge": 5.0,  # unchanged
            },
        ),
        # Test 5: billing_pge.csv with peak window expansion
        (
            "billing_pge.csv",
            {
                "individual_ratios": {
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
        ),
    ],
)
def test_parametrize_rate_data(billing_file, variant_params, expected_checks):
    """Test the parametrize_rate_data function with different files and variants."""

    rate_data = pd.read_csv(input_dir + billing_file)
    variant_data = costs.parametrize_rate_data(rate_data, **variant_params)

    # Test demand charges
    if "peak_demand_charge" in expected_checks:
        peak_demand = variant_data[
            (variant_data["type"] == costs.DEMAND)
            & (variant_data["name"] == "peak-summer")
        ]
        assert np.allclose(
            peak_demand[CHARGE].values, expected_checks["peak_demand_charge"]
        )

    if "half_peak_demand_charge" in expected_checks:
        half_peak_demand = variant_data[
            (variant_data["type"] == costs.DEMAND)
            & (variant_data["name"] == "half-peak-summer")
        ]
        assert np.allclose(
            half_peak_demand[CHARGE].values,
            expected_checks["half_peak_demand_charge"],
        )

    if "off_peak_demand_charge" in expected_checks:
        off_peak_demand = variant_data[
            (variant_data["type"] == costs.DEMAND)
            & (variant_data["name"] == "off-peak")
        ]
        assert np.allclose(
            off_peak_demand[CHARGE].values, expected_checks["off_peak_demand_charge"]
        )

    if "on_peak_demand_charge" in expected_checks:
        on_peak_demand = variant_data[
            (variant_data["type"] == costs.DEMAND) & (variant_data["name"] == "on-peak")
        ]
        assert np.isclose(
            on_peak_demand[CHARGE].values[0], expected_checks["on_peak_demand_charge"]
        )

    if "all_day_demand_charge" in expected_checks:
        all_day_demand = variant_data[
            (variant_data["type"] == costs.DEMAND) & (variant_data["name"] == "all-day")
        ]
        assert np.isclose(
            all_day_demand[CHARGE].values[0], expected_checks["all_day_demand_charge"]
        )

    # Test energy charges
    if "peak_energy_charge" in expected_checks:
        if "peak_energy_window" in expected_checks:
            # Window expansion test
            start_hour, end_hour = expected_checks["peak_energy_window"]
            peak_energy = variant_data[
                (variant_data["type"] == costs.ENERGY)
                & (variant_data[HOUR_START] == start_hour)
                & (variant_data[HOUR_END] == end_hour)
            ]
        else:
            # Charge scaling test
            peak_energy = variant_data[
                (variant_data["type"] == costs.ENERGY)
                & (variant_data[HOUR_START] == 12)
                & (variant_data[HOUR_END] == 18)
            ]
        assert np.isclose(
            peak_energy[CHARGE].values[0], expected_checks["peak_energy_charge"]
        )

    if "half_peak_energy_charge" in expected_checks:
        # Charge scaling test
        half_peak_energy = variant_data[
            (variant_data["type"] == costs.ENERGY)
            & (variant_data[HOUR_START] == 8.5)
            & (variant_data[HOUR_END] == 12)
            & (variant_data[MONTH_START] == 5)
            & (variant_data[WEEKDAY_START] == 0)
        ]
        assert np.allclose(
            half_peak_energy[CHARGE], expected_checks["half_peak_energy_charge"]
        )

    if "off_peak_energy_rates" in expected_checks:
        off_peak_energy = variant_data[
            (variant_data["type"] == costs.ENERGY)
            & ((variant_data[HOUR_START] == 0) | (variant_data[HOUR_START] == 21.5))
            & ((variant_data[HOUR_END] == 8.5) | (variant_data[HOUR_END] == 24))
        ]
        unique_off_peak_rates = off_peak_energy[CHARGE].unique()
        for rate in unique_off_peak_rates:
            assert rate in expected_checks["off_peak_energy_rates"]

    # Test window expansions for energy charges
    if "peak_energy_window" in expected_checks:
        start_hour, end_hour = expected_checks["peak_energy_window"]
        peak_energy = variant_data[
            (variant_data["type"] == costs.ENERGY)
            & (variant_data[HOUR_START] == start_hour)
            & (variant_data[HOUR_END] == end_hour)
            & (variant_data[MONTH_START] == 5)
            & (variant_data[WEEKDAY_START] == 0)
        ]
        assert (
            not peak_energy.empty
        ), f"Peak energy window should be {start_hour}-{end_hour}"

    # Test window expansions for demand charges
    if "peak_demand_window" in expected_checks:
        start_hour, end_hour = expected_checks["peak_demand_window"]
        peak_demand = variant_data[
            (variant_data["type"] == costs.DEMAND)
            & (variant_data["name"] == "peak-summer")
        ]
        assert not peak_demand.empty, "Peak demand should exist"
        peak_demand_row = peak_demand.iloc[0]
        assert (
            peak_demand_row[HOUR_START] == start_hour
        ), f"Peak demand should start at {start_hour}"
        assert (
            peak_demand_row[HOUR_END] == end_hour
        ), f"Peak demand should end at {end_hour}"


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_parametrize_charge_dict():
    """Test the parametrize_charge_dict function with multiple variants."""

    rate_data = pd.read_csv(input_dir + "billing_pge.csv")
    start_dt = np.datetime64("2024-07-10")
    end_dt = np.datetime64("2024-07-11")

    # Define test variants
    variants = [
        # Double peak charges
        {
            "individual_ratios": {
                DEMAND: {PEAK: 2.0, HALF_PEAK: 2.0, OFF_PEAK: 1.0, SUPER_OFF_PEAK: 1.0},
                ENERGY: {PEAK: 2.0, HALF_PEAK: 2.0, OFF_PEAK: 1.0, SUPER_OFF_PEAK: 1.0},
            },
            "variant_name": "double_peak",
        },
        # Scale all demand by 1.5
        {
            "scale_all_demand": 1.5,
            "scale_all_energy": 1.0,
            "variant_name": "scale_demand",
        },
    ]

    # Get parametrized charge dicts
    charge_dicts = costs.parametrize_charge_dict(start_dt, end_dt, rate_data, variants)

    # Test that we have original + 2 variants
    assert "original" in charge_dicts
    assert "double_peak" in charge_dicts
    assert "scale_demand" in charge_dicts

    # Test that charge dicts have different structures
    original_keys = set(charge_dicts["original"].keys())
    double_peak_keys = set(charge_dicts["double_peak"].keys())
    scale_demand_keys = set(charge_dicts["scale_demand"].keys())

    # All should have the same keys (same time periods)
    assert original_keys == double_peak_keys
    assert original_keys == scale_demand_keys

    # Test that charges are different between variants
    # Find a peak demand charge key
    peak_demand_key = None
    for key in original_keys:
        if "peak-summer" in key and "demand" in key:
            peak_demand_key = key
            break

    assert peak_demand_key is not None, "Should find peak demand charge"

    # Original and scale_demand should have different charge values
    # Double_peak should have doubled charge values
    original_charge = charge_dicts["original"][peak_demand_key]
    double_peak_charge = charge_dicts["double_peak"][peak_demand_key]
    scale_demand_charge = charge_dicts["scale_demand"][peak_demand_key]

    # Check that variants have different charges where they apply
    assert np.any(
        double_peak_charge != original_charge
    ), "Double peak variant should have different charges"

    assert np.any(
        scale_demand_charge != original_charge
    ), "Scale demand variant should have different charges"


# TODO: write test_calculate_itemized_cost

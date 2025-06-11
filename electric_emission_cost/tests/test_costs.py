import os
import pytest
import numpy as np
import cvxpy as cp
import pandas as pd
import pyomo.environ as pyo

from electric_emission_cost import costs

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
                "charge": 0.05,
                "month_start": 1,
                "month_end": 12,
                "weekday_start": 0,
                "weekday_end": 6,
                "hour_start": 0,
                "hour_end": 24,
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
                "charge": 0.05,
                "month_start": 1,
                "month_end": 12,
                "weekday_start": 0,
                "weekday_end": 6,
                "hour_start": 0,
                "hour_end": 24,
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
                "charge": 0.05,
                "month_start": 1,
                "month_end": 12,
                "weekday_start": 0,
                "weekday_end": 6,
                "hour_start": 0,
                "hour_end": 24,
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
                "charge": 0.05,
                "month_start": 1,
                "month_end": 12,
                "weekday_start": 0,
                "weekday_end": 6,
                "hour_start": 0,
                "hour_end": 24,
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
                "electric_energy_0_2024-07-10_2024-07-10_0": np.ones(96) * 0.05,
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
                "electric_energy_0_2024-07-10_2024-07-10_0": np.ones(288) * 0.05,
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
                "electric_energy_0_2024-07-10_2024-07-10_0": np.ones(24) * 0.05,
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
                "electric_energy_0_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.zeros(32),
                    ]
                ),
                "electric_energy_1_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 0.1,
                        np.zeros(12),
                    ]
                ),
                "electric_energy_2_2024-07-10_2024-07-10_0": np.concatenate(
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
                "electric_energy_off-peak_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.zeros(20),
                        np.ones(12) * 0.05,
                    ]
                ),
                "electric_energy_on-peak_2024-07-10_2024-07-10_0": np.concatenate(
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
                "electric_energy_all-day_2024-07-10_2024-07-10_0": np.concatenate(
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
                "electric_demand_all-day_2024-07-10_2024-07-10_0": np.ones(96) * 5,
                "electric_demand_on-peak_2024-07-10_2024-07-10_0": np.concatenate(
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
                "electric_demand_all-day_2024-07-10_2024-07-10_0": np.ones(96) * 5,
                "electric_demand_on-peak_2024-07-10_2024-07-10_0": np.concatenate(
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
                "electric_demand_all-day_2024-07-10_2024-07-10_0": np.ones(96) * 5,
                "electric_demand_on-peak_2024-07-10_2024-07-10_0": np.concatenate(
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
                "electric_demand_all-day_2024-07-10_2024-07-11_0": np.ones(192) * 5,
                "electric_demand_on-peak_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 20,
                        np.zeros(108),
                    ]
                ),
                "electric_demand_on-peak_2024-07-11_2024-07-11_0": np.concatenate(
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
                "electric_export_0_2024-07-10_2024-07-11_0": np.ones(192) * 0.025,
            },
        ),
        # customer payments for any number of days
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-08-01"),  # Summer weekdays
            input_dir + "billing_customer.csv",
            "15m",
            {
                "electric_customer_0_2024-07-10_2024-07-31_0": np.array([1000]),
            },
        ),
        # effective start/end dates
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-08-01"),  # Summer weekdays
            input_dir + "billing_effective.csv",
            "1h",
            {
                "electric_energy_0_2024-01-01_2024-07-20_0": np.concatenate(
                    [
                        np.ones(264) * 0.05,
                        np.zeros(264),
                    ]
                ),
                "electric_energy_0_2024-07-21_2024-12-31_0": np.concatenate(
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
                "electric_customer_0_2024-05-31_2024-06-01_0": np.array([300]),
                "electric_energy_0_2024-05-31_2024-06-01_0": np.concatenate(
                    [np.ones(24) * 0.019934, np.zeros(24)]
                ),
                "electric_energy_1_2024-05-31_2024-06-01_0": np.zeros(48),
                "electric_energy_2_2024-05-31_2024-06-01_0": np.zeros(48),
                "electric_energy_3_2024-05-31_2024-06-01_0": np.zeros(48),
                "electric_energy_4_2024-05-31_2024-06-01_0": np.concatenate(
                    [
                        np.zeros(24),
                        np.ones(24) * 0.021062,
                    ]
                ),
                "electric_energy_5_2024-05-31_2024-06-01_0": np.zeros(48),
                "electric_energy_6_2024-05-31_2024-06-01_0": np.zeros(48),
                "electric_demand_maximum_2024-05-31_2024-06-01_0": np.ones(48) * 7.128,
                "gas_customer_0_2024-05-31_2024-06-01_0": np.array([93.14]),
                "gas_energy_0_2024-05-31_2024-06-01_0": np.ones(48) * 0.2837,
                "gas_energy_1_2024-05-31_2024-06-01_0": np.zeros(48),
            },
        ),
        # switch between years
        (
            np.datetime64("2023-12-31"),  # Summer weekdays
            np.datetime64("2024-01-02"),  # Summer weekdays
            input_dir + "billing.csv",
            "1h",
            {
                "electric_customer_0_2023-12-31_2024-01-01_0": np.array([300]),
                "electric_energy_0_2023-12-31_2024-01-01_0": np.concatenate(
                    [
                        np.zeros(24),
                        np.ones(24) * 0.019934,
                    ]
                ),
                "electric_energy_1_2023-12-31_2024-01-01_0": np.zeros(48),
                "electric_energy_2_2023-12-31_2024-01-01_0": np.zeros(48),
                "electric_energy_3_2023-12-31_2024-01-01_0": np.zeros(48),
                "electric_energy_4_2023-12-31_2024-01-01_0": np.zeros(48),
                "electric_energy_5_2023-12-31_2024-01-01_0": np.zeros(48),
                "electric_energy_6_2023-12-31_2024-01-01_0": np.concatenate(
                    [np.ones(24) * 0.022552, np.zeros(24)]
                ),
                "electric_demand_maximum_2023-12-31_2024-01-01_0": np.ones(48) * 7.128,
                "gas_customer_0_2023-12-31_2024-01-01_0": np.array([93.14]),
                "gas_energy_0_2023-12-31_2024-01-01_0": np.concatenate(
                    [
                        np.zeros(24),
                        np.ones(24) * 0.2837,
                    ]
                ),
                "gas_energy_1_2023-12-31_2024-01-01_0": np.concatenate(
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
                "electric_energy_0_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.ones(64) * 0.05,
                        np.zeros(32),
                    ]
                ),
                "electric_energy_0_2024-07-10_2024-07-10_100": np.concatenate(
                    [
                        np.ones(64) * 0.1,
                        np.zeros(32),
                    ]
                ),
                "electric_energy_1_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 0.1,
                        np.zeros(12),
                    ]
                ),
                "electric_energy_1_2024-07-10_2024-07-10_100": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 0.15,
                        np.zeros(12),
                    ]
                ),
                "electric_energy_2_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.zeros(84),
                        np.ones(12) * 0.05,
                    ]
                ),
                "electric_energy_2_2024-07-10_2024-07-10_100": np.concatenate(
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
        ),
        # 2 demand charges with 100 kW charge limits, all-day and on-peak
        (
            np.datetime64("2024-07-10"),  # Summer weekdays
            np.datetime64("2024-07-11"),  # Summer weekdays
            input_dir + "billing_demand_2_charge_limit.csv",
            "15m",
            {
                "electric_demand_all-day_2024-07-10_2024-07-10_0": np.ones(96) * 5,
                "electric_demand_all-day_2024-07-10_2024-07-10_100": np.ones(96) * 10,
                "electric_demand_on-peak_2024-07-10_2024-07-10_0": np.concatenate(
                    [
                        np.zeros(64),
                        np.ones(20) * 20,
                        np.zeros(12),
                    ]
                ),
                "electric_demand_on-peak_2024-07-10_2024-07-10_100": np.concatenate(
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
    rate_df = pd.read_csv(billing_path)
    result = costs.get_charge_dict(start_dt, end_dt, rate_df, resolution=resolution)
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
            {"electric": np.ones(96), "gas": np.ones(96)},
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
            {"electric": np.arange(96), "gas": np.ones(96)},
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
            {"electric": np.ones(96) * 100, "gas": np.ones(96)},
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
            {"electric": np.ones(96) * 100, "gas": np.ones(96)},
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
            {"electric": np.ones(96) * 100, "gas": np.ones(96)},
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
            {"electric": np.arange(96), "gas": np.arange(96)},
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
            {"electric": np.arange(96), "gas": np.arange(96)},
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
    assert model == model


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "start_dt, end_dt, billing_data, utility, consumption_data_dict, "
    "prev_demand_dict, consumption_estimate, expected",
    [
        (
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            input_dir + "billing_pge.csv",
            "electric",
            {"electric": np.arange(96), "gas": np.arange(96)},
            None,
            0,
            np.float64(4027.79),
        ),
        (
            np.datetime64("2024-07-13"),  # Summer weekend
            np.datetime64("2024-07-14"),  # Summer weekend
            input_dir + "billing_pge.csv",
            "electric",
            {"electric": np.arange(96), "gas": np.arange(96)},
            None,
            0,
            np.float64(2023.5),
        ),
        (
            np.datetime64("2024-03-07"),  # Winter weekday
            np.datetime64("2024-03-08"),  # Winter weekday
            input_dir + "billing_pge.csv",
            "electric",
            {"electric": np.arange(96), "gas": np.arange(96)},
            None,
            0,
            np.float64(2028.6),
        ),
        (
            np.datetime64("2024-03-09"),  # Winter weekend
            np.datetime64("2024-03-10"),  # Winter weekend
            input_dir + "billing_pge.csv",
            "electric",
            {"electric": np.arange(96), "gas": np.arange(96)},
            {
                "electric_demand_peak-summer_2024-03-09_2024-03-09_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_half-peak-summer_2024-03-09_2024-03-09_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_off-peak_2024-03-09_2024-03-09_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_half-peak-winter1_2024-03-09_2024-03-09_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_half-peak-winter2_2024-03-09_2024-03-09_0": {
                    "demand": 0,
                    "cost": 0,
                },
            },
            0,
            np.float64(2023.5),
        ),
        (
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            input_dir + "billing_pge.csv",
            "electric",
            {"electric": np.arange(96), "gas": np.arange(96)},
            {
                "electric_demand_peak-summer_2024-07-10_2024-07-10_0": {
                    "demand": 7.078810759792355,
                    "cost": 150,
                },
                "electric_demand_half-peak-summer_2024-07-10_2024-07-10_0": {
                    "demand": 13.605442176870748,
                    "cost": 80,
                },
                "electric_demand_off-peak_2024-07-10_2024-07-10_0": {
                    "demand": 42.253521126760563,
                    "cost": 900,
                },
                "electric_demand_half-peak-winter1_2024-07-10_2024-07-10_0": {
                    "demand": 0,
                    "cost": 0,
                },
                "electric_demand_half-peak-winter2_2024-07-10_2024-07-10_0": {
                    "demand": 0,
                    "cost": 0,
                },
            },
            0,
            np.float64(2897.79),
        ),
        (
            np.datetime64("2024-07-10"),  # Summer weekday
            np.datetime64("2024-07-11"),  # Summer weekday
            input_dir + "billing_pge.csv",
            "gas",
            {"electric": np.arange(96), "gas": np.arange(96)},
            None,
            0,
            np.float64(0),
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
            "electric",
            {"electric": np.arange(96), "gas": np.arange(96)},
            {
                "gas_energy_0_2024-07-10_2024-07-10_0": 0,
                "gas_energy_0_2024-07-10_2024-07-10_5000": 0,
                "electric_customer_0_2024-07-10_2024-07-10_0": 0,
                "electric_energy_0_2024-07-10_2024-07-10_0": 0,
                "electric_energy_1_2024-07-10_2024-07-10_0": 0,
                "electric_energy_2_2024-07-10_2024-07-10_0": 0,
                "electric_energy_3_2024-07-10_2024-07-10_0": 0,
                "electric_energy_4_2024-07-10_2024-07-10_0": 0,
                "electric_energy_5_2024-07-10_2024-07-10_0": 0,
                "electric_energy_6_2024-07-10_2024-07-10_0": 0,
                "electric_energy_7_2024-07-10_2024-07-10_0": 0,
                "electric_energy_8_2024-07-10_2024-07-10_0": 0,
                "electric_energy_9_2024-07-10_2024-07-10_0": 0,
                "electric_energy_10_2024-07-10_2024-07-10_0": 0,
                "electric_energy_11_2024-07-10_2024-07-10_0": 0,
                "electric_energy_12_2024-07-10_2024-07-10_0": 0,
                "electric_energy_13_2024-07-10_2024-07-10_0": 0,
            },
            0,
            pytest.approx(140.916195),
        ),
        (
            np.datetime64("2024-07-13"),  # Summer weekend
            np.datetime64("2024-07-14"),  # Summer weekend
            input_dir + "billing_pge.csv",
            "electric",
            {"electric": np.arange(96), "gas": np.arange(96)},
            None,
            0,
            pytest.approx(102.3834),
        ),
        (
            np.datetime64("2024-03-07"),  # Winter weekday
            np.datetime64("2024-03-08"),  # Winter weekday
            input_dir + "billing_pge.csv",
            "electric",
            {"electric": np.arange(96), "gas": np.arange(96)},
            None,
            0,
            pytest.approx(123.24669),
        ),
        (
            np.datetime64("2024-03-09"),  # Winter weekend
            np.datetime64("2024-03-10"),  # Winter weekend
            input_dir + "billing_pge.csv",
            "electric",
            {"electric": np.arange(96), "gas": np.arange(96)},
            None,
            0,
            pytest.approx(110.7624),
        ),
        (
            np.datetime64("2024-03-09"),  # Winter weekend
            np.datetime64("2024-03-10"),  # Winter weekend
            input_dir + "billing_pge.csv",
            "gas",
            {"electric": np.arange(96), "gas": np.arange(96)},
            None,
            0,
            pytest.approx(0),
        ),
        (
            np.datetime64("2024-03-09"),  # Winter weekend
            np.datetime64("2024-03-10"),  # Winter weekend
            input_dir + "billing_pge.csv",
            "gas",
            {"electric": np.arange(96), "gas": np.repeat(np.array([5100]), 96)},
            None,
            0,
            pytest.approx(59.1),
        ),
        (
            np.datetime64("2024-03-09"),  # Winter weekend
            np.datetime64("2024-03-10"),  # Winter weekend
            input_dir + "billing_pge.csv",
            "gas",
            {"electric": np.arange(96), "gas": np.ones(96)},
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


# TODO: write test_calculate_itemized_cost

import json
import os

import numpy as np
import pandas as pd
import pytest
import cvxpy as cp
import pyomo.environ as pyo

from eeco import demandresponse as dr

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
skip_all_tests = False

np.random.seed(0)

with open(os.path.join("tests", "data", "input", "cbp_payment_function.json")) as f:
    CBP_PAYMENT_FUNCTION = json.load(f)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_event_window_mask():
    index = pd.date_range("2024-01-01", "2024-01-03", freq="1h", inclusive="left")
    mask = dr._event_window_mask(index, "2024-01-02", 13, 2)
    selected = index[mask]
    assert list(selected) == [
        pd.Timestamp("2024-01-02 13:00"),
        pd.Timestamp("2024-01-02 14:00"),
    ]


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_add_event():
    events = dr.add_event(
        None,
        event_date="2024-01-08",
        start_hour=13,
        duration_hours=2,
        notification_hours=17,
        baseline_days=["2024-01-01", "2024-01-02"],
        bid_capacity_kW=100,
        capacity_price=10,
    )
    assert len(events) == 1
    events = dr.add_event(
        events,
        event_date="2024-01-09",
        start_hour=14,
        duration_hours=1,
        notification_hours=17,
        baseline_days=["2024-01-01", "2024-01-02"],
        bid_capacity_kW=200,
        capacity_price=20,
    )
    assert len(events) == 2
    assert events[0][dr.EVENT_DATE] == pd.Timestamp("2024-01-08")

    with pytest.raises(ValueError):
        dr.add_event(
            None, "2024-01-08", 13, 0, 17, ["2024-01-01"], 100, 10
        )  # duration_hours == 0
    with pytest.raises(ValueError):
        dr.add_event(
            None, "2024-01-08", 13, 2, -1, ["2024-01-01"], 100, 10
        )  # notification_hours < 0
    with pytest.raises(ValueError):
        dr.add_event(None, "2024-01-08", 13, 2, 17, [], 100, 10)  # empty baseline_days
    with pytest.raises(ValueError):
        dr.add_event(
            None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 0, 10
        )  # bid_capacity_kW == 0
    with pytest.warns(UserWarning):
        dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 0)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_events_to_dataframe():
    events = dr.add_event(None, "2024-01-09", 13, 2, 17, ["2024-01-01"], 100, 10)
    events = dr.add_event(events, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)
    events_df = dr.events_to_dataframe(events)
    assert list(events_df[dr.EVENT_DATE]) == [
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-09"),
    ]


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_make_baseline_parameters():
    params = dr.make_baseline_parameters()
    assert params[dr.N_BASELINE_DAYS] == 10
    assert params[dr.ADJUSTMENT_HOURS] == 3

    with pytest.raises(ValueError):
        dr.make_baseline_parameters(baseline_method="high_x_of_y")
    with pytest.raises(ValueError):
        dr.make_baseline_parameters(n_baseline_days=0)


def _flat_power_series(value_by_hour=100, adj_value_by_hour=None):
    """Builds an hourly power series over Jan 2024 where every day has the
    same value during hours [13, 15) (the event window used by these tests)
    and, if given, a separate constant value during hours [10, 13)
    (the day-of adjustment window)."""
    index = pd.date_range("2024-01-01", "2024-02-01", freq="1h", inclusive="left")
    values = np.full(len(index), 50.0)
    hours = index.hour
    values[(hours >= 13) & (hours < 15)] = value_by_hour
    if adj_value_by_hour is not None:
        values[(hours >= 10) & (hours < 13)] = adj_value_by_hour
    return pd.Series(values, index=index)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_simple_average():
    power_kW = _flat_power_series(value_by_hour=100)
    baseline_days = [f"2024-01-0{d}" for d in range(1, 6)]  # 5 weekdays
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, baseline_days, 100, 10)[0]
    params = dr.make_baseline_parameters(n_baseline_days=5, adjustment_hours=None)
    baseline_kW = dr.calculate_event_baseline(power_kW, event, params)
    assert baseline_kW == pytest.approx(100)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_day_of_adjustment():
    power_kW = _flat_power_series(value_by_hour=100, adj_value_by_hour=100)
    # bump the event day's own adjustment-window consumption to create a 1.1x factor
    event_adj_start = pd.Timestamp("2024-01-08 10:00")
    event_adj_end = event_adj_start + pd.Timedelta(hours=2)
    power_kW.loc[event_adj_start:event_adj_end] = 110

    baseline_days = [f"2024-01-0{d}" for d in range(1, 6)]
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, baseline_days, 100, 10)[0]
    params = dr.make_baseline_parameters(
        n_baseline_days=5, adjustment_hours=3, adjustment_clip=(0.8, 1.2)
    )
    baseline_kW = dr.calculate_event_baseline(power_kW, event, params)
    assert baseline_kW == pytest.approx(110, rel=1e-3)

    # push the factor beyond the clip bounds
    power_kW.loc[event_adj_start:event_adj_end] = 1000
    baseline_kW_clipped = dr.calculate_event_baseline(power_kW, event, params)
    assert baseline_kW_clipped == pytest.approx(100 * 1.2, rel=1e-3)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_zero_denominator_warns():
    power_kW = _flat_power_series(value_by_hour=100, adj_value_by_hour=0)
    baseline_days = [f"2024-01-0{d}" for d in range(1, 6)]
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, baseline_days, 100, 10)[0]
    params = dr.make_baseline_parameters(n_baseline_days=5, adjustment_hours=3)
    with pytest.warns(UserWarning):
        baseline_kW = dr.calculate_event_baseline(power_kW, event, params)
    assert baseline_kW == pytest.approx(100)  # factor skipped (left at 1.0)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_insufficient_days():
    power_kW = _flat_power_series(value_by_hour=100)
    baseline_days = ["2024-01-01", "2024-01-02"]  # only 2, fewer than requested 5
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, baseline_days, 100, 10)[0]
    params = dr.make_baseline_parameters(n_baseline_days=5, adjustment_hours=None)
    with pytest.warns(UserWarning):
        dr.calculate_event_baseline(power_kW, event, params)

    weekend_only_days = [
        "2024-01-06",
        "2024-01-07",
    ]  # both weekend, excluded by default
    event2 = dr.add_event(None, "2024-01-08", 13, 2, 17, weekend_only_days, 100, 10)[0]
    with pytest.raises(ValueError):
        dr.calculate_event_baseline(power_kW, event2, params)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_dynamic_day_fully_in_horizon():
    historical_power_kW = _flat_power_series(
        value_by_hour=100
    )  # unused: day is in-horizon
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    params = dr.make_baseline_parameters(n_baseline_days=1, adjustment_hours=None)

    model = pyo.ConcreteModel()
    model_datetime_index = pd.date_range(
        "2024-01-01 12:00", "2024-01-01 17:00", freq="1h", inclusive="left"
    )
    model.t = pyo.RangeSet(0, len(model_datetime_index) - 1)
    model.power = pyo.Var(model.t)
    for t in model.t:
        model.power[t].fix(0)
    model.power[1].fix(100)  # 13:00
    model.power[2].fix(140)  # 14:00 -> baseline day's [13, 15) window mean = 120

    baseline_var, model = dr.calculate_event_baseline(
        historical_power_kW,
        event,
        params,
        model=model,
        model_power_kW=model.power,
        model_datetime_index=model_datetime_index,
        varstr="test_baseline_1",
    )
    baseline_var.fix(120)
    constraint = model.find_component("test_baseline_1_constraint")
    assert pyo.value(constraint.body) == pytest.approx(0, abs=1e-6)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_dynamic_day_fully_out_of_horizon():
    historical_power_kW = _flat_power_series(value_by_hour=100)
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    params = dr.make_baseline_parameters(n_baseline_days=1, adjustment_hours=None)

    model = pyo.ConcreteModel()
    model_datetime_index = pd.date_range(
        "2024-02-01", "2024-02-02", freq="1h", inclusive="left"
    )
    model.t = pyo.RangeSet(0, len(model_datetime_index) - 1)
    model.power = pyo.Var(model.t)
    for t in model.t:
        model.power[t].fix(9999)  # obviously wrong if mistakenly consulted

    baseline_kW, model = dr.calculate_event_baseline(
        historical_power_kW,
        event,
        params,
        model=model,
        model_power_kW=model.power,
        model_datetime_index=model_datetime_index,
        varstr="test_baseline_2",
    )
    expected = dr.calculate_event_baseline(historical_power_kW, event, params)
    assert baseline_kW == pytest.approx(expected)
    assert model.find_component("test_baseline_2") is None  # no new component created


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_dynamic_day_partial_overlap():
    historical_power_kW = _flat_power_series(value_by_hour=100)
    historical_power_kW.loc["2024-01-01 13:00"] = 50
    historical_power_kW.loc["2024-01-01 14:00"] = 70
    # true historical mean for the [13, 15) window on this day: 60

    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    params = dr.make_baseline_parameters(n_baseline_days=1, adjustment_hours=None)

    model = pyo.ConcreteModel()
    # horizon starts at 14:00 -- the window [13, 15) only partially overlaps it
    model_datetime_index = pd.date_range(
        "2024-01-01 14:00", "2024-01-01 17:00", freq="1h", inclusive="left"
    )
    model.t = pyo.RangeSet(0, len(model_datetime_index) - 1)
    model.power = pyo.Var(model.t)
    for t in model.t:
        model.power[t].fix(9999)  # obviously wrong if the 14:00 slot were used

    baseline_kW, model = dr.calculate_event_baseline(
        historical_power_kW,
        event,
        params,
        model=model,
        model_power_kW=model.power,
        model_datetime_index=model_datetime_index,
        varstr="test_baseline_3",
    )
    assert baseline_kW == pytest.approx(60)
    assert model.find_component("test_baseline_3") is None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_dynamic_mixed_days():
    historical_power_kW = _flat_power_series(value_by_hour=100)
    # day A (2024-01-01) in horizon -> dynamic; day B (2024-01-02) out -> historical (mean 100)
    event = dr.add_event(
        None, "2024-01-08", 13, 2, 17, ["2024-01-01", "2024-01-02"], 100, 10
    )[0]
    params = dr.make_baseline_parameters(n_baseline_days=2, adjustment_hours=None)

    model = pyo.ConcreteModel()
    model_datetime_index = pd.date_range(
        "2024-01-01 12:00", "2024-01-01 17:00", freq="1h", inclusive="left"
    )
    model.t = pyo.RangeSet(0, len(model_datetime_index) - 1)
    model.power = pyo.Var(model.t)
    for t in model.t:
        model.power[t].fix(0)
    model.power[1].fix(80)  # 13:00
    model.power[2].fix(160)  # 14:00 -> day A mean = 120

    baseline_var, model = dr.calculate_event_baseline(
        historical_power_kW,
        event,
        params,
        model=model,
        model_power_kW=model.power,
        model_datetime_index=model_datetime_index,
        varstr="test_baseline_4",
    )
    baseline_var.fix(110)  # (day A 120 + day B 100) / 2
    constraint = model.find_component("test_baseline_4_constraint")
    assert pyo.value(constraint.body) == pytest.approx(0, abs=1e-6)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_adjustment_stays_historical_even_in_horizon():
    historical_power_kW = _flat_power_series(value_by_hour=100, adj_value_by_hour=100)
    event_adj_start = pd.Timestamp("2024-01-08 10:00")
    event_adj_end = event_adj_start + pd.Timedelta(hours=2)
    historical_power_kW.loc[event_adj_start:event_adj_end] = 110

    baseline_days = [f"2024-01-0{d}" for d in range(1, 6)]
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, baseline_days, 100, 10)[0]
    params = dr.make_baseline_parameters(
        n_baseline_days=5, adjustment_hours=3, adjustment_clip=(0.8, 1.2)
    )
    expected = dr.calculate_event_baseline(historical_power_kW, event, params)

    model = pyo.ConcreteModel()
    # horizon covers only the event day's adjustment window (10:00-13:00 on Jan 8),
    # not any baseline day's main window -- proving the adjustment factor never
    # consults model_power_kW even though its window falls inside the horizon.
    model_datetime_index = pd.date_range(
        "2024-01-08 10:00", "2024-01-08 13:00", freq="1h", inclusive="left"
    )
    model.t = pyo.RangeSet(0, len(model_datetime_index) - 1)
    model.power = pyo.Var(model.t)
    for t in model.t:
        model.power[t].fix(99999)  # obviously wrong if mistakenly consulted

    baseline_kW, model = dr.calculate_event_baseline(
        historical_power_kW,
        event,
        params,
        model=model,
        model_power_kW=model.power,
        model_datetime_index=model_datetime_index,
        varstr="test_baseline_5",
    )
    assert baseline_kW == pytest.approx(expected)


def _adjustment_factor_fixture():
    """Baseline days whose adjustment window (10:00-13:00) sits at 100 kW and
    an event day whose own adjustment window sits at 110 kW, so the day-of
    adjustment factor works out to 1.1 against a raw baseline of 100 kW."""
    historical_power_kW = _flat_power_series(value_by_hour=100, adj_value_by_hour=100)
    event_adj_start = pd.Timestamp("2024-01-08 10:00")
    event_adj_end = event_adj_start + pd.Timedelta(hours=2)
    historical_power_kW.loc[event_adj_start:event_adj_end] = 110

    baseline_days = [f"2024-01-0{d}" for d in range(1, 6)]
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, baseline_days, 100, 10)[0]
    return historical_power_kW, event


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_adjustment_factor_exposed_as_fixed_model_var():
    """With `adjustment_in_model=True` the day-of adjustment factor becomes a
    fixed pyomo Var, so the user can retune it and re-solve without
    rebuilding the model."""
    historical_power_kW, event = _adjustment_factor_fixture()
    baseline_method = dr.BaselineMethod(
        n_baseline_days=5, adjustment_hours=3, adjustment_in_model=True
    )

    model = pyo.ConcreteModel()
    # horizon well away from every baseline day -> all days stay historical,
    # proving the Param is created even when the baseline itself is a constant
    model_datetime_index = pd.date_range(
        "2024-02-01", "2024-02-02", freq="1h", inclusive="left"
    )
    model.t = pyo.RangeSet(0, len(model_datetime_index) - 1)
    model.power = pyo.Var(model.t)

    baseline_var, model = baseline_method.compute(
        historical_power_kW,
        event,
        model=model,
        model_power_kW=model.power,
        model_datetime_index=model_datetime_index,
        varstr="adj_baseline",
    )

    factor_var = model.find_component("adj_baseline_adjustment_factor")
    assert factor_var is not None
    assert pyo.value(factor_var) == pytest.approx(1.1, rel=1e-3)
    # Fixed, so the solver treats it as an input rather than choosing it --
    # which also keeps `baseline * factor` linear (degree 1, not bilinear).
    assert factor_var.fixed
    constraint = model.find_component("adj_baseline_constraint")
    assert constraint.body.polynomial_degree() == 1

    baseline_var.fix(110)  # raw baseline 100 * factor 1.1
    assert pyo.value(constraint.body) == pytest.approx(
        pyo.value(constraint.upper), abs=1e-6
    )

    # retune the factor in place -- no rebuild, the baseline follows it
    factor_var.fix(1.2)
    baseline_var.fix(120)
    assert pyo.value(constraint.body) == pytest.approx(
        pyo.value(constraint.upper), abs=1e-6
    )


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_adjustment_in_model_is_inert_without_a_model():
    """Ex-post there is no model to hold a Param, so the flag changes nothing
    and the factor is folded into a plain float as usual."""
    historical_power_kW, event = _adjustment_factor_fixture()
    baseline_method = dr.BaselineMethod(
        n_baseline_days=5, adjustment_hours=3, adjustment_in_model=True
    )
    baseline_kW = baseline_method.compute(historical_power_kW, event)
    assert baseline_kW == pytest.approx(110, rel=1e-3)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_adjustment_in_model_defaults_off():
    """Left at its default the flag adds nothing to the model, preserving the
    plain-float return for an all-historical baseline."""
    historical_power_kW, event = _adjustment_factor_fixture()
    baseline_method = dr.BaselineMethod(n_baseline_days=5, adjustment_hours=3)

    model = pyo.ConcreteModel()
    model_datetime_index = pd.date_range(
        "2024-02-01", "2024-02-02", freq="1h", inclusive="left"
    )
    model.t = pyo.RangeSet(0, len(model_datetime_index) - 1)
    model.power = pyo.Var(model.t)

    baseline_kW, model = baseline_method.compute(
        historical_power_kW,
        event,
        model=model,
        model_power_kW=model.power,
        model_datetime_index=model_datetime_index,
        varstr="no_adj_param",
    )
    assert baseline_kW == pytest.approx(110, rel=1e-3)
    assert model.find_component("no_adj_param_adjustment_factor") is None
    assert model.find_component("no_adj_param") is None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_model_args_incomplete_raises():
    historical_power_kW = _flat_power_series(value_by_hour=100)
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    params = dr.make_baseline_parameters(n_baseline_days=1, adjustment_hours=None)

    model = pyo.ConcreteModel()
    with pytest.raises(ValueError):
        dr.calculate_event_baseline(historical_power_kW, event, params, model=model)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_event_baseline_datetime_index_too_short():
    historical_power_kW = _flat_power_series(value_by_hour=100)
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    params = dr.make_baseline_parameters(n_baseline_days=1, adjustment_hours=None)

    model = pyo.ConcreteModel()
    model.t = pyo.RangeSet(0, 0)
    model.power = pyo.Var(model.t)
    model.power[0].fix(0)
    model_datetime_index = pd.DatetimeIndex(["2024-01-01 13:00"])  # only 1 entry

    with pytest.raises(ValueError):
        dr.calculate_event_baseline(
            historical_power_kW,
            event,
            params,
            model=model,
            model_power_kW=model.power,
            model_datetime_index=model_datetime_index,
            varstr="test_baseline_7",
        )


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_find_payment_region_errors():
    with pytest.raises(ValueError):
        dr._find_payment_region(CBP_PAYMENT_FUNCTION, -1.0)  # below every region
    with pytest.raises(ValueError):
        dr._find_payment_region(CBP_PAYMENT_FUNCTION, region_x1=0.42)  # no such region


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "delivered_ratio, expected_revenue",
    [
        (1.1, 10 * 1.05 * 100),  # capped region: 10 * (0*110 + 1.05*100)
        (0.9, 10 * 90),  # region 1: 10 * (1*90 + 0)
        (0.75, 10 * 75),  # exact boundary lands in region 1, not region 2
        (0.7, 10 * 0.5 * 70),  # region 2: 10 * (0.5*70 + 0)
        (0.6, 10 * 0.5 * 60),  # exact boundary lands in region 2, not region 3
        (0.5, 10 * (50 - 0.6 * 100)),  # region 3 (penalty): 10 * (1*50 - 0.6*100)
    ],
)
def test_evaluate_payment_function(delivered_ratio, expected_revenue):
    bid_capacity_kW = 100
    capacity_price = 10
    reduction_kW = delivered_ratio * bid_capacity_kW
    revenue = dr.evaluate_payment_function(
        CBP_PAYMENT_FUNCTION, reduction_kW, bid_capacity_kW, capacity_price
    )
    assert revenue == pytest.approx(expected_revenue)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_calculate_dr_revenue_multi_event():
    power_kW = _flat_power_series(value_by_hour=100)
    # Event 1: actual drops to 26 kW (delivered ratio 0.74 -> Region 2)
    event_1_start = pd.Timestamp("2024-01-08 13:00")
    event_1_end = event_1_start + pd.Timedelta(hours=1)
    power_kW.loc[event_1_start:event_1_end] = 26
    # Event 2: distinct date/window, actual drops to 10 kW (ratio 0.9 -> Region 1)
    event_2_start = pd.Timestamp("2024-01-15 13:00")
    event_2_end = event_2_start + pd.Timedelta(hours=1)
    power_kW.loc[event_2_start:event_2_end] = 10

    baseline_days = [f"2024-01-0{d}" for d in range(1, 6)]
    events = dr.add_event(None, "2024-01-08", 13, 2, 17, baseline_days, 100, 10)
    events = dr.add_event(events, "2024-01-15", 13, 2, 17, baseline_days, 100, 10)
    params = dr.make_baseline_parameters(n_baseline_days=5, adjustment_hours=None)

    per_event_df, total_revenue = dr.calculate_dr_revenue(
        power_kW, events, params, payment_function=CBP_PAYMENT_FUNCTION
    )
    assert len(per_event_df) == 2

    event_1_row = per_event_df.iloc[0]
    assert event_1_row[dr.DELIVERED_RATIO] == pytest.approx(0.74, abs=1e-6)
    assert event_1_row[dr.REVENUE] == pytest.approx(10 * 0.5 * 74)

    event_2_row = per_event_df.iloc[1]
    assert event_2_row[dr.DELIVERED_RATIO] == pytest.approx(0.9, abs=1e-6)
    assert event_2_row[dr.REVENUE] == pytest.approx(10 * 90)

    assert total_revenue == pytest.approx(
        event_1_row[dr.REVENUE] + event_2_row[dr.REVENUE]
    )


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_build_event_revenue_pyomo():
    model = pyo.ConcreteModel()
    model.t = pyo.RangeSet(0, 2)
    model.power = pyo.Var(model.t)
    for t in model.t:
        model.power[t].fix(30)  # mean actual power = 30 kW

    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    baseline_kW = 90  # reduction = 60 -> delivered ratio 0.6 -> Region 2 boundary

    revenue_var, model = dr.build_event_revenue(
        model.power,
        event,
        baseline_kW,
        payment_function=CBP_PAYMENT_FUNCTION,
        region_x1=0.60,
        model=model,
        varstr="event_2024_01_08",
    )
    expected_revenue = 10 * 0.5 * 60

    # No LP solver is assumed to be installed in this environment. Since every
    # other variable in the model is already fixed, fixing revenue_var to the
    # hand-computed expected value and checking that the defining equality
    # constraint is satisfied (residual ~= 0) verifies correctness without
    # needing to actually solve the model.
    revenue_var.fix(expected_revenue)
    revenue_constraint = model.find_component("event_2024_01_08_revenue_constraint")
    assert pyo.value(revenue_constraint.body) == pytest.approx(0, abs=1e-6)

    lower_bound_constraint = model.find_component(
        "event_2024_01_08_lower_bound_constraint"
    )
    assert pyo.value(lower_bound_constraint.body) == pytest.approx(60)  # reduction_kW
    upper_bound_constraint = model.find_component(
        "event_2024_01_08_upper_bound_constraint"
    )
    assert pyo.value(upper_bound_constraint.body) == pytest.approx(60)  # reduction_kW

    with pytest.raises(ValueError):
        dr.build_event_revenue(
            model.power,
            event,
            baseline_kW,
            payment_function=CBP_PAYMENT_FUNCTION,
            region_x1=None,
            model=model,
            varstr="x",
        )

    with pytest.raises(RuntimeError):
        # reusing the same varstr on the same model raises a pyomo error
        dr.build_event_revenue(
            model.power,
            event,
            baseline_kW,
            payment_function=CBP_PAYMENT_FUNCTION,
            region_x1=0.60,
            model=model,
            varstr="event_2024_01_08",
        )


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_build_event_revenue_cvxpy():
    power = cp.Variable(3)
    power.value = np.array([30.0, 30.0, 30.0])  # mean actual power = 30 kW

    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    baseline_kW = 90  # reduction = 60 -> delivered ratio 0.6 -> Region 2 boundary

    revenue_expr, constraints = dr.build_event_revenue(
        power,
        event,
        baseline_kW,
        payment_function=CBP_PAYMENT_FUNCTION,
        region_x1=0.60,
    )
    assert len(constraints) == 2
    assert revenue_expr.value == pytest.approx(10 * 0.5 * 60)

    with pytest.raises(ValueError):
        dr.build_event_revenue(
            power,
            event,
            baseline_kW,
            payment_function=CBP_PAYMENT_FUNCTION,
            region_x1=None,
        )


def _build_dr_revenue_fixture():
    """Two events, one bid_capacity_kW/capacity_price each, whose windows map
    to distinct positions in a 4-slot `datetime_index`/pyomo model, added out
    of date order to exercise `build_dr_revenue`'s internal sort:
    - "2024-01-08" (positions 0, 1): actual fixed to 30 kW, baseline 90 kW
      -> reduction 60 kW -> delivered ratio 0.6 -> region [0.60, 0.75),
      expected revenue 10 * 0.5 * 60 = 300.
    - "2024-01-15" (positions 2, 3): actual fixed to 0 kW, baseline 90 kW
      -> reduction 90 kW -> delivered ratio 0.9 -> region [0.75, 1.05),
      expected revenue 10 * 90 = 900.
    Total expected revenue: 1200.
    """
    datetime_index = pd.DatetimeIndex(
        [
            "2024-01-08 13:00",
            "2024-01-08 14:00",
            "2024-01-15 13:00",
            "2024-01-15 14:00",
        ]
    )
    historical_power_kW = _flat_power_series(value_by_hour=90)
    baseline_params = dr.make_baseline_parameters(
        n_baseline_days=1, adjustment_hours=None
    )

    events = dr.add_event(None, "2024-01-15", 13, 2, 17, ["2024-01-01"], 100, 10)
    events = dr.add_event(events, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)
    region_x1s = {"2024-01-08": 0.60, "2024-01-15": 0.75}

    model = pyo.ConcreteModel()
    model.t = pyo.RangeSet(0, 3)
    model.power = pyo.Var(model.t)
    model.power[0].fix(30)
    model.power[1].fix(30)
    model.power[2].fix(0)
    model.power[3].fix(0)

    return (
        model,
        datetime_index,
        events,
        historical_power_kW,
        baseline_params,
        region_x1s,
    )


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_build_dr_revenue_pyomo_existing_objective():
    (
        model,
        datetime_index,
        events,
        historical_power_kW,
        baseline_params,
        region_x1s,
    ) = _build_dr_revenue_fixture()

    model.cost = pyo.Var()
    model.cost.fix(500)
    model.objective = pyo.Objective(expr=model.cost, sense=pyo.minimize)

    total_revenue, model = dr.build_dr_revenue(
        model.power,
        datetime_index,
        events,
        historical_power_kW,
        baseline_params,
        model,
        CBP_PAYMENT_FUNCTION,
        region_x1s,
    )

    # dr_event_0 is "2024-01-08" (sorted first, though added second above).
    rev0 = model.find_component("dr_event_0_revenue")
    rev1 = model.find_component("dr_event_1_revenue")
    rev0.fix(300)
    rev1.fix(900)

    assert pyo.value(model.objective.expr) == pytest.approx(500 - 1200)

    revenue_constraint_0 = model.find_component("dr_event_0_revenue_constraint")
    assert pyo.value(revenue_constraint_0.body) == pytest.approx(0, abs=1e-6)
    lower_bound_0 = model.find_component("dr_event_0_lower_bound_constraint")
    assert pyo.value(lower_bound_0.body) == pytest.approx(60)  # reduction_kW

    revenue_constraint_1 = model.find_component("dr_event_1_revenue_constraint")
    assert pyo.value(revenue_constraint_1.body) == pytest.approx(0, abs=1e-6)
    lower_bound_1 = model.find_component("dr_event_1_lower_bound_constraint")
    assert pyo.value(lower_bound_1.body) == pytest.approx(90)  # reduction_kW


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_build_dr_revenue_pyomo_new_objective():
    (
        model,
        datetime_index,
        events,
        historical_power_kW,
        baseline_params,
        region_x1s,
    ) = _build_dr_revenue_fixture()

    assert not hasattr(model, "objective")

    total_revenue, model = dr.build_dr_revenue(
        model.power,
        datetime_index,
        events,
        historical_power_kW,
        baseline_params,
        model,
        CBP_PAYMENT_FUNCTION,
        region_x1s,
    )

    assert model.objective.sense == pyo.minimize
    model.find_component("dr_event_0_revenue").fix(300)
    model.find_component("dr_event_1_revenue").fix(900)
    assert pyo.value(model.objective.expr) == pytest.approx(-1200)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_build_dr_revenue_errors():
    (
        model,
        datetime_index,
        events,
        historical_power_kW,
        baseline_params,
        region_x1s,
    ) = _build_dr_revenue_fixture()

    with pytest.raises(KeyError):
        dr.build_dr_revenue(
            model.power,
            datetime_index,
            events,
            historical_power_kW,
            baseline_params,
            model,
            CBP_PAYMENT_FUNCTION,
            {"2024-01-08": 0.60},  # missing "2024-01-15"
        )

    events_out_of_range = dr.add_event(
        None, "2024-02-01", 13, 2, 17, ["2024-01-01"], 100, 10
    )
    with pytest.raises(ValueError):
        dr.build_dr_revenue(
            model.power,
            datetime_index,  # covers only January 8 and 15, not February 1
            events_out_of_range,
            historical_power_kW,
            baseline_params,
            model,
            CBP_PAYMENT_FUNCTION,
            {"2024-02-01": 0.60},
        )


def _per_day_event_series(day_values, event_start_hour=13, event_duration_hours=2):
    """Builds an hourly power series over Jan 2024 where the event window on
    each date in `day_values` is set to that date's value, and every other
    hour is a constant filler (50)."""
    index = pd.date_range("2024-01-01", "2024-02-01", freq="1h", inclusive="left")
    series = pd.Series(np.full(len(index), 50.0), index=index)
    for day_str, value in day_values.items():
        mask = dr._event_window_mask(
            index, day_str, event_start_hour, event_duration_hours
        )
        series.loc[mask] = value
    return series


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_top_usage_days_baseline_selects_highest_usage_days():
    day_values = {
        "2024-01-01": 50,  # Mon
        "2024-01-02": 200,  # Tue -- highest usage
        "2024-01-03": 60,  # Wed
        "2024-01-04": 150,  # Thu -- second highest usage
        "2024-01-05": 70,  # Fri -- most recent, but low usage
    }
    power_kW = _per_day_event_series(day_values)
    event = dr.add_event(
        None, "2024-01-08", 13, 2, 17, list(day_values.keys()), 100, 10
    )[0]

    top_usage_method = dr.TopUsageDaysBaseline(n_baseline_days=2, adjustment_hours=None)
    assert top_usage_method.compute(power_kW, event) == pytest.approx(
        np.mean([200, 150])
    )

    # The foundation class instead picks the two *most recent* days -- proving
    # the subclass changed only the ranking rule, not the rest of the behavior.
    default_method = dr.BaselineMethod(n_baseline_days=2, adjustment_hours=None)
    assert default_method.compute(power_kW, event) == pytest.approx(np.mean([70, 150]))


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_fixed_level_baseline_ignores_history():
    power_kW = _flat_power_series(value_by_hour=9999)  # deliberately irrelevant
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    baseline_method = dr.FixedLevelBaseline(firm_level_kW=500)

    assert baseline_method.compute(power_kW, event) == 500

    model = pyo.ConcreteModel()
    baseline_kW, returned_model = baseline_method.compute(power_kW, event, model=model)
    assert baseline_kW == 500
    assert returned_model is model

    with pytest.raises(ValueError):
        dr.FixedLevelBaseline(firm_level_kW=-1)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_unilateral_interruption_baseline_requires_model():
    power_kW = _flat_power_series(value_by_hour=100)
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    baseline_method = dr.UnilateralInterruptionBaseline(interruption_level_kW=0.0)

    with pytest.raises(NotImplementedError):
        baseline_method.compute(power_kW, event)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_unilateral_interruption_baseline_constrains_model():
    power_kW = _flat_power_series(value_by_hour=100)
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    baseline_method = dr.UnilateralInterruptionBaseline(interruption_level_kW=0.0)

    model = pyo.ConcreteModel()
    model_datetime_index = pd.date_range(
        "2024-01-08 12:00", "2024-01-08 17:00", freq="1h", inclusive="left"
    )
    model.t = pyo.RangeSet(0, len(model_datetime_index) - 1)
    model.power = pyo.Var(model.t)

    baseline_kW, model = baseline_method.compute(
        power_kW,
        event,
        model=model,
        model_power_kW=model.power,
        model_datetime_index=model_datetime_index,
        varstr="interrupt_1",
    )
    assert baseline_kW == 0.0

    constraint = model.find_component("interrupt_1_interruption_constraint")
    assert constraint is not None
    # event window [13, 15) -> positions 1, 2 in the 12:00-17:00 index
    assert set(constraint.keys()) == {1, 2}
    for idx in constraint:
        # An upper bound only -- the interruption caps what the facility can
        # draw, it does not force it to draw exactly that much.
        assert constraint[idx].lower is None
        assert pyo.value(constraint[idx].upper) == pytest.approx(0.0)

    with pytest.raises(ValueError):
        baseline_method.compute(
            power_kW, event, model=model
        )  # missing other model args


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_capacity_energy_payment_evaluate_adds_energy_term():
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    reduction_kW = 60  # delivered ratio 0.6 -> region [0.60, 0.75)
    payment_structure = dr.CapacityEnergyPayment(
        CBP_PAYMENT_FUNCTION, energy_price=0.09
    )

    revenue = payment_structure.evaluate(event, reduction_kW)

    expected_capacity = dr.PaymentStructure(CBP_PAYMENT_FUNCTION).evaluate(
        event, reduction_kW
    )
    expected_energy = 0.09 * reduction_kW * event[dr.EVENT_DURATION]
    assert revenue == pytest.approx(expected_capacity + expected_energy)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_capacity_energy_payment_build_expression_cvxpy():
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    reduction_kW = cp.Variable()
    reduction_kW.value = 60.0
    payment_structure = dr.CapacityEnergyPayment(
        CBP_PAYMENT_FUNCTION, energy_price=0.09
    )

    revenue_expr, constraints = payment_structure.build_expression(
        event, reduction_kW, region_x1=0.60
    )

    expected_capacity_expr, _ = dr.PaymentStructure(
        CBP_PAYMENT_FUNCTION
    ).build_expression(event, reduction_kW, region_x1=0.60)
    expected_energy = 0.09 * 60.0 * event[dr.EVENT_DURATION]
    assert revenue_expr.value == pytest.approx(
        expected_capacity_expr.value + expected_energy
    )
    assert len(constraints) == 2


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_capacity_energy_payment_build_expression_pyomo():
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    model = pyo.ConcreteModel()
    model.reduction = pyo.Var()
    model.reduction.fix(60.0)

    payment_structure = dr.CapacityEnergyPayment(
        CBP_PAYMENT_FUNCTION, energy_price=0.09
    )
    total_var, model = payment_structure.build_expression(
        event, model.reduction, region_x1=0.60, model=model, varstr="ce_event"
    )

    capacity_var = model.find_component("ce_event_revenue")
    capacity_var.fix(10 * 0.5 * 60)  # region [0.60, 0.75) formula, capacity_price=10
    capacity_constraint = model.find_component("ce_event_revenue_constraint")
    assert pyo.value(capacity_constraint.body) == pytest.approx(0, abs=1e-6)

    energy_var = model.find_component("ce_event_energy_revenue")
    energy_var.fix(0.09 * 60 * event[dr.EVENT_DURATION])
    energy_constraint = model.find_component("ce_event_energy_revenue_constraint")
    assert pyo.value(energy_constraint.body) == pytest.approx(0, abs=1e-6)

    total_var.fix(pyo.value(capacity_var) + pyo.value(energy_var))
    total_constraint = model.find_component("ce_event_total_revenue_constraint")
    assert pyo.value(total_constraint.body) == pytest.approx(0, abs=1e-6)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_market_indexed_payment_resolves_price():
    event = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-01"], 100, 10)[0]
    reduction_kW = 60

    payment_structure = dr.MarketIndexedPayment(
        CBP_PAYMENT_FUNCTION, price_lookup=lambda e: 42.0
    )
    revenue = payment_structure.evaluate(event, reduction_kW)

    resolved_event = {**event, dr.CAPACITY_PRICE: 42.0}
    expected = dr.PaymentStructure(CBP_PAYMENT_FUNCTION).evaluate(
        resolved_event, reduction_kW
    )
    assert revenue == pytest.approx(expected)
    # sanity: the original event's capacity_price (10) would give a different answer
    assert revenue != pytest.approx(
        dr.PaymentStructure(CBP_PAYMENT_FUNCTION).evaluate(event, reduction_kW)
    )


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_build_dr_revenue_with_capacity_energy_payment():
    (
        model,
        datetime_index,
        events,
        historical_power_kW,
        baseline_params,
        region_x1s,
    ) = _build_dr_revenue_fixture()

    payment_structure = dr.CapacityEnergyPayment(
        CBP_PAYMENT_FUNCTION, energy_price=0.09
    )

    total_revenue, model = dr.build_dr_revenue(
        model.power,
        datetime_index,
        events,
        historical_power_kW,
        baseline_params,
        model,
        payment_structure,
        region_x1s,
    )

    # dr_event_0 is "2024-01-08": capacity 300 (see fixture docstring) + energy 0.09*60*2
    capacity_0 = model.find_component("dr_event_0_revenue")
    energy_0 = model.find_component("dr_event_0_energy_revenue")
    total_0 = model.find_component("dr_event_0_total_revenue")
    capacity_0.fix(300)
    energy_0.fix(0.09 * 60 * 2)
    total_0.fix(pyo.value(capacity_0) + pyo.value(energy_0))
    constraint_0 = model.find_component("dr_event_0_total_revenue_constraint")
    assert pyo.value(constraint_0.body) == pytest.approx(0, abs=1e-6)

    # dr_event_1 is "2024-01-15": capacity 900 + energy 0.09*90*2
    capacity_1 = model.find_component("dr_event_1_revenue")
    energy_1 = model.find_component("dr_event_1_energy_revenue")
    total_1 = model.find_component("dr_event_1_total_revenue")
    capacity_1.fix(900)
    energy_1.fix(0.09 * 90 * 2)
    total_1.fix(pyo.value(capacity_1) + pyo.value(energy_1))
    constraint_1 = model.find_component("dr_event_1_total_revenue_constraint")
    assert pyo.value(constraint_1.body) == pytest.approx(0, abs=1e-6)

    assert pyo.value(model.objective.expr) == pytest.approx(
        -(pyo.value(total_0) + pyo.value(total_1))
    )


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_build_dr_revenue_dynamic_baseline():
    """The event's baseline day (2024-01-03) is inside the 4-slot horizon
    alongside the event day itself (2024-01-08), so `build_dr_revenue` should
    add a baseline Var/Constraint defined over the decision variable rather
    than a plain historical float."""
    datetime_index = pd.DatetimeIndex(
        [
            "2024-01-03 13:00",
            "2024-01-03 14:00",
            "2024-01-08 13:00",
            "2024-01-08 14:00",
        ]
    )
    historical_power_kW = _flat_power_series(
        value_by_hour=100
    )  # unused: baseline day is in-horizon
    baseline_params = dr.make_baseline_parameters(
        n_baseline_days=1, adjustment_hours=None
    )

    events = dr.add_event(None, "2024-01-08", 13, 2, 17, ["2024-01-03"], 100, 10)
    region_x1s = {"2024-01-08": 0.75}

    model = pyo.ConcreteModel()
    model.t = pyo.RangeSet(0, 3)
    model.power = pyo.Var(model.t)
    model.power[0].fix(120)  # baseline day 2024-01-03, 13:00
    model.power[1].fix(120)  # baseline day 2024-01-03, 14:00
    model.power[2].fix(30)  # event day 2024-01-08, 13:00 (actual)
    model.power[3].fix(30)  # event day 2024-01-08, 14:00 (actual)

    total_revenue, model = dr.build_dr_revenue(
        model.power,
        datetime_index,
        events,
        historical_power_kW,
        baseline_params,
        model,
        CBP_PAYMENT_FUNCTION,
        region_x1s,
        varstr_prefix="dyn_event",
    )

    baseline_var = model.find_component("dyn_event_0_baseline_kW")
    assert baseline_var is not None
    baseline_var.fix(120)  # hand-computed: mean(120, 120)
    baseline_constraint = model.find_component("dyn_event_0_baseline_kW_constraint")
    assert pyo.value(baseline_constraint.body) == pytest.approx(0, abs=1e-6)

    # reduction = 120 - 30 = 90 -> delivered ratio 0.9 -> region [0.75, 1.05) -> revenue 900
    revenue_var = model.find_component("dyn_event_0_revenue")
    revenue_var.fix(900)
    revenue_constraint = model.find_component("dyn_event_0_revenue_constraint")
    assert pyo.value(revenue_constraint.body) == pytest.approx(0, abs=1e-6)
    lower_bound = model.find_component("dyn_event_0_lower_bound_constraint")
    assert pyo.value(lower_bound.body) == pytest.approx(90)

"""Functions to calculate demand response revenue from electricity consumption data."""

import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
import pyomo.environ as pyo

from . import utils as ut

# Event dict keys
EVENT_DATE = "event_date"
EVENT_START_HOUR = "start_hour"
EVENT_DURATION = "duration_hours"
NOTIFICATION_HOURS = "notification_hours"
BASELINE_DAYS = "baseline_days"
BID_CAPACITY_KW = "bid_capacity_kW"
CAPACITY_PRICE = "capacity_price"

# Baseline parameter dict keys
BASELINE_METHOD = "baseline_method"
N_BASELINE_DAYS = "n_baseline_days"
ADJUSTMENT_HOURS = "adjustment_hours"
ADJUSTMENT_CLIP = "adjustment_clip"
EXCLUDE_WEEKENDS = "exclude_weekends"
EXCLUDE_HOLIDAYS = "exclude_holidays"
HOLIDAY_DATES = "holiday_dates"

# Output/result column keys
BASELINE_KW = "baseline_kW"
ACTUAL_KW = "actual_kW"
REDUCTION_KW = "reduction_kW"
DELIVERED_RATIO = "delivered_ratio"
REVENUE = "revenue"

# Payment function region dict keys.
REGION_X1 = "x1"
REGION_X2 = "x2"
REGION_Y1 = "y1"
REGION_Y2 = "y2"


def _event_window_mask(index, event_date, start_hour, duration_hours):
    """Boolean mask selecting timestamps in `index` within the half-open window
    [event_date + start_hour, event_date + start_hour + duration_hours).

    Use this as opposed to the pandas.Series.between_time method to identify
    the window on a specific event day, as opposed to every day present.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Index of timestamps to select from.

    event_date : datetime.date, datetime.datetime, or str
        Calendar date the window is anchored to.

    start_hour : float
        Hour of day (0-24) the window begins, relative to `event_date`.

    duration_hours : float
        Length of the window in hours.

    Returns
    -------
    numpy.ndarray
        Boolean mask, `True` for timestamps within the window.
    """
    window_start = pd.Timestamp(event_date) + pd.Timedelta(hours=start_hour)
    window_end = window_start + pd.Timedelta(hours=duration_hours)
    return (index >= window_start) & (index < window_end)


def _find_payment_region(payment_function, delivered_ratio=None, region_x1=None):
    """Find a region in `payment_function`, by interval or by `x1`.

    Exactly one of `delivered_ratio`/`region_x1` must be given: `delivered_ratio`
    looks up the region whose `[x1, x2)` interval contains it; `region_x1` looks
    up the region whose `x1` matches it (via `numpy.isclose`, since callers such
    as `build_payment_expression` may pass a value with floating-point noise
    from an upstream solver).

    Parameters
    ----------
    payment_function : list of dict
        Each dict has keys `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`.

    delivered_ratio : float or None
        Known delivered ratio to look up by interval containment.

    region_x1 : float or None
        The `x1` value identifying the region to look up.

    Raises
    ------
    ValueError
        When no region matches.

    Returns
    -------
    dict
        The matching region.
    """
    if region_x1 is not None:
        predicate = lambda r: np.isclose(r[REGION_X1], region_x1)
        error_msg = f"No region with x1 close to {region_x1}"
    else:
        predicate = lambda r: r[REGION_X1] <= delivered_ratio < r[REGION_X2]
        error_msg = (
            f"delivered_ratio {delivered_ratio} is not covered by payment_function"
        )

    region = next((r for r in payment_function if predicate(r)), None)
    if region is None:
        raise ValueError(error_msg)
    return region


def evaluate_payment_function(
    payment_function, reduction_kW, bid_capacity_kW, capacity_price
):
    """Calculate ex-post revenue for a known reduction.

    Automatically determines the applicable region and linearly interpolates
    the payment ratio between its `(x1, y1)` and `(x2, y2)` endpoints, since
    the delivered ratio is already known -- no iterative region search is
    needed here.

    Parameters
    ----------
    payment_function : list of dict
        Each dict has keys `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`.

    reduction_kW : float
        Realized load reduction (baseline minus actual power) in kW.

    bid_capacity_kW : float
        Nominated capacity bid in kW.

    capacity_price : float
        Program capacity price in $/kW.

    Raises
    ------
    ValueError
        When `bid_capacity_kW` is not positive, or `payment_function` has no
        region covering the resulting delivered ratio.

    Returns
    -------
    float
        Revenue (positive) or penalty (negative) in USD.
    """
    if bid_capacity_kW <= 0:
        raise ValueError("bid_capacity_kW must be positive")
    delivered_ratio = reduction_kW / bid_capacity_kW
    region = _find_payment_region(payment_function, delivered_ratio)
    x1, x2, y1, y2 = (region[k] for k in (REGION_X1, REGION_X2, REGION_Y1, REGION_Y2))
    if np.isinf(x2):
        payment_ratio = y1
    else:
        payment_ratio = y1 + (y2 - y1) * (delivered_ratio - x1) / (x2 - x1)
    return payment_ratio * capacity_price * bid_capacity_kW


def build_payment_expression(
    payment_function,
    reduction_kW,
    bid_capacity_kW,
    capacity_price,
    region_x1=None,
    model=None,
    varstr="",
):
    """Build the revenue expression for a single, specified region.

    Dispatches on the type of `reduction_kW`:

    - `numpy.ndarray` or Python number: delegates to `evaluate_payment_function`
      (`region_x1` is ignored, since the actual region is already fully
      determined).
    - `cvxpy.Expression`/`cvxpy.Variable`: builds `revenue_expr` directly and
      a list of 1-2 cvxpy constraints for the region bounds.
    - `pyomo.environ.Var`/expression: adds a `varstr + "_revenue"` `Var` plus
      a defining constraint and 1-2 region-bound constraints onto `model`
      (which may be a `pyomo.environ.Block`) via `model.add_component`.

    Parameters
    ----------
    payment_function : list of dict
        Each dict has keys `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`.

    reduction_kW : numpy.ndarray, float, cvxpy.Expression, or pyomo.environ.Var
        Load reduction, as a realized value or a decision-variable expression.

    bid_capacity_kW : float
        Nominated capacity bid in kW.

    capacity_price : float
        Program capacity price in $/kW.

    region_x1 : float or None
        The `x1` value identifying which region to build. Required (and
        used) only for the cvxpy/pyomo cases.

    model : pyomo.environ.Model or pyomo.environ.Block
        The model or block to add pyomo components to.
        Only used in the pyomo case, so `None` by default.

    varstr : str
        Name prefix for pyomo variables/constraints created on `model`.
        Must be unique per call on a given `model`, since reusing a `varstr`
        will raise a pyomo "component already exists" error.

    Raises
    ------
    TypeError
        When `reduction_kW` is not a supported type.

    Returns
    -------
    tuple
        `(revenue_var, model)` for numpy or pyomo `reduction_kW`, or
        `(revenue_expr, constraints_list)` for cvxpy `reduction_kW`, where
        `constraints_list` holds the region-bound constraints for the caller
        to add to their own `cvxpy.Problem`.
    """
    if ut.check_indexed_np_array(reduction_kW) or ut.check_nonindexed_python_type(
        reduction_kW
    ):
        return (
            evaluate_payment_function(
                payment_function, reduction_kW, bid_capacity_kW, capacity_price
            ),
            model,
        )

    region = _find_payment_region(payment_function, region_x1=region_x1)
    x1, x2, y1, y2 = (region[k] for k in (REGION_X1, REGION_X2, REGION_Y1, REGION_Y2))
    slope_ratio = 0.0 if np.isinf(x2) else (y2 - y1) / (x2 - x1)
    slope = capacity_price * slope_ratio
    intercept = capacity_price * bid_capacity_kW * (y1 - slope_ratio * x1)

    if ut.check_cvx_type(reduction_kW):
        revenue_expr = slope * reduction_kW + intercept
        constraints = [reduction_kW >= x1 * bid_capacity_kW]
        if not np.isinf(x2):
            constraints.append(reduction_kW <= x2 * bid_capacity_kW)
        return revenue_expr, constraints
    elif ut.check_indexed_pyomo_type(reduction_kW) or ut.check_nonindexed_pyomo_type(
        reduction_kW
    ):
        model.add_component(varstr + "_revenue", pyo.Var())
        revenue_var = model.find_component(varstr + "_revenue")

        def revenue_rule(model):
            return revenue_var == slope * reduction_kW + intercept

        model.add_component(
            varstr + "_revenue_constraint", pyo.Constraint(rule=revenue_rule)
        )

        def lower_bound_rule(model):
            return reduction_kW >= x1 * bid_capacity_kW

        model.add_component(
            varstr + "_lower_bound_constraint", pyo.Constraint(rule=lower_bound_rule)
        )
        if not np.isinf(x2):

            def upper_bound_rule(model):
                return reduction_kW <= x2 * bid_capacity_kW

            model.add_component(
                varstr + "_upper_bound_constraint",
                pyo.Constraint(rule=upper_bound_rule),
            )
        return revenue_var, model
    else:
        raise TypeError(
            "reduction_kW must be numpy.ndarray, a Python number, "
            "cvxpy.Expression/Variable, or pyomo.environ.Var"
        )


def add_event(
    events,
    event_date,
    start_hour,
    duration_hours,
    notification_hours,
    baseline_days,
    bid_capacity_kW,
    capacity_price,
):
    """Add a demand response event to an events collection.

    `baseline_days` should already exclude any date that is itself another
    event's date -- this function does not check that for you, since doing so
    would require knowing every other event's date rather than just this one.

    Parameters
    ----------
    events : list of dict or None
        Existing events collection to append to. If `None`, a new list is
        created.

    event_date : datetime.date, datetime.datetime, or str
        Calendar date the event occurs on.

    start_hour : float
        Hour of day (0-24) the event begins.

    duration_hours : float
        Length of the event in hours.

    notification_hours : float
        Advance notice given before the event, in hours.

    baseline_days : list
        Calendar days to be used for this event's baseline calculation.

    bid_capacity_kW : float
        Nominated capacity bid in kW for this event.

    capacity_price : float
        Program capacity price in $/kW for this event.

    Raises
    ------
    ValueError
        When `duration_hours` is not positive, `notification_hours` is
        negative, `baseline_days` is empty, or `bid_capacity_kW` is not
        positive.

    Warnings
        When `capacity_price` is zero.

    Returns
    -------
    list of dict
        A new list with the new event appended (the input `events` is not
        mutated in place).
    """
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    if notification_hours < 0:
        raise ValueError("notification_hours must be non-negative")
    if len(baseline_days) == 0:
        raise ValueError("baseline_days must be non-empty")
    if bid_capacity_kW <= 0:
        raise ValueError("bid_capacity_kW must be positive")
    if capacity_price == 0:
        warnings.warn("capacity_price is zero", UserWarning)

    new_event = {
        EVENT_DATE: pd.Timestamp(event_date),
        EVENT_START_HOUR: start_hour,
        EVENT_DURATION: duration_hours,
        NOTIFICATION_HOURS: notification_hours,
        BASELINE_DAYS: list(baseline_days),
        BID_CAPACITY_KW: bid_capacity_kW,
        CAPACITY_PRICE: capacity_price,
    }
    return (events or []) + [new_event]


def events_to_dataframe(events):
    """Convert an events collection into a `DataFrame` sorted by event date.

    Parameters
    ----------
    events : list of dict or pandas.DataFrame
        Events collection, as produced by `add_event`.

    Returns
    -------
    pandas.DataFrame
        One row per event, sorted by `EVENT_DATE`.
    """
    events_df = events if isinstance(events, pd.DataFrame) else pd.DataFrame(events)
    return events_df.sort_values(EVENT_DATE).reset_index(drop=True)


def make_baseline_parameters(
    baseline_method="average_similar_days",
    n_baseline_days=10,
    adjustment_hours=3,
    adjustment_clip=(0.8, 1.2),
    exclude_weekends=True,
    exclude_holidays=True,
    holiday_dates=None,
):
    """Build a dictionary of program-specific baseline calculation parameters.

    Defaults match PG&E's Capacity Bidding Program (10 similar weekdays,
    3-hour day-of adjustment).

    Parameters
    ----------
    baseline_method : str
        Baseline calculation method. Only `"average_similar_days"` is
        currently supported.

    n_baseline_days : int
        Number of valid baseline days to average over.

    adjustment_hours : int or None
        Number of hours immediately before the event to use for a day-of
        adjustment factor. If `None`, no day-of adjustment is applied.

    adjustment_clip : tuple of float or None
        `(low, high)` bounds the day-of adjustment factor is clipped to.

    exclude_weekends : bool
        If `True`, exclude Saturday/Sunday from candidate baseline days.

    exclude_holidays : bool
        If `True`, exclude dates in `holiday_dates` from candidate baseline
        days.

    holiday_dates : list or None
        Calendar dates treated as holidays. Defaults to an empty list.

    Raises
    ------
    ValueError
        When `baseline_method` is not `"average_similar_days"`,
        or `n_baseline_days` is not positive.

    Returns
    -------
    dict
        Baseline parameters keyed by the module's `BASELINE_METHOD`,
        `N_BASELINE_DAYS`, `ADJUSTMENT_HOURS`, `ADJUSTMENT_CLIP`,
        `EXCLUDE_WEEKENDS`, `EXCLUDE_HOLIDAYS`, and `HOLIDAY_DATES` constants.
    """
    if baseline_method != "average_similar_days":
        raise ValueError(
            "baseline_method must be 'average_similar_days'; "
            "other methods are not yet supported"
        )
    if n_baseline_days <= 0:
        raise ValueError("n_baseline_days must be positive")

    return {
        BASELINE_METHOD: baseline_method,
        N_BASELINE_DAYS: n_baseline_days,
        ADJUSTMENT_HOURS: adjustment_hours,
        ADJUSTMENT_CLIP: adjustment_clip,
        EXCLUDE_WEEKENDS: exclude_weekends,
        EXCLUDE_HOLIDAYS: exclude_holidays,
        HOLIDAY_DATES: list(holiday_dates) if holiday_dates else [],
    }


def _infer_datetime_index_step(datetime_index):
    """Infer the regular spacing of a `DatetimeIndex`.

    Parameters
    ----------
    datetime_index : pandas.DatetimeIndex
        Index to infer the step size of.

    Raises
    ------
    ValueError
        When `datetime_index` has fewer than 2 entries.

    Returns
    -------
    pandas.Timedelta
        The spacing between the first two entries.
    """
    if len(datetime_index) < 2:
        raise ValueError(
            "model_datetime_index must have at least 2 entries to infer its step size"
        )
    return datetime_index[1] - datetime_index[0]


def _baseline_day_in_horizon(day, start_hour, duration_hours, datetime_index, step):
    """Check whether a baseline day's event-window is fully contained in the
    simulation horizon spanned by `datetime_index`.

    A window that only partially overlaps the horizon returns `False`, same
    as a window fully outside it -- both fall back to historical data.

    Parameters
    ----------
    day : pandas.Timestamp
        Calendar date the window is anchored to.

    start_hour : float
        Hour of day (0-24) the window begins.

    duration_hours : float
        Length of the window in hours.

    datetime_index : pandas.DatetimeIndex
        Calendar timestamps spanned by the simulation horizon.

    step : pandas.Timedelta
        Regular spacing of `datetime_index`, as returned by
        `_infer_datetime_index_step`.

    Returns
    -------
    bool
        `True` if the window's `[start, end)` bounds both fall within
        `[datetime_index.min(), datetime_index.max() + step)`.
    """
    window_start = pd.Timestamp(day) + pd.Timedelta(hours=start_hour)
    window_end = window_start + pd.Timedelta(hours=duration_hours)
    horizon_start = datetime_index.min()
    horizon_end = datetime_index.max() + step
    return (window_start >= horizon_start) and (window_end <= horizon_end)


def _baseline_day_terms(
    day,
    start_hour,
    duration_hours,
    historical_power_kW,
    model_power_kW,
    model_var_index,
    model_datetime_index,
    step,
):
    """Compute a single baseline day's window-mean as weighted terms.

    Parameters
    ----------
    day : pandas.Timestamp
        Calendar date the window is anchored to.

    start_hour : float
        Hour of day (0-24) the window begins.

    duration_hours : float
        Length of the window in hours.

    historical_power_kW : pandas.Series
        Historical realized power consumption in kW, indexed by
        `pandas.DatetimeIndex`.

    model_power_kW : pyomo.environ.Var or None
        The model's own decision variable for the full simulation horizon,
        or `None` if no model context was supplied.

    model_var_index : list or None
        `list(model_power_kW.index_set())`, or `None`.

    model_datetime_index : pandas.DatetimeIndex or None
        Calendar timestamp for each position in `model_var_index`, or `None`.

    step : pandas.Timedelta or None
        Regular spacing of `model_datetime_index`, or `None`.

    Raises
    ------
    ValueError
        When the day is in-horizon but no positions matched, or when the
        day is historical but has no data in its window.

    Returns
    -------
    tuple
        `(terms, is_dynamic)`, where `terms` is a list of `(weight, value)`
        pairs whose weights sum to `1.0` and whose weighted sum equals the
        day's window-mean.
    """
    if model_power_kW is not None and _baseline_day_in_horizon(
        day, start_hour, duration_hours, model_datetime_index, step
    ):
        mask = _event_window_mask(model_datetime_index, day, start_hour, duration_hours)
        matched = [idx for idx, keep in zip(model_var_index, mask) if keep]
        if not matched:
            raise ValueError(
                f"Baseline day {day}'s window bounds fall inside the simulation "
                "horizon but no matching positions were found in "
                "model_datetime_index (is it gap-free and regularly spaced?)"
            )
        terms = [(1.0 / len(matched), model_power_kW[idx]) for idx in matched]
        return terms, True

    mask = _event_window_mask(
        historical_power_kW.index, day, start_hour, duration_hours
    )
    day_slice = historical_power_kW.loc[mask]
    if day_slice.empty:
        raise ValueError(f"No data available for baseline day {day}")
    return [(1.0, day_slice.mean())], False


def calculate_event_baseline(
    historical_power_kW,
    event,
    baseline_params,
    *,
    model=None,
    model_power_kW=None,
    model_datetime_index=None,
    varstr=None,
):
    """Calculate the mean baseline power for a single event's window.

    When `model` is `None` (the default), every baseline day is computed
    from `historical_power_kW` and a plain `float` is returned, exactly as
    for a purely ex-post calculation.

    When `model` is given (along with `model_power_kW` and
    `model_datetime_index`), any baseline day whose full event-window is
    contained within the simulation horizon spanned by `model_datetime_index`
    is instead computed as a decision-variable average over `model_power_kW`
    -- a baseline day whose window only partially overlaps the horizon falls
    back fully to historical data, never mixing sources within one day. If
    at least one day was dynamic, a `pyo.Var` + defining linear `Constraint`
    named `varstr`/`varstr + "_constraint"` is added to `model`, and
    `(baseline_var, model)` is returned; if every day stayed historical,
    `(baseline_kW, model)` is returned instead, with no new components
    added.

    The day-of adjustment factor is always computed from historical
    `historical_power_kW`, never from `model_power_kW`, even for events
    whose adjustment window falls inside the simulation horizon -- clipping
    a ratio of decision-variable expressions has no linear pyomo
    representation.

    Parameters
    ----------
    historical_power_kW : pandas.Series
        Historical realized power consumption in kW, indexed by
        `pandas.DatetimeIndex`. Used for every baseline day that isn't fully
        contained in the simulation horizon, and always for the day-of
        adjustment factor.

    event : dict
        A single event, as produced by `add_event`.

    baseline_params : dict
        Baseline parameters, as produced by `make_baseline_parameters`.

    model : pyomo.environ.Model or pyomo.environ.Block or None
        The model to add the baseline `Var`/`Constraint` to, if at least one
        baseline day is dynamic. `None` by default (purely ex-post usage).

    model_power_kW : pyomo.environ.Var or None
        The model's own decision variable for the full simulation/
        optimization horizon -- distinct from `historical_power_kW` above.
        Required when `model` is given.

    model_datetime_index : pandas.DatetimeIndex or None
        Calendar timestamp for each position in `model_power_kW.index_set()`.
        Required when `model` is given.

    varstr : str or None
        Name of the pyomo `Var` to create for the baseline (its defining
        constraint is named `varstr + "_constraint"`). Must be unique per
        call on a given `model`. Required when `model` is given.

    Raises
    ------
    ValueError
        When zero valid baseline days remain after filtering; when a valid
        baseline day has no data in its event window; when `model` is given
        but `model_power_kW`, `model_datetime_index`, or `varstr` is not; or
        when `model_datetime_index` has fewer than 2 entries.

    Warnings
        When fewer valid baseline days remain than `N_BASELINE_DAYS`.

    Returns
    -------
    float or tuple
        `float`: mean baseline power in kW, when `model` is `None`.
        `(baseline_kW, model)`: when `model` is given, where `baseline_kW`
        is a `float` if every baseline day stayed historical, or a
        `pyomo.environ.Var` if at least one was dynamic.
    """
    if model is not None and any(
        a is None for a in (model_power_kW, model_datetime_index, varstr)
    ):
        raise ValueError(
            "model_power_kW, model_datetime_index, and varstr are all required "
            "when model is given"
        )

    candidate_days = [pd.Timestamp(d) for d in event[BASELINE_DAYS]]

    if baseline_params[EXCLUDE_WEEKENDS]:
        candidate_days = [d for d in candidate_days if d.weekday() < 5]
    if baseline_params[EXCLUDE_HOLIDAYS]:
        holidays = {pd.Timestamp(d) for d in baseline_params[HOLIDAY_DATES]}
        candidate_days = [d for d in candidate_days if d not in holidays]

    if len(candidate_days) == 0:
        raise ValueError("No valid baseline days remain after filtering")
    if len(candidate_days) < baseline_params[N_BASELINE_DAYS]:
        warnings.warn(
            f"Only {len(candidate_days)} valid baseline days available, "
            f"fewer than the requested {baseline_params[N_BASELINE_DAYS]}",
            UserWarning,
        )

    valid_days = sorted(candidate_days, reverse=True)[
        : baseline_params[N_BASELINE_DAYS]
    ]

    model_var_index = (
        list(model_power_kW.index_set()) if model_power_kW is not None else None
    )
    step = (
        _infer_datetime_index_step(model_datetime_index)
        if model_datetime_index is not None
        else None
    )

    all_terms = []
    any_dynamic = False
    for day in valid_days:
        terms, is_dynamic = _baseline_day_terms(
            day,
            event[EVENT_START_HOUR],
            event[EVENT_DURATION],
            historical_power_kW,
            model_power_kW,
            model_var_index,
            model_datetime_index,
            step,
        )
        all_terms.extend(terms)
        any_dynamic = any_dynamic or is_dynamic

    baseline_kW = sum(weight * value for weight, value in all_terms) / len(valid_days)
    if not any_dynamic:
        baseline_kW = float(baseline_kW)

    # The day-of adjustment factor is always computed from historical
    # `historical_power_kW`, never from `model_power_kW`, even if its window
    # falls inside the simulation horizon -- see docstring for why.
    adjustment_hours = baseline_params[ADJUSTMENT_HOURS]
    if adjustment_hours is not None:
        event_adj_mask = _event_window_mask(
            historical_power_kW.index,
            event[EVENT_DATE],
            event[EVENT_START_HOUR] - adjustment_hours,
            adjustment_hours,
        )
        event_adj_mean = historical_power_kW.loc[event_adj_mask].mean()

        baseline_adj_means = []
        for day in valid_days:
            mask = _event_window_mask(
                historical_power_kW.index,
                day,
                event[EVENT_START_HOUR] - adjustment_hours,
                adjustment_hours,
            )
            baseline_adj_means.append(historical_power_kW.loc[mask].mean())
        baseline_adj_mean = np.mean(baseline_adj_means)

        if np.isclose(baseline_adj_mean, 0, atol=1e-9):
            warnings.warn(
                "Day-of adjustment denominator is near zero; skipping adjustment",
                UserWarning,
            )
            factor = 1.0
        else:
            factor = event_adj_mean / baseline_adj_mean
            factor = np.clip(factor, *baseline_params[ADJUSTMENT_CLIP])
        baseline_kW *= factor

    if model is None:
        return baseline_kW
    if not any_dynamic:
        return baseline_kW, model

    model.add_component(varstr, pyo.Var())
    baseline_var = model.find_component(varstr)

    def baseline_rule(m):
        return baseline_var == baseline_kW

    model.add_component(varstr + "_constraint", pyo.Constraint(rule=baseline_rule))
    return baseline_var, model


def build_event_revenue(
    power_kW,
    event,
    baseline_kW,
    payment_function,
    region_x1=None,
    model=None,
    varstr="",
):
    """Calculate or build the demand response revenue for a single event.

    `power_kW` must already be sliced to this event's own time window
    (the caller's responsibility), matching the convention in `costs.py` of
    passing in already-relevant consumption slices.

    When calling this repeatedly (e.g., once per event) with a pyomo `model`,
    `varstr` must be unique per call -- reusing a `varstr` on the same model
    raises a pyomo "component already exists" error.

    Parameters
    ----------
    power_kW : numpy.ndarray, cvxpy.Expression, cvxpy.Variable, or pyomo.environ.Var
        Actual power consumption during the event window, either a realized
        numpy array or a decision-variable expression.

    event : dict
        A single event, as produced by `add_event`.

    baseline_kW : float
        Precomputed baseline power in kW for this event's window (e.g., from
        `calculate_event_baseline`). Always a constant, never a decision
        variable.

    payment_function : list of dict
        Payment/penalty schedule to apply, as a list of region dicts (see
        `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`).

    region_x1 : float or None
        The `x1` value identifying which payment-function region to build.
        Required when `power_kW` is a cvxpy or pyomo type, since the
        applicable region cannot be known before solving. Ignored when
        `power_kW` is numpy/scalar, since the actual region is already fully
        determined.

    model : pyomo.environ.Model or pyomo.environ.Block
        The model or block to add pyomo components to.
        Only used in the pyomo case, so `None` by default.

    varstr : str
        Name prefix for pyomo variables/constraints created on `model`.

    Raises
    ------
    ValueError
        When `power_kW` is a cvxpy or pyomo type and `region_x1` is `None`.

    TypeError
        When `power_kW` is not a supported type.

    Returns
    -------
    tuple
        `(revenue, model)` for numpy `power_kW`, `(revenue_var, model)` for
        pyomo `power_kW`, or `(revenue_expr, constraints_list)` for cvxpy
        `power_kW`.
    """
    bid_capacity_kW = event[BID_CAPACITY_KW]
    capacity_price = event[CAPACITY_PRICE]

    if ut.check_indexed_np_array(power_kW) or ut.check_nonindexed_python_type(power_kW):
        reduction_kW = baseline_kW - np.mean(power_kW)
        return (
            evaluate_payment_function(
                payment_function, reduction_kW, bid_capacity_kW, capacity_price
            ),
            model,
        )
    elif ut.check_cvx_type(power_kW):
        if region_x1 is None:
            raise ValueError("region_x1 must be specified for cvxpy power_kW")
        mean_power = cp.sum(power_kW) / power_kW.size
        reduction_kW = baseline_kW - mean_power
        return build_payment_expression(
            payment_function,
            reduction_kW,
            bid_capacity_kW,
            capacity_price,
            region_x1=region_x1,
            model=model,
            varstr=varstr,
        )
    elif ut.check_indexed_pyomo_type(power_kW) or ut.check_nonindexed_pyomo_type(
        power_kW
    ):
        if region_x1 is None:
            raise ValueError("region_x1 must be specified for pyomo power_kW")
        n = len(power_kW)
        mean_power = pyo.quicksum(power_kW[t] for t in power_kW.index_set()) / n
        reduction_kW = baseline_kW - mean_power
        return build_payment_expression(
            payment_function,
            reduction_kW,
            bid_capacity_kW,
            capacity_price,
            region_x1=region_x1,
            model=model,
            varstr=varstr,
        )
    else:
        raise TypeError(
            "power_kW must be numpy.ndarray, a Python number, "
            "cvxpy.Expression/Variable, or pyomo.environ.Var"
        )


def calculate_event_revenue(
    historical_power_kW, event, baseline_params, payment_function
):
    """Calculate ex-post demand response revenue for a single event.

    Parameters
    ----------
    historical_power_kW : pandas.Series
        Realized power consumption in kW, indexed by `pandas.DatetimeIndex`,
        covering at least the event's window and its baseline days.

    event : dict
        A single event, as produced by `add_event`.

    baseline_params : dict
        Baseline parameters, as produced by `make_baseline_parameters`.

    payment_function : list of dict
        Payment/penalty schedule to apply, as a list of region dicts (see
        `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`).

    Raises
    ------
    ValueError
        When `historical_power_kW` has no data in the event's window.

    Returns
    -------
    dict
        Per-event results with keys `EVENT_DATE`, `BASELINE_KW`, `ACTUAL_KW`,
        `REDUCTION_KW`, `DELIVERED_RATIO`, and `REVENUE`.
    """
    mask = _event_window_mask(
        historical_power_kW.index,
        event[EVENT_DATE],
        event[EVENT_START_HOUR],
        event[EVENT_DURATION],
    )
    actual_kW = historical_power_kW.loc[mask].values
    if actual_kW.size == 0:
        raise ValueError("No data available for event window")

    baseline_kW = calculate_event_baseline(historical_power_kW, event, baseline_params)
    revenue, _ = build_event_revenue(
        actual_kW, event, baseline_kW, payment_function=payment_function
    )

    actual_mean = float(np.mean(actual_kW))
    reduction_kW = baseline_kW - actual_mean

    return {
        EVENT_DATE: event[EVENT_DATE],
        BASELINE_KW: baseline_kW,
        ACTUAL_KW: actual_mean,
        REDUCTION_KW: reduction_kW,
        DELIVERED_RATIO: reduction_kW / event[BID_CAPACITY_KW],
        REVENUE: revenue,
    }


def calculate_dr_revenue(
    historical_power_kW, events, baseline_params, payment_function
):
    """Calculate ex-post demand response revenue across all events.

    Each event is processed independently: its own window is re-sliced from
    `historical_power_kW` and its own baseline is recomputed, so results
    never mix time windows across different events.

    Parameters
    ----------
    historical_power_kW : pandas.Series
        Realized power consumption in kW, indexed by `pandas.DatetimeIndex`.

    events : list of dict or pandas.DataFrame
        Events collection, as produced by `add_event`.

    baseline_params : dict
        Baseline parameters, as produced by `make_baseline_parameters`.

    payment_function : list of dict
        Payment/penalty schedule to apply, as a list of region dicts (see
        `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`).

    Returns
    -------
    tuple
        `(per_event_df, total_revenue)`, where `per_event_df` is a
        `pandas.DataFrame` with one row per event (see
        `calculate_event_revenue`) and `total_revenue` is the sum of the
        `REVENUE` column in USD.
    """
    events_df = events_to_dataframe(events)
    results = [
        calculate_event_revenue(
            historical_power_kW, row.to_dict(), baseline_params, payment_function
        )
        for _, row in events_df.iterrows()
    ]
    per_event_df = pd.DataFrame(results)
    total_revenue = per_event_df[REVENUE].sum()
    return per_event_df, total_revenue


def build_dr_revenue(
    power_kW,
    datetime_index,
    events,
    historical_power_kW,
    baseline_params,
    model,
    payment_function,
    region_x1s,
    varstr_prefix="dr_event",
):
    """Build pyomo revenue expressions for all events and net them into model.objective.

    Slices `power_kW` to each event's window internally (via `datetime_index`)
    and computes each event's baseline internally (via `historical_power_kW`
    and `calculate_event_baseline`) -- the caller only needs to supply the
    model's full power variable, a historical consumption series, and an
    assumed payment-function region per event.

    Processes events in `EVENT_DATE` order (like `calculate_dr_revenue`).

    If `model` already has an `objective` component (e.g. built by
    `costs.build_pyomo_costing`), the total DR revenue is subtracted from it
    in place, the same way `calculate_cost` already nets export revenue
    against cost (`cost -= new_cost`). Otherwise a new minimize objective of
    `-total_revenue` is created (minimizing negative revenue == maximizing
    revenue).

    Parameters
    ----------
    power_kW : pyomo.environ.Var
        Full time-indexed decision variable for actual power consumption
        over the optimization horizon.

    datetime_index : pandas.DatetimeIndex
        Calendar timestamp for each position in `power_kW`'s index set, in
        the same order as `list(power_kW.index_set())` -- pyomo index sets
        carry no calendar information of their own, so this is required to
        determine which positions fall in each event's window.

    events : list of dict or pandas.DataFrame
        Events collection, as produced by `add_event`.

    historical_power_kW : pandas.Series
        Realized historical power consumption, indexed by
        `pandas.DatetimeIndex`. Used as a fallback source for each event's
        baseline: a baseline day whose window is fully contained in the
        simulation horizon spanned by `power_kW`/`datetime_index` is instead
        computed as a decision-variable average over `power_kW` itself (see
        `calculate_event_baseline`), and the day-of adjustment factor (if
        configured) is always computed from `historical_power_kW` -- see
        `calculate_event_baseline`'s docstring for why.

    baseline_params : dict
        Baseline parameters, as produced by `make_baseline_parameters`.

    model : pyomo.environ.Model or pyomo.environ.Block
        The model to add components to and whose objective is updated.

    payment_function : list of dict
        Payment/penalty schedule shared by all events.

    region_x1s : dict
        Assumed payment-function region's `x1`, keyed by event date (any
        value `pandas.Timestamp` can parse), e.g.
        `{"2024-01-08": 0.6, "2024-01-15": 0.75}`.

    varstr_prefix : str
        Prefix for the per-event `varstr` passed to `build_payment_expression`
        (combined with the event's position in date order). Must be unique
        per call on a given `model`.

    Raises
    ------
    ValueError
        When an event's window has no matching positions in `datetime_index`.

    KeyError
        When `region_x1s` has no entry for an event's date.

    Returns
    -------
    tuple
        `(total_revenue, model)`.
    """
    events_df = events_to_dataframe(events)
    var_index = list(power_kW.index_set())
    region_x1_by_date = {pd.Timestamp(k): v for k, v in region_x1s.items()}

    total_revenue = 0
    for i, row in events_df.iterrows():
        event = row.to_dict()
        baseline_kW, model = calculate_event_baseline(
            historical_power_kW,
            event,
            baseline_params,
            model=model,
            model_power_kW=power_kW,
            model_datetime_index=datetime_index,
            varstr=f"{varstr_prefix}_{i}_baseline_kW",
        )

        mask = _event_window_mask(
            datetime_index,
            event[EVENT_DATE],
            event[EVENT_START_HOUR],
            event[EVENT_DURATION],
        )
        matched_indices = [idx for idx, keep in zip(var_index, mask) if keep]
        if not matched_indices:
            raise ValueError(
                f"No data available for event window on {event[EVENT_DATE]}"
            )
        mean_power = pyo.quicksum(power_kW[idx] for idx in matched_indices) / len(
            matched_indices
        )
        reduction_kW = baseline_kW - mean_power

        revenue_var, model = build_payment_expression(
            payment_function,
            reduction_kW,
            event[BID_CAPACITY_KW],
            event[CAPACITY_PRICE],
            region_x1=region_x1_by_date[event[EVENT_DATE]],
            model=model,
            varstr=f"{varstr_prefix}_{i}",
        )
        total_revenue += revenue_var

    if hasattr(model, "objective"):
        model.objective.expr -= total_revenue
    else:
        model.objective = pyo.Objective(expr=-total_revenue, sense=pyo.minimize)

    return total_revenue, model

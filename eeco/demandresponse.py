"""Calculate incentive-based demand response revenue.

See :doc:`/demandresponse` for a description of the module.
"""

import warnings
import holidays
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
ADJUSTMENT_FACTOR = "adjustment_factor"

# Baseline parameter dict keys
BASELINE_METHOD = "baseline_method"
N_BASELINE_DAYS = "n_baseline_days"
ADJUSTMENT_OFFSET_HOURS = "adjustment_offset_hours"
ADJUSTMENT_DURATION_HOURS = "adjustment_duration_hours"
ADJUSTMENT_CLIP = "adjustment_clip"
EXCLUDE_WEEKENDS = "exclude_weekends"
EXCLUDE_HOLIDAYS = "exclude_holidays"
HOLIDAY_COUNTRY = "holiday_country"
HOLIDAY_SUBDIV = "holiday_subdiv"
HOLIDAY_DATES = "holiday_dates"
RESOLUTION = "resolution"

# Output/result column keys
BASELINE_KW = "baseline_kW"
ACTUAL_KW = "actual_kW"
REDUCTION_KW = "reduction_kW"
DELIVERED_RATIO = "delivered_ratio"
REVENUE = "revenue"
# Per-interval companions to the scalar (event-window-mean) keys above.
BASELINE_PROFILE_KW = "baseline_profile_kW"
ACTUAL_PROFILE_KW = "actual_profile_kW"
REDUCTION_PROFILE_KW = "reduction_profile_kW"
INTERVAL_DATETIME = "interval_datetime"

# Payment function region dict keys.
REGION_X1 = "x1"
REGION_X2 = "x2"
REGION_Y1 = "y1"
REGION_Y2 = "y2"

# PaymentStructure settlement modes.
SETTLEMENT_AVERAGE = "average"
SETTLEMENT_INTERVAL = "interval"
SETTLEMENT_MODES = (SETTLEMENT_AVERAGE, SETTLEMENT_INTERVAL)

# PaymentStructure payment_basis / payout_basis values.
BASIS_PER_EVENT = "per_event"
BASIS_PER_HOUR = "per_hour"
PAYMENT_BASES = (BASIS_PER_EVENT, BASIS_PER_HOUR)

# Pyomo component suffixes, appended to a `varstr` as `varstr + "_" + suffix`
CONSTRAINT_SUFFIX = "constraint"
ADJUSTMENT_FACTOR_SUFFIX = "adjustment_factor"
INTERRUPTION_CONSTRAINT_SUFFIX = "interruption_constraint"
REGION_ACTIVE_SUFFIX = "region_active"
REGION_SELECT_CONSTRAINT_SUFFIX = "region_select_constraint"
REGION_REDUCTION_SUFFIX = "region_reduction"
REGION_REDUCTION_LOWER_CONSTRAINT_SUFFIX = "region_reduction_lower_constraint"
REGION_REDUCTION_UPPER_CONSTRAINT_SUFFIX = "region_reduction_upper_constraint"
REGION_REDUCTION_SUM_CONSTRAINT_SUFFIX = "region_reduction_sum_constraint"
INTERVAL_REVENUE_SUFFIX = "interval_revenue"
INTERVAL_REVENUE_CONSTRAINT_SUFFIX = "interval_revenue_constraint"
REVENUE_SUFFIX = "revenue"
REVENUE_CONSTRAINT_SUFFIX = "revenue_constraint"
ENERGY_REVENUE_SUFFIX = "energy_revenue"
ENERGY_REVENUE_CONSTRAINT_SUFFIX = "energy_revenue_constraint"
TOTAL_REVENUE_SUFFIX = "total_revenue"
TOTAL_REVENUE_CONSTRAINT_SUFFIX = "total_revenue_constraint"
BASELINE_COMPONENT_SUFFIXES = (
    CONSTRAINT_SUFFIX,
    ADJUSTMENT_FACTOR_SUFFIX,
    INTERRUPTION_CONSTRAINT_SUFFIX,
)
PAYMENT_COMPONENT_SUFFIXES = (
    REGION_ACTIVE_SUFFIX,
    REGION_SELECT_CONSTRAINT_SUFFIX,
    REGION_REDUCTION_SUFFIX,
    REGION_REDUCTION_LOWER_CONSTRAINT_SUFFIX,
    REGION_REDUCTION_UPPER_CONSTRAINT_SUFFIX,
    REGION_REDUCTION_SUM_CONSTRAINT_SUFFIX,
    INTERVAL_REVENUE_SUFFIX,
    INTERVAL_REVENUE_CONSTRAINT_SUFFIX,
    REVENUE_SUFFIX,
    REVENUE_CONSTRAINT_SUFFIX,
    ENERGY_REVENUE_SUFFIX,
    ENERGY_REVENUE_CONSTRAINT_SUFFIX,
    TOTAL_REVENUE_SUFFIX,
    TOTAL_REVENUE_CONSTRAINT_SUFFIX,
)


def _component_name(varstr, suffix):
    """Join a `varstr` and a component suffix into a pyomo component name.

    Parameters
    ----------
    varstr : str
        Name prefix shared by a group of pyomo components.

    suffix : str
        One of the module's `*_SUFFIX` constants.

    Returns
    -------
    str
        `varstr + "_" + suffix`.
    """
    return varstr + "_" + suffix


class BaselineMethod:
    """Average-of-similar-days baseline with an optional day-of adjustment.

    Parameters
    ----------
    n_baseline_days : int
        Number of eligible baseline days to average. `0` disables baselining
        and yields an all-zero baseline.

    adjustment_offset_hours : int or None
        Hours before the event start at which the day-of adjustment window
        begins. `None` disables the day-of adjustment.

    adjustment_duration_hours : int
        Length of the day-of adjustment window in hours. Ignored when
        `adjustment_offset_hours` is `None`.

    adjustment_clip : tuple of float
        `(low, high)` bounds on the day-of adjustment factor.

    exclude_weekends : bool
        If `True`, drop Saturdays and Sundays from the candidate baseline days.

    exclude_holidays : bool
        If `True`, drop holidays from the candidate baseline days.

    holiday_country : str or None
        ISO 3166-1 country code passed to `holidays.country_holidays`.
        `None` uses only `holiday_dates`.

    holiday_subdiv : str or None
        Subdivision code (e.g., `"CA"`) passed to `holidays.country_holidays`.

    holiday_dates : list or None
        Additional dates to treat as holidays.

    adjustment_in_model : bool
        If `True` and a `model` is given to `compute`, add the day-of
        adjustment factor to the model as a fixed `pyomo.environ.Var`.

    resolution : str or None
        Settlement interval width (e.g., `"15m"`, `"1h"`). `None` infers it
        from the data or model index.

    Raises
    ------
    ValueError
        If `n_baseline_days` is negative, or if `adjustment_offset_hours` is
        not `None` and `adjustment_duration_hours` is not positive or exceeds
        `adjustment_offset_hours`.
    """

    def __init__(
        self,
        n_baseline_days=10,
        adjustment_offset_hours=3,
        adjustment_duration_hours=3,
        adjustment_clip=(0.8, 1.2),
        exclude_weekends=True,
        exclude_holidays=True,
        holiday_country=None,
        holiday_subdiv=None,
        holiday_dates=None,
        adjustment_in_model=False,
        resolution=None,
    ):
        if n_baseline_days < 0:
            raise ValueError("n_baseline_days must be non-negative")
        if adjustment_offset_hours is not None:
            if adjustment_duration_hours <= 0:
                raise ValueError("adjustment_duration_hours must be positive")
            if adjustment_duration_hours > adjustment_offset_hours:
                raise ValueError(
                    "adjustment_duration_hours must not exceed "
                    "adjustment_offset_hours, so the adjustment window ends at "
                    "or before the event start"
                )
        self.n_baseline_days = n_baseline_days
        self.adjustment_offset_hours = adjustment_offset_hours
        self.adjustment_duration_hours = adjustment_duration_hours
        self.adjustment_clip = adjustment_clip
        self.exclude_weekends = exclude_weekends
        self.exclude_holidays = exclude_holidays
        self.holiday_country = holiday_country
        self.holiday_subdiv = holiday_subdiv
        self.holiday_dates = list(holiday_dates) if holiday_dates else []
        self.adjustment_in_model = adjustment_in_model
        self.resolution = resolution

    def _is_holiday(self, day):
        """Check whether a day is a holiday under this method's configuration.

        Parameters
        ----------
        day : pandas.Timestamp
            Calendar day to check.

        Returns
        -------
        bool
            `True` if `day` is a `holiday_country`/`holiday_subdiv` holiday or
            is in `holiday_dates`.
        """
        if day in {pd.Timestamp(d) for d in self.holiday_dates}:
            return True
        if self.holiday_country is None:
            return False
        calendar = holidays.country_holidays(
            self.holiday_country, subdiv=self.holiday_subdiv
        )
        return day in calendar

    def _rank_days(self, candidate_days, historical_power_kW, event):
        """Filter out ineligible days and rank the rest, most preferred first.

        Parameters
        ----------
        candidate_days : list of pandas.Timestamp
            Days proposed for this event's baseline.

        historical_power_kW : pandas.Series
            Historical power consumption in kW, indexed by timestamp.

        event : dict
            Single event dict. See `add_event` for the keys.

        Returns
        -------
        list of pandas.Timestamp
            Eligible days, most recent first.
        """
        if self.exclude_weekends:
            candidate_days = [d for d in candidate_days if d.weekday() < 5]
        if self.exclude_holidays:
            candidate_days = [d for d in candidate_days if not self._is_holiday(d)]
        return sorted(candidate_days, reverse=True)

    def select_days(self, candidate_days, historical_power_kW, event):
        """Select the baseline days to average for an event.

        Parameters
        ----------
        candidate_days : list of pandas.Timestamp
            Days proposed for this event's baseline.

        historical_power_kW : pandas.Series
            Historical power consumption in kW, indexed by timestamp.

        event : dict
            Single event dict. See `add_event` for the keys.

        Raises
        ------
        ValueError
            If no eligible days remain after filtering.

        Warns
        -----
        UserWarning
            If fewer eligible days remain than `n_baseline_days`.

        Returns
        -------
        list of pandas.Timestamp
            At most `n_baseline_days` days, most preferred first.
        """
        ranked = self._rank_days(candidate_days, historical_power_kW, event)
        if len(ranked) == 0:
            raise ValueError("No valid baseline days remain after filtering")
        if len(ranked) < self.n_baseline_days:
            warnings.warn(
                f"Only {len(ranked)} valid baseline days available, "
                f"fewer than the requested {self.n_baseline_days}",
                UserWarning,
            )
        return ranked[: self.n_baseline_days]

    def _adjustment_factor(self, valid_days, historical_power_kW, event):
        """Calculate the day-of adjustment factor for an event.

        Parameters
        ----------
        valid_days : list of pandas.Timestamp
            Baseline days selected for this event.

        historical_power_kW : pandas.Series
            Historical power consumption in kW, indexed by timestamp.

        event : dict
            Single event dict. See `add_event` for the keys.

        Warns
        -----
        UserWarning
            If the baseline days' mean power in the adjustment window is near
            zero, in which case a factor of `1.0` is returned.

        Returns
        -------
        float or None
            Clipped adjustment factor, or `None` if `adjustment_offset_hours`
            is `None`.
        """
        if self.adjustment_offset_hours is None:
            return None
        index = historical_power_kW.index
        window_start_hour = event[EVENT_START_HOUR] - self.adjustment_offset_hours
        event_adj_mask = _event_window_mask(
            index,
            event[EVENT_DATE],
            window_start_hour,
            self.adjustment_duration_hours,
        )
        event_adj_mean = historical_power_kW.loc[event_adj_mask].mean()

        # Pool every baseline day's adjustment window into one mean
        baseline_adj_mask = np.zeros(len(index), dtype=bool)
        for day in valid_days:
            baseline_adj_mask |= _event_window_mask(
                index, day, window_start_hour, self.adjustment_duration_hours
            )
        baseline_adj_mean = historical_power_kW.loc[baseline_adj_mask].mean()

        if np.isclose(baseline_adj_mean, 0, atol=1e-9):
            warnings.warn(
                "Day-of adjustment denominator is near zero; skipping adjustment",
                UserWarning,
            )
            factor = 1.0
        else:
            factor = event_adj_mean / baseline_adj_mean
            factor = np.clip(factor, *self.adjustment_clip)
        return float(factor)

    def compute(
        self,
        historical_power_kW,
        event,
        *,
        model=None,
        model_power_kW=None,
        model_datetime_index=None,
        varstr=None,
        adjustment_factor=None,
    ):
        """Calculate the per-interval baseline power for a single event.

        Parameters
        ----------
        historical_power_kW : pandas.Series
            Historical power consumption in kW, indexed by timestamp.

        event : dict
            Single event dict. See `add_event` for the keys.

        model : pyomo.environ.Model or pyomo.environ.Block or None
            Model to add baseline components to. `None` for ex-post use.

        model_power_kW : pyomo.environ.Var or None
            Power decision variable over the optimization horizon. Required
            when `model` is given.

        model_datetime_index : pandas.DatetimeIndex or None
            Timestamp of each entry of `model_power_kW`. Required when `model`
            is given.

        varstr : str or None
            Name of the baseline `Var` created on `model`. Required when
            `model` is given.

        adjustment_factor : float or None
            Day-of adjustment factor to apply as-is. `None` uses
            `event["adjustment_factor"]` if set, otherwise calculates it with
            `_adjustment_factor`.

        Raises
        ------
        ValueError
            If `model` is given without `model_power_kW`,
            `model_datetime_index`, and `varstr`, if a selected baseline day
            has missing data, or if `adjustment_factor` is not positive or
            differs from `event["adjustment_factor"]`.

        Returns
        -------
        numpy.ndarray or tuple
            Per-interval baseline in kW when `model` is `None`, otherwise
            `(baseline_kW, model)` where `baseline_kW` is a `numpy.ndarray` or
            an indexed `pyomo.environ.Var`.
        """
        if model is not None and any(
            a is None for a in (model_power_kW, model_datetime_index, varstr)
        ):
            raise ValueError(
                "model_power_kW, model_datetime_index, and varstr are all required "
                "when model is given"
            )

        step = _resolve_step(
            self.resolution,
            model_datetime_index,
            getattr(historical_power_kW, "index", None),
        )
        n_intervals = _window_interval_count(event[EVENT_DURATION], step)

        if self.n_baseline_days == 0:
            # No baselining: there are no days to select or adjust against
            baseline_kW = np.zeros(n_intervals)
            if model is None:
                return baseline_kW
            return baseline_kW, model

        candidate_days = [pd.Timestamp(d) for d in event[BASELINE_DAYS]]
        valid_days = self.select_days(candidate_days, historical_power_kW, event)

        model_var_index = (
            list(model_power_kW.index_set()) if model_power_kW is not None else None
        )

        # interval_values[k] collects interval k's value from every baseline day
        interval_values = [[] for _ in range(n_intervals)]
        any_dynamic = False
        for day in valid_days:
            values, is_dynamic = _baseline_day_interval_values(
                day,
                event[EVENT_START_HOUR],
                event[EVENT_DURATION],
                step,
                n_intervals,
                historical_power_kW,
                model_power_kW,
                model_var_index,
                model_datetime_index,
            )
            for k, value in enumerate(values):
                interval_values[k].append(value)
            any_dynamic = any_dynamic or is_dynamic

        if any_dynamic:
            baseline_kW = [
                pyo.quicksum(vals) / len(valid_days) for vals in interval_values
            ]
        else:
            baseline_kW = np.array(
                [sum(vals) / len(valid_days) for vals in interval_values]
            )

        factor = _resolve_adjustment_factor(event, adjustment_factor)
        if factor is None:
            factor = self._adjustment_factor(valid_days, historical_power_kW, event)
        factor_in_model = (
            factor is not None and model is not None and self.adjustment_in_model
        )
        if factor_in_model:
            factor_name = _component_name(varstr, ADJUSTMENT_FACTOR_SUFFIX)
            model.add_component(factor_name, pyo.Var(initialize=factor))
            factor_var = model.find_component(factor_name)
            # Fixing is what makes this an input rather than a decision. Left
            # free, the solver would choose whichever factor maximizes revenue
            # (inflating the baseline without bound), and `baseline * factor`
            # would be bilinear; fixed, it collapses to a linear coefficient.
            # Retune with `.fix(new_value)` and re-solve -- no rebuild needed.
            factor_var.fix(factor)
            baseline_kW = [value * factor_var for value in baseline_kW]
            any_dynamic = True
        elif factor is not None:
            if any_dynamic:
                baseline_kW = [value * factor for value in baseline_kW]
            else:
                baseline_kW = baseline_kW * factor

        if model is None:
            return baseline_kW
        # A factor-scaled baseline is a symbolic expression, so it needs a Var
        # to stand for it even when every baseline day was historical.
        if not any_dynamic:
            return baseline_kW, model

        interval_idx = range(n_intervals)
        model.add_component(varstr, pyo.Var(interval_idx))
        baseline_var = model.find_component(varstr)

        def baseline_rule(m, t):
            return baseline_var[t] == baseline_kW[t]

        model.add_component(
            _component_name(varstr, CONSTRAINT_SUFFIX),
            pyo.Constraint(interval_idx, rule=baseline_rule),
        )
        return baseline_var, model


class TopUsageDaysBaseline(BaselineMethod):
    """Baseline that averages the highest-usage eligible days.

    Takes the same parameters as `BaselineMethod`.
    """

    def _rank_days(self, candidate_days, historical_power_kW, event):
        """Filter out ineligible days and rank the rest by event-window usage.

        Parameters
        ----------
        candidate_days : list of pandas.Timestamp
            Days proposed for this event's baseline.

        historical_power_kW : pandas.Series
            Historical power consumption in kW, indexed by timestamp.

        event : dict
            Single event dict. See `add_event` for the keys.

        Returns
        -------
        list of pandas.Timestamp
            Eligible days, highest mean event-window power first.
        """
        eligible_days = super()._rank_days(candidate_days, historical_power_kW, event)

        def day_mean(day):
            mask = _event_window_mask(
                historical_power_kW.index,
                day,
                event[EVENT_START_HOUR],
                event[EVENT_DURATION],
            )
            day_slice = historical_power_kW.loc[mask]
            return day_slice.mean() if not day_slice.empty else -np.inf

        return sorted(eligible_days, key=day_mean, reverse=True)


class FixedLevelBaseline(BaselineMethod):
    """Baseline fixed at a contracted firm service level.

    Parameters
    ----------
    firm_level_kW : float
        Contracted firm demand level in kW.

    resolution : str or None
        Settlement interval width (e.g., `"15m"`, `"1h"`). `None` infers it
        from the data or model index.

    Raises
    ------
    ValueError
        If `firm_level_kW` is negative.
    """

    def __init__(self, firm_level_kW, resolution=None):
        if firm_level_kW < 0:
            raise ValueError("firm_level_kW must be non-negative")
        self.firm_level_kW = firm_level_kW
        self.resolution = resolution

    def compute(
        self,
        historical_power_kW,
        event,
        *,
        model=None,
        model_power_kW=None,
        model_datetime_index=None,
        varstr=None,
        adjustment_factor=None,
    ):
        """Return the firm service level as a flat per-interval baseline.

        Parameters
        ----------
        historical_power_kW : pandas.Series
            Historical power consumption in kW. Only its index is used, to
            infer the interval width.

        event : dict
            Single event dict. See `add_event` for the keys.

        model : pyomo.environ.Model or pyomo.environ.Block or None
            Determines the return type only. No components are added.

        model_power_kW : pyomo.environ.Var or None
            Unused.

        model_datetime_index : pandas.DatetimeIndex or None
            Used only to infer the interval width.

        varstr : str or None
            Unused.

        adjustment_factor : float or None
            Unused.

        Returns
        -------
        numpy.ndarray or tuple
            `firm_level_kW` repeated once per interval when `model` is `None`,
            otherwise `(baseline_kW, model)`.
        """
        step = _resolve_step(
            self.resolution,
            model_datetime_index,
            getattr(historical_power_kW, "index", None),
        )
        n_intervals = _window_interval_count(event[EVENT_DURATION], step)
        baseline_kW = np.full(n_intervals, self.firm_level_kW, dtype=float)
        if model is None:
            return baseline_kW
        return baseline_kW, model


class UnilateralInterruptionBaseline(BaselineMethod):
    """Utility-controlled interruption enforced as an upper bound on power.

    Parameters
    ----------
    interruption_level_kW : float
        Power level in kW the load is held at or below during an event.

    resolution : str or None
        Settlement interval width (e.g., `"15m"`, `"1h"`). `None` infers it
        from the model index.
    """

    def __init__(self, interruption_level_kW=0.0, resolution=None):
        self.interruption_level_kW = interruption_level_kW
        self.resolution = resolution

    def compute(
        self,
        historical_power_kW,
        event,
        *,
        model=None,
        model_power_kW=None,
        model_datetime_index=None,
        varstr=None,
        adjustment_factor=None,
    ):
        """Constrain modeled power to the interruption level during an event.

        Parameters
        ----------
        historical_power_kW : pandas.Series
            Unused.

        event : dict
            Single event dict. See `add_event` for the keys.

        model : pyomo.environ.Model or pyomo.environ.Block
            Model to add the interruption constraint to.

        model_power_kW : pyomo.environ.Var
            Power decision variable over the optimization horizon.

        model_datetime_index : pandas.DatetimeIndex
            Timestamp of each entry of `model_power_kW`.

        varstr : str
            Name prefix for the constraint created on `model`.

        adjustment_factor : float or None
            Unused.

        Raises
        ------
        NotImplementedError
            If `model` is `None`.

        ValueError
            If `model_power_kW`, `model_datetime_index`, or `varstr` is
            missing, or if the event window matches no entries of
            `model_datetime_index`.

        Returns
        -------
        tuple
            `(baseline_kW, model)`, where `baseline_kW` is
            `interruption_level_kW` repeated once per interval.
        """
        if model is None:
            raise NotImplementedError(
                "UnilateralInterruptionBaseline has no ex-post/numpy evaluation -- "
                "it only applies within an optimization model, as a hard constraint."
            )
        if any(a is None for a in (model_power_kW, model_datetime_index, varstr)):
            raise ValueError(
                "model_power_kW, model_datetime_index, and varstr are all required "
                "when model is given"
            )

        var_index = list(model_power_kW.index_set())
        mask = _event_window_mask(
            model_datetime_index,
            event[EVENT_DATE],
            event[EVENT_START_HOUR],
            event[EVENT_DURATION],
        )
        matched_indices = [idx for idx, keep in zip(var_index, mask) if keep]
        if not matched_indices:
            raise ValueError(
                f"No data available for event window on {event[EVENT_DATE]}"
            )

        step = _resolve_step(self.resolution, model_datetime_index)
        n_intervals = _window_interval_count(event[EVENT_DURATION], step)
        baseline_kW = np.full(n_intervals, self.interruption_level_kW, dtype=float)

        def interruption_rule(m, idx):
            return model_power_kW[idx] <= self.interruption_level_kW

        model.add_component(
            _component_name(varstr, INTERRUPTION_CONSTRAINT_SUFFIX),
            pyo.Constraint(matched_indices, rule=interruption_rule),
        )
        return baseline_kW, model


def _resolve_adjustment_factor(event, adjustment_factor):
    """Reconcile an adjustment factor given on the event and as an argument.

    Parameters
    ----------
    event : dict
        Single event dict. See `add_event` for the keys.

    adjustment_factor : float or None
        Adjustment factor passed directly to `compute`.

    Raises
    ------
    ValueError
        If a given factor is not positive, or if both are given and differ.

    Returns
    -------
    float or None
        The supplied factor, or `None` if neither is given.
    """
    event_factor = event.get(ADJUSTMENT_FACTOR)
    # Events read back from a DataFrame carry NaN where no factor was given
    if event_factor is not None and pd.isna(event_factor):
        event_factor = None
    if adjustment_factor is None:
        adjustment_factor = event_factor
    elif event_factor is not None and not np.isclose(adjustment_factor, event_factor):
        raise ValueError(
            f"adjustment_factor ({adjustment_factor}) conflicts with "
            f"event[ADJUSTMENT_FACTOR] ({event_factor}); give one or make them match"
        )
    if adjustment_factor is None:
        return None
    if adjustment_factor <= 0:
        raise ValueError("adjustment_factor must be positive")
    return float(adjustment_factor)


def _coerce_baseline_method(baseline_params):
    """Convert a baseline parameter dict into a `BaselineMethod`.

    Parameters
    ----------
    baseline_params : dict or BaselineMethod
        Baseline parameter dict from `make_baseline_parameters`, or a
        `BaselineMethod` instance.

    Returns
    -------
    BaselineMethod
        `baseline_params` if it is already a `BaselineMethod`, otherwise a
        new `BaselineMethod` with the dict's settings.
    """
    if isinstance(baseline_params, BaselineMethod):
        return baseline_params
    return BaselineMethod(
        n_baseline_days=baseline_params[N_BASELINE_DAYS],
        adjustment_offset_hours=baseline_params[ADJUSTMENT_OFFSET_HOURS],
        adjustment_duration_hours=baseline_params[ADJUSTMENT_DURATION_HOURS],
        adjustment_clip=baseline_params[ADJUSTMENT_CLIP],
        exclude_weekends=baseline_params[EXCLUDE_WEEKENDS],
        exclude_holidays=baseline_params[EXCLUDE_HOLIDAYS],
        holiday_country=baseline_params.get(HOLIDAY_COUNTRY),
        holiday_subdiv=baseline_params.get(HOLIDAY_SUBDIV),
        holiday_dates=baseline_params[HOLIDAY_DATES],
        resolution=baseline_params.get(RESOLUTION),
    )


def _event_window_mask(index, event_date, start_hour, duration_hours):
    """Select the timestamps that fall within an event window on one day.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Timestamps to select from.

    event_date : datetime.date, datetime.datetime, or str
        Calendar date of the window.

    start_hour : float
        Hour of day (0-24) the window begins.

    duration_hours : float
        Length of the window in hours.

    Returns
    -------
    numpy.ndarray
        Boolean mask, `True` for timestamps in
        `[event_date + start_hour, event_date + start_hour + duration_hours)`.
    """
    window_start = pd.Timestamp(event_date) + pd.Timedelta(hours=start_hour)
    window_end = window_start + pd.Timedelta(hours=duration_hours)
    return (index >= window_start) & (index < window_end)


def _index_step(index):
    """Infer the spacing of a regularly spaced `pandas.DatetimeIndex`.

    Parameters
    ----------
    index : pandas.DatetimeIndex or None
        Index to infer the spacing of.

    Raises
    ------
    ValueError
        If `index` has fewer than 2 entries.

    Returns
    -------
    pandas.Timedelta or None
        Gap between the first two entries of `index`, or `None` if `index` is
        `None`.
    """
    if index is None:
        return None
    if len(index) < 2:
        raise ValueError("index must have at least 2 entries to infer its step size")
    return index[1] - index[0]


def _resolve_step(resolution, *indices):
    """Determine the settlement interval width for a baseline calculation.

    Parameters
    ----------
    resolution : str or None
        Interval width (e.g., `"15m"`, `"1h"`). Takes precedence over
        `indices`.

    *indices : pandas.DatetimeIndex or None
        Indices to infer the width from, in order of preference.

    Raises
    ------
    ValueError
        If `resolution` is `None` and no index has at least 2 entries.

    Returns
    -------
    pandas.Timedelta
        Settlement interval width.
    """
    if resolution is not None:
        return pd.Timedelta(minutes=ut.get_freq_binsize_minutes(resolution))
    for index in indices:
        if index is None:
            continue
        return _index_step(index)
    raise ValueError(
        "Could not infer a settlement interval width: pass resolution explicitly, "
        "or supply an index with at least 2 entries"
    )


def _window_interval_count(duration_hours, step):
    """Count the settlement intervals in an event window.

    Parameters
    ----------
    duration_hours : float
        Length of the event window in hours.

    step : pandas.Timedelta
        Settlement interval width.

    Raises
    ------
    ValueError
        If `duration_hours` is not an integer multiple of `step`.

    Returns
    -------
    int
        Number of intervals.
    """
    n = duration_hours * 3600.0 / step.total_seconds()
    if not np.isclose(n, round(n), atol=1e-6):
        raise ValueError(
            f"duration_hours ({duration_hours}) is not an integer multiple of the "
            f"settlement interval ({step}); pass a resolution that evenly divides "
            "the event duration"
        )
    return int(round(n))


def _window_interval_starts(event_date, start_hour, duration_hours, step):
    """List the start timestamp of each settlement interval in an event window.

    Parameters
    ----------
    event_date : datetime.date, datetime.datetime, or str
        Calendar date of the window.

    start_hour : float
        Hour of day (0-24) the window begins.

    duration_hours : float
        Length of the window in hours.

    step : pandas.Timedelta
        Settlement interval width.

    Returns
    -------
    pandas.DatetimeIndex
        Start timestamp of each interval.
    """
    window_start = pd.Timestamp(event_date) + pd.Timedelta(hours=start_hour)
    n_intervals = _window_interval_count(duration_hours, step)
    return pd.DatetimeIndex([window_start + i * step for i in range(n_intervals)])


def _window_interval_buckets(index, event_date, start_hour, duration_hours, step):
    """Group the positions of `index` within an event window by interval.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Timestamps to group.

    event_date : datetime.date, datetime.datetime, or str
        Calendar date of the window.

    start_hour : float
        Hour of day (0-24) the window begins.

    duration_hours : float
        Length of the window in hours.

    step : pandas.Timedelta
        Settlement interval width.

    Raises
    ------
    ValueError
        If `duration_hours` is not an integer multiple of `step`.

    Returns
    -------
    list of numpy.ndarray
        Integer positions of `index` in each interval. Intervals with no
        matching positions are empty arrays.
    """
    window_start = pd.Timestamp(event_date) + pd.Timedelta(hours=start_hour)
    n_intervals = _window_interval_count(duration_hours, step)
    mask = _event_window_mask(index, event_date, start_hour, duration_hours)
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return [np.empty(0, dtype=int) for _ in range(n_intervals)]
    offsets = (index[positions] - window_start) / step
    interval_idx = np.floor(offsets.values.astype(float) + 1e-9).astype(int)
    buckets = [np.empty(0, dtype=int) for _ in range(n_intervals)]
    for k in range(n_intervals):
        buckets[k] = positions[interval_idx == k]
    return buckets


def _as_pyomo_terms(reduction_kW):
    """Convert a pyomo reduction into a list of per-interval terms.

    Parameters
    ----------
    reduction_kW : pyomo.environ.Var, pyomo expression, list, or tuple
        Indexed or scalar `Var`/expression, or a list/tuple of per-interval
        expressions.

    Returns
    -------
    list
        Per-interval terms.
    """
    if isinstance(reduction_kW, (list, tuple)):
        return list(reduction_kW)
    if ut.check_indexed_pyomo_type(reduction_kW):
        return [reduction_kW[i] for i in reduction_kW.index_set()]
    return [reduction_kW]


class PaymentStructure:
    """Piecewise-linear capacity payment with an optional flat payout.

    Parameters
    ----------
    regions : list of dict or None
        Payment schedule. Each dict has float values under the keys `"x1"`,
        `"x2"` (delivered ratio bounds) and `"y1"`, `"y2"` (payment ratios at
        those bounds). Bounds may be `"Infinity"`/`"-Infinity"` strings.
        `None` or `[]` means no capacity payment.

    settlement : str
        `"average"` (default) or `"interval"`.

    payment_basis : str
        `"per_event"` (default) or `"per_hour"`.

    payout : float
        Flat participation payment in $/kW of bid capacity.

    payout_basis : str
        `"per_event"` (default) or `"per_hour"`.

    Raises
    ------
    ValueError
        If `settlement`, `payment_basis`, or `payout_basis` is not a
        supported value.
    """

    def __init__(
        self,
        regions,
        settlement=SETTLEMENT_AVERAGE,
        payment_basis=BASIS_PER_EVENT,
        payout=0.0,
        payout_basis=BASIS_PER_EVENT,
    ):
        if settlement not in SETTLEMENT_MODES:
            raise ValueError(f"settlement must be one of {SETTLEMENT_MODES}")
        if payment_basis not in PAYMENT_BASES:
            raise ValueError(f"payment_basis must be one of {PAYMENT_BASES}")
        if payout_basis not in PAYMENT_BASES:
            raise ValueError(f"payout_basis must be one of {PAYMENT_BASES}")
        self.regions = (
            [{k: float(v) for k, v in region.items()} for region in regions]
            if regions
            else []
        )
        self.settlement = settlement
        self.payment_basis = payment_basis
        self.payout = payout
        self.payout_basis = payout_basis

    def find_region(self, delivered_ratio=None, region_x1=None):
        """Look up a payment region by delivered ratio or by its `x1` bound.

        Parameters
        ----------
        delivered_ratio : float or None
            Delivered ratio contained in the region.

        region_x1 : float or None
            `x1` bound of the region.

        Raises
        ------
        ValueError
            If neither argument is given, if both are given and
            `delivered_ratio` is outside the region identified by
            `region_x1`, if no region matches, or if there are no regions.

        Returns
        -------
        dict
            The matching region.
        """
        if delivered_ratio is None and region_x1 is None:
            raise ValueError("One of delivered_ratio or region_x1 must be given")
        if not self.regions:
            raise ValueError(
                "This PaymentStructure has no regions (payout-only); there is no "
                "capacity payment region to look up"
            )
        if region_x1 is not None:
            region = next(
                (r for r in self.regions if np.isclose(r[REGION_X1], region_x1)),
                None,
            )
            if region is None:
                raise ValueError(f"No region with x1 close to {region_x1}")
            if delivered_ratio is not None and not (
                region[REGION_X1] <= delivered_ratio < region[REGION_X2]
            ):
                raise ValueError(
                    f"delivered_ratio {delivered_ratio} is outside the region with "
                    f"x1={region_x1}"
                )
            return region

        region = next(
            (r for r in self.regions if r[REGION_X1] <= delivered_ratio < r[REGION_X2]),
            None,
        )
        if region is None:
            raise ValueError(
                f"delivered_ratio {delivered_ratio} is not covered by payment_function"
            )
        return region

    def _basis_multiplier(self, event, basis, label):
        """Get the multiplier for a payment or payout basis.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        basis : str
            `"per_event"` or `"per_hour"`.

        label : str
            Attribute name used in the error message.

        Raises
        ------
        ValueError
            If `basis` is `"per_hour"` and `event` has no `"duration_hours"`.

        Returns
        -------
        float
            `1.0` for `"per_event"`, `event["duration_hours"]` for
            `"per_hour"`.
        """
        if basis == BASIS_PER_EVENT:
            return 1.0
        if EVENT_DURATION not in event:
            raise ValueError(
                f"{label}='per_hour' requires event[EVENT_DURATION] (duration_hours) "
                "-- pass duration_hours to evaluate_payment_function/"
                "build_payment_expression, or use the full event dict via "
                "build_event_revenue/calculate_dr_revenue"
            )
        return event[EVENT_DURATION]

    def _payout_amount(self, event):
        """Calculate the flat participation payout for an event.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        Raises
        ------
        ValueError
            If `payout_basis` is `"per_hour"`, `payout` is nonzero, and
            `event` has no `"duration_hours"`.

        Returns
        -------
        float
            Payout in USD.
        """
        if self.payout == 0.0:
            return 0.0
        basis_mult = self._basis_multiplier(event, self.payout_basis, "payout_basis")
        return self.payout * event[BID_CAPACITY_KW] * basis_mult

    def _payment_ratio(self, delivered_ratio):
        """Interpolate the payment ratio at a delivered ratio.

        Parameters
        ----------
        delivered_ratio : float
            Delivered ratio to evaluate.

        Raises
        ------
        ValueError
            If no region contains `delivered_ratio`.

        Returns
        -------
        float
            Payment ratio.
        """
        region = self.find_region(delivered_ratio=delivered_ratio)
        x1, x2, y1, y2 = (
            region[k] for k in (REGION_X1, REGION_X2, REGION_Y1, REGION_Y2)
        )
        if np.isinf(x2):
            return y1
        return y1 + (y2 - y1) * (delivered_ratio - x1) / (x2 - x1)

    def _region_coefficients(self, event):
        """Calculate each region's revenue slope and intercept for an event.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        Raises
        ------
        ValueError
            If `payment_basis` is `"per_hour"` and `event` has no
            `"duration_hours"`.

        Returns
        -------
        tuple of list
            `(slopes, intercepts)`, one entry per region, in $/kW and $.
        """
        bid_capacity_kW = event[BID_CAPACITY_KW]
        capacity_price = event[CAPACITY_PRICE]
        basis_mult = self._basis_multiplier(event, self.payment_basis, "payment_basis")
        slopes = []
        intercepts = []
        for region in self.regions:
            x1, x2, y1, y2 = (
                region[k] for k in (REGION_X1, REGION_X2, REGION_Y1, REGION_Y2)
            )
            slope_ratio = 0.0 if np.isinf(x2) else (y2 - y1) / (x2 - x1)
            slopes.append(capacity_price * slope_ratio * basis_mult)
            intercepts.append(
                capacity_price * bid_capacity_kW * (y1 - slope_ratio * x1) * basis_mult
            )
        return slopes, intercepts

    @staticmethod
    def _as_interval_terms(reduction_kW):
        """Convert a numeric reduction into a list of per-interval floats.

        Parameters
        ----------
        reduction_kW : float or numpy.ndarray
            Scalar or 1-D array of reductions in kW.

        Returns
        -------
        tuple
            `(terms, n_intervals)`, where `terms` is a list of float.
        """
        arr = np.atleast_1d(np.asarray(reduction_kW, dtype=float))
        return list(arr), arr.size

    def evaluate(self, event, reduction_kW):
        """Calculate realized revenue for a known reduction.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        reduction_kW : float or numpy.ndarray
            Load reduction in kW, as a window mean or per interval.

        Raises
        ------
        ValueError
            If `event["bid_capacity_kW"]` is not positive, if no region
            contains a resulting delivered ratio, or if a `"per_hour"` basis
            is used and `event` has no `"duration_hours"`.

        Returns
        -------
        float
            Revenue (positive) or penalty (negative) in USD.
        """
        bid_capacity_kW = event[BID_CAPACITY_KW]
        if bid_capacity_kW <= 0:
            raise ValueError("bid_capacity_kW must be positive")

        if not self.regions:
            capacity_payment = 0.0
        else:
            capacity_price = event[CAPACITY_PRICE]
            basis_mult = self._basis_multiplier(
                event, self.payment_basis, "payment_basis"
            )
            terms, _ = self._as_interval_terms(reduction_kW)
            if self.settlement == SETTLEMENT_INTERVAL and len(terms) > 1:
                ratios = [self._payment_ratio(t / bid_capacity_kW) for t in terms]
                payment_ratio = float(np.mean(ratios))
            else:
                mean_reduction = float(np.mean(terms))
                payment_ratio = self._payment_ratio(mean_reduction / bid_capacity_kW)
            capacity_payment = payment_ratio * capacity_price * bid_capacity_kW
            capacity_payment *= basis_mult

        return capacity_payment + self._payout_amount(event)

    def build_expression(
        self, event, reduction_kW, region_x1=None, model=None, varstr=""
    ):
        """Build the revenue expression for an event.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        reduction_kW : float, numpy.ndarray, cvxpy.Expression, or pyomo.environ.Var
            Load reduction in kW, as a realized value or a decision-variable
            expression, per interval or as a window mean. A list or tuple of
            pyomo expressions is also accepted.

        region_x1 : float, list of float, or None
            `x1` bound of the region to fix. Required for cvxpy. Optional for
            pyomo, where `None` lets the solver choose.

        model : pyomo.environ.Model or pyomo.environ.Block or None
            Model to add components to. Required for pyomo.

        varstr : str
            Name prefix for pyomo components created on `model`.

        Raises
        ------
        ValueError
            If `reduction_kW` is cvxpy and `region_x1` matches no region, if
            `reduction_kW` is pyomo and `model` is `None` or a given
            `region_x1` matches no region.

        TypeError
            If `reduction_kW` is not a supported type.

        Returns
        -------
        tuple
            `(revenue, model)` for numpy or scalar `reduction_kW`,
            `(revenue_var, model)` for pyomo, or
            `(revenue_expr, constraints)` for cvxpy.
        """
        if ut.check_indexed_np_array(reduction_kW) or ut.check_nonindexed_python_type(
            reduction_kW
        ):
            return self.evaluate(event, reduction_kW), model

        if ut.check_cvx_type(reduction_kW):
            return self._build_cvxpy_expression(event, reduction_kW, region_x1)
        elif (
            ut.check_indexed_pyomo_type(reduction_kW)
            or ut.check_nonindexed_pyomo_type(reduction_kW)
            or isinstance(reduction_kW, (list, tuple))
        ):
            if model is None:
                raise ValueError("model is required for pyomo expressions")
            terms = _as_pyomo_terms(reduction_kW)
            if not self.regions:
                return self._build_payout_only_pyomo(event, model, varstr)
            if self.settlement == SETTLEMENT_INTERVAL:
                return self._build_pyomo_regions_indexed(
                    event, terms, region_x1, model, varstr
                )
            return self._build_pyomo_regions_scalar(
                event, terms, region_x1, model, varstr
            )
        else:
            raise TypeError(
                "reduction_kW must be numpy.ndarray, a Python number, "
                "cvxpy.Expression/Variable, pyomo.environ.Var, or a list/tuple of "
                "pyomo expressions"
            )

    def _build_cvxpy_expression(self, event, reduction_kW, region_x1):
        """Build the cvxpy revenue expression for a fixed region.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        reduction_kW : cvxpy.Expression or cvxpy.Variable
            Load reduction in kW, scalar or per interval.

        region_x1 : float or None
            `x1` bound of the region to fix.

        Raises
        ------
        ValueError
            If `region_x1` matches no region.

        Returns
        -------
        tuple
            `(revenue_expr, constraints)`.
        """
        payout_amt = self._payout_amount(event)
        if not self.regions:
            return payout_amt, []

        bid_capacity_kW = event[BID_CAPACITY_KW]
        region = self.find_region(region_x1=region_x1)
        x1, x2 = region[REGION_X1], region[REGION_X2]
        slopes, intercepts = self._region_coefficients(event)
        idx = next(i for i, r in enumerate(self.regions) if r is region)
        slope, intercept = slopes[idx], intercepts[idx]

        mean_reduction = cp.sum(reduction_kW) / reduction_kW.size
        # The payment ratio is affine within one fixed region, so meaning the
        # per-interval payments equals paying the mean reduction: the revenue
        # expression is identical in both settlement modes. Only the region
        # bound constraints differ -- "interval" settlement requires every
        # interval (not just the mean) to fall in the fixed region.
        revenue_expr = slope * mean_reduction + intercept + payout_amt
        if self.settlement == SETTLEMENT_INTERVAL:
            constraints = [reduction_kW >= x1 * bid_capacity_kW]
            if not np.isinf(x2):
                constraints.append(reduction_kW <= x2 * bid_capacity_kW)
        else:
            constraints = [mean_reduction >= x1 * bid_capacity_kW]
            if not np.isinf(x2):
                constraints.append(mean_reduction <= x2 * bid_capacity_kW)
        return revenue_expr, constraints

    def _build_payout_only_pyomo(self, event, model, varstr):
        """Build a pyomo revenue variable equal to the flat payout.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        model : pyomo.environ.Model or pyomo.environ.Block
            Model to add components to.

        varstr : str
            Name prefix for pyomo components created on `model`.

        Returns
        -------
        tuple
            `(revenue_var, model)`.
        """
        revenue_name = _component_name(varstr, REVENUE_SUFFIX)
        model.add_component(revenue_name, pyo.Var())
        revenue_var = model.find_component(revenue_name)
        model.add_component(
            _component_name(varstr, REVENUE_CONSTRAINT_SUFFIX),
            pyo.Constraint(expr=revenue_var == self._payout_amount(event)),
        )
        return revenue_var, model

    def _build_pyomo_regions_scalar(self, event, terms, region_x1, model, varstr):
        """Build the pyomo revenue expression for `"average"` settlement.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        terms : list
            Per-interval reduction expressions.

        region_x1 : float or None
            `x1` bound of the region to fix, or `None` to let the solver choose.

        model : pyomo.environ.Model or pyomo.environ.Block
            Model to add components to.

        varstr : str
            Name prefix for pyomo components created on `model`.

        Raises
        ------
        ValueError
            If `region_x1` matches no region.

        Returns
        -------
        tuple
            `(revenue_var, model)`.
        """
        bid_capacity_kW = event[BID_CAPACITY_KW]
        mean_reduction = pyo.quicksum(terms) / len(terms)
        region_idx = range(len(self.regions))
        slopes, intercepts = self._region_coefficients(event)

        z_name = _component_name(varstr, REGION_ACTIVE_SUFFIX)
        model.add_component(z_name, pyo.Var(region_idx, within=pyo.Binary))
        z = model.find_component(z_name)
        model.add_component(
            _component_name(varstr, REGION_SELECT_CONSTRAINT_SUFFIX),
            pyo.Constraint(expr=pyo.quicksum(z[r] for r in region_idx) == 1),
        )

        region_reduction_name = _component_name(varstr, REGION_REDUCTION_SUFFIX)
        model.add_component(region_reduction_name, pyo.Var(region_idx))
        region_reduction = model.find_component(region_reduction_name)

        def lower_rule(m, r):
            x1 = self.regions[r][REGION_X1]
            # implicitly bounds the DR bid to be > 0.1% of max power production
            if np.isinf(x1):
                x1 = -1000.0
            return region_reduction[r] >= x1 * bid_capacity_kW * z[r]

        model.add_component(
            _component_name(varstr, REGION_REDUCTION_LOWER_CONSTRAINT_SUFFIX),
            pyo.Constraint(region_idx, rule=lower_rule),
        )

        def upper_rule(m, r):
            x2 = self.regions[r][REGION_X2]
            # implicitly bounds the DR bid to be > 0.1% of max power consumption
            if np.isinf(x2):
                x2 = 1000.0
            return region_reduction[r] <= x2 * bid_capacity_kW * z[r]

        model.add_component(
            _component_name(varstr, REGION_REDUCTION_UPPER_CONSTRAINT_SUFFIX),
            pyo.Constraint(region_idx, rule=upper_rule),
        )

        model.add_component(
            _component_name(varstr, REGION_REDUCTION_SUM_CONSTRAINT_SUFFIX),
            pyo.Constraint(
                expr=mean_reduction
                == pyo.quicksum(region_reduction[r] for r in region_idx)
            ),
        )

        revenue_name = _component_name(varstr, REVENUE_SUFFIX)
        model.add_component(revenue_name, pyo.Var())
        revenue_var = model.find_component(revenue_name)
        payout_amt = self._payout_amount(event)
        model.add_component(
            _component_name(varstr, REVENUE_CONSTRAINT_SUFFIX),
            pyo.Constraint(
                expr=revenue_var
                == pyo.quicksum(
                    slopes[r] * region_reduction[r] + intercepts[r] * z[r]
                    for r in region_idx
                )
                + payout_amt
            ),
        )

        if region_x1 is not None:
            matched_region = self.find_region(region_x1=region_x1)
            fixed_idx = next(
                i for i, r in enumerate(self.regions) if r is matched_region
            )
            for r in region_idx:
                z[r].fix(1 if r == fixed_idx else 0)

        return revenue_var, model

    def _build_pyomo_regions_indexed(self, event, terms, region_x1, model, varstr):
        """Build the pyomo revenue expression for `"interval"` settlement.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        terms : list
            Per-interval reduction expressions.

        region_x1 : float, list of float, or None
            `x1` bound of the region to fix for every interval (float), for
            each interval (list), or `None` to let the solver choose.

        model : pyomo.environ.Model or pyomo.environ.Block
            Model to add components to.

        varstr : str
            Name prefix for pyomo components created on `model`.

        Raises
        ------
        ValueError
            If `region_x1` is a list whose length differs from `len(terms)`,
            or if a `region_x1` value matches no region.

        Returns
        -------
        tuple
            `(revenue_var, model)`.
        """
        bid_capacity_kW = event[BID_CAPACITY_KW]
        n_intervals = len(terms)
        interval_idx = range(n_intervals)
        region_idx = range(len(self.regions))
        slopes, intercepts = self._region_coefficients(event)

        z_name = _component_name(varstr, REGION_ACTIVE_SUFFIX)
        model.add_component(
            z_name, pyo.Var(interval_idx, region_idx, within=pyo.Binary)
        )
        z = model.find_component(z_name)
        model.add_component(
            _component_name(varstr, REGION_SELECT_CONSTRAINT_SUFFIX),
            pyo.Constraint(
                interval_idx,
                rule=lambda m, t: pyo.quicksum(z[t, r] for r in region_idx) == 1,
            ),
        )

        region_reduction_name = _component_name(varstr, REGION_REDUCTION_SUFFIX)
        model.add_component(region_reduction_name, pyo.Var(interval_idx, region_idx))
        region_reduction = model.find_component(region_reduction_name)

        def lower_rule(m, t, r):
            x1 = self.regions[r][REGION_X1]
            if np.isinf(x1):
                x1 = -1000.0
            return region_reduction[t, r] >= x1 * bid_capacity_kW * z[t, r]

        model.add_component(
            _component_name(varstr, REGION_REDUCTION_LOWER_CONSTRAINT_SUFFIX),
            pyo.Constraint(interval_idx, region_idx, rule=lower_rule),
        )

        def upper_rule(m, t, r):
            x2 = self.regions[r][REGION_X2]
            if np.isinf(x2):
                x2 = 1000.0
            return region_reduction[t, r] <= x2 * bid_capacity_kW * z[t, r]

        model.add_component(
            _component_name(varstr, REGION_REDUCTION_UPPER_CONSTRAINT_SUFFIX),
            pyo.Constraint(interval_idx, region_idx, rule=upper_rule),
        )

        model.add_component(
            _component_name(varstr, REGION_REDUCTION_SUM_CONSTRAINT_SUFFIX),
            pyo.Constraint(
                interval_idx,
                rule=lambda m, t: terms[t]
                == pyo.quicksum(region_reduction[t, r] for r in region_idx),
            ),
        )

        interval_revenue_name = _component_name(varstr, INTERVAL_REVENUE_SUFFIX)
        model.add_component(interval_revenue_name, pyo.Var(interval_idx))
        interval_revenue = model.find_component(interval_revenue_name)
        model.add_component(
            _component_name(varstr, INTERVAL_REVENUE_CONSTRAINT_SUFFIX),
            pyo.Constraint(
                interval_idx,
                rule=lambda m, t: interval_revenue[t]
                == pyo.quicksum(
                    slopes[r] * region_reduction[t, r] + intercepts[r] * z[t, r]
                    for r in region_idx
                ),
            ),
        )

        revenue_name = _component_name(varstr, REVENUE_SUFFIX)
        model.add_component(revenue_name, pyo.Var())
        revenue_var = model.find_component(revenue_name)
        payout_amt = self._payout_amount(event)
        model.add_component(
            _component_name(varstr, REVENUE_CONSTRAINT_SUFFIX),
            pyo.Constraint(
                expr=revenue_var
                == pyo.quicksum(interval_revenue[t] for t in interval_idx) / n_intervals
                + payout_amt
            ),
        )

        if region_x1 is not None:
            if isinstance(region_x1, (list, tuple)):
                if len(region_x1) != n_intervals:
                    raise ValueError(
                        f"region_x1 must have length {n_intervals} to match "
                        f"reduction_kW's intervals, got {len(region_x1)}"
                    )
                region_x1_per_t = list(region_x1)
            else:
                region_x1_per_t = [region_x1] * n_intervals
            for t in interval_idx:
                matched_region = self.find_region(region_x1=region_x1_per_t[t])
                fixed_idx = next(
                    i for i, r in enumerate(self.regions) if r is matched_region
                )
                for r in region_idx:
                    z[t, r].fix(1 if r == fixed_idx else 0)

        return revenue_var, model


class CapacityEnergyPayment(PaymentStructure):
    """Piecewise-linear capacity payment plus a flat payment per kWh curtailed.

    Parameters
    ----------
    regions : list of dict or None
        Capacity payment schedule, as for `PaymentStructure`.

    energy_price : float
        Energy payment rate in $/kWh.

    **payment_kwargs
        Keyword arguments passed to `PaymentStructure`.
    """

    def __init__(self, regions, energy_price, **payment_kwargs):
        super().__init__(regions, **payment_kwargs)
        self.energy_price = energy_price

    def evaluate(self, event, reduction_kW):
        """Calculate realized capacity and energy revenue for a known reduction.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        reduction_kW : float or numpy.ndarray
            Load reduction in kW, as a window mean or per interval.

        Raises
        ------
        ValueError
            In the cases listed for `PaymentStructure.evaluate`.

        Returns
        -------
        float
            Revenue (positive) or penalty (negative) in USD.
        """
        capacity_payment = super().evaluate(event, reduction_kW)
        terms, _ = self._as_interval_terms(reduction_kW)
        energy_payment = (
            self.energy_price * float(np.mean(terms)) * event[EVENT_DURATION]
        )
        return capacity_payment + energy_payment

    def build_expression(
        self, event, reduction_kW, region_x1=None, model=None, varstr=""
    ):
        """Build the capacity and energy revenue expression for an event.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        reduction_kW : float, numpy.ndarray, cvxpy.Expression, or pyomo.environ.Var
            Load reduction in kW, as a realized value or a decision-variable
            expression, per interval or as a window mean. A list or tuple of
            pyomo expressions is also accepted.

        region_x1 : float, list of float, or None
            `x1` bound of the region to fix. Required for cvxpy.

        model : pyomo.environ.Model or pyomo.environ.Block or None
            Model to add components to. Required for pyomo.

        varstr : str
            Name prefix for pyomo components created on `model`.

        Raises
        ------
        ValueError
            In the cases listed for `PaymentStructure.build_expression`.

        TypeError
            If `reduction_kW` is not a supported type.

        Returns
        -------
        tuple
            `(revenue, model)` for numpy or scalar `reduction_kW`,
            `(total_revenue_var, model)` for pyomo, or
            `(revenue_expr, constraints)` for cvxpy.
        """
        if ut.check_indexed_np_array(reduction_kW) or ut.check_nonindexed_python_type(
            reduction_kW
        ):
            return self.evaluate(event, reduction_kW), model

        if ut.check_cvx_type(reduction_kW):
            mean_reduction = cp.sum(reduction_kW) / reduction_kW.size
            energy_term = self.energy_price * mean_reduction * event[EVENT_DURATION]
            capacity_expr, constraints = super().build_expression(
                event, reduction_kW, region_x1=region_x1, model=model, varstr=varstr
            )
            return capacity_expr + energy_term, constraints

        terms = _as_pyomo_terms(reduction_kW)
        mean_reduction = pyo.quicksum(terms) / len(terms)
        energy_term = self.energy_price * mean_reduction * event[EVENT_DURATION]

        capacity_var, model = super().build_expression(
            event, reduction_kW, region_x1=region_x1, model=model, varstr=varstr
        )

        energy_name = _component_name(varstr, ENERGY_REVENUE_SUFFIX)
        model.add_component(energy_name, pyo.Var())
        energy_var = model.find_component(energy_name)

        def energy_rule(m):
            return energy_var == energy_term

        model.add_component(
            _component_name(varstr, ENERGY_REVENUE_CONSTRAINT_SUFFIX),
            pyo.Constraint(rule=energy_rule),
        )

        total_name = _component_name(varstr, TOTAL_REVENUE_SUFFIX)
        model.add_component(total_name, pyo.Var())
        total_var = model.find_component(total_name)

        def total_rule(m):
            return total_var == capacity_var + energy_var

        model.add_component(
            _component_name(varstr, TOTAL_REVENUE_CONSTRAINT_SUFFIX),
            pyo.Constraint(rule=total_rule),
        )
        return total_var, model


class MarketIndexedPayment(PaymentStructure):
    """Piecewise-linear capacity payment priced from a market index.

    Parameters
    ----------
    regions : list of dict or None
        Payment schedule, as for `PaymentStructure`.

    price_lookup : callable
        Function `price_lookup(event) -> float` returning the capacity price
        in $/kW for an event.

    **payment_kwargs
        Keyword arguments passed to `PaymentStructure`.
    """

    def __init__(self, regions, price_lookup, **payment_kwargs):
        super().__init__(regions, **payment_kwargs)
        self.price_lookup = price_lookup  # callable: price_lookup(event) -> float

    def _resolve_event(self, event):
        """Copy an event with its capacity price replaced by the market price.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        Returns
        -------
        dict
            Shallow copy of `event` with `"capacity_price"` from
            `price_lookup(event)`.
        """
        resolved = dict(event)
        resolved[CAPACITY_PRICE] = self.price_lookup(event)
        return resolved

    def evaluate(self, event, reduction_kW):
        """Calculate realized revenue at the market price for a known reduction.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        reduction_kW : float or numpy.ndarray
            Load reduction in kW, as a window mean or per interval.

        Raises
        ------
        ValueError
            In the cases listed for `PaymentStructure.evaluate`.

        Returns
        -------
        float
            Revenue (positive) or penalty (negative) in USD.
        """
        return super().evaluate(self._resolve_event(event), reduction_kW)

    def build_expression(
        self, event, reduction_kW, region_x1=None, model=None, varstr=""
    ):
        """Build the revenue expression for an event at the market price.

        Parameters
        ----------
        event : dict
            Single event dict. See `add_event` for the keys.

        reduction_kW : float, numpy.ndarray, cvxpy.Expression, or pyomo.environ.Var
            Load reduction in kW, as a realized value or a decision-variable
            expression, per interval or as a window mean. A list or tuple of
            pyomo expressions is also accepted.

        region_x1 : float, list of float, or None
            `x1` bound of the region to fix. Required for cvxpy.

        model : pyomo.environ.Model or pyomo.environ.Block or None
            Model to add components to. Required for pyomo.

        varstr : str
            Name prefix for pyomo components created on `model`.

        Raises
        ------
        ValueError
            In the cases listed for `PaymentStructure.build_expression`.

        TypeError
            If `reduction_kW` is not a supported type.

        Returns
        -------
        tuple
            As for `PaymentStructure.build_expression`.
        """
        return super().build_expression(
            self._resolve_event(event),
            reduction_kW,
            region_x1=region_x1,
            model=model,
            varstr=varstr,
        )


def _coerce_payment_structure(payment_function):
    """Convert a list of payment regions into a `PaymentStructure`.

    Parameters
    ----------
    payment_function : list of dict or PaymentStructure
        Payment regions, or a `PaymentStructure` instance.

    Returns
    -------
    PaymentStructure
        `payment_function` if it is already a `PaymentStructure`, otherwise a
        new `PaymentStructure` over those regions.
    """
    if isinstance(payment_function, PaymentStructure):
        return payment_function
    return PaymentStructure(payment_function)


def evaluate_payment_function(
    payment_function,
    reduction_kW,
    bid_capacity_kW,
    capacity_price,
    duration_hours=None,
):
    """Calculate realized revenue for a known reduction without an event dict.

    Parameters
    ----------
    payment_function : list of dict or PaymentStructure
        Payment regions, each a dict with float values under the keys `"x1"`,
        `"x2"`, `"y1"`, and `"y2"`, or a `PaymentStructure` instance.

    reduction_kW : float or numpy.ndarray
        Load reduction in kW, as a window mean or per interval.

    bid_capacity_kW : float
        Bid capacity in kW.

    capacity_price : float
        Capacity price in $/kW.

    duration_hours : float or None
        Event duration in hours. Required for a `"per_hour"` basis.

    Raises
    ------
    ValueError
        If `bid_capacity_kW` is not positive, if no region contains a
        resulting delivered ratio, or if a `"per_hour"` basis is used without
        `duration_hours`.

    Returns
    -------
    float
        Revenue (positive) or penalty (negative) in USD.
    """
    event = {BID_CAPACITY_KW: bid_capacity_kW, CAPACITY_PRICE: capacity_price}
    if duration_hours is not None:
        event[EVENT_DURATION] = duration_hours
    return _coerce_payment_structure(payment_function).evaluate(event, reduction_kW)


def build_payment_expression(
    payment_function,
    reduction_kW,
    bid_capacity_kW,
    capacity_price,
    region_x1=None,
    model=None,
    varstr="",
    duration_hours=None,
):
    """Build the revenue expression for a reduction without an event dict.

    Parameters
    ----------
    payment_function : list of dict or PaymentStructure
        Payment regions, each a dict with float values under the keys `"x1"`,
        `"x2"`, `"y1"`, and `"y2"`, or a `PaymentStructure` instance.

    reduction_kW : float, numpy.ndarray, cvxpy.Expression, or pyomo.environ.Var
        Load reduction in kW, as a realized value or a decision-variable
        expression, per interval or as a window mean. A list or tuple of
        pyomo expressions is also accepted.

    bid_capacity_kW : float
        Bid capacity in kW.

    capacity_price : float
        Capacity price in $/kW.

    region_x1 : float, list of float, or None
        `x1` bound of the region to fix. Required for cvxpy.

    model : pyomo.environ.Model or pyomo.environ.Block or None
        Model to add components to. Required for pyomo.

    varstr : str
        Name prefix for pyomo components created on `model`.

    duration_hours : float or None
        Event duration in hours. Required for a `"per_hour"` basis.

    Raises
    ------
    ValueError
        If a `"per_hour"` basis is used without `duration_hours`, or in the
        cases listed for `PaymentStructure.build_expression`.

    TypeError
        If `reduction_kW` is not a supported type.

    Returns
    -------
    tuple
        `(revenue, model)` for numpy or scalar `reduction_kW`,
        `(revenue_var, model)` for pyomo, or `(revenue_expr, constraints)` for
        cvxpy.
    """
    event = {BID_CAPACITY_KW: bid_capacity_kW, CAPACITY_PRICE: capacity_price}
    if duration_hours is not None:
        event[EVENT_DURATION] = duration_hours
    return _coerce_payment_structure(payment_function).build_expression(
        event, reduction_kW, region_x1=region_x1, model=model, varstr=varstr
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
    adjustment_factor=None,
):
    """Append a demand response event to an events list.

    Parameters
    ----------
    events : list of dict or None
        Events list to append to. `None` starts a new list.

    event_date : datetime.date, datetime.datetime, or str
        Calendar date of the event.

    start_hour : float
        Hour of day (0-24) the event begins.

    duration_hours : float
        Length of the event in hours.

    notification_hours : float
        Advance notice before the event in hours.

    baseline_days : list
        Candidate baseline days for the event.

    bid_capacity_kW : float
        Bid capacity in kW.

    capacity_price : float
        Capacity price in $/kW.

    adjustment_factor : float or None
        Day-of adjustment factor to apply as-is. `None` calculates it from
        historical data.

    Raises
    ------
    ValueError
        If `duration_hours`, `bid_capacity_kW`, or a given
        `adjustment_factor` is not positive, `notification_hours` is
        negative, or `baseline_days` is empty.

    Warns
    -----
    UserWarning
        If `capacity_price` is zero.

    Returns
    -------
    list of dict
        New list with the event appended. Each event dict has the keys

        - `"event_date"` : pandas.Timestamp
        - `"start_hour"` : float
        - `"duration_hours"` : float
        - `"notification_hours"` : float
        - `"baseline_days"` : list
        - `"bid_capacity_kW"` : float
        - `"capacity_price"` : float
        - `"adjustment_factor"` : float or None
    """
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    if notification_hours < 0:
        raise ValueError("notification_hours must be non-negative")
    if len(baseline_days) == 0:
        raise ValueError("baseline_days must be non-empty")
    if bid_capacity_kW <= 0:
        raise ValueError("bid_capacity_kW must be positive")
    if adjustment_factor is not None and adjustment_factor <= 0:
        raise ValueError("adjustment_factor must be positive")
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
        ADJUSTMENT_FACTOR: adjustment_factor,
    }
    return (events or []) + [new_event]


def events_to_dataframe(events):
    """Convert an events list into a `DataFrame` sorted by event date.

    Parameters
    ----------
    events : list of dict or pandas.DataFrame
        Events from `add_event`.

    Returns
    -------
    pandas.DataFrame
        One row per event, sorted by `"event_date"`.
    """
    events_df = events if isinstance(events, pd.DataFrame) else pd.DataFrame(events)
    return events_df.sort_values(EVENT_DATE).reset_index(drop=True)


def make_baseline_parameters(
    baseline_method="average_similar_days",
    n_baseline_days=10,
    adjustment_offset_hours=3,
    adjustment_duration_hours=3,
    adjustment_clip=(0.8, 1.2),
    exclude_weekends=True,
    exclude_holidays=True,
    holiday_country=None,
    holiday_subdiv=None,
    holiday_dates=None,
    resolution=None,
):
    """Build a dictionary of baseline calculation parameters.

    Parameters
    ----------
    baseline_method : str
        Baseline method. Only `"average_similar_days"` is supported.

    n_baseline_days : int
        Number of eligible baseline days to average. `0` disables baselining.

    adjustment_offset_hours : int or None
        Hours before the event start at which the day-of adjustment window
        begins. `None` disables the day-of adjustment.

    adjustment_duration_hours : int
        Length of the day-of adjustment window in hours.

    adjustment_clip : tuple of float
        `(low, high)` bounds on the day-of adjustment factor.

    exclude_weekends : bool
        If `True`, drop Saturdays and Sundays from the candidate baseline days.

    exclude_holidays : bool
        If `True`, drop holidays from the candidate baseline days.

    holiday_country : str or None
        ISO 3166-1 country code passed to `holidays.country_holidays`.

    holiday_subdiv : str or None
        Subdivision code (e.g., `"CA"`) passed to `holidays.country_holidays`.

    holiday_dates : list or None
        Additional dates to treat as holidays.

    resolution : str or None
        Settlement interval width (e.g., `"15m"`, `"1h"`).

    Raises
    ------
    ValueError
        If `baseline_method` is not `"average_similar_days"`, if
        `n_baseline_days` is negative, or if `adjustment_offset_hours` is not
        `None` and `adjustment_duration_hours` is not positive or exceeds
        `adjustment_offset_hours`.

    Returns
    -------
    dict
        Baseline parameters with the keys

        - `"baseline_method"` : str
        - `"n_baseline_days"` : int
        - `"adjustment_offset_hours"` : int or None
        - `"adjustment_duration_hours"` : int
        - `"adjustment_clip"` : tuple of float
        - `"exclude_weekends"` : bool
        - `"exclude_holidays"` : bool
        - `"holiday_country"` : str or None
        - `"holiday_subdiv"` : str or None
        - `"holiday_dates"` : list
        - `"resolution"` : str or None
    """
    if baseline_method != "average_similar_days":
        raise ValueError(
            "baseline_method must be 'average_similar_days'; "
            "other methods are not yet supported"
        )
    if n_baseline_days < 0:
        raise ValueError("n_baseline_days must be non-negative")
    if adjustment_offset_hours is not None:
        if adjustment_duration_hours <= 0:
            raise ValueError("adjustment_duration_hours must be positive")
        if adjustment_duration_hours > adjustment_offset_hours:
            raise ValueError(
                "adjustment_duration_hours must not exceed "
                "adjustment_offset_hours, so the adjustment window ends at or "
                "before the event start"
            )

    return {
        BASELINE_METHOD: baseline_method,
        N_BASELINE_DAYS: n_baseline_days,
        ADJUSTMENT_OFFSET_HOURS: adjustment_offset_hours,
        ADJUSTMENT_DURATION_HOURS: adjustment_duration_hours,
        ADJUSTMENT_CLIP: adjustment_clip,
        EXCLUDE_WEEKENDS: exclude_weekends,
        EXCLUDE_HOLIDAYS: exclude_holidays,
        HOLIDAY_COUNTRY: holiday_country,
        HOLIDAY_SUBDIV: holiday_subdiv,
        HOLIDAY_DATES: list(holiday_dates) if holiday_dates else [],
        RESOLUTION: resolution,
    }


def _baseline_day_in_horizon(day, start_hour, duration_hours, datetime_index, step):
    """Check whether a baseline day's event window lies within the model horizon.

    Parameters
    ----------
    day : pandas.Timestamp
        Calendar date of the window.

    start_hour : float
        Hour of day (0-24) the window begins.

    duration_hours : float
        Length of the window in hours.

    datetime_index : pandas.DatetimeIndex
        Timestamps of the model horizon.

    step : pandas.Timedelta
        Spacing of `datetime_index`.

    Returns
    -------
    bool
        `True` if the whole window lies within
        `[datetime_index.min(), datetime_index.max() + step)`.
    """
    window_start = pd.Timestamp(day) + pd.Timedelta(hours=start_hour)
    window_end = window_start + pd.Timedelta(hours=duration_hours)
    horizon_start = datetime_index.min()
    horizon_end = datetime_index.max() + step
    return (window_start >= horizon_start) and (window_end <= horizon_end)


def _baseline_day_interval_values(
    day,
    start_hour,
    duration_hours,
    step,
    n_intervals,
    historical_power_kW,
    model_power_kW,
    model_var_index,
    model_datetime_index,
):
    """Calculate one baseline day's mean power in each settlement interval.

    Parameters
    ----------
    day : pandas.Timestamp
        Calendar date of the baseline day.

    start_hour : float
        Hour of day (0-24) the event window begins.

    duration_hours : float
        Length of the event window in hours.

    step : pandas.Timedelta
        Settlement interval width.

    n_intervals : int
        Number of settlement intervals in the event window.

    historical_power_kW : pandas.Series
        Historical power consumption in kW, indexed by timestamp.

    model_power_kW : pyomo.environ.Var or None
        Power decision variable over the optimization horizon.

    model_var_index : list or None
        `list(model_power_kW.index_set())`.

    model_datetime_index : pandas.DatetimeIndex or None
        Timestamp of each entry of `model_var_index`.

    Raises
    ------
    ValueError
        If any interval has no data, or if a historical interval has `NaN`
        data.

    Returns
    -------
    tuple
        `(values, is_dynamic)`, where `values` holds one float (historical
        day) or pyomo expression (day in the model horizon) per interval.
    """
    if model_power_kW is not None and _baseline_day_in_horizon(
        day, start_hour, duration_hours, model_datetime_index, step
    ):
        buckets = _window_interval_buckets(
            model_datetime_index, day, start_hour, duration_hours, step
        )
        values = []
        for k, bucket in enumerate(buckets):
            if bucket.size == 0:
                raise ValueError(
                    f"Baseline day {day}'s window bounds fall inside the simulation "
                    f"horizon but interval {k} matched no positions in "
                    "model_datetime_index (is it gap-free and regularly spaced?)"
                )
            terms = [model_power_kW[model_var_index[i]] for i in bucket]
            values.append(pyo.quicksum(terms) / len(terms))
        return values, True

    buckets = _window_interval_buckets(
        historical_power_kW.index, day, start_hour, duration_hours, step
    )
    values = []
    for k, bucket in enumerate(buckets):
        if bucket.size == 0:
            raise ValueError(f"No data available for baseline day {day}, interval {k}")
        day_values = historical_power_kW.values[bucket]
        if np.any(pd.isna(day_values)):
            raise ValueError(f"Baseline day {day}, interval {k} has missing (NaN) data")
        values.append(float(np.mean(day_values)))
    return values, False


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
    """Calculate the per-interval baseline power for a single event.

    Parameters
    ----------
    historical_power_kW : pandas.Series
        Historical power consumption in kW, indexed by timestamp.

    event : dict
        Single event dict with the keys `"event_date"` (pandas.Timestamp),
        `"start_hour"` (float), `"duration_hours"` (float),
        `"notification_hours"` (float), `"baseline_days"` (list),
        `"bid_capacity_kW"` (float), `"capacity_price"` (float), and
        `"adjustment_factor"` (float or None), as returned by `add_event`.

    baseline_params : dict or BaselineMethod
        Baseline parameter dict from `make_baseline_parameters`, or a
        `BaselineMethod` instance.

    model : pyomo.environ.Model or pyomo.environ.Block or None
        Model to add baseline components to. `None` for ex-post use.

    model_power_kW : pyomo.environ.Var or None
        Power decision variable over the optimization horizon. Required when
        `model` is given.

    model_datetime_index : pandas.DatetimeIndex or None
        Timestamp of each entry of `model_power_kW`. Required when `model` is
        given.

    varstr : str or None
        Name of the baseline `Var` created on `model`. Required when `model`
        is given.

    Raises
    ------
    ValueError
        If no eligible baseline days remain, if a selected baseline day has
        missing data, or if `model` is given without `model_power_kW`,
        `model_datetime_index`, and `varstr`.

    Warns
    -----
    UserWarning
        If fewer eligible baseline days remain than `n_baseline_days`.

    Returns
    -------
    numpy.ndarray or tuple
        Per-interval baseline in kW when `model` is `None`, otherwise
        `(baseline_kW, model)` where `baseline_kW` is a `numpy.ndarray` or an
        indexed `pyomo.environ.Var`.
    """
    baseline_method = _coerce_baseline_method(baseline_params)
    return baseline_method.compute(
        historical_power_kW,
        event,
        model=model,
        model_power_kW=model_power_kW,
        model_datetime_index=model_datetime_index,
        varstr=varstr,
    )


def _baseline_terms(baseline_kW, n):
    """Convert a baseline into a list of `n` per-interval values.

    Parameters
    ----------
    baseline_kW : float, numpy.ndarray, list, tuple, or pyomo.environ.Var
        Single baseline value or one value per interval.

    n : int
        Number of intervals.

    Raises
    ------
    ValueError
        If `baseline_kW` has more than one value and its length is not `n`.

    Returns
    -------
    list
        `n` values or expressions.
    """
    if ut.check_indexed_pyomo_type(baseline_kW):
        idx = list(baseline_kW.index_set())
        if len(idx) != n:
            raise ValueError(
                f"baseline_kW has {len(idx)} entries but power_kW has {n}; "
                "they must match"
            )
        return [baseline_kW[i] for i in idx]
    if isinstance(baseline_kW, (list, tuple, np.ndarray)):
        values = list(baseline_kW)
        if len(values) == 1:
            return values * n
        if len(values) != n:
            raise ValueError(
                f"baseline_kW has {len(values)} entries but power_kW has {n}; "
                "they must match, or baseline_kW must be a scalar"
            )
        return values
    # A scalar: Python number, 0-d array, or nonindexed pyomo Var/expression.
    return [baseline_kW] * n


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

    Parameters
    ----------
    power_kW : numpy.ndarray, cvxpy.Expression, cvxpy.Variable, or pyomo.environ.Var
        Power consumption in kW during the event window only.

    event : dict
        Single event dict with the keys

        - `"event_date"` : pandas.Timestamp
        - `"start_hour"` : float
        - `"duration_hours"` : float
        - `"notification_hours"` : float
        - `"baseline_days"` : list
        - `"bid_capacity_kW"` : float
        - `"capacity_price"` : float
        - `"adjustment_factor"` : float or None

        as returned by `add_event`.

    baseline_kW : float, numpy.ndarray, or pyomo.environ.Var
        Baseline power in kW, as a single value or one value per interval.

    payment_function : list of dict or PaymentStructure
        Payment regions, each a dict with float values under the keys `"x1"`,
        `"x2"`, `"y1"`, and `"y2"`, or a `PaymentStructure` instance.

    region_x1 : float, list of float, or None
        `x1` bound of the region to fix. Required for cvxpy. Optional for
        pyomo, where `None` lets the solver choose.

    model : pyomo.environ.Model or pyomo.environ.Block or None
        Model to add components to. Required for pyomo.

    varstr : str
        Name prefix for pyomo components created on `model`.

    Raises
    ------
    ValueError
        If `power_kW` is cvxpy and `region_x1` is `None`, or if the length of
        `baseline_kW` does not match `power_kW`.

    TypeError
        If `power_kW` is not a supported type.

    Returns
    -------
    tuple
        `(revenue, model)` for numpy `power_kW`, `(revenue_var, model)` for
        pyomo, or `(revenue_expr, constraints)` for cvxpy.
    """
    payment_structure = _coerce_payment_structure(payment_function)

    if ut.check_indexed_np_array(power_kW) or ut.check_nonindexed_python_type(power_kW):
        power = np.atleast_1d(np.asarray(power_kW, dtype=float))
        baseline_arr = np.asarray(_baseline_terms(baseline_kW, power.size), dtype=float)
        reduction_kW = baseline_arr - power
        return payment_structure.evaluate(event, reduction_kW), model
    elif ut.check_cvx_type(power_kW):
        if region_x1 is None:
            raise ValueError("region_x1 must be specified for cvxpy power_kW")
        n = power_kW.size if hasattr(power_kW, "size") else 1
        baseline_arr = np.asarray(_baseline_terms(baseline_kW, n), dtype=float)
        reduction_kW = baseline_arr - power_kW
        return payment_structure.build_expression(
            event, reduction_kW, region_x1=region_x1, model=model, varstr=varstr
        )
    elif ut.check_indexed_pyomo_type(power_kW) or ut.check_nonindexed_pyomo_type(
        power_kW
    ):
        if ut.check_indexed_pyomo_type(power_kW):
            var_index = list(power_kW.index_set())
            power_terms = [power_kW[idx] for idx in var_index]
        else:
            power_terms = [power_kW]
        baseline_values = _baseline_terms(baseline_kW, len(power_terms))
        reduction_terms = [b - p for b, p in zip(baseline_values, power_terms)]
        return payment_structure.build_expression(
            event, reduction_terms, region_x1=region_x1, model=model, varstr=varstr
        )
    else:
        raise TypeError(
            "power_kW must be numpy.ndarray, a Python number, "
            "cvxpy.Expression/Variable, or pyomo.environ.Var"
        )


def calculate_event_revenue(
    historical_power_kW, event, baseline_params, payment_function
):
    """Calculate realized demand response revenue for a single event.

    Parameters
    ----------
    historical_power_kW : pandas.Series
        Power consumption in kW, indexed by timestamp, covering the event and
        its baseline days.

    event : dict
        Single event dict with the keys `"event_date"` (pandas.Timestamp),
        `"start_hour"` (float), `"duration_hours"` (float),
        `"notification_hours"` (float), `"baseline_days"` (list),
        `"bid_capacity_kW"` (float), `"capacity_price"` (float), and
        `"adjustment_factor"` (float or None), as returned by `add_event`.

    baseline_params : dict or BaselineMethod
        Baseline parameter dict from `make_baseline_parameters`, or a
        `BaselineMethod` instance.

    payment_function : list of dict or PaymentStructure
        Payment regions, each a dict with float values under the keys `"x1"`,
        `"x2"`, `"y1"`, and `"y2"`, or a `PaymentStructure` instance.

    Raises
    ------
    ValueError
        If `historical_power_kW` has no data in the event window.

    Returns
    -------
    dict
        Event results with the keys

        - `"event_date"` : pandas.Timestamp
        - `"baseline_kW"` : float, event-window mean
        - `"actual_kW"` : float, event-window mean
        - `"reduction_kW"` : float, event-window mean
        - `"delivered_ratio"` : float
        - `"revenue"` : float, in USD
        - `"baseline_profile_kW"` : numpy.ndarray, per interval
        - `"actual_profile_kW"` : numpy.ndarray, per interval
        - `"reduction_profile_kW"` : numpy.ndarray, per interval
        - `"interval_datetime"` : pandas.DatetimeIndex
    """
    mask = _event_window_mask(
        historical_power_kW.index,
        event[EVENT_DATE],
        event[EVENT_START_HOUR],
        event[EVENT_DURATION],
    )
    actual_profile = historical_power_kW.loc[mask].values
    if actual_profile.size == 0:
        raise ValueError("No data available for event window")
    interval_datetime = historical_power_kW.index[mask]

    baseline_profile = calculate_event_baseline(
        historical_power_kW, event, baseline_params
    )
    baseline_profile = np.asarray(
        _baseline_terms(baseline_profile, actual_profile.size), dtype=float
    )
    revenue, _ = build_event_revenue(
        actual_profile, event, baseline_profile, payment_function=payment_function
    )

    reduction_profile = baseline_profile - actual_profile
    baseline_mean = float(np.mean(baseline_profile))
    actual_mean = float(np.mean(actual_profile))
    reduction_mean = baseline_mean - actual_mean

    return {
        EVENT_DATE: event[EVENT_DATE],
        BASELINE_KW: baseline_mean,
        ACTUAL_KW: actual_mean,
        REDUCTION_KW: reduction_mean,
        DELIVERED_RATIO: reduction_mean / event[BID_CAPACITY_KW],
        REVENUE: revenue,
        BASELINE_PROFILE_KW: baseline_profile,
        ACTUAL_PROFILE_KW: actual_profile,
        REDUCTION_PROFILE_KW: reduction_profile,
        INTERVAL_DATETIME: interval_datetime,
    }


def _as_power_series(power_kW, datetime_index):
    """Convert power data into a `pandas.Series` indexed by timestamp.

    Parameters
    ----------
    power_kW : numpy.ndarray or pandas.Series
        Power consumption in kW.

    datetime_index : pandas.DatetimeIndex or None
        Timestamp of each entry of `power_kW`. Required when `power_kW` is a
        `numpy.ndarray`.

    Raises
    ------
    ValueError
        If `power_kW` is a `numpy.ndarray` and `datetime_index` is `None` or a
        different length.

    Returns
    -------
    pandas.Series
        `power_kW` indexed by timestamp.
    """
    if ut.check_pandas_type(power_kW):
        return power_kW
    if datetime_index is None:
        raise ValueError(
            "datetime_index is required when power_kW is a numpy.ndarray; "
            "otherwise pass power_kW as a pandas.Series indexed by timestamps"
        )
    if len(power_kW) != len(datetime_index):
        raise ValueError(
            f"power_kW has {len(power_kW)} entries but datetime_index has "
            f"{len(datetime_index)}; they must be the same length"
        )
    return pd.Series(power_kW, index=datetime_index)


def calculate_itemized_dr_revenue(
    power_kW, events, baseline_params, payment_function, datetime_index=None
):
    """Calculate realized demand response revenue for each event.

    Parameters
    ----------
    power_kW : pandas.Series or numpy.ndarray
        Power consumption in kW.

    events : list of dict or pandas.DataFrame
        Events from `add_event`.

    baseline_params : dict or BaselineMethod
        Baseline parameter dict from `make_baseline_parameters`, or a
        `BaselineMethod` instance.

    payment_function : list of dict or PaymentStructure
        Payment regions, each a dict with float values under the keys `"x1"`,
        `"x2"`, `"y1"`, and `"y2"`, or a `PaymentStructure` instance.

    datetime_index : pandas.DatetimeIndex or None
        Timestamp of each entry of `power_kW`. Required when `power_kW` is a
        `numpy.ndarray`.

    Raises
    ------
    ValueError
        If `datetime_index` is missing or mismatched for a `numpy.ndarray`
        `power_kW`, or if an event window has no data.

    Returns
    -------
    tuple
        `(per_event_df, total_revenue)`, where `per_event_df` has one row per
        event with the columns returned by `calculate_event_revenue` and
        `total_revenue` is in USD.
    """
    historical_power_kW = _as_power_series(power_kW, datetime_index)
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


def calculate_dr_revenue(
    power_kW,
    events,
    baseline_params,
    payment_function,
    historical_power_kW=None,
    datetime_index=None,
    model=None,
    region_x1s=None,
    varstr_prefix="dr_event",
):
    """Calculate or build demand response revenue across all events.

    Parameters
    ----------
    power_kW : pandas.Series, numpy.ndarray, or pyomo.environ.Var
        Power consumption in kW, as realized data or a decision variable.

    events : list of dict or pandas.DataFrame
        Events from `add_event`.

    baseline_params : dict or BaselineMethod
        Baseline parameter dict from `make_baseline_parameters`, or a
        `BaselineMethod` instance.

    payment_function : list of dict or PaymentStructure
        Payment regions, each a dict with float values under the keys `"x1"`,
        `"x2"`, `"y1"`, and `"y2"`, or a `PaymentStructure` instance.

    historical_power_kW : pandas.Series or None
        Historical power consumption in kW, indexed by timestamp. Required
        for pyomo.

    datetime_index : pandas.DatetimeIndex or None
        Timestamp of each entry of `power_kW`. Required for pyomo and for a
        `numpy.ndarray` `power_kW`.

    model : pyomo.environ.Model or pyomo.environ.Block or None
        Model to add components to. Required for pyomo.

    region_x1s : dict or None
        `x1` bound of the region to fix for each event, keyed by event date
        (e.g., `{"2024-01-08": 0.6}`). Events not in the dict let the solver
        choose. Only used for pyomo.

    varstr_prefix : str
        Name prefix for pyomo components created on `model`.

    Raises
    ------
    NotImplementedError
        If `power_kW` is a cvxpy type.

    ValueError
        For pyomo, if `datetime_index`, `historical_power_kW`, or `model` is
        missing, or if an event window matches no entries of
        `datetime_index`.

    TypeError
        If `power_kW` is not a supported type.

    Returns
    -------
    tuple
        `(total_revenue, model)`, where `total_revenue` is a float in USD for
        realized data or a pyomo expression for a decision variable.
    """
    if ut.check_cvx_type(power_kW):
        raise NotImplementedError(
            "cvxpy power_kW is not supported for demand response revenue. Pass a "
            "pandas.Series or numpy.ndarray for ex-post evaluation, or a "
            "pyomo.environ.Var to build an optimization model."
        )
    elif ut.check_indexed_pyomo_type(power_kW) or ut.check_nonindexed_pyomo_type(
        power_kW
    ):
        return _build_dr_revenue_components(
            power_kW,
            datetime_index,
            events,
            historical_power_kW,
            baseline_params,
            model,
            payment_function,
            region_x1s,
            varstr_prefix,
        )
    elif ut.check_indexed_np_array(power_kW) or ut.check_pandas_type(power_kW):
        _, total_revenue = calculate_itemized_dr_revenue(
            power_kW, events, baseline_params, payment_function, datetime_index
        )
        return total_revenue, model
    else:
        raise TypeError(
            "power_kW must be of type pandas.Series, numpy.ndarray, "
            "or pyomo.environ.Var"
        )


def _build_dr_revenue_components(
    power_kW,
    datetime_index,
    events,
    historical_power_kW,
    baseline_params,
    model,
    payment_function,
    region_x1s,
    varstr_prefix,
):
    """Build each event's baseline and revenue components on a pyomo model.

    Parameters
    ----------
    power_kW : pyomo.environ.Var
        Power decision variable over the optimization horizon.

    datetime_index : pandas.DatetimeIndex
        Timestamp of each entry of `power_kW`.

    events : list of dict or pandas.DataFrame
        Events from `add_event`.

    historical_power_kW : pandas.Series
        Historical power consumption in kW, indexed by timestamp.

    baseline_params : dict or BaselineMethod
        Baseline parameter dict from `make_baseline_parameters`, or a
        `BaselineMethod` instance.

    model : pyomo.environ.Model or pyomo.environ.Block
        Model to add components to.

    payment_function : list of dict or PaymentStructure
        Payment regions, or a `PaymentStructure` instance.

    region_x1s : dict or None
        `x1` bound of the region to fix for each event, keyed by event date.

    varstr_prefix : str
        Name prefix for pyomo components created on `model`.

    Raises
    ------
    ValueError
        If `datetime_index`, `historical_power_kW`, or `model` is `None`, or
        if an event window matches no entries of `datetime_index`.

    Returns
    -------
    tuple
        `(total_revenue, model)`, where `total_revenue` is a pyomo expression.
    """
    if any(a is None for a in (datetime_index, historical_power_kW, model)):
        raise ValueError(
            "datetime_index, historical_power_kW, and model are all "
            "required when power_kW is a pyomo variable"
        )
    varstr_prefix = ut.sanitize_varstr(varstr_prefix)
    events_df = events_to_dataframe(events)
    var_index = list(power_kW.index_set())
    region_x1_by_date = (
        {pd.Timestamp(k): v for k, v in region_x1s.items()} if region_x1s else {}
    )
    payment_structure = _coerce_payment_structure(payment_function)

    total_revenue = 0
    for i, row in events_df.iterrows():
        event = row.to_dict()
        event_varstr = f"{varstr_prefix}_{i}"
        baseline_kW, model = calculate_event_baseline(
            historical_power_kW,
            event,
            baseline_params,
            model=model,
            model_power_kW=power_kW,
            model_datetime_index=datetime_index,
            varstr=_component_name(event_varstr, BASELINE_KW),
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
        power_terms = [power_kW[idx] for idx in matched_indices]
        baseline_values = _baseline_terms(baseline_kW, len(power_terms))
        reduction_terms = [b - p for b, p in zip(baseline_values, power_terms)]

        revenue_var, model = payment_structure.build_expression(
            event,
            reduction_terms,
            region_x1=region_x1_by_date.get(event[EVENT_DATE]),
            model=model,
            varstr=event_varstr,
        )
        total_revenue += revenue_var

    return total_revenue, model


def build_dr_revenue(
    power_kW,
    datetime_index,
    events,
    historical_power_kW,
    baseline_params,
    model,
    payment_function,
    region_x1s=None,
    varstr_prefix="dr_event",
):
    """Build demand response revenue on a pyomo model and subtract it from the
    objective.

    Parameters
    ----------
    power_kW : pyomo.environ.Var
        Power decision variable over the optimization horizon.

    datetime_index : pandas.DatetimeIndex
        Timestamp of each entry of `power_kW`.

    events : list of dict or pandas.DataFrame
        Events from `add_event`.

    historical_power_kW : pandas.Series
        Historical power consumption in kW, indexed by timestamp.

    baseline_params : dict or BaselineMethod
        Baseline parameter dict from `make_baseline_parameters`, or a
        `BaselineMethod` instance.

    model : pyomo.environ.Model or pyomo.environ.Block
        Model to add components to.

    payment_function : list of dict or PaymentStructure
        Payment regions, each a dict with float values under the keys `"x1"`,
        `"x2"`, `"y1"`, and `"y2"`, or a `PaymentStructure` instance.

    region_x1s : dict or None
        `x1` bound of the region to fix for each event, keyed by event date
        (e.g., `{"2024-01-08": 0.6}`). Events not in the dict let the solver
        choose.

    varstr_prefix : str
        Name prefix for pyomo components created on `model`.

    Raises
    ------
    ValueError
        If an event window matches no entries of `datetime_index`.

    Returns
    -------
    tuple
        `(total_revenue, model)`, where `total_revenue` is a pyomo expression.
    """
    total_revenue, model = calculate_dr_revenue(
        power_kW,
        events,
        baseline_params,
        payment_function,
        historical_power_kW=historical_power_kW,
        datetime_index=datetime_index,
        model=model,
        region_x1s=region_x1s,
        varstr_prefix=varstr_prefix,
    )
    if hasattr(model, "objective"):
        model.objective.expr -= total_revenue
    else:
        model.objective = pyo.Objective(expr=-total_revenue, sense=pyo.minimize)
    return total_revenue, model

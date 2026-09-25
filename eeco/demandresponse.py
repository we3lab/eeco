"""Calculate demand response revenue from electricity consumption data.

This uses two classes to define the baselining procedure and the payment
structure. The `BaselineMethod` class defines how to calculate the
counterfactual baseline consumption. `PaymentStructure` class defines how to
calculate the revenue based on the actual consumption and the baseline.
These are used by `calculate_dr_revenue`, `build_dr_revenue` (pyomo only),
and `calculate_itemized_dr_revenue` to calculate costs or add accounting
equations to models.
"""

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
ADJUSTMENT_OFFSET_HOURS = "adjustment_offset_hours"
ADJUSTMENT_DURATION_HOURS = "adjustment_duration_hours"
ADJUSTMENT_CLIP = "adjustment_clip"
EXCLUDE_WEEKENDS = "exclude_weekends"
EXCLUDE_HOLIDAYS = "exclude_holidays"
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


class BaselineMethod:
    """Default baseline method. See :doc:`/demandresponse` for a description
    of this class and how to extend it.

    Parameters
    ----------
    n_baseline_days : int
        Number of valid baseline days to average over. `0` means "no
        baselining": `compute` skips day selection and the day-of adjustment
        entirely and returns an all-zero profile, for programs (e.g. some
        interruption-based ones) whose settlement has no historical
        counterfactual. A zero baseline makes `reduction_kW` negative, so
        pair it with a payment function whose lowest region extends to
        `-Infinity`, or with a payout-only `PaymentStructure`
        (`regions=None`) -- the CBP-style schedules bundled with this module
        do not cover a negative delivered ratio and `find_region` will raise.

    adjustment_offset_hours : int or None
        Number of hours before the event start where the day-of adjustment
        window begins. If `None`, no adjustment is applied.

    adjustment_duration_hours : int
        Length, in hours, of the day-of adjustment window. Combined with
        `adjustment_offset_hours`, the window is
        `[event_start - adjustment_offset_hours,
        event_start - adjustment_offset_hours + adjustment_duration_hours)`
        -- e.g. `adjustment_offset_hours=4, adjustment_duration_hours=2`
        looks at the window starting 4 hours before the event and ending 2
        hours before it. Ignored when `adjustment_offset_hours` is `None`.

    adjustment_clip : tuple of float
        `(low, high)` bounds the day-of adjustment factor is clipped to,
        limiting how far one anomalous morning can move the baseline.

    exclude_weekends : bool
        If `True`, drop Saturday/Sunday from candidate baseline days.

    exclude_holidays : bool
        If `True`, drop dates in `holiday_dates` from candidate baseline
        days.

    holiday_dates : list or None
        Calendar dates treated as holidays. Defaults to an empty list.

    adjustment_in_model : bool
        If `True` and `compute` is called with a `model`, the day-of
        adjustment factor is added to that model as a **fixed**
        `pyomo.environ.Var` named `varstr + "_adjustment_factor"` and
        multiplied into the baseline symbolically, rather than being folded
        in as a hard-coded number. This lets the caller retune the factor
        (`model.<varstr>_adjustment_factor.fix(1.15)`) and re-solve without
        rebuilding the model. The `Var` is fixed on creation, which is what
        keeps it an input rather than a decision: retune it with
        `.fix(...)` rather than `.unfix()`, since unfixing it would both let
        the solver choose the revenue-maximizing factor and make
        `baseline * factor` bilinear. `False` by default, which folds the
        factor in as a constant and keeps `compute`'s return value a plain
        `numpy.ndarray` whenever every baseline day is historical.

    resolution : str or None
        Settlement interval width, as a string of the form `"[int][unit]"`
        (e.g. `"15m"`, `"1h"`), parsed by `utils.get_freq_binsize_minutes`.
        The event window is divided into intervals of this width, and
        `compute` returns one baseline value per interval. If `None`
        (default), the width is inferred from the spacing of
        `model_datetime_index` (when a model is given) or
        `historical_power_kW.index` otherwise -- pass this explicitly when
        that index is coarser or finer than the settlement interval you
        actually want (e.g. hourly meter data settled at 15-minute
        resolution).

    Raises
    ------
    ValueError
        When `n_baseline_days` is negative, or when
        `adjustment_offset_hours` is not `None` and `adjustment_duration_hours`
        is not positive or exceeds `adjustment_offset_hours` (the adjustment
        window must fall strictly before the event start).
    """

    def __init__(
        self,
        n_baseline_days=10,
        adjustment_offset_hours=3,
        adjustment_duration_hours=3,
        adjustment_clip=(0.8, 1.2),
        exclude_weekends=True,
        exclude_holidays=True,
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
        self.holiday_dates = list(holiday_dates) if holiday_dates else []
        self.adjustment_in_model = adjustment_in_model
        self.resolution = resolution

    def _rank_days(self, candidate_days, historical_power_kW, event):
        """Drop ineligible candidate days and rank the remainder by
        preference, most-preferred first.

        `select_days` keeps the first `n_baseline_days` of
        whatever this returns, so this method alone decides both which days
        are eligible and which survive when more are available than are
        needed. The foundation implementation applies the
        `exclude_weekends`/`exclude_holidays` configuration, then prefers
        the most recent days; override to change either rule (see
        `TopUsageDaysBaseline`, which calls `super()._rank_days(...)` to
        reuse the eligibility rule and only replaces the ranking).

        Parameters
        ----------
        candidate_days : list of pandas.Timestamp
            Days proposed for this event's baseline.

        historical_power_kW : pandas.Series
            Historical realized power consumption in kW, indexed by
            `pandas.DatetimeIndex`. Unused by the foundation ranking, but
            available to subclasses that rank by consumption.

        event : dict
            A single event, as produced by `add_event`. Unused by the
            foundation ranking, but available to subclasses that need the
            event window to rank days.

        Returns
        -------
        list of pandas.Timestamp
            The eligible subset of `candidate_days`, most-preferred first.
        """
        if self.exclude_weekends:
            candidate_days = [d for d in candidate_days if d.weekday() < 5]
        if self.exclude_holidays:
            holidays = {pd.Timestamp(d) for d in self.holiday_dates}
            candidate_days = [d for d in candidate_days if d not in holidays]
        return sorted(candidate_days, reverse=True)

    def select_days(self, candidate_days, historical_power_kW, event):
        """Filter, rank, and truncate candidate days to the ones actually
        used in the baseline average.

        Composes the `_rank_days` method, then keeps the top
        `n_baseline_days`. Prefer overriding `_rank_days` instead of this
        method.

        Parameters
        ----------
        candidate_days : list of pandas.Timestamp
            Days proposed for this event's baseline, before filtering.

        historical_power_kW : pandas.Series
            Historical realized power consumption in kW, indexed by
            `pandas.DatetimeIndex`. Passed through to `_rank_days`.

        event : dict
            A single event, as produced by `add_event`. Passed through to
            `_rank_days`.

        Raises
        ------
        ValueError
            When zero eligible days remain after filtering.

        Warnings
        --------
        When fewer eligible days remain than `n_baseline_days`.

        Returns
        -------
        list of pandas.Timestamp
            At most `n_baseline_days` days, most-preferred first.
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
        """Compute the day-of adjustment factor for this event.

        Compares the event day's own consumption over the window
        `[event_start - adjustment_offset_hours, event_start -
        adjustment_offset_hours + adjustment_duration_hours)` against the
        same pre-event window averaged across the baseline days, clipped to
        `adjustment_clip`. Always computed from `historical_power_kW`.

        Parameters
        ----------
        valid_days : list of pandas.Timestamp
            The baseline days selected for this event, as returned by
            `select_days`. Their pre-event windows form the denominator.

        historical_power_kW : pandas.Series
            Historical realized power consumption in kW, indexed by
            `pandas.DatetimeIndex`.

        event : dict
            A single event, as produced by `add_event`.

        Warnings
        --------
        When the denominator is near zero, in which case the adjustment is
        skipped by returning a factor of `1.0`.

        Returns
        -------
        float or None
            The clipped multiplicative factor, or `None` when
            `adjustment_offset_hours` is `None` (no adjustment configured).
        """
        if self.adjustment_offset_hours is None:
            return None
        window_start_hour = event[EVENT_START_HOUR] - self.adjustment_offset_hours
        event_adj_mask = _event_window_mask(
            historical_power_kW.index,
            event[EVENT_DATE],
            window_start_hour,
            self.adjustment_duration_hours,
        )
        event_adj_mean = historical_power_kW.loc[event_adj_mask].mean()

        baseline_adj_means = []
        for day in valid_days:
            mask = _event_window_mask(
                historical_power_kW.index,
                day,
                window_start_hour,
                self.adjustment_duration_hours,
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
    ):
        """Calculate the per-interval baseline power for a single event's window.

        This is the implementation behind the module-level
        `calculate_event_baseline`, whose docstring carries the full
        parameter and return-value contract: pass no `model` for a plain
        `numpy.ndarray`, or a `model` (plus `model_power_kW`,
        `model_datetime_index`, and `varstr`) to compute in-horizon baseline
        days from the decision variable, returning `(baseline, model)`.

        `t`, the index of the returned array/`Var`, is a **position within
        the event window** (`0` at `event[EVENT_START_HOUR]`), not a
        position in `model_datetime_index` or in calendar time.
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
            # "No baselining": skip day selection and the day-of adjustment
            # entirely (there are no days to compute either from). See the
            # class docstring for the payment-function implications of a
            # zero baseline.
            baseline_kW = np.zeros(n_intervals)
            if model is None:
                return baseline_kW
            return baseline_kW, model

        candidate_days = [pd.Timestamp(d) for d in event[BASELINE_DAYS]]
        valid_days = self.select_days(candidate_days, historical_power_kW, event)

        model_var_index = (
            list(model_power_kW.index_set()) if model_power_kW is not None else None
        )

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

        factor = self._adjustment_factor(valid_days, historical_power_kW, event)
        factor_in_model = (
            factor is not None and model is not None and self.adjustment_in_model
        )
        if factor_in_model:
            factor_name = varstr + "_adjustment_factor"
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
            varstr + "_constraint", pyo.Constraint(interval_idx, rule=baseline_rule)
        )
        return baseline_var, model


class TopUsageDaysBaseline(BaselineMethod):
    """See :doc:`/demandresponse` for a description of this class. Takes the
    same constructor parameters as `BaselineMethod`.
    """

    def _rank_days(self, candidate_days, historical_power_kW, event):
        """Rank eligible days by mean power over the event window, highest
        first, keeping the base eligibility rule (weekend/holiday exclusion).

        A day with no data in the event window sorts last (rather than
        raising); if such a day is still selected, `_baseline_day_terms`
        raises on it downstream.

        Parameters
        ----------
        candidate_days : list of pandas.Timestamp
            Days proposed for this event's baseline, before filtering.

        historical_power_kW : pandas.Series
            Historical realized power consumption in kW, indexed by
            `pandas.DatetimeIndex`. Supplies the consumption being ranked.

        event : dict
            A single event, as produced by `add_event`. Supplies the window
            (`EVENT_START_HOUR`, `EVENT_DURATION`) each day is scored over.

        Returns
        -------
        list of pandas.Timestamp
            The eligible subset of `candidate_days`, highest mean
            event-window power first.
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
    """See :doc:`/demandresponse` for a description of this class. Does not
    call `super().__init__()`, so none of `BaselineMethod`'s day-selection or
    adjustment configuration applies.

    Parameters
    ----------
    firm_level_kW : float
        The contracted firm demand level in kW, used as the baseline for
        every event.

    resolution : str or None
        Settlement interval width, as for `BaselineMethod`. Only consulted
        to decide how many (identical) values `compute` returns; if `None`,
        the width is inferred from `model_datetime_index` or
        `historical_power_kW.index`.

    Raises
    ------
    ValueError
        When `firm_level_kW` is negative.
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
    ):
        """Return the contracted firm level as a flat per-interval baseline.

        Adds nothing to `model`: the baseline is a constant, so there is no
        expression for a `Var` to stand in for. Every parameter other than
        `event` and `model` is accepted for interface compatibility with
        `BaselineMethod.compute`.

        Parameters
        ----------
        historical_power_kW : pandas.Series
            Not used as a data source -- this baseline is not derived from
            history -- but its index is consulted to infer the settlement
            interval width when `resolution` is `None` and no
            `model_datetime_index` is given.

        event : dict
            A single event, as produced by `add_event`. Supplies
            `EVENT_DURATION`, used to size the returned array.

        model : pyomo.environ.Model or pyomo.environ.Block or None
            Only consulted to decide the return shape, matching
            `BaselineMethod.compute`'s contract.

        model_power_kW : pyomo.environ.Var or None
            Ignored.

        model_datetime_index : pandas.DatetimeIndex or None
            Consulted only to infer the settlement interval width when
            `resolution` is `None`.

        varstr : str or None
            Ignored -- no components are created.

        Returns
        -------
        numpy.ndarray or tuple
            An array of `firm_level_kW` repeated once per settlement
            interval when `model` is `None`, otherwise `(that array, model)`.
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
    """See :doc:`/demandresponse` for a description of this class.

    Modeled as a hard constraint (an upper bound on `model_power_kW`), not a
    revenue opportunity: the operator has no decision to make, so `compute`
    raises `NotImplementedError` when called without a `model` (there is no
    ex-post evaluation path).

    Parameters
    ----------
    interruption_level_kW : float
        Power level in kW the load is held at during an event. Defaults to
        `0.0` (a full interruption).

    resolution : str or None
        Settlement interval width, as for `BaselineMethod`. Only consulted
        to decide how many (identical) values `compute` returns; if `None`,
        the width is inferred from `model_datetime_index`.
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
    ):
        """Constrain modeled power to the interruption level over the event
        window.

        Adds an indexed `Constraint` named
        `varstr + "_interruption_constraint"` over exactly the positions
        falling inside the event window, holding each at or below
        `interruption_level_kW`.

        Parameters
        ----------
        historical_power_kW : pandas.Series
            Ignored -- there is no historical counterfactual to draw on.

        event : dict
            A single event, as produced by `add_event`. Supplies the window
            to constrain.

        model : pyomo.environ.Model or pyomo.environ.Block or None
            The model to add the interruption constraint to. Required.

        model_power_kW : pyomo.environ.Var
            The model's power decision variable to cap. Required.

        model_datetime_index : pandas.DatetimeIndex
            Calendar timestamp for each position in
            `model_power_kW.index_set()`. Required.

        varstr : str
            Name prefix for the constraint created on `model`. Must be
            unique per call on a given `model`. Required.

        Raises
        ------
        NotImplementedError
            When `model` is `None`, since this program has no ex-post or
            numpy evaluation.

        ValueError
            When `model_power_kW`, `model_datetime_index`, or `varstr` is
            missing, or when the event window matches no positions in
            `model_datetime_index`.

        Returns
        -------
        tuple
            `(baseline_kW, model)`, where `baseline_kW` is a `numpy.ndarray`
            of `interruption_level_kW` repeated once per settlement
            interval. The level is returned in the baseline's position so
            downstream payment logic keeps a consistent interface, though
            for this program the reduction it implies is not paid per-event
            -- pair a payout-only `PaymentStructure` with this baseline to
            pay for the interruption itself.
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
            varstr + "_interruption_constraint",
            pyo.Constraint(matched_indices, rule=interruption_rule),
        )
        return baseline_kW, model


def _coerce_baseline_method(baseline_params):
    """Normalize a baseline configuration into a `BaselineMethod`.

    Lets every public entry point accept either a dict from
    `make_baseline_parameters` or an already-constructed `BaselineMethod`
    instance (e.g. `TopUsageDaysBaseline`).

    Parameters
    ----------
    baseline_params : dict or BaselineMethod
        Baseline parameters as produced by `make_baseline_parameters`, or
        an already-constructed baseline method.

    Returns
    -------
    BaselineMethod
        `baseline_params` itself when it is already a `BaselineMethod`,
        otherwise a new foundation `BaselineMethod` carrying the dict's
        settings.
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
        holiday_dates=baseline_params[HOLIDAY_DATES],
        resolution=baseline_params.get(RESOLUTION),
    )


def _event_window_mask(index, event_date, start_hour, duration_hours):
    """Boolean mask selecting timestamps in `index` within the half-open window
    [event_date + start_hour, event_date + start_hour + duration_hours).

    Use this over `pandas.Series.between_time`, which would match the window
    on every day present rather than the one event day.

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


def _index_step(index):
    """Infer a `pandas.DatetimeIndex`'s regular spacing as a `pandas.Timedelta`.

    Parameters
    ----------
    index : pandas.DatetimeIndex or None
        Index to infer the spacing of. `None` is passed through as `None`,
        so callers can try a fallback index without a separate `is None`
        check.

    Raises
    ------
    ValueError
        When `index` has fewer than 2 entries.

    Returns
    -------
    pandas.Timedelta or None
        `None` if `index` is `None`, otherwise the gap between its first two
        entries.
    """
    if index is None:
        return None
    if len(index) < 2:
        raise ValueError("index must have at least 2 entries to infer its step size")
    return index[1] - index[0]


def _resolve_step(resolution, *indices):
    """Pick the settlement interval width for a baseline calculation.

    Precedence: an explicit `resolution` string first, then the step of the
    first non-`None` index in `indices`.

    Parameters
    ----------
    resolution : str or None
        Interval width as a string of the form `"[int][unit]"` (e.g.
        `"15m"`, `"1h"`), parsed by `utils.get_freq_binsize_minutes`. Takes
        precedence over every index when given.

    *indices : pandas.DatetimeIndex or None
        Candidate indices to infer the step from, in preference order.
        `None` entries are skipped.

    Raises
    ------
    ValueError
        When `resolution` is `None` and every index is either `None` or has
        fewer than 2 entries.

    Returns
    -------
    pandas.Timedelta
        The resolved interval width.
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
    """Number of equal-width settlement intervals spanning an event window.

    Parameters
    ----------
    duration_hours : float
        Length of the event window in hours.

    step : pandas.Timedelta
        Width of one settlement interval.

    Raises
    ------
    ValueError
        When `duration_hours` is not (approximately) an integer multiple of
        `step`.

    Returns
    -------
    int
        The number of intervals.
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
    """Calendar start timestamp of each settlement interval in an event window.

    Parameters
    ----------
    event_date : datetime.date, datetime.datetime, or str
        Calendar date the window is anchored to.

    start_hour : float
        Hour of day (0-24) the window begins.

    duration_hours : float
        Length of the window in hours.

    step : pandas.Timedelta
        Width of one settlement interval.

    Returns
    -------
    pandas.DatetimeIndex
        One entry per interval, its start timestamp.
    """
    window_start = pd.Timestamp(event_date) + pd.Timedelta(hours=start_hour)
    n_intervals = _window_interval_count(duration_hours, step)
    return pd.DatetimeIndex([window_start + i * step for i in range(n_intervals)])


def _window_interval_buckets(index, event_date, start_hour, duration_hours, step):
    """Bucket `index`'s positions within an event window by settlement interval.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Index of timestamps to bucket.

    event_date : datetime.date, datetime.datetime, or str
        Calendar date the window is anchored to.

    start_hour : float
        Hour of day (0-24) the window begins.

    duration_hours : float
        Length of the window in hours.

    step : pandas.Timedelta
        Width of one settlement interval; must evenly divide `duration_hours`
        (see `_window_interval_count`).

    Returns
    -------
    list of numpy.ndarray
        One entry per interval, holding the positions of `index` (as used by
        `.iloc`/`values[...]`) whose timestamp falls in that interval. An
        interval with no matching positions is an empty array -- callers
        decide whether that is an error.
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
    """Normalize a pyomo-side `reduction_kW` into a list of per-interval terms.

    Parameters
    ----------
    reduction_kW : pyomo.environ.Var, pyomo expression, or list/tuple
        An indexed `Var`/expression (one entry per settlement interval), a
        scalar `Var`/expression (treated as a single interval), or an
        already-built list/tuple of per-interval expressions (as
        `build_event_revenue` passes).

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
    """Foundation payment structure. See :doc:`/demandresponse` for a
    description of this class and how to extend it.

    The capacity payment is `payment_ratio * capacity_price * bid_capacity_kW`,
    where the `regions` list maps the delivered ratio (`reduction_kW /
    bid_capacity_kW`) to a payment ratio through consecutive linear segments.

    Parameters
    ----------
    regions : list of dict or None
        Payment schedule, each dict having keys `REGION_X1`, `REGION_X2`,
        `REGION_Y1`, and `REGION_Y2`. Expected to cover the delivered
        ratios that can occur; `find_region` raises if one is uncovered.
        A bound may be given as the string `"Infinity"`/`"-Infinity"`
        (as produced by `json.load` on a quoted JSON value) instead of a
        float; these are coerced to `inf`/`-inf` on construction. `None`
        (or an empty list) means no capacity payment at all -- a
        payout-only structure, for programs whose revenue is entirely the
        flat `payout` below.

    settlement : str
        How the capacity payment is aggregated over the event window's
        settlement intervals when `reduction_kW` carries more than one
        (see `calculate_event_baseline`'s `resolution`):

        - `SETTLEMENT_AVERAGE` (`"average"`, default): mean the per-interval
          reduction to a scalar delivered ratio, then evaluate the
          piecewise function once. This is the historical behavior of this
          class, and stays the default because it keeps the pyomo
          formulation to one region-selection block per event.
        - `SETTLEMENT_INTERVAL` (`"interval"`): evaluate the piecewise
          function at each interval's own delivered ratio, then mean the
          resulting payments (equivalently, a duration-weighted mean, since
          this module's settlement intervals are equal-width). This is the
          more rigorous reading whenever the payment function is
          nonlinear over the realized range -- the two modes agree exactly
          when it is linear there, and always agree when `reduction_kW` is
          a single scalar. In the pyomo path, this multiplies the number of
          region-selection binaries by the number of intervals; budget for
          that when choosing it.

    payment_basis : str
        Whether `capacity_price` is settled once per event
        (`BASIS_PER_EVENT`, `"per_event"`, default) or scales with the
        event's duration (`BASIS_PER_HOUR`, `"per_hour"`, i.e.
        `capacity_price` is $/kW-hour rather than $/kW-event). `"per_hour"`
        requires `event[EVENT_DURATION]`.

    payout : float
        A flat, performance-independent participation payment in
        **$/kW of `event[BID_CAPACITY_KW]`** (not a flat dollar amount --
        this structure is shared across events with different bid sizes),
        added on top of the capacity payment. `0.0` (default) adds nothing.

    payout_basis : str
        Whether `payout` is settled once per event (`BASIS_PER_EVENT`,
        default) or scales with the event's duration (`BASIS_PER_HOUR`,
        requires `event[EVENT_DURATION]`).

    Raises
    ------
    ValueError
        When `settlement`, `payment_basis`, or `payout_basis` is not one of
        its documented values.
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
        """Look up the applicable payment region.

        Parameters
        ----------
        delivered_ratio : float or None
            Known delivered ratio to look up by interval containment.

        region_x1 : float or None
            The `x1` value identifying the region to look up. Takes
            precedence when both are given.

        Raises
        ------
        ValueError
            When no region matches, or when this structure has no regions
            at all (a payout-only structure).

        Returns
        -------
        dict
            The matching region.
        """
        if not self.regions:
            raise ValueError(
                "This PaymentStructure has no regions (payout-only); there is no "
                "capacity payment region to look up"
            )
        if region_x1 is not None:

            def predicate(r):
                return np.isclose(r[REGION_X1], region_x1)

            error_msg = f"No region with x1 close to {region_x1}"
        else:

            def predicate(r):
                return r[REGION_X1] <= delivered_ratio < r[REGION_X2]

            error_msg = (
                f"delivered_ratio {delivered_ratio} is not covered by payment_function"
            )
        region = next((r for r in self.regions if predicate(r)), None)
        if region is None:
            raise ValueError(error_msg)
        return region

    def _basis_multiplier(self, event, basis, label):
        """1.0 for `BASIS_PER_EVENT`, `event[EVENT_DURATION]` for
        `BASIS_PER_HOUR`.

        Parameters
        ----------
        event : dict
            A single event, or the minimal synthetic dict built by
            `evaluate_payment_function`/`build_payment_expression`.

        basis : str
            `self.payment_basis` or `self.payout_basis`.

        label : str
            Name of the attribute being resolved, used only to phrase the
            error message (`"payment_basis"` or `"payout_basis"`).

        Raises
        ------
        ValueError
            When `basis` is `BASIS_PER_HOUR` and `event` has no
            `EVENT_DURATION`.

        Returns
        -------
        float
            The multiplier.
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
        """Flat participation payout for this event, in USD.

        Parameters
        ----------
        event : dict
            A single event, or the minimal synthetic dict built by
            `evaluate_payment_function`/`build_payment_expression`.

        Returns
        -------
        float
            `0.0` when `self.payout == 0.0` (regardless of whether
            `event[EVENT_DURATION]` is available), otherwise
            `payout * bid_capacity_kW * basis_multiplier`.
        """
        if self.payout == 0.0:
            return 0.0
        basis_mult = self._basis_multiplier(event, self.payout_basis, "payout_basis")
        return self.payout * event[BID_CAPACITY_KW] * basis_mult

    def _payment_ratio(self, delivered_ratio):
        """Piecewise-linear payment ratio `f(delivered_ratio)`.

        Parameters
        ----------
        delivered_ratio : float
            A single interval's (or the window-mean) delivered ratio.

        Returns
        -------
        float
            The interpolated payment ratio.
        """
        region = self.find_region(delivered_ratio=delivered_ratio)
        x1, x2, y1, y2 = (
            region[k] for k in (REGION_X1, REGION_X2, REGION_Y1, REGION_Y2)
        )
        if np.isinf(x2):
            return y1
        return y1 + (y2 - y1) * (delivered_ratio - x1) / (x2 - x1)

    def _region_coefficients(self, event):
        """Per-region `(slope, intercept)` pairs, with `capacity_price`,
        `bid_capacity_kW`, and the `payment_basis` multiplier folded in, so
        every backend (cvxpy, pyomo) and `evaluate` compute revenue from the
        same numbers.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`.

        Returns
        -------
        tuple
            `(slopes, intercepts)`, parallel lists, one entry per region in
            `self.regions`.
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
        """Normalize a numeric `reduction_kW` into per-interval terms.

        Parameters
        ----------
        reduction_kW : float or numpy.ndarray
            A scalar or a 1-D array of per-interval reductions.

        Returns
        -------
        tuple
            `(terms, n_intervals)`, where `terms` is a `list` of `float`.
        """
        arr = np.atleast_1d(np.asarray(reduction_kW, dtype=float))
        return list(arr), arr.size

    def evaluate(self, event, reduction_kW):
        """Calculate realized revenue for a known reduction. Interpolates
        based on the payment function, honoring `self.settlement`.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`. Supplies
            `BID_CAPACITY_KW` and `CAPACITY_PRICE`, and `EVENT_DURATION` if
            `payment_basis`/`payout_basis` is `"per_hour"`.

        reduction_kW : float or numpy.ndarray
            Realized load reduction (baseline minus actual power) in kW, as
            a window-mean scalar or a 1-D array of per-interval values. A
            scalar is always evaluated the same way regardless of
            `self.settlement`.

        Raises
        ------
        ValueError
            When the event's `BID_CAPACITY_KW` is not positive; when no
            region covers a resulting delivered ratio; or when a `"per_hour"`
            basis is configured and `event` has no `EVENT_DURATION`.

        Returns
        -------
        float
            Revenue (positive) or penalty (negative) in USD, including any
            configured `payout`.
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
        """Build the revenue expression for this event.

        Based on the type of `reduction_kW`:
        - `numpy.ndarray` or Python number: the region is already
          determined, via `evaluate`.
        - `cvxpy.Expression`/`cvxpy.Variable`: still requires a known
          `region_x1` (raises if missing) and builds only that one region.
          A vector `reduction_kW` (one entry per settlement interval) is
          meaned for `"average"` settlement; for `"interval"` settlement the
          region bound constraints apply to every interval elementwise,
          though the revenue expression is identical in both modes (the
          payment ratio is affine within one fixed region, so its mean
          equals the mean reduction's payment ratio).
        - `pyomo.environ.Var`/expression, or a `list`/`tuple` of pyomo
          expressions (one per settlement interval): builds *every* region
          onto `model` at once, using a disaggregated binary-selection
          formulation (Balas' extended form). To optimize over the regions,
          use `region_x1=None`.

          For `"average"` settlement, components added under `varstr`
          (`R` = `len(self.regions)`, indexed `0..R-1`):
          - `_region_active`: binary `Var`, 1 iff region `r` is active.
          - `_region_select_constraint`: exactly one region is active.
          - `_region_reduction`: `Var`, region `r`'s disaggregated share of
            the mean reduction -- forced to `0` when region `r` is
            inactive, bounded by region `r`'s own `[x1, x2] *
            bid_capacity_kW` box when active.
          - `_region_reduction_lower_constraint` / `_upper_constraint`:
            the bounds above.
          - `_region_reduction_sum_constraint`: the mean reduction equals
            the sum of the per-region shares.
          - `_revenue` / `_revenue_constraint`: total revenue (capacity
            payment, summed from each region's own contribution, plus any
            `payout`).

          For `"interval"` settlement, every component above gains a
          leading interval index `t` (`0..n_intervals-1`), one
          region-selection block per interval --
          `_region_active`/`_region_reduction` become `Var(interval_idx,
          region_idx, ...)`, `_region_select_constraint`/
          `_region_reduction_sum_constraint` become one-per-interval, and
          an `_interval_revenue : Var(interval_idx)` is added so a solved
          model can be audited per interval; `_revenue` is their mean plus
          `payout`. This multiplies the number of binaries by
          `n_intervals` relative to `"average"` settlement -- the cost of
          the more rigorous formulation. A scalar `region_x1` fixes the
          same region for every interval; passing a sequence of length
          `n_intervals` fixes them individually. Fixing one region for
          every interval is a *tighter* feasible set than `"average"`
          settlement (which only constrains the mean), and can render a
          previously-feasible plan infeasible -- prefer `region_x1=None`
          in `"interval"` settlement and let the solver choose per
          interval.

          If `self.regions` is empty (a payout-only structure), no region
          components are created in either backend; `_revenue` (or the
          cvxpy/numpy return value) is just the constant payout.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`. Supplies
            `BID_CAPACITY_KW` and `CAPACITY_PRICE`, and `EVENT_DURATION` if
            `payment_basis`/`payout_basis` is `"per_hour"`.

        reduction_kW : numpy.ndarray, float, cvxpy.Expression, pyomo.environ.Var,
            or list/tuple of pyomo expressions
            Load reduction, as a realized value or a decision-variable
            expression, per settlement interval (or a single scalar/window
            mean).

        region_x1 : float, list of float, or None
            The `x1` value identifying which region to fix. Required for
            the cvxpy case. For the pyomo case under `"interval"`
            settlement, a list of length `n_intervals` fixes each
            interval's region individually; a scalar fixes the same region
            for every interval.

        model : pyomo.environ.Model or pyomo.environ.Block or None
            Only used in the pyomo case.

        varstr : str
            Name prefix for pyomo components created on `model`. Must be
            unique per call on a given `model` or `block`.

        Raises
        ------
        ValueError
            When `reduction_kW` is a cvxpy type and no region matches
            `region_x1` (including when it's `None`), or when `reduction_kW`
            is a pyomo type and a given `region_x1` matches no region.

        TypeError
            When `reduction_kW` is not a supported type.

        Returns
        -------
        tuple
            `(revenue, model)` for numpy/scalar, `(revenue_var, model)` for
            pyomo, or `(revenue_expr, constraints_list)` for cvxpy.
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
        """The cvxpy branch of `build_expression`.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`.

        reduction_kW : cvxpy.Expression or cvxpy.Variable
            Load reduction, scalar or a vector over settlement intervals.

        region_x1 : float or None
            The `x1` value identifying which region to fix. Required.

        Raises
        ------
        ValueError
            When no region matches `region_x1` (including when it's `None`
            and `self.regions` is non-empty).

        Returns
        -------
        tuple
            `(revenue_expr, constraints_list)`.
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
        """Build a constant-revenue expression for a payout-only structure
        (`self.regions` is empty), for the pyomo backend.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`.

        model : pyomo.environ.Model or pyomo.environ.Block
            The model to add components to.

        varstr : str
            Name prefix for the `_revenue`/`_revenue_constraint` components.

        Returns
        -------
        tuple
            `(revenue_var, model)`.
        """
        model.add_component(varstr + "_revenue", pyo.Var())
        revenue_var = model.find_component(varstr + "_revenue")
        model.add_component(
            varstr + "_revenue_constraint",
            pyo.Constraint(expr=revenue_var == self._payout_amount(event)),
        )
        return revenue_var, model

    def _build_pyomo_regions_scalar(self, event, terms, region_x1, model, varstr):
        """`"average"` settlement: one region-selection block for the
        window-mean reduction. See `build_expression` for the component
        names this creates.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`.

        terms : list
            Per-interval reduction expressions (length 1 for a scalar
            reduction); meaned here before region selection.

        region_x1 : float or None
            The `x1` value identifying which region to fix, or `None` to
            leave the choice to the solver.

        model : pyomo.environ.Model or pyomo.environ.Block
            The model to add components to.

        varstr : str
            Name prefix for the components created on `model`.

        Returns
        -------
        tuple
            `(revenue_var, model)`.
        """
        bid_capacity_kW = event[BID_CAPACITY_KW]
        mean_reduction = pyo.quicksum(terms) / len(terms)
        region_idx = range(len(self.regions))
        slopes, intercepts = self._region_coefficients(event)

        model.add_component(
            varstr + "_region_active", pyo.Var(region_idx, within=pyo.Binary)
        )
        z = model.find_component(varstr + "_region_active")
        model.add_component(
            varstr + "_region_select_constraint",
            pyo.Constraint(expr=pyo.quicksum(z[r] for r in region_idx) == 1),
        )

        model.add_component(varstr + "_region_reduction", pyo.Var(region_idx))
        region_reduction = model.find_component(varstr + "_region_reduction")

        def lower_rule(m, r):
            x1 = self.regions[r][REGION_X1]
            # implicitly bounds the DR bid to be > 0.1% of max power production
            if np.isinf(x1):
                x1 = -1000.0
            return region_reduction[r] >= x1 * bid_capacity_kW * z[r]

        model.add_component(
            varstr + "_region_reduction_lower_constraint",
            pyo.Constraint(region_idx, rule=lower_rule),
        )

        def upper_rule(m, r):
            x2 = self.regions[r][REGION_X2]
            # implicitly bounds the DR bid to be > 0.1% of max power consumption
            if np.isinf(x2):
                x2 = 1000.0
            return region_reduction[r] <= x2 * bid_capacity_kW * z[r]

        model.add_component(
            varstr + "_region_reduction_upper_constraint",
            pyo.Constraint(region_idx, rule=upper_rule),
        )

        model.add_component(
            varstr + "_region_reduction_sum_constraint",
            pyo.Constraint(
                expr=mean_reduction
                == pyo.quicksum(region_reduction[r] for r in region_idx)
            ),
        )

        model.add_component(varstr + "_revenue", pyo.Var())
        revenue_var = model.find_component(varstr + "_revenue")
        payout_amt = self._payout_amount(event)
        model.add_component(
            varstr + "_revenue_constraint",
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
        """`"interval"` settlement: one region-selection block per settlement
        interval. See `build_expression` for the component names this
        creates and the feasibility caveat around fixing `region_x1`.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`.

        terms : list
            Per-interval reduction expressions, length `n_intervals`.

        region_x1 : float, list of float, or None
            The `x1` value fixing every interval's region (scalar), each
            interval's region individually (a length-`n_intervals` list),
            or `None` to leave every interval's choice to the solver.

        model : pyomo.environ.Model or pyomo.environ.Block
            The model to add components to.

        varstr : str
            Name prefix for the components created on `model`.

        Raises
        ------
        ValueError
            When `region_x1` is a sequence whose length does not match
            `len(terms)`.

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

        model.add_component(
            varstr + "_region_active",
            pyo.Var(interval_idx, region_idx, within=pyo.Binary),
        )
        z = model.find_component(varstr + "_region_active")
        model.add_component(
            varstr + "_region_select_constraint",
            pyo.Constraint(
                interval_idx,
                rule=lambda m, t: pyo.quicksum(z[t, r] for r in region_idx) == 1,
            ),
        )

        model.add_component(
            varstr + "_region_reduction", pyo.Var(interval_idx, region_idx)
        )
        region_reduction = model.find_component(varstr + "_region_reduction")

        def lower_rule(m, t, r):
            x1 = self.regions[r][REGION_X1]
            if np.isinf(x1):
                x1 = -1000.0
            return region_reduction[t, r] >= x1 * bid_capacity_kW * z[t, r]

        model.add_component(
            varstr + "_region_reduction_lower_constraint",
            pyo.Constraint(interval_idx, region_idx, rule=lower_rule),
        )

        def upper_rule(m, t, r):
            x2 = self.regions[r][REGION_X2]
            if np.isinf(x2):
                x2 = 1000.0
            return region_reduction[t, r] <= x2 * bid_capacity_kW * z[t, r]

        model.add_component(
            varstr + "_region_reduction_upper_constraint",
            pyo.Constraint(interval_idx, region_idx, rule=upper_rule),
        )

        model.add_component(
            varstr + "_region_reduction_sum_constraint",
            pyo.Constraint(
                interval_idx,
                rule=lambda m, t: terms[t]
                == pyo.quicksum(region_reduction[t, r] for r in region_idx),
            ),
        )

        model.add_component(varstr + "_interval_revenue", pyo.Var(interval_idx))
        interval_revenue = model.find_component(varstr + "_interval_revenue")
        model.add_component(
            varstr + "_interval_revenue_constraint",
            pyo.Constraint(
                interval_idx,
                rule=lambda m, t: interval_revenue[t]
                == pyo.quicksum(
                    slopes[r] * region_reduction[t, r] + intercepts[r] * z[t, r]
                    for r in region_idx
                ),
            ),
        )

        model.add_component(varstr + "_revenue", pyo.Var())
        revenue_var = model.find_component(varstr + "_revenue")
        payout_amt = self._payout_amount(event)
        model.add_component(
            varstr + "_revenue_constraint",
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
    """See :doc:`/demandresponse` for a description of this class.

    Parameters
    ----------
    regions : list of dict or None
        Capacity payment schedule, as for `PaymentStructure`.

    energy_price : float
        Energy payment rate in $/kWh applied to the curtailed energy. When
        `reduction_kW` carries more than one settlement interval, this is
        `energy_price * mean(reduction_kW) * event[EVENT_DURATION]` --
        equivalently, the sum of each interval's own energy
        (`reduction_t * interval_hours`), since the intervals are
        equal-width and `interval_hours = EVENT_DURATION / n_intervals`.
        Independent of `self.settlement`, which only affects the capacity
        term.

    **payment_kwargs
        Forwarded to `PaymentStructure.__init__` (`settlement`,
        `payment_basis`, `payout`, `payout_basis`).
    """

    def __init__(self, regions, energy_price, **payment_kwargs):
        super().__init__(regions, **payment_kwargs)
        self.energy_price = energy_price

    def evaluate(self, event, reduction_kW):
        """Realized capacity payment plus the energy payment.

        Needs the full `event` dict for `EVENT_DURATION`, which is why the
        scalar-argument `evaluate_payment_function` wrapper cannot be used
        with this class.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`. Supplies
            `EVENT_DURATION` in addition to the fields the capacity term
            needs.

        reduction_kW : float or numpy.ndarray
            Realized load reduction in kW, window-mean or per-interval.

        Returns
        -------
        float
            Combined revenue in USD.
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
        """Build the combined capacity-plus-energy revenue expression.

        Returns
        -------
        tuple
            `(total_revenue_var, model)` for pyomo, `(revenue, model)` for
            numpy/scalar, or `(revenue_expr, constraints_list)` for cvxpy.
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

        model.add_component(varstr + "_energy_revenue", pyo.Var())
        energy_var = model.find_component(varstr + "_energy_revenue")

        def energy_rule(m):
            return energy_var == energy_term

        model.add_component(
            varstr + "_energy_revenue_constraint", pyo.Constraint(rule=energy_rule)
        )

        model.add_component(varstr + "_total_revenue", pyo.Var())
        total_var = model.find_component(varstr + "_total_revenue")

        def total_rule(m):
            return total_var == capacity_var + energy_var

        model.add_component(
            varstr + "_total_revenue_constraint", pyo.Constraint(rule=total_rule)
        )
        return total_var, model


class MarketIndexedPayment(PaymentStructure):
    """See :doc:`/demandresponse` for a description of this class.

    Parameters
    ----------
    regions : list of dict or None
        Payment schedule, as for `PaymentStructure`.

    price_lookup : callable
        Called as `price_lookup(event)` and must return the capacity price
        in $/kW to use for that event. Typically closes over a price series
        and keys off `event[EVENT_DATE]`.

    **payment_kwargs
        Forwarded to `PaymentStructure.__init__` (`settlement`,
        `payment_basis`, `payout`, `payout_basis`).
    """

    def __init__(self, regions, price_lookup, **payment_kwargs):
        super().__init__(regions, **payment_kwargs)
        self.price_lookup = price_lookup  # callable: price_lookup(event) -> float

    def _resolve_event(self, event):
        """Return a shallow copy of `event` with `CAPACITY_PRICE` replaced
        by the looked-up market price.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`.

        Returns
        -------
        dict
            A shallow copy carrying the resolved price.
        """
        resolved = dict(event)
        resolved[CAPACITY_PRICE] = self.price_lookup(event)
        return resolved

    def evaluate(self, event, reduction_kW):
        """Realized revenue at the looked-up market price. Takes and
        returns the same things as `PaymentStructure.evaluate`."""
        return super().evaluate(self._resolve_event(event), reduction_kW)

    def build_expression(
        self, event, reduction_kW, region_x1=None, model=None, varstr=""
    ):
        """Revenue expression at the looked-up market price. Takes and
        returns the same things as `PaymentStructure.build_expression`.

        The price is resolved once, at build time, so a model built this
        way is tied to the prices in force when it was built.
        """
        return super().build_expression(
            self._resolve_event(event),
            reduction_kW,
            region_x1=region_x1,
            model=model,
            varstr=varstr,
        )


def _coerce_payment_structure(payment_function):
    """Normalize a payment configuration into a `PaymentStructure`.

    The payment-side counterpart to `_coerce_baseline_method`: lets every
    public entry point accept either a list of region dicts or an
    already-constructed `PaymentStructure` instance (e.g.
    `CapacityEnergyPayment`).

    Parameters
    ----------
    payment_function : list of dict or PaymentStructure
        Payment/penalty schedule as a list of region dicts, or an
        already-constructed payment structure.

    Returns
    -------
    PaymentStructure
        `payment_function` itself when it is already a `PaymentStructure`,
        otherwise a new foundation `PaymentStructure` over those regions.
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
    """Calculate ex-post revenue for a known reduction.

    Note this function passes only `bid_capacity_kW`, `capacity_price`, and
    (if given) `duration_hours` through to `payment_function` -- a subclass
    needing other event fields (e.g. `CapacityEnergyPayment`, or any
    `"per_hour"` basis without `duration_hours` supplied here) must be used
    via `build_event_revenue` or `calculate_dr_revenue` instead, which have
    the full event to hand.

    Parameters
    ----------
    payment_function : list of dict or PaymentStructure
        Each dict has keys `REGION_X1`, `REGION_X2`, `REGION_Y1`,
        `REGION_Y2`. A `PaymentStructure` instance is also accepted.

    reduction_kW : float or numpy.ndarray
        Realized load reduction (baseline minus actual power) in kW,
        window-mean or per-interval.

    bid_capacity_kW : float
        Nominated capacity bid in kW.

    capacity_price : float
        Program capacity price in $/kW.

    duration_hours : float or None
        Event duration in hours, populating `EVENT_DURATION` in the
        synthetic event dict. Required only when `payment_function`'s
        `payment_basis` or `payout_basis` is `"per_hour"`.

    Raises
    ------
    ValueError
        When `bid_capacity_kW` is not positive; when `payment_function` has
        no region covering the resulting delivered ratio; or when a
        `"per_hour"` basis is configured and `duration_hours` is not given.

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
    """Build the revenue expression for a single, specified region.

    Dispatches on `reduction_kW`'s type (numpy/scalar, cvxpy, or pyomo); see
    `PaymentStructure.build_expression`, which this delegates to.

    Parameters
    ----------
    payment_function : list of dict or PaymentStructure
        Each dict has keys `REGION_X1`, `REGION_X2`, `REGION_Y1`,
        `REGION_Y2`. A `PaymentStructure` instance is also accepted, but
        note that this function passes only `bid_capacity_kW`,
        `capacity_price`, and (if given) `duration_hours` through -- a
        subclass needing other event fields (as `CapacityEnergyPayment`
        needs `EVENT_DURATION`) must be used via `build_event_revenue`,
        `calculate_dr_revenue`, or `build_dr_revenue`, which have the full
        event to hand.

    reduction_kW : numpy.ndarray, float, cvxpy.Expression, pyomo.environ.Var,
        or list/tuple of pyomo expressions
        Load reduction, as a realized value or a decision-variable
        expression, per settlement interval (or a single scalar/window mean).

    bid_capacity_kW : float
        Nominated capacity bid in kW.

    capacity_price : float
        Program capacity price in $/kW.

    region_x1 : float, list of float, or None
        The `x1` value identifying which region to build. Required (and
        used) only for the cvxpy/pyomo cases.

    model : pyomo.environ.Model or pyomo.environ.Block
        The model or block to add pyomo components to.
        Only used in the pyomo case, so `None` by default.

    varstr : str
        Name prefix for pyomo variables/constraints created on `model`.
        Must be unique per call on a given `model`, since reusing a `varstr`
        will raise a pyomo "component already exists" error.

    duration_hours : float or None
        Event duration in hours, populating `EVENT_DURATION` in the
        synthetic event dict. Required only when `payment_function`'s
        `payment_basis` or `payout_basis` is `"per_hour"`.

    Raises
    ------
    ValueError
        When a `"per_hour"` basis is configured and `duration_hours` is not
        given, in addition to the cases documented on
        `PaymentStructure.build_expression`.

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
):
    """Adds a demand response event to an events collection, returning a new
    list.

    `baseline_days` should already exclude any date that is itself another
    event's date; this is not checked here.

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
    --------
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
    adjustment_offset_hours=3,
    adjustment_duration_hours=3,
    adjustment_clip=(0.8, 1.2),
    exclude_weekends=True,
    exclude_holidays=True,
    holiday_dates=None,
    resolution=None,
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
        Number of valid baseline days to average over. `0` means "no
        baselining" -- see `BaselineMethod`'s docstring for what this
        implies for the resulting delivered ratio.

    adjustment_offset_hours : int or None
        Number of hours before the event start where the day-of adjustment
        window begins. If `None`, no day-of adjustment is applied.

    adjustment_duration_hours : int
        Length, in hours, of the day-of adjustment window. Combined with
        `adjustment_offset_hours`, the window is `[event_start -
        adjustment_offset_hours, event_start - adjustment_offset_hours +
        adjustment_duration_hours)` -- e.g. `adjustment_offset_hours=4,
        adjustment_duration_hours=2` looks at the window starting 4 hours
        before the event and ending 2 hours before it. Ignored when
        `adjustment_offset_hours` is `None`.

    adjustment_clip : tuple of float or None
        `(low, high)` bounds the day-of adjustment factor is clipped to.

    exclude_weekends : bool
        If `True`, exclude Saturday/Sunday from candidate baseline days.

    exclude_holidays : bool
        If `True`, exclude dates in `holiday_dates` from candidate baseline
        days.

    holiday_dates : list or None
        Calendar dates treated as holidays. Defaults to an empty list.

    resolution : str or None
        Settlement interval width, as for `BaselineMethod`. If `None`
        (default), inferred from the data/model index at compute time.

    Raises
    ------
    ValueError
        When `baseline_method` is not `"average_similar_days"`; when
        `n_baseline_days` is negative; or when `adjustment_offset_hours`
        is not `None` and `adjustment_duration_hours` is not positive or
        exceeds `adjustment_offset_hours`.

    Returns
    -------
    dict
        Baseline parameters keyed by the module's `BASELINE_METHOD`,
        `N_BASELINE_DAYS`, `ADJUSTMENT_OFFSET_HOURS`,
        `ADJUSTMENT_DURATION_HOURS`, `ADJUSTMENT_CLIP`, `EXCLUDE_WEEKENDS`,
        `EXCLUDE_HOLIDAYS`, `HOLIDAY_DATES`, and `RESOLUTION` constants.
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
        HOLIDAY_DATES: list(holiday_dates) if holiday_dates else [],
        RESOLUTION: resolution,
    }


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
        Regular spacing of `datetime_index` (e.g. the gap between its first
        two entries).

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
    """Compute a single baseline day's per-interval values.

    Parameters
    ----------
    day : pandas.Timestamp
        Calendar date the window is anchored to.

    start_hour : float
        Hour of day (0-24) the window begins.

    duration_hours : float
        Length of the window in hours.

    step : pandas.Timedelta
        Settlement interval width, as resolved by `_resolve_step`.

    n_intervals : int
        Number of settlement intervals in the event window
        (`_window_interval_count(duration_hours, step)`).

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

    Raises
    ------
    ValueError
        When the day is in-horizon but some interval matched no positions in
        `model_datetime_index`; when the day is historical and some interval
        has no data or only `NaN` data.

    Returns
    -------
    tuple
        `(values, is_dynamic)`, where `values` is a list of length
        `n_intervals`: floats for a historical day, pyomo expressions (each
        interval's own mean, as a `quicksum`) for a day inside the
        optimization horizon.
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
    """Calculate the per-interval baseline power for a single event's window.

    The event window is divided into settlement intervals (see
    `resolution` in `make_baseline_parameters`/`BaselineMethod`) and one
    baseline value is returned per interval, not a single window-mean.

    Pass no `model` for a plain `numpy.ndarray`, ex-post. Pass a `model`
    (plus `model_power_kW` and `model_datetime_index`) to compute any
    baseline day fully inside the simulation horizon from the decision
    variable instead of history, returning `(baseline, model)`. The day-of
    adjustment factor is always computed from `historical_power_kW`, never
    from the model.

    Parameters
    ----------
    historical_power_kW : pandas.Series
        Historical realized power consumption in kW, indexed by
        `pandas.DatetimeIndex`. Used for every baseline day that isn't fully
        contained in the simulation horizon, and always for the day-of
        adjustment factor.

    event : dict
        A single event, as produced by `add_event`.

    baseline_params : dict or BaselineMethod
        Baseline parameters, as produced by `make_baseline_parameters`, or
        a `BaselineMethod` instance (e.g. `TopUsageDaysBaseline`) to use a
        non-default baselining strategy.

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
    --------
    When fewer valid baseline days remain than `N_BASELINE_DAYS`.

    Returns
    -------
    numpy.ndarray or tuple
        `numpy.ndarray`: per-interval baseline power in kW, when `model` is
        `None`. `(baseline_kW, model)`: when `model` is given, where
        `baseline_kW` is a `numpy.ndarray` if every baseline day stayed
        historical, or a `pyomo.environ.Var` indexed `0..n_intervals-1` if
        at least one was dynamic.
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
    """Normalize `baseline_kW` into a list of `n` per-interval values.

    Parameters
    ----------
    baseline_kW : float, numpy.ndarray, list/tuple, or pyomo.environ.Var
        A single value (broadcast to every interval), or one value per
        interval -- as an array/list, or an indexed pyomo `Var` (from the
        dynamic-baseline path of `calculate_event_baseline`).

    n : int
        Number of intervals `power_kW` (the event-window slice) covers.

    Raises
    ------
    ValueError
        When `baseline_kW` carries more than one value and its count does
        not match `n`.

    Returns
    -------
    list
        Length-`n` list of values/expressions.
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
        numpy array or a decision-variable expression. Its length (or
        `.size`) sets the number of settlement intervals.

    event : dict
        A single event, as produced by `add_event`.

    baseline_kW : float, numpy.ndarray, or pyomo.environ.Var
        Precomputed baseline power in kW for this event's window (e.g.,
        from `calculate_event_baseline`), as a single value (broadcast
        across the window) or one value per interval, matching `power_kW`'s
        length. Always a constant with respect to the model's decision
        variables (a `numpy.ndarray`, or a `pyomo.environ.Var` fixed
        upstream), never itself a free decision variable.

    payment_function : list of dict or PaymentStructure
        Payment/penalty schedule to apply, as a list of region dicts (see
        `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`), or a
        `PaymentStructure` instance (e.g. `CapacityEnergyPayment`) to use a
        non-default payment structure.

    region_x1 : float, list of float, or None
        The `x1` value identifying which payment-function region to fix.
        Required when `power_kW` is a cvxpy type, since the applicable
        region cannot be known before solving. Optional when `power_kW` is
        a pyomo type: fixes that region when given (a list, under
        `settlement="interval"`, fixes each interval's region individually),
        or leaves the region choice to the solver when `None`. Ignored when
        `power_kW` is numpy/scalar, since the actual region is already
        fully determined.

    model : pyomo.environ.Model or pyomo.environ.Block
        The model or block to add pyomo components to.
        Only used in the pyomo case, so `None` by default.

    varstr : str
        Name prefix for pyomo variables/constraints created on `model`.

    Raises
    ------
    ValueError
        When `power_kW` is a cvxpy type and `region_x1` is `None`, or when
        `baseline_kW` carries more than one value and its count does not
        match `power_kW`'s.

    TypeError
        When `power_kW` is not a supported type.

    Returns
    -------
    tuple
        `(revenue, model)` for numpy `power_kW`, `(revenue_var, model)` for
        pyomo `power_kW`, or `(revenue_expr, constraints_list)` for cvxpy
        `power_kW`.
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
    """Calculate ex-post demand response revenue for a single event.

    Parameters
    ----------
    historical_power_kW : pandas.Series
        Power consumption in kW, indexed by `pandas.DatetimeIndex`,
        covering at least the event's window and its baseline days.

    event : dict
        A single event, as produced by `add_event`.

    baseline_params : dict or BaselineMethod
        Baseline parameters, as produced by `make_baseline_parameters`, or
        a `BaselineMethod` instance (e.g. `TopUsageDaysBaseline`) to use a
        non-default baselining strategy.

    payment_function : list of dict or PaymentStructure
        Payment/penalty schedule to apply, as a list of region dicts (see
        `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`), or a
        `PaymentStructure` instance (e.g. `CapacityEnergyPayment`) to use a
        non-default payment structure.

    Raises
    ------
    ValueError
        When `historical_power_kW` has no data in the event's window.

    Returns
    -------
    dict
        Per-event results with keys `EVENT_DATE`, `BASELINE_KW`, `ACTUAL_KW`,
        `REDUCTION_KW`, `DELIVERED_RATIO`, and `REVENUE` -- each an
        event-window mean -- plus the per-interval `BASELINE_PROFILE_KW`,
        `ACTUAL_PROFILE_KW`, `REDUCTION_PROFILE_KW`, and `INTERVAL_DATETIME`.
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
    """Normalizes realized power data into a pandas Series indexed by timestamp.

    A `pandas.Series`/`DataFrame` is returned unchanged (it is assumed to
    already carry a timestamp index); a bare `numpy.ndarray` is paired with
    `datetime_index` to build one.

    Parameters
    ----------
    power_kW : numpy.ndarray or pandas.Series
        Realized power consumption in kW. If already a `pandas.Series` (or
        `DataFrame`), `datetime_index` is ignored.

    datetime_index : pandas.DatetimeIndex or None
        Calendar timestamp for each entry of `power_kW`. Required, and used,
        only when `power_kW` is a bare `numpy.ndarray`.

    Raises
    ------
    ValueError
        When `power_kW` is a `numpy.ndarray` and `datetime_index` is `None`,
        or when their lengths differ.

    Returns
    -------
    pandas.Series
        `power_kW` unchanged if already pandas, otherwise
        `pandas.Series(power_kW, index=datetime_index)`.
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
    """Calculates ex-post demand response revenue with a row per event.

    Each event is sliced from `power_kW` and re-baselined independently, so
    results never mix time windows across events.

    Parameters
    ----------
    power_kW : pandas.Series or numpy.ndarray
        Realized power consumption in kW. A `pandas.Series` must be indexed
        by `pandas.DatetimeIndex`; a bare `numpy.ndarray` is paired with
        `datetime_index` via `_as_power_series`.

    events : list of dict or pandas.DataFrame
        Events collection, as produced by `add_event`.

    baseline_params : dict or BaselineMethod
        Baseline parameters, as produced by `make_baseline_parameters`, or
        a `BaselineMethod` instance (e.g. `TopUsageDaysBaseline`) to use a
        non-default baselining strategy.

    payment_function : list of dict or PaymentStructure
        Payment/penalty schedule to apply, as a list of region dicts (see
        `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`), or a
        `PaymentStructure` instance (e.g. `CapacityEnergyPayment`) to use a
        non-default payment structure.

    datetime_index : pandas.DatetimeIndex or None
        Calendar timestamp for each entry of `power_kW`. Only needed (and
        only used) when `power_kW` is a bare `numpy.ndarray` rather than a
        `pandas.Series`.

    Raises
    ------
    ValueError
        When `power_kW` is a `numpy.ndarray` and `datetime_index` is missing
        or mismatched in length (see `_as_power_series`), or when
        `historical_power_kW` has no data in some event's window.

    Returns
    -------
    tuple
        `(per_event_df, total_revenue)`, where `per_event_df` is a
        `pandas.DataFrame` with one row per event (see
        `calculate_event_revenue`) and `total_revenue` is the sum of the
        `REVENUE` column in USD.
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
    """Calculates demand response revenue across all events.

    Dispatches on `power_kW`: realized numpy/pandas data is settled ex-post,
    while a pyomo variable has its baseline and revenue components built onto
    `model`. Use `build_dr_revenue` to also net the result into the objective.

    Parameters
    ----------
    power_kW : pandas.Series, numpy.ndarray, or pyomo.environ.Var
        Power consumption in kW. A `pandas.Series`/`numpy.ndarray` is data and
        uses `calculate_itemized_dr_revenue`. A `pyomo.environ.Var` is a model
        decision variable and uses `_build_dr_revenue_components`. CVXPy vars
        are not currently implemented.

    events : list of dict or pandas.DataFrame
        Events collection, as produced by `add_event`. Used in both branches.

    baseline_params : dict or BaselineMethod
        Baseline parameters, as produced by `make_baseline_parameters`, or a
        `BaselineMethod` instance (e.g. `TopUsageDaysBaseline`). Used in both
        branches.

    payment_function : list of dict or PaymentStructure
        Payment/penalty schedule to apply, as a list of region dicts or a
        `PaymentStructure` instance (e.g. `CapacityEnergyPayment`). Used in
        both branches.

    historical_power_kW : pandas.Series or None
        Realized historical power consumption, indexed by
        `pandas.DatetimeIndex`. Only used (and required) in the pyomo
        branch, as the fallback baseline source -- see
        `calculate_event_baseline`.

    datetime_index : pandas.DatetimeIndex or None
        Calendar timestamp for each position in `power_kW`. In the
        numpy/pandas branch, only needed (and used) when `power_kW` is a
        bare `numpy.ndarray` (see `_as_power_series`). In the pyomo branch,
        required, and must align with `list(power_kW.index_set())`.

    model : pyomo.environ.Model, pyomo.environ.Block, or None
        The model to add components to, in the pyomo branch. Required (and
        only used) there; passed through unchanged in the numpy/pandas
        branch.

    region_x1s : dict or None
        Assumed payment-function region's `x1`, keyed by event date (any
        value `pandas.Timestamp` can parse). Only used in the pyomo branch,
        and optional there: an event missing from the dict (or the dict
        being `None` entirely) leaves that event's region choice to the
        solver, via `PaymentStructure.build_expression`'s all-regions
        formulation; an event with an entry fixes that region instead.

    varstr_prefix : str
        Prefix for the per-event `varstr` passed to `build_payment_expression`
        in the pyomo branch. Must be unique per call on a given `model`.
        Unused in the numpy/pandas branch.

    Raises
    ------
    NotImplementedError
        When `power_kW` is a `cvxpy` type, which this function does not
        support.

    ValueError
        In the pyomo branch, when `datetime_index`, `historical_power_kW`,
        or `model` is missing, or when an event's window has no matching
        positions in `datetime_index` (delegated from
        `_build_dr_revenue_components`).

    TypeError
        When `power_kW` is not a `pandas.Series`, `numpy.ndarray`, or
        `pyomo.environ.Var`.

    Returns
    -------
    tuple
        `(total_revenue, model)` in both branches. In the numpy/pandas
        branch, `total_revenue` is a `float` and `model` is passed through
        unchanged (typically `None`). In the pyomo branch, `total_revenue`
        is a pyomo expression built from newly-added revenue variables, and
        `model` has those components (and each event's baseline) added to it.
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
    """Builds each event's baseline and revenue components onto a pyomo model.

    Slices `power_kW` to each event's window internally (via `datetime_index`)
    and computes each event's baseline internally (via `historical_power_kW`
    and `calculate_event_baseline`).

    Processes events in `EVENT_DATE` order (like `calculate_dr_revenue`).

    Parameters
    ----------
    power_kW : pyomo.environ.Var
        Time-indexed decision variable for actual power consumption
        over the optimization horizon.

    datetime_index : pandas.DatetimeIndex
        Calendar timestamp for each position in `power_kW`'s index set.
    events : list of dict or pandas.DataFrame
        Events collection, as produced by `add_event`.

    historical_power_kW : pandas.Series
        Historical power consumption, indexed by `pandas.DatetimeIndex`.

    baseline_params : dict or BaselineMethod
        Baseline parameters, as produced by `make_baseline_parameters`, or
        a `BaselineMethod` instance (e.g. `TopUsageDaysBaseline`) to use a
        non-default baselining strategy.

    model : pyomo.environ.Model or pyomo.environ.Block
        The model to add components to.

    payment_function : list of dict or PaymentStructure
        Payment/penalty schedule shared by all events, as a list of region
        dicts or a `PaymentStructure` instance (e.g.
        `CapacityEnergyPayment`).

    region_x1s : dict or None
        Assumed payment-function region's `x1`, keyed by event date (any
        value `pandas.Timestamp` can parse), e.g.
        `{"2024-01-08": 0.6, "2024-01-15": 0.75}`. Optional: an event
        missing from the dict (or `None` entirely) leaves that event's
        region choice to the solver instead of fixing it -- see
        `PaymentStructure.build_expression`.

    varstr_prefix : str
        Prefix for the per-event `varstr` passed to `build_payment_expression`
        (combined with the event's position in date order). Must be unique
        per call on a given `model`.

    Raises
    ------
    ValueError
        When `datetime_index`, `historical_power_kW`, or `model` is `None`,
        or when an event's window has no matching positions in
        `datetime_index`.

    Returns
    -------
    tuple
        `(total_revenue, model)`, where `total_revenue` is a pyomo
        expression summing each event's revenue variable.
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
        power_terms = [power_kW[idx] for idx in matched_indices]
        baseline_values = _baseline_terms(baseline_kW, len(power_terms))
        reduction_terms = [b - p for b, p in zip(baseline_values, power_terms)]

        revenue_var, model = payment_structure.build_expression(
            event,
            reduction_terms,
            region_x1=region_x1_by_date.get(event[EVENT_DATE]),
            model=model,
            varstr=f"{varstr_prefix}_{i}",
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
    """Wrapper for `calculate_dr_revenue` that adds DR revenue to the objective.

    Parameters
    ----------
    power_kW : pyomo.environ.Var
        Time-indexed decision variable for actual power consumption
        over the optimization horizon.

    datetime_index : pandas.DatetimeIndex
        Calendar timestamp for each position in `power_kW`'s index set.

    events : list of dict or pandas.DataFrame
        Events collection, as produced by `add_event`.

    historical_power_kW : pandas.Series
        Realized historical power consumption, indexed by
        `pandas.DatetimeIndex`.

    baseline_params : dict or BaselineMethod
        Baseline parameters, as produced by `make_baseline_parameters`, or
        a `BaselineMethod` instance (e.g. `TopUsageDaysBaseline`) to use a
        non-default baselining strategy.

    model : pyomo.environ.Model or pyomo.environ.Block
        The model to add components to and whose objective is updated.

    payment_function : list of dict or PaymentStructure
        Payment/penalty schedule shared by all events, as a list of region
        dicts or a `PaymentStructure` instance (e.g.
        `CapacityEnergyPayment`).

    region_x1s : dict or None
        Assumed payment-function region's `x1`, keyed by event date (any
        value `pandas.Timestamp` can parse), e.g.
        `{"2024-01-08": 0.6, "2024-01-15": 0.75}`. Optional: an event
        missing from the dict (or `None` entirely) leaves that event's
        region choice to the solver instead of fixing it -- see
        `PaymentStructure.build_expression`.

    varstr_prefix : str
        Prefix for the per-event `varstr` passed to `build_payment_expression`
        (combined with the event's position in date order). Must be unique
        per call on a given `model`.

    Raises
    ------
    ValueError
        When an event's window has no matching positions in `datetime_index`
        (delegated from `_build_dr_revenue_components`).

    Returns
    -------
    tuple
        `(total_revenue, model)`.
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

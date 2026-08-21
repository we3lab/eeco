"""Calculate demand response revenue from electricity consumption data.

A demand response program is modeled along two independent axes, each with
a base class that captures the common case and subclasses for the edge
cases found in real US tariffs:

- **How the counterfactual is measured** -- `BaselineMethod`, defaulting to
  an average of recent similar days with a day-of adjustment. See
  `TopUsageDaysBaseline`, `FixedLevelBaseline`, and
  `UnilateralInterruptionBaseline`.
- **How that measurement is paid for** -- `PaymentStructure`, defaulting to
  a piecewise-linear capacity payment keyed on the delivered fraction of
  the bid. See `CapacityEnergyPayment` and `MarketIndexedPayment`.

The two compose freely: any baseline method pairs with any payment
structure. Every public function accepts either a plain configuration (a
dict from `make_baseline_parameters`, a list of payment region dicts) or an
instance of these classes, so the defaults need no ceremony and the edge
cases need no new arguments.

Typical use:

- Ex-post settlement of realized data -- `calculate_dr_revenue`.
- Building DR revenue into a pyomo optimization -- `build_dr_revenue`.
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


class BaselineMethod:
    """Foundation baselining strategy: the average of the N most recent
    valid similar days, optionally scaled by a day-of adjustment factor.

    A "baseline" is the counterfactual power the site *would* have drawn
    during an event window had the event not been called. Revenue is paid
    on the gap between that baseline and metered power, so the baselining
    rule is as economically significant as the payment rate itself.

    This class implements the most common rule in US demand response
    programs, and matches the defaults of PG&E's Capacity Bidding Program
    (10 similar weekdays, 3-hour day-of adjustment). `compute` runs four
    steps in order, each of which is an overridable hook:

    1. `_filter_days`  -- drop ineligible candidate days (weekends/holidays)
    2. `_order_days`   -- rank the survivors by preference (default: recency)
    3. `_baseline_day_terms` (module-level) -- average each kept day's
       event window, transparently sourcing from either historical data or
       an optimization decision variable
    4. `_adjustment_factor` -- derive a same-day correction to scale by

    Extending this class
    --------------------
    Override the smallest hook that captures the difference, not `compute`:

    - Different *ranking* of candidate days (e.g. by usage rather than
      recency)? Override `_order_days` -- see `TopUsageDaysBaseline`.
    - Different *eligibility* rules (e.g. also exclude prior event days)?
      Override `_filter_days`.
    - Different *same-day correction*? Override `_adjustment_factor`.
    - A baseline that isn't an average of historical days at all (e.g. a
      contracted firm level)? Override `compute` -- see `FixedLevelBaseline`
      and `UnilateralInterruptionBaseline`.

    Subclasses must preserve `compute`'s return-value contract (a bare
    value when `model` is `None`, a `(value, model)` tuple otherwise), since
    callers such as `calculate_event_baseline` and `build_dr_revenue`
    unpack it positionally.

    Parameters
    ----------
    n_baseline_days : int
        Number of valid baseline days to average over.

    adjustment_hours : int or None
        Number of hours immediately before the event used to compute the
        day-of adjustment factor. If `None`, no adjustment is applied.

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
        rebuilding the model -- useful for sensitivity analysis on a
        baseline assumption that materially moves revenue.

        The `Var` is fixed on creation, which is what keeps it an input
        rather than a decision: pyomo treats a fixed variable as a constant
        coefficient, so `baseline * factor` stays linear even when the
        baseline is itself a decision-variable expression. Unfixing it
        would both make that product bilinear and let the solver choose the
        factor that maximizes revenue, so callers should retune it with
        `.fix(...)` rather than `.unfix()`.

        `False` by default, which folds the factor in as a constant and
        keeps `compute`'s return value a plain `float` whenever every
        baseline day is historical.

    Raises
    ------
    ValueError
        When `n_baseline_days` is not positive.
    """

    def __init__(
        self,
        n_baseline_days=10,
        adjustment_hours=3,
        adjustment_clip=(0.8, 1.2),
        exclude_weekends=True,
        exclude_holidays=True,
        holiday_dates=None,
        adjustment_in_model=False,
    ):
        if n_baseline_days <= 0:
            raise ValueError("n_baseline_days must be positive")
        self.n_baseline_days = n_baseline_days
        self.adjustment_hours = adjustment_hours
        self.adjustment_clip = adjustment_clip
        self.exclude_weekends = exclude_weekends
        self.exclude_holidays = exclude_holidays
        self.holiday_dates = list(holiday_dates) if holiday_dates else []
        self.adjustment_in_model = adjustment_in_model

    def _filter_days(self, candidate_days):
        """Drop candidate days that are ineligible for use in a baseline.

        Hook method. The foundation implementation applies the
        `exclude_weekends`/`exclude_holidays` configuration; override to add
        program-specific eligibility rules (e.g. also excluding days on
        which a prior event was called).

        Parameters
        ----------
        candidate_days : list of pandas.Timestamp
            Days proposed for this event's baseline.

        Returns
        -------
        list of pandas.Timestamp
            The eligible subset, in the order given.
        """
        if self.exclude_weekends:
            candidate_days = [d for d in candidate_days if d.weekday() < 5]
        if self.exclude_holidays:
            holidays = {pd.Timestamp(d) for d in self.holiday_dates}
            candidate_days = [d for d in candidate_days if d not in holidays]
        return candidate_days

    def _order_days(self, candidate_days, historical_power_kW, event):
        """Rank eligible days by preference, most-preferred first.

        Hook method. `select_days` keeps the first `n_baseline_days` of
        whatever this returns, so this method alone decides *which* days
        survive when more are available than are needed. The foundation
        implementation prefers the most recent days; override this rather
        than `select_days` to change the rule (see `TopUsageDaysBaseline`).

        Parameters
        ----------
        candidate_days : list of pandas.Timestamp
            Eligible days, as returned by `_filter_days`.

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
            All of `candidate_days`, reordered most-preferred first.
        """
        return sorted(candidate_days, reverse=True)

    def select_days(self, candidate_days, historical_power_kW, event):
        """Filter, rank, and truncate candidate days to the ones actually
        used in the baseline average.

        Composes the `_filter_days` and `_order_days` hooks, then keeps the
        top `n_baseline_days`. Prefer overriding one of those two hooks
        instead of this method.

        Parameters
        ----------
        candidate_days : list of pandas.Timestamp
            Days proposed for this event's baseline, before filtering.

        historical_power_kW : pandas.Series
            Historical realized power consumption in kW, indexed by
            `pandas.DatetimeIndex`. Passed through to `_order_days`.

        event : dict
            A single event, as produced by `add_event`. Passed through to
            `_order_days`.

        Raises
        ------
        ValueError
            When zero eligible days remain after filtering.

        Warnings
            When fewer eligible days remain than `n_baseline_days`.

        Returns
        -------
        list of pandas.Timestamp
            At most `n_baseline_days` days, most-preferred first.
        """
        filtered = self._filter_days(candidate_days)
        if len(filtered) == 0:
            raise ValueError("No valid baseline days remain after filtering")
        if len(filtered) < self.n_baseline_days:
            warnings.warn(
                f"Only {len(filtered)} valid baseline days available, "
                f"fewer than the requested {self.n_baseline_days}",
                UserWarning,
            )
        ordered = self._order_days(filtered, historical_power_kW, event)
        return ordered[: self.n_baseline_days]

    def _adjustment_factor(self, valid_days, historical_power_kW, event):
        """Compute the day-of adjustment factor for this event.

        The multi-day average is a lagging estimate: it cannot know whether
        the event day itself is running hot or cold. This factor corrects
        for that by comparing the event day's own consumption over the
        `adjustment_hours` immediately preceding the event against the same
        pre-event window averaged across the baseline days. A factor of
        1.1, for instance, means the site was drawing 10% above its usual
        pre-event level that morning, so the baseline is scaled up 10% to
        match. The result is clipped to `adjustment_clip` so one anomalous
        morning cannot distort the baseline without bound.

        Hook method. Override to derive the factor differently; `compute`
        applies whatever scalar this returns multiplicatively. A subclass
        needing a non-multiplicative correction should override `compute`
        instead.

        This is always computed from `historical_power_kW`, never from a
        model decision variable, even when the adjustment window falls
        inside an optimization horizon: the factor is a ratio of two power
        averages, and both dividing and clipping decision-variable
        expressions are nonlinear operations with no linear pyomo
        representation.

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
            When the denominator is near zero, in which case the adjustment
            is skipped by returning a factor of `1.0`.

        Returns
        -------
        float or None
            The clipped multiplicative factor, or `None` when
            `adjustment_hours` is `None` (no adjustment configured).
        """
        if self.adjustment_hours is None:
            return None
        adjustment_hours = self.adjustment_hours
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
        """Calculate the mean baseline power for a single event's window.

        This is the implementation behind the module-level
        `calculate_event_baseline`, whose docstring carries the full
        parameter and return-value contract. In brief: pass no `model` for
        an ex-post calculation returning a plain `float`; pass a `model`
        (plus `model_power_kW`, `model_datetime_index`, and `varstr`) to
        have any baseline day falling entirely inside the optimization
        horizon computed from the decision variable instead of history,
        returning `(baseline, model)`.

        When `adjustment_in_model` is set, the day-of adjustment factor is
        additionally registered on `model` as a fixed `Var` and applied
        symbolically, so the baseline is always materialized as a `Var`
        with a defining constraint the caller can retune post-build.
        """
        if model is not None and any(
            a is None for a in (model_power_kW, model_datetime_index, varstr)
        ):
            raise ValueError(
                "model_power_kW, model_datetime_index, and varstr are all required "
                "when model is given"
            )

        candidate_days = [pd.Timestamp(d) for d in event[BASELINE_DAYS]]
        valid_days = self.select_days(candidate_days, historical_power_kW, event)

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

        baseline_kW = sum(weight * value for weight, value in all_terms) / len(
            valid_days
        )
        if not any_dynamic:
            baseline_kW = float(baseline_kW)

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
            baseline_kW = baseline_kW * factor_var
        elif factor is not None:
            baseline_kW = baseline_kW * factor

        if model is None:
            return baseline_kW
        # A factor-scaled baseline is a symbolic expression, so it needs a Var
        # to stand for it even when every baseline day was historical.
        if not any_dynamic and not factor_in_model:
            return baseline_kW, model

        model.add_component(varstr, pyo.Var())
        baseline_var = model.find_component(varstr)

        def baseline_rule(m):
            return baseline_var == baseline_kW

        model.add_component(varstr + "_constraint", pyo.Constraint(rule=baseline_rule))
        return baseline_var, model


class TopUsageDaysBaseline(BaselineMethod):
    """Ranks candidate days by consumption (highest first) rather than by
    recency, so the baseline averages the site's busiest similar days.

    This raises the baseline relative to the foundation method, which
    generally raises the measured reduction and therefore the payment --
    which is precisely why programs that use it specify the rule tightly.
    Matches LADWP ("average of the three highest energy usage days over the
    past 10 weekdays"), Appalachian Power TN ("highest 4 of the 5 most
    recent similar weekdays"), and Delmarva DE ("three days with the
    highest energy usage during the past 30 days").

    Takes the same constructor parameters as `BaselineMethod`; only the
    day-ranking hook differs, so `n_baseline_days` still controls how many
    of the top days are kept.
    """

    def _order_days(self, candidate_days, historical_power_kW, event):
        """Rank eligible days by mean power over the event window,
        highest first.

        A day with no data in the event window sorts last (rather than
        raising) so that a single gap in the historical series degrades the
        ranking instead of failing the whole calculation; if such a day is
        still selected, `_baseline_day_terms` raises on it downstream.

        Parameters
        ----------
        candidate_days : list of pandas.Timestamp
            Eligible days, as returned by `_filter_days`.

        historical_power_kW : pandas.Series
            Historical realized power consumption in kW, indexed by
            `pandas.DatetimeIndex`. Supplies the consumption being ranked.

        event : dict
            A single event, as produced by `add_event`. Supplies the window
            (`EVENT_START_HOUR`, `EVENT_DURATION`) each day is scored over.

        Returns
        -------
        list of pandas.Timestamp
            All of `candidate_days`, highest mean event-window power first.
        """

        def day_mean(day):
            mask = _event_window_mask(
                historical_power_kW.index,
                day,
                event[EVENT_START_HOUR],
                event[EVENT_DURATION],
            )
            day_slice = historical_power_kW.loc[mask]
            return day_slice.mean() if not day_slice.empty else -np.inf

        return sorted(candidate_days, key=day_mean, reverse=True)


class FixedLevelBaseline(BaselineMethod):
    """Baseline is a constant contracted "firm" demand level agreed with
    the utility up front, never inferred from history.

    Under these tariffs the customer commits to holding load at or below a
    nominated level during events, and performance is measured against that
    number directly -- so there is no similar-day average, no day-of
    adjustment, and no dependence on `historical_power_kW` at all. Matches
    VT Curtailable Load Rider, WI Commercial and Industrial Interruptible
    Rider, IA Interruptible Service Option, and SC Large Load Curtailable
    Rider.

    Deliberately does not call `super().__init__()`: none of the
    day-selection or adjustment configuration on `BaselineMethod` is
    meaningful here, so those attributes are intentionally absent. Because
    `compute` is fully overridden, no inherited method reads them.

    Parameters
    ----------
    firm_level_kW : float
        The contracted firm demand level in kW, used as the baseline for
        every event.

    Raises
    ------
    ValueError
        When `firm_level_kW` is negative.
    """

    def __init__(self, firm_level_kW):
        if firm_level_kW < 0:
            raise ValueError("firm_level_kW must be non-negative")
        self.firm_level_kW = firm_level_kW

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
        """Return the contracted firm level as the baseline.

        Adds nothing to `model`: the baseline is a constant, so there is no
        expression for a `Var` to stand in for. Every parameter other than
        `model` is accepted for interface compatibility with
        `BaselineMethod.compute` and is unused.

        Parameters
        ----------
        historical_power_kW : pandas.Series
            Ignored -- this baseline is not derived from history.

        event : dict
            Ignored -- the same firm level applies to every event.

        model : pyomo.environ.Model or pyomo.environ.Block or None
            Only consulted to decide the return shape, matching
            `BaselineMethod.compute`'s contract.

        model_power_kW : pyomo.environ.Var or None
            Ignored.

        model_datetime_index : pandas.DatetimeIndex or None
            Ignored.

        varstr : str or None
            Ignored -- no components are created.

        Returns
        -------
        float or tuple
            `firm_level_kW` when `model` is `None`, otherwise
            `(firm_level_kW, model)`.
        """
        if model is None:
            return self.firm_level_kW
        return self.firm_level_kW, model


class UnilateralInterruptionBaseline(BaselineMethod):
    """The utility interrupts service itself, holding load at a fixed level
    for the duration of an event.

    Modeled as a hard constraint rather than a revenue opportunity, because
    the operator has no decision to make: the utility physically interrupts
    the load, so consumption during the window is imposed, not chosen. The
    Alaska Chugach Electric interruptible tariff is the motivating example
    -- its own documentation notes the power company interrupts the
    operator's service itself and the operator does not choose how much to
    reduce load by. Compensation there takes the form of an eliminated
    demand charge, not a per-event payment.

    Consequently there is no ex-post evaluation path: with no counterfactual
    and no operator choice, "what revenue did this event earn" is not a
    meaningful question to ask of realized data, so `compute` raises
    `NotImplementedError` when called without a `model`. Inside a model, it
    caps `model_power_kW` at `interruption_level_kW` across the event
    window so the rest of the optimization plans around the outage.

    The cap is an upper bound rather than an equality: the utility's
    interruption puts a ceiling on what the facility *can* draw, and going
    below it stays a legitimate operating choice. Forcing equality would
    also make the window infeasible for any site whose other constraints
    (a minimum charge rate, a process that cannot idle at exactly that
    level) prevent it from landing precisely on the interruption level.

    Deliberately does not call `super().__init__()`, for the same reason as
    `FixedLevelBaseline`.

    Parameters
    ----------
    interruption_level_kW : float
        Power level in kW the load is held at during an event. Defaults to
        `0.0` (a full interruption).
    """

    def __init__(self, interruption_level_kW=0.0):
        self.interruption_level_kW = interruption_level_kW

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
        """Constrain modeled power to the interruption level over the
        event window.

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
            `(interruption_level_kW, model)`. The level is returned in the
            baseline's position so downstream payment logic keeps a
            consistent interface, though for this program the reduction it
            implies is not paid per-event.
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

        def interruption_rule(m, idx):
            return model_power_kW[idx] <= self.interruption_level_kW

        model.add_component(
            varstr + "_interruption_constraint",
            pyo.Constraint(matched_indices, rule=interruption_rule),
        )
        return self.interruption_level_kW, model


def _coerce_baseline_method(baseline_params):
    """Normalize a baseline configuration into a `BaselineMethod`.

    Lets every public entry point accept either representation: the
    original dict from `make_baseline_parameters` (kept working for
    backward compatibility, and wrapped here into a foundation
    `BaselineMethod`) or a `BaselineMethod` instance, which is how callers
    opt into a subclass such as `TopUsageDaysBaseline`.

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
        adjustment_hours=baseline_params[ADJUSTMENT_HOURS],
        adjustment_clip=baseline_params[ADJUSTMENT_CLIP],
        exclude_weekends=baseline_params[EXCLUDE_WEEKENDS],
        exclude_holidays=baseline_params[EXCLUDE_HOLIDAYS],
        holiday_dates=baseline_params[HOLIDAY_DATES],
    )


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


class PaymentStructure:
    """Foundation payment structure: a piecewise-linear capacity payment
    keyed on how much of the bid the site actually delivered.

    Revenue is expressed as a function of the *delivered ratio* --
    `reduction_kW / bid_capacity_kW`, i.e. the fraction of the nominated
    capacity the site actually shed. The `regions` list maps that ratio to
    a payment ratio through consecutive linear segments, and revenue is
    `payment_ratio * capacity_price * bid_capacity_kW`. Casting payment in
    terms of ratios keeps one schedule reusable across events with
    different bid sizes and prices.

    Each region is a dict with keys `REGION_X1`/`REGION_X2` (the half-open
    delivered-ratio interval `[x1, x2)` it covers) and
    `REGION_Y1`/`REGION_Y2` (the payment ratio at each end, interpolated
    linearly between). Regions express real tariff structure: a negative
    `y` encodes an underdelivery penalty, and a final region with
    `x2 = inf` encodes a cap where overdelivery earns no more.

    Extending this class
    --------------------
    Override `evaluate` and `build_expression` as a pair -- `evaluate` is
    the realized-value path and `build_expression` the optimization path,
    and they must agree or an optimized plan will not reconcile with its
    ex-post settlement. Call `super()` for the capacity term and add to it,
    as `CapacityEnergyPayment` does, or transform the event before
    delegating, as `MarketIndexedPayment` does.

    Both methods take the whole `event` dict rather than loose scalars
    precisely so subclasses can reach other event fields (duration, date)
    without changing the signature.

    Parameters
    ----------
    regions : list of dict
        Payment schedule, each dict having keys `REGION_X1`, `REGION_X2`,
        `REGION_Y1`, and `REGION_Y2`. Expected to cover the delivered
        ratios that can occur; `find_region` raises if one is uncovered.
    """

    def __init__(self, regions):
        self.regions = regions  # list of dicts with REGION_X1/X2/Y1/Y2 keys

    def find_region(self, delivered_ratio=None, region_x1=None):
        """Look up the applicable payment region.

        Exactly one of `delivered_ratio`/`region_x1` should be given.
        Looking up by `delivered_ratio` finds the region whose `[x1, x2)`
        interval contains it, and is used ex-post once the realized ratio
        is known. Looking up by `region_x1` finds the region whose `x1`
        matches, and is used when building an optimization expression,
        where the region must be assumed before solving; the match uses
        `numpy.isclose` because such a value often carries floating-point
        noise from an upstream solver.

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
        region = next((r for r in self.regions if predicate(r)), None)
        if region is None:
            raise ValueError(error_msg)
        return region

    def evaluate(self, event, reduction_kW):
        """Calculate realized revenue for a known reduction.

        Determines the applicable region from the delivered ratio and
        interpolates the payment ratio linearly between that region's
        `(x1, y1)` and `(x2, y2)` endpoints. In a capped region
        (`x2 = inf`) the payment ratio is flat at `y1`.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`. Supplies
            `BID_CAPACITY_KW` and `CAPACITY_PRICE`.

        reduction_kW : float or numpy.ndarray
            Realized load reduction (baseline minus actual power) in kW.

        Raises
        ------
        ValueError
            When the event's `BID_CAPACITY_KW` is not positive, or no
            region covers the resulting delivered ratio.

        Returns
        -------
        float
            Revenue (positive) or penalty (negative) in USD.
        """
        bid_capacity_kW = event[BID_CAPACITY_KW]
        capacity_price = event[CAPACITY_PRICE]
        if bid_capacity_kW <= 0:
            raise ValueError("bid_capacity_kW must be positive")
        delivered_ratio = reduction_kW / bid_capacity_kW
        region = self.find_region(delivered_ratio=delivered_ratio)
        x1, x2, y1, y2 = (
            region[k] for k in (REGION_X1, REGION_X2, REGION_Y1, REGION_Y2)
        )
        if np.isinf(x2):
            payment_ratio = y1
        else:
            payment_ratio = y1 + (y2 - y1) * (delivered_ratio - x1) / (x2 - x1)
        return payment_ratio * capacity_price * bid_capacity_kW

    def build_expression(
        self, event, reduction_kW, region_x1=None, model=None, varstr=""
    ):
        """Build the revenue expression for a single, specified region.

        The optimization counterpart to `evaluate`. Because the delivered
        ratio is unknown before solving, the caller must nominate which
        region applies via `region_x1`; this method then emits that
        region's linear payment expression together with bound constraints
        holding the solution inside the region, so the assumption cannot be
        silently violated. Sweeping `region_x1` across regions and keeping
        the best solution is how a caller explores the full schedule.

        Dispatches on the type of `reduction_kW`:

        - `numpy.ndarray` or Python number: the region is already
          determined, so this delegates to `evaluate` and `region_x1` is
          ignored.
        - `cvxpy.Expression`/`cvxpy.Variable`: returns the expression plus
          a list of 1-2 region-bound constraints for the caller to add to
          their own `cvxpy.Problem`.
        - `pyomo.environ.Var`/expression: adds a `varstr + "_revenue"`
          `Var`, its defining constraint, and 1-2 region-bound constraints
          to `model` (which may be a `pyomo.environ.Block`).

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`. Supplies
            `BID_CAPACITY_KW` and `CAPACITY_PRICE`.

        reduction_kW : numpy.ndarray, float, cvxpy.Expression, or pyomo.environ.Var
            Load reduction, as a realized value or a decision-variable
            expression.

        region_x1 : float or None
            The `x1` value identifying which region to build. Required for
            the cvxpy and pyomo cases.

        model : pyomo.environ.Model or pyomo.environ.Block or None
            The model or block to add pyomo components to. Only used in the
            pyomo case.

        varstr : str
            Name prefix for pyomo components created on `model`. Must be
            unique per call on a given `model`, since reusing one raises a
            pyomo "component already exists" error.

        Raises
        ------
        ValueError
            When no region matches `region_x1`.

        TypeError
            When `reduction_kW` is not a supported type.

        Returns
        -------
        tuple
            `(revenue, model)` for numpy/scalar, `(revenue_var, model)` for
            pyomo, or `(revenue_expr, constraints_list)` for cvxpy.
        """
        bid_capacity_kW = event[BID_CAPACITY_KW]
        capacity_price = event[CAPACITY_PRICE]

        if ut.check_indexed_np_array(reduction_kW) or ut.check_nonindexed_python_type(
            reduction_kW
        ):
            return self.evaluate(event, reduction_kW), model

        region = self.find_region(region_x1=region_x1)
        x1, x2, y1, y2 = (
            region[k] for k in (REGION_X1, REGION_X2, REGION_Y1, REGION_Y2)
        )
        slope_ratio = 0.0 if np.isinf(x2) else (y2 - y1) / (x2 - x1)
        slope = capacity_price * slope_ratio
        intercept = capacity_price * bid_capacity_kW * (y1 - slope_ratio * x1)

        if ut.check_cvx_type(reduction_kW):
            revenue_expr = slope * reduction_kW + intercept
            constraints = [reduction_kW >= x1 * bid_capacity_kW]
            if not np.isinf(x2):
                constraints.append(reduction_kW <= x2 * bid_capacity_kW)
            return revenue_expr, constraints
        elif ut.check_indexed_pyomo_type(
            reduction_kW
        ) or ut.check_nonindexed_pyomo_type(reduction_kW):
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
                varstr + "_lower_bound_constraint",
                pyo.Constraint(rule=lower_bound_rule),
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


class CapacityEnergyPayment(PaymentStructure):
    """Two-part payment: the foundation capacity payment plus a flat $/kWh
    payment on the energy actually curtailed.

    The most common structure among the surveyed US programs. The capacity
    term pays for *availability* (scaled by the bid), while the energy term
    pays for *delivery* (scaled by kWh shed, i.e. `reduction_kW` times the
    event's duration in hours).

    Parameters
    ----------
    regions : list of dict
        Capacity payment schedule, as for `PaymentStructure`.

    energy_price : float
        Energy payment rate in $/kWh applied to the curtailed energy.
    """

    def __init__(self, regions, energy_price):
        super().__init__(regions)
        self.energy_price = energy_price

    def evaluate(self, event, reduction_kW):
        """Realized capacity payment plus the energy payment.

        Parameters
        ----------
        event : dict
            A single event, as produced by `add_event`. Supplies
            `EVENT_DURATION` in addition to the fields the capacity term
            needs -- note this is why a full event is required here, and
            why the scalar-argument `evaluate_payment_function` wrapper
            cannot be used with this class.

        reduction_kW : float or numpy.ndarray
            Realized load reduction in kW.

        Returns
        -------
        float
            Combined revenue in USD.
        """
        capacity_payment = super().evaluate(event, reduction_kW)
        energy_payment = self.energy_price * reduction_kW * event[EVENT_DURATION]
        return capacity_payment + energy_payment

    def build_expression(
        self, event, reduction_kW, region_x1=None, model=None, varstr=""
    ):
        """Build the combined capacity-plus-energy revenue expression.

        Takes the same parameters as `PaymentStructure.build_expression`.
        For pyomo, the capacity and energy terms each get their own `Var`
        and defining constraint (named `varstr + "_revenue"` and
        `varstr + "_energy_revenue"`), and a third `varstr +
        "_total_revenue"` nets them, so the objective sees one combined
        revenue variable per event while each component stays separately
        inspectable after solving.

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

        energy_term = self.energy_price * reduction_kW * event[EVENT_DURATION]

        if ut.check_cvx_type(reduction_kW):
            capacity_expr, constraints = super().build_expression(
                event, reduction_kW, region_x1=region_x1, model=model, varstr=varstr
            )
            return capacity_expr + energy_term, constraints

        # pyomo case: capacity_var and energy_var each get their own defining
        # Var + Constraint, then a third total_var nets them together, so the
        # model's objective sees one combined revenue variable per event.
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
    """Resolves the capacity price from a lookup at calculation time rather
    than reading a static value off the event.

    Under market-referenced programs the price is not known when the event
    is defined -- it comes from a clearing price or index (CAISO RTD/FMM
    real-time dispatch, PJM Capacity Performance auction results, or an
    LMP-referenced credit such as Appalachian Power's). Rather than
    duplicating the payment maths, this substitutes the resolved price into
    a copy of the event and delegates to the foundation implementation.

    The event is copied, never mutated, so the caller's events collection
    is unchanged and re-running with a different `price_lookup` is safe.

    Parameters
    ----------
    regions : list of dict
        Payment schedule, as for `PaymentStructure`.

    price_lookup : callable
        Called as `price_lookup(event)` and must return the capacity price
        in $/kW to use for that event. Typically closes over a price series
        and keys off `event[EVENT_DATE]`.
    """

    def __init__(self, regions, price_lookup):
        super().__init__(regions)
        self.price_lookup = price_lookup  # callable: price_lookup(event) -> float

    def _resolve_event(self, event):
        """Return a copy of `event` with `CAPACITY_PRICE` replaced by the
        looked-up market price.

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

        The price is resolved once, at build time, and enters the model as
        a constant coefficient -- so a model built this way is tied to the
        prices in force when it was built.
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
    public entry point accept either the original list of region dicts
    (kept working for backward compatibility, and wrapped here into a
    foundation `PaymentStructure`) or a `PaymentStructure` instance, which
    is how callers opt into a subclass such as `CapacityEnergyPayment`.

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


def _find_payment_region(payment_function, delivered_ratio=None, region_x1=None):
    """Find a region in `payment_function`, by interval or by `x1`.

    Exactly one of `delivered_ratio`/`region_x1` must be given: `delivered_ratio`
    looks up the region whose `[x1, x2)` interval contains it; `region_x1` looks
    up the region whose `x1` matches it (via `numpy.isclose`, since callers such
    as `build_payment_expression` may pass a value with floating-point noise
    from an upstream solver).

    Parameters
    ----------
    payment_function : list of dict or PaymentStructure
        Each dict has keys `REGION_X1`, `REGION_X2`, `REGION_Y1`,
        `REGION_Y2`. A `PaymentStructure` instance is also accepted, in
        which case its own regions are searched.

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
    return _coerce_payment_structure(payment_function).find_region(
        delivered_ratio=delivered_ratio, region_x1=region_x1
    )


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
    payment_function : list of dict or PaymentStructure
        Each dict has keys `REGION_X1`, `REGION_X2`, `REGION_Y1`,
        `REGION_Y2`. A `PaymentStructure` instance is also accepted, but
        note that this function passes only `bid_capacity_kW` and
        `capacity_price` through -- a subclass needing other event fields
        (as `CapacityEnergyPayment` needs `EVENT_DURATION`) must be used
        via `build_event_revenue`, `calculate_dr_revenue`, or
        `build_dr_revenue`, which have the full event to hand.

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
    event = {BID_CAPACITY_KW: bid_capacity_kW, CAPACITY_PRICE: capacity_price}
    return _coerce_payment_structure(payment_function).evaluate(event, reduction_kW)


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
    payment_function : list of dict or PaymentStructure
        Each dict has keys `REGION_X1`, `REGION_X2`, `REGION_Y1`,
        `REGION_Y2`. A `PaymentStructure` instance is also accepted, but
        note that this function passes only `bid_capacity_kW` and
        `capacity_price` through -- a subclass needing other event fields
        (as `CapacityEnergyPayment` needs `EVENT_DURATION`) must be used
        via `build_event_revenue`, `calculate_dr_revenue`, or
        `build_dr_revenue`, which have the full event to hand.

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
    event = {BID_CAPACITY_KW: bid_capacity_kW, CAPACITY_PRICE: capacity_price}
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
        When fewer valid baseline days remain than `N_BASELINE_DAYS`.

    Returns
    -------
    float or tuple
        `float`: mean baseline power in kW, when `model` is `None`.
        `(baseline_kW, model)`: when `model` is given, where `baseline_kW`
        is a `float` if every baseline day stayed historical, or a
        `pyomo.environ.Var` if at least one was dynamic.
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

    payment_function : list of dict or PaymentStructure
        Payment/penalty schedule to apply, as a list of region dicts (see
        `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`), or a
        `PaymentStructure` instance (e.g. `CapacityEnergyPayment`) to use a
        non-default payment structure.

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
    payment_structure = _coerce_payment_structure(payment_function)

    if ut.check_indexed_np_array(power_kW) or ut.check_nonindexed_python_type(power_kW):
        reduction_kW = baseline_kW - np.mean(power_kW)
        return payment_structure.evaluate(event, reduction_kW), model
    elif ut.check_cvx_type(power_kW):
        if region_x1 is None:
            raise ValueError("region_x1 must be specified for cvxpy power_kW")
        mean_power = cp.sum(power_kW) / power_kW.size
        reduction_kW = baseline_kW - mean_power
        return payment_structure.build_expression(
            event, reduction_kW, region_x1=region_x1, model=model, varstr=varstr
        )
    elif ut.check_indexed_pyomo_type(power_kW) or ut.check_nonindexed_pyomo_type(
        power_kW
    ):
        if region_x1 is None:
            raise ValueError("region_x1 must be specified for pyomo power_kW")
        n = len(power_kW)
        mean_power = pyo.quicksum(power_kW[t] for t in power_kW.index_set()) / n
        reduction_kW = baseline_kW - mean_power
        return payment_structure.build_expression(
            event, reduction_kW, region_x1=region_x1, model=model, varstr=varstr
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

    baseline_params : dict or BaselineMethod
        Baseline parameters, as produced by `make_baseline_parameters`, or
        a `BaselineMethod` instance (e.g. `TopUsageDaysBaseline`) to use a
        non-default baselining strategy.

    payment_function : list of dict or PaymentStructure
        Payment/penalty schedule to apply, as a list of region dicts (see
        `REGION_X1`, `REGION_X2`, `REGION_Y1`, `REGION_Y2`), or a
        `PaymentStructure` instance (e.g. `CapacityEnergyPayment`) to use a
        non-default payment structure.

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
        mean_power = pyo.quicksum(power_kW[idx] for idx in matched_indices) / len(
            matched_indices
        )
        reduction_kW = baseline_kW - mean_power

        revenue_var, model = payment_structure.build_expression(
            event,
            reduction_kW,
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

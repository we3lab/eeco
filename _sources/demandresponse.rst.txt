.. contents::

.. _demandresponse:

**************
demandresponse
**************

This module computes incentive-based demand response (DR) revenue from
electricity consumption data, modeled along two composable axes --
:class:`BaselineMethod` for how the counterfactual is measured and
:class:`PaymentStructure` for how it is paid -- with three entry points:
``calculate_dr_revenue``, ``build_pyomo_dr_revenue`` (pyomo only, is netted
into the model's active objective), and ``calculate_itemized_dr_revenue`` (post-optimization,
per-event breakdown).

Baselines are computed per settlement interval (see ``resolution`` on
:class:`BaselineMethod`), so the shape of the counterfactual is preserved
for reporting and comparison against metered data. ``PaymentStructure`` adds
``settlement`` mode (``"average"`` settles the window-mean reduction once;
``"interval"`` settles each interval against the payment schedule), and a
payout axis (``payment_basis``/``payout_basis`` choose ``"per_event"`` vs.
``"per_hour"`` pricing, and ``payout`` adds a flat, performance-independent
participation payment in $/kW of bid).

Input formats
=============

Events
------

An event is a ``dict`` built by ``add_event``. A collection of events is a
``list`` of these dicts, or a ``pandas.DataFrame`` with one row per event
(``events_to_dataframe``). The keys are available as Python string variables
denoted in all caps since they are global constants.

.. list-table::
   :header-rows: 1

   * - Key (constant)
     - Type
     - Description
   * - ``"event_date"`` (``EVENT_DATE``)
     - ``pandas.Timestamp``
     - Calendar date of the event
   * - ``"start_hour"`` (``EVENT_START_HOUR``)
     - ``float``
     - Hour of day (0-24) the event begins
   * - ``"duration_hours"`` (``EVENT_DURATION``)
     - ``float``
     - Length of the event in hours
   * - ``"notification_hours"`` (``NOTIFICATION_HOURS``)
     - ``float``
     - Advance notice before the event in hours
   * - ``"baseline_days"`` (``BASELINE_DAYS``)
     - ``list``
     - Candidate baseline days (anything ``pandas.Timestamp`` can parse)
   * - ``"bid_capacity_kW"`` (``BID_CAPACITY_KW``)
     - ``float``
     - Bid capacity in kW
   * - ``"capacity_price"`` (``CAPACITY_PRICE``)
     - ``float``
     - Capacity price in $/kW (or $/kW-hour with ``payment_basis="per_hour"``,
       or $/kW-month with ``payment_basis="monthly_share"``)
   * - ``"adjustment_factor"`` (``ADJUSTMENT_FACTOR``)
     - ``float`` or ``None``
     - Day-of adjustment factor to apply as-is; ``None`` calculates it
   * - ``"monthly_called_capacity_kWh"`` (``MONTHLY_CALLED_CAPACITY_KWH``)
     - ``float`` or ``None``
     - Called capacity summed over every event hour in the event's month, used
       by ``payment_basis="monthly_share"``; ``None`` sums it over the events
       passed in

``baseline_days`` should already exclude any date that is itself another
event's date; ``add_event`` does not check this.

Time zones
----------

Power data may be indexed by a tz-naive or a tz-aware ``DatetimeIndex``.
Event dates, start hours, and baseline days are always read as **local
wall-clock** times, as utility programs define them: an event at
``start_hour=16`` covers 4 pm local time on every day, on either side of a
daylight-saving change. A tz-aware ``event_date``, baseline day, or
``region_x1s`` key keeps its local wall-clock date and drops the time zone.
``"interval_datetime"`` in the per-event results is returned in the data's
time zone.

On a daylight-saving day, a window that covers the skipped spring-forward
hour has no data in that hour and raises ``ValueError``. In a window that
covers the repeated fall-back hour, both occurrences fall in the same
wall-clock interval and are averaged. Windows away from the change (such as
afternoon or evening events) are unaffected.

Payment function
----------------

A payment function is a ``list`` of region dicts, each with ``float`` values
under the keys ``"x1"``, ``"x2"``, ``"y1"``, and ``"y2"`` (``REGION_X1``,
``REGION_X2``, ``REGION_Y1``, ``REGION_Y2``). Region ``[x1, x2)`` covers a
span of delivered ratio (reduction divided by bid capacity) and maps it
linearly onto payment ratios ``y1`` to ``y2``. A region whose ``x2`` is
infinite pays a flat ``y1``. Bounds may be given as the strings
``"Infinity"``/``"-Infinity"`` (as produced by ``json.load`` on a quoted JSON
value); these are coerced to ``inf``/``-inf``. Entry points that take
``payment_function`` expect a :class:`PaymentStructure` (or subclass)
instance, built from the regions list, e.g.
``PaymentStructure(regions)``, not the raw list.

Baseline methods
=================

A :class:`BaselineMethod` computes the counterfactual power the site
*would have* drawn absent the event, which is subtracted from actual power
to get the reduction a program pays for. ``BaselineMethod`` is an abstract
base class: it holds the shared ``resolution`` setting and ``replace``, and
each subclass implements ``calculate_event_baseline``. The module has
built-in subclasses for four baseline methods.

``SimilarDaysBaseline``
------------------------

The default baseline: the average power across a configurable number of the
most recent eligible "similar days" (candidate days are supplied per-event
via ``BASELINE_DAYS``), with Saturdays/Sundays and configured holidays
excluded by default. Its defaults (10 similar weekdays, 3-hour day-of
adjustment) match PG&E's Capacity Bidding Program (CBP).

**Day selection.** ``_rank_days`` drops ineligible candidate days and ranks
the rest, most preferred first; ``select_days`` keeps the first
``n_baseline_days`` of that ranking. The default ranking prefers the most
recent days. ``select_days`` raises ``ValueError`` when no eligible day
remains and warns when fewer remain than ``n_baseline_days``.

**Holidays.** When ``exclude_holidays=True``, a day is excluded if it is a
holiday in the `holidays <https://pypi.org/project/holidays/>`_ package's
calendar for ``holiday_country`` (an ISO 3166-1 code such as ``"US"``) and
optional ``holiday_subdiv`` (such as ``"CA"``), or if it appears in
``holiday_dates``. ``holiday_country`` defaults to ``"US"``, as in CBP; set it
to ``None`` to use ``holiday_dates`` alone.

**Per-interval averaging.** The event window is split into equal-width
settlement intervals of width ``resolution`` (a string of the form
``"[int][unit]"``, e.g. ``"15m"`` or ``"1h"``, parsed by
``utils.get_freq_binsize_minutes``). For each baseline day, the mean power in
each interval is computed; the baseline for interval ``k`` is then the mean of
interval ``k`` across all selected days. The result is one value per
interval, not one value per day. When ``resolution`` is ``None``, the width is
inferred from the spacing of ``model_datetime_index`` (when a model is given)
or ``historical_power_kW.index``, as the smallest gap between timestamps.
Gaps in the data are allowed, but an index whose gaps are not all whole
multiples of the smallest one raises ``ValueError``. Pass it explicitly when that spacing differs
from the settlement interval you want (e.g. hourly meter data settled at
15-minute resolution). The event duration must be an integer multiple of the
interval width. Index ``t`` of the returned array (or ``Var``) is a position
within the event window, with ``0`` at the event start.

**Day-of adjustment.** The averaged baseline can be scaled by a day-of
adjustment factor: the event day's mean power over the adjustment window,
divided by the mean power over the same window pooled across all selected
baseline days. The adjustment window is
``[event_start - adjustment_offset_hours,
event_start - adjustment_offset_hours + adjustment_duration_hours)``; for
example, ``adjustment_offset_hours=4, adjustment_duration_hours=2`` uses the
window from 4 hours to 2 hours before the event. The window must end at or
before the event start. The factor is clipped to ``adjustment_clip`` so one
anomalous morning can't swing the baseline too far, and is skipped (factor
``1.0``, with a warning) when the denominator is near zero. It is always
computed from ``historical_power_kW``, never from the model.
``adjustment_offset_hours=None`` disables the adjustment. To use a factor
calculated elsewhere (e.g. one published by the utility), pass
``adjustment_factor`` to ``add_event`` or directly to ``calculate_event_baseline``. It is
applied as-is, without clipping, in place of the calculated factor. If
both are given they must match, or ``calculate_event_baseline`` raises ``ValueError``.

**Adjustment factor in the model.** With ``adjustment_in_model=True`` and a
``model``, the factor is added as a **fixed** ``pyomo.environ.Var`` named
``varstr + "_adjustment_factor"`` and multiplied into the baseline
symbolically. Retune it with ``model.<varstr>_adjustment_factor.fix(1.15)``
and re-solve without rebuilding the model. Do not ``.unfix()`` it: the solver
would then pick the revenue-maximizing factor, and ``baseline * factor``
would become bilinear. With the default ``False``, the factor is folded in as
a constant, and the baseline ``calculate_event_baseline`` returns is a plain ``numpy.ndarray``
whenever every baseline day is historical.

**Baseline days inside the optimization horizon.** When ``calculate_event_baseline`` is
given a ``model``, ``model_power_kW``, and
``model_datetime_index``, any baseline day whose event window lies fully
inside the model horizon is computed from the decision variable rather than
from history. A window that only partially overlaps the horizon falls back to
history. If at least one day is computed from the model (or the adjustment
factor is in the model), the baseline is returned as an indexed ``Var`` named
``varstr`` with a defining constraint ``varstr + "_constraint"``.

**No baselining.** Passing ``n_baseline_days=0`` turns off baselining
entirely -- day selection and the day-of adjustment are both skipped and the
baseline is a flat zero -- for programs (such as some interruption-based
ones) whose settlement has no historical counterfactual at all. Because a
zero baseline makes every kW of actual consumption look like a negative
reduction, pair it with either a payout-only payment structure or a payment
schedule whose lowest region extends to ``-Infinity`` (as the bundled
CBP-style schedule does). A schedule whose lowest ``x1`` is finite does not
cover a negative delivered ratio: ``find_region`` raises a ValueError, and the
Pyomo/cvxpy formulations enforce ``reduction >= x1 * bid_capacity_kW``, so the
optimizer can never consume above baseline during an event.

``TopUsageDaysBaseline``
-------------------------

A subclass of ``SimilarDaysBaseline`` that keeps the same averaging mechanism but
ranks candidate days by highest mean power draw over the event window
instead of by recency, so the baseline reflects the days the site used the
most power rather than the most recent ones. This is the "high-X-of-Y"
style baseline used by some utility programs, and is otherwise configured
identically to ``SimilarDaysBaseline`` (same day-of adjustment, weekend/holiday
exclusion, etc.). A day with no data in the event window is ranked last; if
it is still selected, the baseline calculation raises on it.

``FixedLevelBaseline``
------------------------

The baseline is a constant, contracted "firm service level" agreed with the
utility ahead of time, rather than anything derived from historical
consumption. Every event uses the same ``firm_level_kW`` value, and there is
no day selection, ranking, or day-of adjustment to configure (it subclasses
``BaselineMethod`` directly, not ``SimilarDaysBaseline``).
``historical_power_kW`` and
``model_datetime_index`` are consulted only to infer the interval width when
``resolution`` is ``None``, and no components are added to a ``model``.

``UnilateralInterruptionBaseline``
------------------------------------

Models programs where the utility -- not the site -- decides when and how
much load is interrupted, holding consumption at or below a fixed level
(``0`` by default, i.e. a full interruption) for the event's duration. This
is enforced as a hard constraint, ``varstr + "_interruption_constraint"``,
on the entries of ``model_power_kW`` inside the event window, rather than
treated as a revenue opportunity the operator chooses to pursue. It
therefore has no ex-post/numpy evaluation path: ``calculate_event_baseline`` raises
``NotImplementedError`` without a ``model``. The interruption level is
returned in the baseline's position to keep a consistent interface, but the
reduction it implies is not the basis for payment, so pair it with a
payout-only payment structure. ``interruption_level_kW`` must be
non-negative.

Payment structures
====================

A :class:`PaymentStructure` turns a reduction (baseline minus actual power)
into a dollar amount. The module has built-in classes for three Payment 
methods.

``PaymentStructure``
----------------------

The default payment structure: a piecewise-linear capacity payment keyed on
the delivered ratio (reduction as a fraction of the bid capacity), through
the regions of the payment function. The capacity payment is
``payment_ratio * capacity_price * bid_capacity_kW``. On top of that, an
optional flat, performance-independent ``payout`` can be added -- or used
alone, by passing ``regions=None``, for programs whose revenue is entirely a
flat participation payment (for example, paired with
``UnilateralInterruptionBaseline``). ``payout`` is in **$/kW of
``bid_capacity_kW``**, not a flat dollar amount, since one structure is
shared across events with different bid sizes.

.. note::

   The default ``settlement="interval"`` matches PG&E's CBP, which settles
   each hour separately, but the default ``payment_basis="per_event"`` does
   not: CBP pays a monthly capacity payment. To model CBP, also set
   ``payment_basis="monthly_share"`` (see below).

**Settlement.** When the reduction carries more than one settlement
interval, ``settlement`` controls how the capacity payment is aggregated:

- ``"interval"`` (default): evaluate the payment function at each interval's
  own delivered ratio, then average the resulting payments. This matches
  PG&E's CBP, which settles each event hour separately, and is the more
  rigorous reading when the payment function is nonlinear over the realized
  range. In pyomo, it multiplies the number of region-selection binaries by
  the number of intervals.
- ``"average"``: average the per-interval reduction into one delivered
  ratio, then evaluate the payment function once. This keeps the pyomo
  formulation to one region-selection block per event. The two modes agree
  exactly when the function is linear over the realized range, and always
  agree for a single scalar reduction.

**Payment basis.** ``payment_basis`` and ``payout_basis`` choose whether the
capacity price and payout are settled once per event (``"per_event"``,
default), scale with the event's duration (``"per_hour"``, i.e. $/kW-hour),
or split a monthly payment across the month's called hours
(``"monthly_share"``). ``"per_hour"`` requires ``"duration_hours"`` on the
event.

``"monthly_share"`` models programs, such as PG&E's CBP, that pay a monthly
Base Capacity Payment (``capacity_price`` in $/kW-month times
``bid_capacity_kW``) and allocate it to each called hour in proportion to
that hour's called capacity. Each event's capacity payment is

.. code-block:: text

   payment_ratio * capacity_price * bid_capacity_kW
       * bid_capacity_kW * duration_hours / monthly_called_capacity_kWh

so the same-month events together earn exactly one Base Capacity Payment
at full delivery. ``bid_capacity_kW`` is taken as the called capacity in
every hour of the event; split an event whose hours have different called
capacities into separate events. ``events_to_dataframe`` (and so every
entry point that takes an events list) fills in a missing
``"monthly_called_capacity_kWh"`` by summing ``bid_capacity_kW *
duration_hours`` over the events in the same calendar month. A value given
on any event (via ``add_event``) applies to its whole month, and events in
the same month that give different values raise ``ValueError``. Give it
explicitly when the events passed in don't cover the whole month -- for
example, an optimization horizon that ends mid-month, or a total taken from
a utility settlement statement. Pair it with ``settlement="interval"`` and an
hourly ``resolution`` to apply the performance adjustment hour by hour, as
CBP does.

**Region lookup.** ``find_region`` takes either a ``delivered_ratio``
(returns the region containing it) or a ``region_x1`` (returns the region
whose ``x1`` matches). Passing neither raises ``ValueError``; passing both
raises unless the delivered ratio falls in the region identified by
``region_x1``.

``CapacityEnergyPayment``
----------------------------

A two-part payment: the same piecewise capacity payment as
``PaymentStructure``, plus a flat ``$/kWh`` payment on the energy actually
curtailed. This models programs that pay separately for having capacity
available (the capacity term) and for the energy actually reduced during
the event (the energy term). The energy term is
``energy_price * mean(reduction_kW) * duration_hours``, which equals the sum
of each interval's own energy since intervals are equal-width. It is
independent of ``settlement``.

``MarketIndexedPayment``
---------------------------

The same piecewise capacity payment as ``PaymentStructure``, except the
capacity price is resolved at evaluation/build time from a caller-supplied
``price_lookup(event)`` callable rather than a fixed value on the event.
This models programs whose price tracks a wholesale or day-ahead market
index rather than a flat contracted rate. The price is resolved once, when
the expression is built, so a model built this way is tied to the prices in
force at build time.

Optimization formulations
=========================

``PaymentStructure.build_expression`` dispatches on the type of
``reduction_kW``.

**Realized values** (``numpy.ndarray`` or a Python number): the region is
already determined, so this calls ``evaluate``.

**cvxpy** (``cvxpy.Expression``/``cvxpy.Variable``): requires a known
``region_x1`` and builds only that region, returning
``(revenue_expr, constraints)`` for the caller to add to their own
``cvxpy.Problem``. A vector reduction is averaged for ``"average"``
settlement; for ``"interval"`` settlement the region bound constraints apply
to every interval. The revenue expression is identical in both modes, since
the payment ratio is affine within one fixed region. To search across
regions with cvxpy, re-build the expression for a new ``region_x1`` when the
solution falls outside the assumed one. ``calculate_dr_revenue`` does not
support cvxpy.

**pyomo** (``pyomo.environ.Var``/expression, or a list/tuple of per-interval
expressions): builds *every* region onto ``model`` at once using a
disaggregated binary-selection formulation (Balas' extended form), so the
solver chooses the region when ``region_x1=None``. Passing ``region_x1``
fixes the region binaries instead. Infinite region bounds are replaced by
``±1000`` times the bid capacity.

Every region component is indexed by settlement interval ``t`` and region.
``"average"`` settlement builds a single interval (``t = 0``) holding the
window-mean reduction; ``"interval"`` settlement builds one region-selection
block per interval. A per-interval revenue ``Expression`` lets a solved model
be audited per interval. A scalar ``region_x1`` fixes the same region for
every interval; a list of length ``n_intervals`` fixes them individually.
Fixing one region for every interval is a tighter feasible set than
``"average"`` settlement (which only constrains the mean) and can make a
previously feasible plan infeasible, so prefer ``region_x1=None`` with
``"interval"`` settlement.

If the structure has no regions (payout-only), no region components are
created and the revenue is the constant payout.

Pyomo component names
---------------------

Every pyomo component is named ``varstr + "_" + suffix``, with the suffixes
available as module constants. ``BASELINE_COMPONENT_SUFFIXES`` and
``PAYMENT_COMPONENT_SUFFIXES`` collect them. ``varstr`` must be unique per
call on a given model, since reusing it raises a pyomo "component already
exists" error. ``calculate_dr_revenue`` and ``build_pyomo_dr_revenue`` check
for such clashes before adding anything and raise ``ValueError``, so calling
them twice on one model needs a different ``varstr_prefix``. They name each
event ``f"{varstr_prefix}_{i}"`` (``i`` is the event's position in date
order) and its baseline ``f"{varstr_prefix}_{i}_baseline_kW"``.

A model is meant to price all its events under one payment structure. The
first structure built into a model is recorded on it, and building revenue
into the same model with a structure of a different class or different
settings raises a ``UserWarning``.

.. list-table::
   :header-rows: 1

   * - Constant name
     - Description
     - Value
     - Created by
   * - ``CONSTRAINT_SUFFIX``
     - Baseline defining constraint
     - ``"constraint"``
     - ``SimilarDaysBaseline.calculate_event_baseline``
   * - ``ADJUSTMENT_FACTOR_SUFFIX``
     - Fixed day-of adjustment factor ``Var``
     - ``"adjustment_factor"``
     - ``SimilarDaysBaseline.calculate_event_baseline``
   * - ``INTERRUPTION_CONSTRAINT_SUFFIX``
     - Interruption upper bound
     - ``"interruption_constraint"``
     - ``UnilateralInterruptionBaseline``
   * - ``REGION_ACTIVE_SUFFIX``
     - Binary, 1 iff the region is active
     - ``"region_active"``
     - ``PaymentStructure``
   * - ``REGION_SELECT_CONSTRAINT_SUFFIX``
     - Exactly one region is active
     - ``"region_select_constraint"``
     - ``PaymentStructure``
   * - ``REGION_REDUCTION_SUFFIX``
     - Region's share of the reduction
     - ``"region_reduction"``
     - ``PaymentStructure``
   * - ``REGION_REDUCTION_LOWER_CONSTRAINT_SUFFIX``
     - Share ``>= x1 * bid_capacity_kW * active``
     - ``"region_reduction_lower_constraint"``
     - ``PaymentStructure``
   * - ``REGION_REDUCTION_UPPER_CONSTRAINT_SUFFIX``
     - Share ``<= x2 * bid_capacity_kW * active``
     - ``"region_reduction_upper_constraint"``
     - ``PaymentStructure``
   * - ``REGION_REDUCTION_SUM_CONSTRAINT_SUFFIX``
     - Reduction equals the sum of shares
     - ``"region_reduction_sum_constraint"``
     - ``PaymentStructure``
   * - ``INTERVAL_REVENUE_SUFFIX``
     - Per-interval revenue ``Expression``
     - ``"interval_revenue"``
     - ``PaymentStructure``
   * - ``REVENUE_SUFFIX``
     - Revenue ``Var``, including payout
     - ``"adjustment_factor"``
     - ``PaymentStructure``
   * - ``REVENUE_CONSTRAINT_SUFFIX``
     - Defines revenue
     - ``"revenue"``
     - ``PaymentStructure``
   * - ``ENERGY_REVENUE_SUFFIX``
     - Energy revenue ``Var``
     - ``"energy_revenue"``
     - ``CapacityEnergyPayment``
   * - ``ENERGY_REVENUE_CONSTRAINT_SUFFIX``
     - Defines energy revenue
     - ``"energy_revenue_constraint"``
     - ``CapacityEnergyPayment``
   * - ``TOTAL_REVENUE_SUFFIX``
     - Capacity plus energy revenue ``Var``
     - ``"total_revenue"``
     - ``CapacityEnergyPayment``
   * - ``TOTAL_REVENUE_CONSTRAINT_SUFFIX``
     - Defines total revenue
     - ``"total_revenue_constraint"``
     - ``CapacityEnergyPayment``

Entry points
============

- ``calculate_dr_revenue`` dispatches on ``power_kW``: realized
  ``pandas``/``numpy`` data is settled ex-post via
  ``calculate_itemized_dr_revenue``, while a ``pyomo.environ.Var`` has each
  event's baseline and revenue components built onto ``model``, returning the
  total revenue expression. ``region_x1s`` optionally fixes each event's
  region, keyed by event date (e.g. ``{"2024-01-08": 0.6}``); events missing
  from it leave the region choice to the solver.
- ``build_pyomo_dr_revenue`` calls ``calculate_dr_revenue`` and nets the total
  revenue into the model's single active objective, whatever its name:
  subtracted from a minimization objective, added to a maximization one. If
  the model has no active objective, it adds a minimization objective named
  ``objective`` (or a unique variant if that name is taken). More than one
  active objective raises ``ValueError`` before anything is added.
- ``calculate_itemized_dr_revenue`` returns a ``pandas.DataFrame`` with one
  row per event (see ``calculate_event_revenue`` for the columns). Each event
  is sliced and re-baselined independently, so results never mix time windows
  across events.
- ``build_event_revenue`` handles a single event and expects ``power_kW``
  already sliced to that event's window, matching the convention in
  ``costs.py`` of passing in already-relevant consumption slices.

Combining baselines and payment structures
=============================================

Because baseline method and payment structure are independent axes, real
programs are represented by pairing whichever combination matches their
rules. A few examples:

- **A day-ahead capacity bidding program** (e.g. PG&E's CBP) pairs the
  default ``SimilarDaysBaseline`` (10 similar weekdays, 3-hour day-of
  adjustment, US holidays excluded, ``resolution="1h"``) with
  ``PaymentStructure(regions, payment_basis="monthly_share")``, whose regions ramp payment up with
  delivered ratio. The monthly Base Capacity Payment is split across the
  month's called hours and each hour is adjusted for its own performance.
- **A firm service level program**, where the customer commits to staying at
  or below a contracted demand level, pairs ``FixedLevelBaseline`` with a
  ``PaymentStructure`` using ``payment_basis="per_hour"`` so the payment (or
  penalty) scales with how long the level was exceeded.
- **An emergency curtailment or interruptible program**, where the utility
  itself cuts load, pairs ``UnilateralInterruptionBaseline`` with a
  payout-only ``PaymentStructure`` (``regions=None``) that pays a flat
  $/kW participation payment regardless of the (utility-controlled)
  reduction.
- **A wholesale-indexed capacity program** pairs ``TopUsageDaysBaseline``
  (so the baseline reflects the site's highest-usage days) with
  ``MarketIndexedPayment``, so the price paid tracks a day-ahead market
  index instead of a flat rate.
- **A combined capacity-and-energy incentive program** pairs the default
  ``SimilarDaysBaseline`` with ``CapacityEnergyPayment``, paying both a
  capacity-based incentive and a ``$/kWh`` rate for the energy actually
  curtailed.

Extending BaselineMethod and PaymentStructure
=================================================

Both axes are designed to be subclassed, and every public entry point
(``calculate_dr_revenue``, ``build_pyomo_dr_revenue``,
``calculate_itemized_dr_revenue``, and friends) takes a ``BaselineMethod``
instance and accepts a ``PaymentStructure`` instance in place of a list of
regions, so a custom subclass drops in without further plumbing.

To add a new similar-days variant, subclass ``SimilarDaysBaseline`` and
override one of:

- ``_rank_days(candidate_days, historical_power_kW, event)`` to change which
  candidate days are eligible and how they're ranked, while keeping the
  averaging mechanism (see ``TopUsageDaysBaseline``, which calls
  ``super()._rank_days(...)`` to reuse the eligibility rule and only
  replaces the ranking). Prefer this over overriding ``select_days``.
- ``_adjustment_factor(valid_days, historical_power_kW, event)`` to change
  how the day-of scaling factor is computed.

For a baseline that isn't a historical day-average at all, subclass
``BaselineMethod`` directly, call ``super().__init__(resolution)``, and
implement ``calculate_event_baseline(historical_power_kW, event, *,
model=None, model_power_kW=None, model_datetime_index=None, varstr=None,
adjustment_factor=None)`` (see ``FixedLevelBaseline`` and
``UnilateralInterruptionBaseline``). The implementation must honor
``calculate_event_baseline``'s calling contract: return a ``(value, model)``
pair, with ``model`` left ``None`` ex-post (or raise), adding any pyomo
components under ``varstr`` when a model is given.

To add a new payment structure, subclass ``PaymentStructure`` and override
``evaluate`` and ``build_expression`` as a pair -- ``evaluate`` is used for
ex-post settlement and ``build_expression`` is used to build the same
payment into an optimization model, and if the two diverge, an optimized
plan will not reconcile with its ex-post settlement. ``CapacityEnergyPayment``
and ``MarketIndexedPayment`` show two ways to keep them in sync: the former
adds an extra term to both methods after delegating the capacity portion to
``super()``; the latter resolves a field on ``event`` (the market price)
before delegating both methods to ``super()`` unchanged.

.. automodule:: eeco.demandresponse
   :members:

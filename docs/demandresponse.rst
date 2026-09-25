.. contents::

.. _demandresponse:

**************
demandresponse
**************

This module computes demand response revenue from electricity consumption
data, modeled along two composable axes -- :class:`BaselineMethod` for how
the counterfactual is measured and :class:`PaymentStructure` for how it is
paid -- with three entry points: ``calculate_dr_revenue``,
``build_dr_revenue`` (pyomo only, nets into ``model.objective``), and
``calculate_itemized_dr_revenue`` (post-optimization, per-event breakdown).

Baselines are computed per settlement interval (see ``resolution`` on
:class:`BaselineMethod`), so the shape of the counterfactual is preserved
for reporting and comparison against metered data. ``PaymentStructure`` adds
``settlement`` mode (``"average"`` settles the window-mean reduction once;
``"interval"`` settles each interval against the payment schedule), and a
payout axis (``payment_basis``/``payout_basis`` choose ``"per_event"`` vs.
``"per_hour"`` pricing, and ``payout`` adds a flat, performance-independent
participation payment in $/kW of bid.

Baseline methods
=================

A :class:`BaselineMethod` computes the counterfactual power the site
*would have* drawn absent the event, which is subtracted from actual power
to get the reduction a program pays for. The module bundles four of them.

``BaselineMethod``
-------------------

The default baseline: the average power across a configurable number of the
most recent eligible "similar days" (candidate days are supplied per-event
via ``BASELINE_DAYS``), with Saturdays/Sundays and configured holidays
excluded by default. The average can be scaled by a day-of adjustment
factor, which compares the event day's own pre-event consumption against
the same pre-event window averaged across the baseline days, clipped to a
configurable range so one anomalous morning can't swing the baseline too
far. Its defaults (10 similar weekdays, 3-hour day-of adjustment) match
PG&E's Capacity Bidding Program (CBP).

Passing ``n_baseline_days=0`` turns off baselining entirely -- day
selection and the day-of adjustment are both skipped and the baseline is a
flat zero -- for programs (such as some interruption-based ones) whose
settlement has no historical counterfactual at all. Because a zero baseline
makes every kW of actual consumption look like a reduction, pair it with
either a payout-only payment structure or a payment schedule whose lowest
region extends to ``-Infinity``.

``TopUsageDaysBaseline``
-------------------------

A subclass of ``BaselineMethod`` that keeps the same averaging mechanism but
ranks candidate days by highest mean power draw over the event window
instead of by recency, so the baseline reflects the days the site used the
most power rather than the most recent ones. This is the "high-X-of-Y"
style baseline used by some utility tariffs, and is otherwise configured
identically to ``BaselineMethod`` (same day-of adjustment, weekend/holiday
exclusion, etc.).

``FixedLevelBaseline``
------------------------

The baseline is a constant, contracted "firm service level" agreed with the
utility ahead of time, rather than anything derived from historical
consumption. Every event uses the same ``firm_level_kW`` value, and there is
no day selection, ranking, or day-of adjustment to configure.

``UnilateralInterruptionBaseline``
------------------------------------

Models programs where the utility -- not the site -- decides when and how
much load is interrupted, holding consumption at a fixed level (``0`` by
default, i.e. a full interruption) for the event's duration. This is
enforced as a hard constraint on the optimization model rather than treated
as a revenue opportunity the operator chooses to pursue, so it has no
ex-post/numpy evaluation path -- it only applies inside a pyomo model. Pair
it with a payout-only payment structure, since the "reduction" it implies
is not itself the basis for payment.

Payment structures
====================

A :class:`PaymentStructure` turns a reduction (baseline minus actual power)
into a dollar amount. The module bundles three of them.

``PaymentStructure``
----------------------

The default payment structure: a piecewise-linear capacity payment keyed on
the delivered ratio (reduction as a fraction of the bid capacity), through a
list of regions each mapping a span of delivered ratio to a payment ratio.
The capacity payment is ``payment_ratio * capacity_price * bid_capacity_kW``.
On top of that, an optional flat, performance-independent ``payout`` in
$/kW of bid can be added -- or used alone, by passing ``regions=None``, for
programs whose revenue is entirely a flat participation payment (for
example, paired with ``UnilateralInterruptionBaseline``, which has no
delivered-ratio payment of its own).

``CapacityEnergyPayment``
----------------------------

A two-part payment: the same piecewise capacity payment as
``PaymentStructure``, plus a flat ``$/kWh`` payment on the energy actually
curtailed. This models programs that pay separately for having capacity
available (the capacity term) and for the energy actually reduced during
the event (the energy term).

``MarketIndexedPayment``
---------------------------

The same piecewise capacity payment as ``PaymentStructure``, except the
capacity price is resolved at evaluation/build time from a caller-supplied
``price_lookup(event)`` callable rather than a fixed value on the event.
This models programs whose price tracks a wholesale or day-ahead market
index rather than a flat contracted rate.

Combining baselines and payment structures
=============================================

Because baseline method and payment structure are independent axes, real
programs are represented by pairing whichever combination matches their
rules. A few examples:

- **A day-ahead capacity bidding program** (e.g. PG&E's CBP) pairs the
  default ``BaselineMethod`` (10 similar weekdays, 3-hour day-of adjustment)
  with a ``PaymentStructure`` whose regions ramp payment up with delivered
  ratio, settled ``"average"``.
- **A firm service level tariff**, where the customer commits to staying at
  or below a contracted demand level, pairs ``FixedLevelBaseline`` with a
  ``PaymentStructure`` using ``payment_basis="per_hour"`` so the payment (or
  penalty) scales with how long the level was exceeded.
- **An emergency curtailment or interruptible tariff**, where the utility
  itself cuts load, pairs ``UnilateralInterruptionBaseline`` with a
  payout-only ``PaymentStructure`` (``regions=None``) that pays a flat
  $/kW participation payment regardless of the (utility-controlled)
  reduction.
- **A wholesale-indexed capacity program** pairs ``TopUsageDaysBaseline``
  (so the baseline reflects the site's highest-usage days) with
  ``MarketIndexedPayment``, so the price paid tracks a day-ahead market
  index instead of a flat rate.
- **A combined capacity-and-energy incentive program** pairs the default
  ``BaselineMethod`` with ``CapacityEnergyPayment``, paying both a
  capacity-based incentive and a ``$/kWh`` rate for the energy actually
  curtailed.

Extending BaselineMethod and PaymentStructure
=================================================

Both axes are designed to be subclassed, and every public entry point
(``calculate_dr_revenue``, ``build_dr_revenue``,
``calculate_itemized_dr_revenue``, and friends) accepts an already-constructed
instance in place of the dict/list configuration it otherwise builds one
from, so a custom subclass drops in without further plumbing.

To add a new baseline method, subclass ``BaselineMethod`` and override one
of:

- ``_rank_days(candidate_days, historical_power_kW, event)`` to change which
  candidate days are eligible and how they're ranked, while keeping the
  averaging mechanism (see ``TopUsageDaysBaseline``, which calls
  ``super()._rank_days(...)`` to reuse the eligibility rule and only
  replaces the ranking).
- ``_adjustment_factor(valid_days, historical_power_kW, event)`` to change
  how the day-of scaling factor is computed.
- ``compute(historical_power_kW, event, *, model=None, model_power_kW=None,
  model_datetime_index=None, varstr=None)`` to replace the baseline
  calculation entirely, for a baseline that isn't a historical day-average
  at all (see ``FixedLevelBaseline`` and ``UnilateralInterruptionBaseline``).
  An override must honor ``compute``'s calling contract: return a plain
  array (or raise) when ``model`` is ``None``, and a ``(value, model)`` pair
  when a model is given, adding any pyomo components under ``varstr``.

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

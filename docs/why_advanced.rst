.. contents::

.. _why-advanced:

***********************************
Why Do the Advanced Features Exist?
***********************************

We know that the "advanced features" outlined in :ref:`how-to-advanced` may be unintuitive, 
so we assure you that we had very good reasons for writing each of them.

Most of the advanced features are designed to handle the peculiarities of computing a monthly electricity bill
in a time window of only a few days for moving horizon optmization.

.. _why-moving-horizon:

Why Use Moving Horizon Optimization?
====================================

1. computational complexity
2. avoid perfect foresight

.. _why-prev-consumption:

What Is the Purpose of `prev_demand_dict` and `prev_consumption_dict`?
======================================================================

FILL IN STUB: when performing moving horizon optimization, we need to know the previous consumption and costs

WHY BOTH?: for some demand charges there are different costs at different times, so we cannot just use the `prev_consumption_dict`

.. _why-consumption-est:

What Is the Purpose of `consumption_estimate`?
==============================================

If a utility uses tiered charges, then the price per unit of electricity varies depending on total monthly consumption (:ref:`complexities`).
This means that the correct tier is a function of the monthly electricity consumption profile, which is a variable in the optimization problem.
This makes computing the cost of electricity directly non-convex,
so we assume the total monthly consumption using the `consumption_estimate` flag to simplify the problem into a convex form. 
Moving horizon optimization omplicates this issue even further, since the total monthly consumption will not be known in a single time window.

We have found that the convex simplification works quite well in our energy flexibility research since the timing but not magnitude of electricity consumption changes between simulations.
However, in other applications this convex approxmiation may not fare as well. 

.. _why-scale-demand:

What Is the Purpose of `demand_scale_factor`?
=============================================

[cite Bolorinos ES&T (2023) & Chapin ES&T (2025)]: gist is that it is necessary for moving horizon optimization 
because the certainty around demand charges and therefore relative weight between demand and energy charges
varies throughout the month.

TODO: INSERT FORMULATION FROM SI OF PAPER
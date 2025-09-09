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

We believe the two main benefits to moving horizon optimization are (1) computational complexity and (2) avoiding perfect foresight.

Sometimes running optimization for an entire billing period (usually a month) at the necessary control intervals is computationally infeasible.
In those cases, moving horizon optimization allows an analyst to define a sliding window (the horizon period) and step through the entire billing period sequentially.

Even when computationally feasible, running an entire billing period may be undesirable because it provides too much foresight to the model.
Moving horizon optimization can be used to realistically simulated model-predictive control (MPC), 
which ensures that the benefits of improved control strategies are correctly quantified.

.. _why-prev-consumption:

What Is the Purpose of `prev_demand_dict` and `prev_consumption_dict`?
======================================================================

When performing moving horizon optimization, we need to know the previous consumption and costs from within the same billing period.
For example, if I have already hit 100 kW of peak demand, then going up to 99 kW will not add any cost to the demand charge on my bill.

Both `prev_demand_dict` and `prev_consumption_dict` are necessary due to the complexity of the tariffs. 
For some demand charges there are different costs at different times, so we cannot just use the `prev_consumption_dict` directly to compute
the previous maximum charge billed.

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

The demand scale factor was designed in our academic papers to balance demand and energy charges during moving horizon optimization. 

The basic concept is that the certainty around demand charges and therefore relative weight between demand and energy charges
varies throughout the month. In other words, as time goes on the demand charge becomes more certain and should be given a higher weight.

The specific mathematical formulation that we recommend is as follows, but the scale factor should be tuned for each application
by adjusting `lambda_energy`:

.. code-block:: python

    n_billing = 4  # number of days in the horizon window
    n_horizon = 31  # number of days in the billing period
    lambda_energy = 0.7  # tunable parameter 
    for d in range(n_billing):
        energy_scale_factor = lambda_energy * (n_billing - d) / n_horizon
        demand_scale_factor = 1 / energy_scale_factor

You can read more about this concept in "Integrated energy flexibility management at wastewater treatment facilities" [`Bolorinos 𝐸𝑛𝑣𝑖𝑟𝑜𝑛. 𝑆𝑐𝑖. 𝑇𝑒𝑐ℎ𝑛𝑜𝑙. (2023) <https://doi.org/10.1021/acs.est.3c00365>`_]
and "Load-Shifting Strategies for Cost-Effective Emission Reductions at Wastewater Facilities" [`Chapin 𝐸𝑛𝑣𝑖𝑟𝑜𝑛. 𝑆𝑐𝑖. 𝑇𝑒𝑐ℎ𝑛𝑜𝑙. (2025) <https://doi.org/10.1021/acs.est.4c09773>`_].
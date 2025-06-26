.. contents::

.. _tutorial-cost:

**************************
Optimize Electricity Costs
**************************

This tutorial will walkthrough how to optimize electricity bill savings for a simple battery model in both CVXPY and Pyomo.
It assumes that you already have the following data or models:

  - An optimization model of the electricity consumer with system constraints

    - Consider using flexibility metrics to encode system constraints (:ref:`why-metrics`)!
    - To start with, you can use our examples:

      - `pyomo_battery_model.py <https://github.com/we3lab/electric-emission-cost/blob/main/examples/pyomo_battery_model.py>`_
      - `cvxpy_battery_model.py <https://github.com/we3lab/electric-emission-cost/blob/main/examples/cvxpy_battery_model.py>`_
  - The electricity tariff to which the electricity consumer is subject

    - We publish a nationwide tariff dataset: https://github.com/we3lab/industrial-electricity-tariffs
    - In this case, we use `tariff.csv <https://github.com/we3lab/electric-emission-cost/blob/main/electric_emission_cost/data/tariff.csv>`_ from the EEC `data` folder

CVXPY
=====

Set up simple battery model:

# TODO: insert code block

Always compute the ex-post cost using numpy due to the convex relaxations that we apply in our optimization code:

# TODO: insert code block

Pyomo
=====


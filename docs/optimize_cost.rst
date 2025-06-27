.. contents::

.. _tutorial-cost:

**************************
Optimize Electricity Costs
**************************

This tutorial will walkthrough how to optimize electricity bill savings for a simple battery model in either :ref:`cvx-cost` or :ref:`pyo-cost` .
It will go through the following steps:

  #. Load an electricity tariff spreadsheet

     - We publish a nationwide tariff dataset: https://github.com/we3lab/industrial-electricity-tariffs
     - In this case, we use `tariff.csv <https://github.com/we3lab/electric-emission-cost/blob/main/electric_emission_cost/data/tariff.csv>`_ from the EEC `data` folder
  #. Configure an optimization model of the electricity consumer with system constraints
  
     - Consider using flexibility metrics to encode system constraints (:ref:`tutorial-metrics`)!
     - The models are presented step-by-step to demonstrate the model building process, 
       but the complete models are available in the `examples` folder:
       
       - `pyomo_battery_model.py <https://github.com/we3lab/electric-emission-cost/blob/main/examples/pyomo_battery_model.py>`_
       - `cvxpy_battery_model.py <https://github.com/we3lab/electric-emission-cost/blob/main/examples/cvxpy_battery_model.py>`_
  #. Create an objective function of electricity costs based on this tariff sheet
  #. Minimize the electriciy costs of this consumer given the system constraints and base load consumption
  #. Display the results to validate that the optimization is correct

.. _cvx-cost:

CVXPY
=====

Load data

Set up simple battery model:

# TODO: insert code block

Always compute the ex-post cost using numpy due to the convex relaxations that we apply in our optimization code:

# TODO: insert code block

.. _pyo-cost:

Pyomo
=====


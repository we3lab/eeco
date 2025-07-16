.. contents::

.. _tutorial-emit:

**************************
Optimize Scope 2 Emissions
**************************

This tutorial will walkthrough how to optimize Scope 2 emissions for a simple battery model in either :ref:`cvx-emit` or :ref:`pyo-emit` by going through the following steps:

  #. Load an emissions spreadsheet
  #. Configure an optimization model of the electricity consumer with system constraints
  
     - Consider using flexibility metrics to encode system constraints (:ref:`tutorial-metrics`)!
     - The models are presented step-by-step to demonstrate the model building process, 
       but the complete models are available in the `examples` folder:

       - `pyomo_battery_model.py <https://github.com/we3lab/electric-emission-cost/blob/main/examples/pyomo_battery_model.py>`_
       - `cvxpy_battery_model.py <https://github.com/we3lab/electric-emission-cost/blob/main/examples/cvxpy_battery_model.py>`_
  #. Create an objective function of Scope 2 emissions using the emissions factors
  #. Minimize the Scope 2 emissions of this consumer given the system constraints and base load consumption
  #. Display the results to validate that the optimization is correct


.. _cvx-emit:

CVXPY
=====


.. _pyo-emit:

Pyomo
=====

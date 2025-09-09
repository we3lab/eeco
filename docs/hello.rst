.. contents::

.. _helloworld:

***************
Getting Started
***************

.. _installation:

Installing Electric Emissions & Costs Optimizer (EECO)
======================================================

For most users, the first step to using EECO will be to pip install the Python package:

.. code-block:: python

    pip install eeco

Core Functionality
==================

The EECO package has three main functions: 

(1) calculate the electricity bill of a facility given a tariff and user consumption data. 
(2) calculate the Scope 2 emissions implications given grid emissions and user consumption data.
(3) incorporate flexibility metrics (e.g., round-trip efficiency) as a constraint or objective.

These functions can be performed in three different modes:

(1) ``NumPy``
(2) ``CVXPY``
(3) ``Pyomo``

More information about how to correctly format data inputs can be found in :ref:`data-format`.

.. _batteryoptimization:

Sample Model: Battery Optimization
====================================

Besides the core functionality and utility functions, a simple electric battery model (Pyomo) is included as an example.
The model uses a linear program to minimize the electricity cost of a facility + battery subject to the baseline power consumption of the facility, dynamic constraints associated with battery charging and discharging, and the rules of the electricity tariff. 
This example is not intended to be a comprehensive model of battery dynamics, but rather to illustrate how to use the ``eeco`` package within and outside of an optimization problem.

The model file is located in the ``examples`` directory, and can be run with the following command after installing in editable mode:
.. code-block:: bash

    python examples/battery_optimization.py

A full walkthrough of this example is available in the Jupyter notebook within the `github repository <https://github.com/we3lab/eeco/blob/main/examples/example_pyomo_jupyter.ipynb>`_ at ``examples/example_pyomo_jupyter.ipynb``.

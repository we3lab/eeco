.. contents::

.. _helloworld:

***************
Getting Started
***************

.. _installation:

Installing Electric Emissions & Costs (EEC)
===========================================

For most users, the first step to using EEC will be to pip install the Python package:

.. code-block:: python

    pip install electric-emission-cost

Core Functionality
==================

The EEC package has two main functions: 

(1) calculate the electricity bill of a facility given a tariff and user consumption data. 
(2) calculate the Scope 2 emissions implications given grid emissions and user consumption data.

These functions can be performed in three different modes:

(1) ``numpy``
(2) ``CVXPY``
(3) ``Pyomo``

More information about how to correctly format data inputs can be found in :ref:`dataformat`.

.. _virtualbattery:

Sample Model: Virtual Battery
=============================

Besides the core functionality and utility functions, a simple virtual battery model is included as an example.
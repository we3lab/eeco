.. contents::

.. _calculatehowto:

************************************
How to Calculate Costs and Emissions
************************************

.. _calculate-tariff:

Electricity Tariffs
===================

.. _calculate-emissions:

Scope 2 Emissions
=================

This package is not designed to calculate Scope 2 emissions that are a complete timeseries.
We feel that is simple enough that it does not warrant a function.

In Pyomo:

# TODO: insert code snippet

In CVXPY:

# TODO: insert code snippet

However, many data sources report emissions factors as monthly/hourly averages (:ref:`_data-format-emissions`).
Our package is designed to unpack data in that format into a timeseries the same length as the consumption variable.

# TODO: example of using `calculate_grid_emissions`, `calculate_grid_emissions_cvx`, and `calculate_grid_emissions_pyo`

The `get_carbon_intensity` function can be used for those interested in getting the timeseries directly:

# TODO: example of using `get_carbon_intensity`

Units
*****

Unit conversions will be handled by the EEC package as long as the proper `emissions_units` and 
`consumption_units` arguments are provided.
Based on the most common data sources we have used, the consumption units are in kW
and emissions units in kg / MWh, so `consumption_units=u.kW` and `emissions_units=u.kg / u.MWh`.
This defaults to a 0.001 conversion factor.

The temporal resolution of the consumption data should be provided as a string. 
The default is 15-minute intervals, so `resolution="15m"`.

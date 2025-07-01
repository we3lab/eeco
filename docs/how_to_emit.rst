.. contents::

.. WARNING::

  Site under construction!    
  Documentation incomplete :( 

.. _how-to-emissions:

**************************
How to Calculate Emissions
**************************

This package is not designed to calculate Scope 2 emissions that are a complete timeseries.
We feel that is simple enough that it does not warrant its own function.

As an example, if you assume that `emission_arr` is a `NumPy` array of Scope 2 emissions factors (in tons CO_2 / MWh)  
and `consumption_arr` is a `NumPy` array of electricity consumption (in MWh), 
you would simply dot product the two arrays to find the total emissions (i.e., multiply and sum):

.. code-block:: python

    total_emissions = np.dot(emission_arr, consumption_arr)

However, many data sources report emissions factors as monthly/hourly averages (:ref:`data-format-emissions`).
Our package is designed to unpack data in that format into a timeseries the same length as the consumption variable.

Get Carbon Intensity
====================

The `get_carbon_intensity` function can be used for those interested in getting the timeseries directly.
For example, we can use this function to get the carbon intensity as a 15-minute timeseries 
for the entire month of May 2025 from our sample emisisons data:

.. code-block:: python

    start_date, end_date = datetime.datetime(2025, 5, 1), datetime.datetime(2025, 6, 1)
    emissions_df = pd.read_csv("electric_emission_cost/data/emissions.csv")
    carbon_intensity = emissions.get_carbon_intensity(start_date, end_date, emissions_df)

The optional argument `resolution` should be used to specify the temporal resolution of the consumption data
as a string in the from `<binsize><unit>`, 
where units are either `'m'` for minutes, `'h'` for hours, or `'d'` / `'D'` for days.
The default is `"15m"`, so the timeseries will be on 15-minute intervals if not otherwise specified.

Calculate Scope 2 Emissions
===========================

# TODO: example of using `calculate_grid_emissions`, `calculate_grid_emissions_cvx`, and `calculate_grid_emissions_pyo`

numpy
*****

CVXPY
*****

Pyomo
*****

Units
=====

The EEC package uses `Pint <https://pint.readthedocs.io/en/stable/>`_ to handle nit conversions automaitcally. 
The logic depends on the proper `emissions_units` and `consumption_units` arguments being provided.
Based on the most common data sources we have used, the consumption units are in kW
and emissions units in kg / MWh, so `consumption_units=u.kW` and `emissions_units=u.kg / u.MWh`.
This defaults to a 0.001 conversion factor.

The temporal resolution of the consumption data should be provided as a string. 
The default is 15-minute intervals, so `resolution="15m"`.
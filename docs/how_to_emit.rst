.. contents::

.. _how-to-emissions:

**************************
How to Calculate Emissions
**************************

This package is designed to calculate cumulative emissions from a given electricity consumption array and monthly/hourly average emissions factors.
Many data sources report emissions factors as monthly/hourly averages (:ref:`data-format-emissions`), 
so it is useful to convert data in monthly/hourly format into a time series the same length as the consumption variable.

Note that if you already have a complete time series of Scope 2 emissions factors, this package is not necessary. 
We feel that is simple enough that it does not warrant its own function.

As an example, if you assume that `emission_arr` is a `NumPy` array of Scope 2 emissions factors (in tons CO_2 / MWh)  
and `consumption_arr` is a `NumPy` array of electricity consumption (in MWh), 
you would simply dot product the two arrays to find the total emissions (i.e., multiply and sum):

.. code-block:: python

    total_emissions = np.dot(emission_arr, consumption_arr)

=================
Import Statements
=================

To make this how-to guide clear, below are the import statements used throughout:

.. code-block:: python

    import datetime
    import cvxpy as cp
    import numpy as np
    import pandas as pd
    import pyomo.environ as pyo
    from eeco.units as u
    from eeco import emissions

====================
Get Carbon Intensity
====================

The `get_carbon_intensity` function can be used for those interested in getting the timeseries directly.
For example, we can use this function to get the carbon intensity as a 15-minute timeseries 
for the entire month of May 2025 from our sample emisisons data:

.. code-block:: python

    start_date, end_date = datetime.datetime(2025, 5, 1), datetime.datetime(2025, 6, 1)
    emissions_df = pd.read_csv("eeco/data/emissions.csv")
    carbon_intensity = emissions.get_carbon_intensity(start_date, end_date, emissions_df)

The optional argument `resolution` should be used to specify the temporal resolution of the consumption data
as a string in the from `<binsize><unit>`, 
where units are either `'m'` for minutes, `'h'` for hours, or `'d'` / `'D'` for days.
The default is `"15m"`, so the timeseries will be on 15-minute intervals if not otherwise specified.

===========================
Calculate Scope 2 Emissions
===========================

It is straightforward to compute Scope 2 emissions after retrieving the carbon intensity array. 
As with electricity bill calculations, 
we show an example in `NumPy`, `CVXPY`, and `Pyomo` since EECO supports all three libraries.

NumPy
*****

Given a historical electricity consumption as a `NumPy` array:

.. code-block:: python

    # one month of 15-min intervals
    num_timesteps = len(carbon_intensity)
    # this is synthetic consumption data, but a user could provide real historical meter data
    consumption_arr = np.ones(num_timesteps)
    total_monthly_emissions, _ = emissions.calculate_grid_emissions(carbon_intensity, consumption_arr)

Note that we ignore the second value of the tuple returned by `calculate_grid_emissions`.
This entry in the tuple is reserved for the `Pyomo` model object.

Units will be handled automatically since `get_carbon_intensity` returns a `pint.Quantity`. 
However, a user could also explicitly define the units if using unitless arrays:

.. code-block:: python
  
    total_monthly_emissions, _ = emissions.calculate_grid_emissions(
          carbon_intensity.magnitude,
          consumption_arr,
          emission_units=carbon_intensity.units
    )


CVXPY
*****

If instead we want to optimize electricity consumption to minimize Scope 2 emissions, we can use a `CVXPY` variable:

.. code-block:: python

    consumption_var = cp.Variable(num_timesteps)
    total_monthly_emissions, _ = emissions.calculate_grid_emissions(
          carbon_intensity, consumption_var
    )

Note that we ignore the second value of the tuple returned by `calculate_grid_emissions`.
This entry in the tuple is reserved for the `Pyomo` model object.


Pyomo
*****

This optimization problem could be solved in `Pyomo` instead of `CVXPY`:

.. code-block:: python

    consumption_var = pyo.Var(
        range(num_timesteps), 
        initialize=np.zeros(num_timesteps), 
        bounds=(0, None)
    )
    total_monthly_emissions, model = costs.calculate_grid_emissions(
        carbon_intensity, consumption_var, model=model
    )

We must pass in and retrieve the `Pyomo` model object for the eletricity bill to be calculated correctly.

.. WARNING::

  For the `Pyomo` code to work properly, we require the `model` object has an attribute `t` that is the range of the time period.
  
  We usually set `model.t = range(model.T)` where `model.T = len(consumption_data_dict["electric"])`.


=====
Units
=====

EECO uses `Pint <https://pint.readthedocs.io/en/stable/>`_ to handle nit conversions automaitcally. 
The logic depends on the proper `emissions_units` and `consumption_units` arguments being provided.
Based on the most common data sources we have used, the consumption units are in kW
and emissions units in kg / MWh, so `consumption_units=u.kW` and `emissions_units=u.kg / u.MWh`.
This defaults to a 0.001 conversion factor.

The temporal resolution of the consumption data should be provided as a string. 
The default is 15-minute intervals, so `resolution="15m"`.

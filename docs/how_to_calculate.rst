.. contents::

.. WARNING::

  Site under construction!    
  Documentation incomplete :( 

.. _how-to-calculate:

************************************
How to Calculate Costs and Emissions
************************************

.. _calculate-tariff:

Electricity Tariffs
===================

The EEC package supports computing the electricity bill of a consumer with the following data:

  - Consumption data: `numpy` array, `CVXPY` variable, or `Pyomo` parameter/variable
  - Tariff sheet: `Pandas` array that can be loaded from our CSV format

This how-to guide assumes that you have already loaded the tariff sheet into a `Pandas.DataFrame` called `tariff_df`.
Further guidance on how to load the data can be found at :ref:`data-format-tariff`.

Create Charge Dictionary
************************

The first step to computing the cost of electricity is converting the tariff `DataFrame` to `dict`.
We will use the built-in function `get_charge_dict`:

.. code-block:: python

    start_date, end_date = datetime.datetime(2025, 5, 1), datetime.datetime(2025, 6, 1)
    charge_dict = costs.get_charge_dict(start_date, end_date, tariff_df, resolution="15m")

In this case, `start_date` and `end_date` must be of a datetime type (i.e., `datetime.datetime`, `numpy.datetime64`, or `pandas.Timestamp`).
Note that `end_date` is exclusive, so in the example above the `charge_dict` will be 1-month long.

The optional argument resolution should be used to specify the temporal resolution of the consumption data.
We use the default value of `"15m"` in this example.

Calculate Cost
**************

Next, we will calculate the cost for the given period (from `start_date` to `end_date`, *exclusively*). 
We show an example in `numpy`, `CVXPY`, and `Pyomo` since the EEC package supports all three libraries.

In numpy:

.. code-block:: python

    # this is synthetic consumption data, but a user could provide real historical meter data
    consumption_data_dict = {"electric": np.ones(96) * 100, "gas": np.ones(96)}
    result, _ = costs.calculate_cost(charge_dict, consumption_data_dict)

Note that we ignore the second value of the tuple returned by `calculate_cost`.
This entry in the tuple is reserved for the `Pyomo` model object.

In CVXPY:

.. code-block:: python

    result, _ = costs.calculate_cost(
        charge_dict,
        consumption_data_dict,
    )

Note that we ignore the second value of the tuple returned by `calculate_cost`.
This entry in the tuple is reserved for the `Pyomo` model object.

In Pyomo:

.. code-block:: python

    result, _ = costs.calculate_cost(
        charge_dict,
        consumption_data_dict,
        resolution=resolution,
        prev_demand_dict=prev_demand_dict,
        consumption_estimate=consumption_estimate,
        desired_utility=desired_utility,
        desired_charge_type=desired_charge_type,
    )

The above examples exclude some more advanced features available via optional arguments and flags, which are particularly useful for moving horizon optimization.
:ref:`how-to-advanced` offers a more complete overview of those advanced features.

:ref:`tutorial-cost` offers a more complete look at how to use this functionality in an optimization problem.

.. _calculate-emissions:

Scope 2 Emissions
=================

This package is not designed to calculate Scope 2 emissions that are a complete timeseries.
We feel that is simple enough that it does not warrant a function.

In numpy:

# TODO: insert code snippet

In CVXPY:

# TODO: insert code snippet

In Pyomo:

# TODO: insert code snippet

However, many data sources report emissions factors as monthly/hourly averages (:ref:`data-format-emissions`).
Our package is designed to unpack data in that format into a timeseries the same length as the consumption variable.

# TODO: example of using `calculate_grid_emissions`, `calculate_grid_emissions_cvx`, and `calculate_grid_emissions_pyo`

Get Carbon Intensity
********************

The `get_carbon_intensity` function can be used for those interested in getting the timeseries directly:

# TODO: example of using `get_carbon_intensity`

Units
=====

The EEC package uses `Pint <https://pint.readthedocs.io/en/stable/>`_ to handle nit conversions automaitcally. 
The logic depends on the proper `emissions_units` and `consumption_units` arguments being provided.
Based on the most common data sources we have used, the consumption units are in kW
and emissions units in kg / MWh, so `consumption_units=u.kW` and `emissions_units=u.kg / u.MWh`.
This defaults to a 0.001 conversion factor.

The temporal resolution of the consumption data should be provided as a string. 
The default is 15-minute intervals, so `resolution="15m"`.
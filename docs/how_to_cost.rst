.. contents::

.. _how-to-cost:

**********************
How to Calculate Costs
**********************

EECO supports computing the electricity bill of a consumer with the following data:

  - Consumption data: `NumPy` array, `CVXPY` variable, or `Pyomo` parameter/variable
  - Tariff sheet: `Pandas` dataframe that can be loaded from our CSV format

This how-to guide assumes that you have already loaded the tariff sheet into a `Pandas.DataFrame` called `tariff_df`.
Further guidance on how to load the data can be found at :ref:`data-format-tariff`.

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
    from eeco import costs

========================
Create Charge Dictionary
========================

The first step to computing the cost of electricity is converting the tariff `DataFrame` to `dict`.
We will use the built-in function `get_charge_dict`:

.. code-block:: python

    start_date, end_date = datetime.datetime(2025, 5, 1), datetime.datetime(2025, 6, 1)
    charge_dict = costs.get_charge_dict(start_date, end_date, tariff_df, resolution="15m")

In this case, `start_date` and `end_date` must be of a datetime type (i.e., `datetime.datetime`, `numpy.datetime64`, or `pandas.Timestamp`).
Note that `end_date` is exclusive, so in the example above the `charge_dict` will be 1-month long.

The optional argument `resolution` should be used to specify the temporal resolution of the consumption data
as a string in the from `<binsize><unit>`, 
where units are either `'m'` for minutes, `'h'` for hours, or `'d'` / `'D'` for days.
The default is `"15m"`, so the timeseries will be on 15-minute intervals if not otherwise specified.

==========================
Calculate Electricity Bill
==========================

Next, we will calculate the cost for the given period (from `start_date` to `end_date`, *exclusively*). 
We show an example in `NumPy`, `CVXPY`, and `Pyomo` since EECO supports all three libraries.

The below examples exclude some more advanced features available via optional arguments and flags, which are particularly useful for moving horizon optimization.
:ref:`how-to-advanced` offers a more complete overview of those advanced features.

:ref:`tutorial-cost` offers a more complete look at how to use this functionality in an optimization problem.

Basic Usage
***********

NumPy
=====

.. code-block:: python

    # one month of 15-min intervals
    num_timesteps = 24 * 4 * 31
    # this is synthetic consumption data, but a user could provide real historical meter data
    # Positive values represent imports, negative values represent exports in consumption data
    consumption_data_dict = {"electric": np.ones(num_timesteps) * 100, "gas": np.ones(num_timesteps))}
    total_monthly_bill, _ = costs.calculate_cost(charge_dict, consumption_data_dict)

Note that we ignore the second value of the tuple returned by `calculate_cost`.
This entry in the tuple is reserved for the `Pyomo` model object.

CVXPY
=====

.. code-block:: python

    consumption_data_dict = {"electric": cp.Variable(num_timesteps), "gas": cp.Variable(num_timesteps)}
    total_monthly_bill, _ = costs.calculate_cost(
        charge_dict, consumption_data_dict, consumption_estimate=sum(np.ones(num_timesteps) * 100)
    )

.. TIP::

  You must use the `consumption_estimate` argument when using an optimization variable for consumption
  in order to determine the appropriate charge tier of the customer.
  For `NumPy`, the charge tiers can be calculated directly from the data so the `consumption_estimate` is ignored.

Note that we ignore the second value of the tuple returned by `calculate_cost`.
This entry in the tuple is reserved for the `Pyomo` model object.

This cost would be the objective function of the optimization problem, 
but the user will still have to provide constraints to bound the cost minimization.
See the :ref:`cvx-cost` tutorial about how to :ref:`tutorial-cost` for more information!  

Pyomo
=====

.. code-block:: python

    consumption_data_dict = {
        "electric": pyo.Var(range(num_timesteps), initialize=np.zeros(num_timesteps), bounds=(0, None))
        "gas": pyo.Var(range(num_timesteps), initialize=np.zeros(num_timesteps), bounds=(0, None))
    }
    total_monthly_bill, model = costs.calculate_cost(
        charge_dict, consumption_data_dict, consumption_estimate=sum(np.ones(num_timesteps) * 100), model=model
    )

.. TIP::

  You must use the `consumption_estimate` argument when using an optimization variable for consumption
  in order to determine the appropriate charge tier of the customer. 
  For `NumPy`, the charge tiers can be calculated directly from the data so the `consumption_estimate` is ignored.

We must pass in and retrieve the `Pyomo` model object for the eletricity bill to be calculated correctly.
The tutorial on :ref:`pyo-cost` cost optimization has more examples of how to use the model object with the functions

.. WARNING::

  For the `Pyomo` code to work properly, we require the `model` object has an attribute `t` that is the range of the time period.
  
  We usually set `model.t = range(model.T)` where `model.T = len(consumption_data_dict["electric"])`.

Specify Resolution
******************

The temporal resolution of the consumption data should be provided as a string. 
The default is 15-minute intervals, so `resolution="15m"`.

.. code-block:: python

    charge_dict = costs.get_charge_dict(start_date, end_date, tariff_df, resolution="1h")
    num_timesteps = 24 * 31
    consumption_data_dict = {"electric": cp.Variable(num_timesteps), "gas": cp.Variable(num_timesteps)}
    total_monthly_bill, _ = costs.calculate_cost(
        charge_dict, 
        consumption_data_dict, 
        consumption_estimate=sum(np.ones(num_timesteps) * 100), 
        resolution="1h",
    )

Specify Utility
****************

Users can select between electric and natural gas utilties by using the `desired_utility` optional argument. 
The accepted arguments are `"electric"`, `"gas"`, or `None`.
By default, the combined costs across both utilities is calculated (i.e., `desired_utility=None`).

.. code-block:: python

    consumption_data_dict = {"electric": np.ones(num_timesteps) * 100, "gas": np.ones(num_timesteps))}
    monthly_elec_bill, _ = costs.calculate_cost(charge_dict, consumption_data_dict, desired_utility="electric")

Specify Charge Type
*******************

Users can select between customer, energy, and demand charges by using the `desired_charge_type` optional argument. 
The accepted arguments are `"customer"`, `"energy"`, `"demand"`, or `None`.
By default, the combined costs across both utilities is calculated (i.e., `desired_utility=desired_charge_type`).

.. code-block:: python

    consumption_data_dict = {"electric": np.ones(num_timesteps) * 100, "gas": np.ones(num_timesteps))}
    monthly_elec_bill, _ = costs.calculate_cost(charge_dict, consumption_data_dict, desired_charge_type="demand")

=====
Units
=====

EECO uses `Pint <https://pint.readthedocs.io/en/stable/>`_ to handle unit conversions automaitcally. 
The logic depends on the proper `electric_consumption_units` and `gas_consumption_units` arguments being provided.
The electric consumption units are in kW and gas consumption units in cubic meters per hour,
so `electric_consumption_units=u.kW` and `gas_consumption_units=u.m ** 3 / u.hour`,
to be consistent with our published natural gas tariff dataset (:ref:`data-format-tariff`).

For example, if `electric_consumption_units` are in megawatts instead of the default kilowatts
and `gas_consumption_units` are in cubic meters per day instead of per hour:

.. code-block:: python

    total_monthly_bill, _ = costs.calculate_cost(
        charge_dict, consumption_data_dict, electric_consumption_units=u.MW, gas_consumption_units=u.m**3/u.day
    )

================
Itemized Charges
================

The function `calculate_itemized_cost` will give you a breakdown of electricity, demand, and customer charges 
to analyze the customer's electricity bill in more detail.

.. code-block:: python

    consumption_data_dict = {"electric": np.ones(num_timesteps) * 100, "gas": np.ones(num_timesteps))}
    monthly_elec_bill, _ = costs.calculate_cost(charge_dict, consumption_data_dict, desired_charge_type="demand")
    itemized_cost_dict = costs.calculate_itemized_cost(charge_dict, consumption_data_dict)

The above example is quite simple, but you can use the same optional arguments that we demonstrated above, 
such as `resolution`, `desired_utility`, and `consumption_estimate`. 
(`desired_charge_type` is not an option since the dictionary uses `charge_type` as a key.)

=================
Advanced Features
=================

See :ref:`how-to-advanced` for an explanation of more advanced features, especially for performing moving horizon optimization.

If you have not done it already, we recommend the walkthrough to practice using this functions: :ref:`tutorial-cost`.
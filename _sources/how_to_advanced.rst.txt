.. contents::

.. _how-to-advanced:

****************************
How to Use Advanced Features
****************************

There are a few advanced features that can be used via flags in `calculate_cost`.
These features are particularly useful for moving horizon optimization. 
Check out :ref:`why-advanced` for more background as to why we recommend moving horizon optimization.

.. _prev-consumption:

How to Use `prev_demand_dict` and `prev_consumption_dict`
=========================================================

By default, `prev_demand_dict=None` and `prev_consumption_dict=None`. 
However, a user may want to optimize their energy bill starting partway through a billing period.
In this case, it is important to take into account previously consumed electricity when optimizing the electricity bill (:ref:`why-prev-consumption`).

To do so, simply provide the total energy consumption (in kWh, therms, or cubic meters) for each charge string from `costs.get_charge_dict`.
For example, using `billing_pge.csv`:

.. code-block:: python

    from eeco import costs

    tariff_path = "tests/data/input/billing_pge.csv"
    start_dt = np.datetime64("2024-07-10")
    end_dt = np.datetime64("2024-07-11")
    billing_data = pd.read_csv(tariff_path)
    charge_dict = costs.get_charge_dict(start_dt, end_dt, billing_data)

    # one day of 15-min intervals
    num_timesteps = 96

    # this is just a CVXPY variable, but a user would provide constraints to the optimization problem
    consumption_data_dict = {"electric": cp.Variable(num_timesteps), "gas": cp.Variable(num_timesteps)}

    prev_consumption_dict = {
        "gas_energy_0_20240710_20240710_0": 960,
        "gas_energy_0_20240710_20240710_5000": 0,
        "electric_energy_0_20240710_20240710_0": 34000,
        "electric_energy_1_20240710_20240710_0": 14000,
        "electric_energy_2_20240710_20240710_0": 24000,
        "electric_energy_3_20240710_20240710_0": 14000,
        "electric_energy_4_20240710_20240710_0": 14000,
        "electric_energy_5_20240710_20240710_0": 19200,
        "electric_energy_6_20240710_20240710_0": 0,
        "electric_energy_7_20240710_20240710_0": 0,
        "electric_energy_8_20240710_20240710_0": 0,
        "electric_energy_9_20240710_20240710_0": 0,
        "electric_energy_10_20240710_20240710_0": 0,
        "electric_energy_11_20240710_20240710_0": 0,
        "electric_energy_12_20240710_20240710_0": 0,
        "electric_energy_13_20240710_20240710_0": 0,
    }

    # see below sections with more detail on how to define the `consumption_estimate``
    # this is just synthetic data, but a real estimate from the facility historical data should be used
    consumption_estimate={"gas": sum(np.ones(num_timesteps)), "electric": sum(np.ones(num_timesteps) * 100)}

    total_monthly_bill, _ = costs.calculate_costs(
        charge_dict,
        consumption_data_dict,
        prev_consumption_dict=prev_consumption_dict,
        consumption_estimate=consumption_estimate
    )

.. _consumption-est:

How to Use `consumption_estimate`
=================================

By default `consumption_estimate=0`, meaning that it will be ignored.
This behavior is not an issue for most electricity and natural gas tariffs.
However, if a tariff has charge tiers, then a consumption estimate is required to estimate which tier the customer will be subject to.
There are four different ways to input `consumption_estimate`:

  - `dict` of `float`, `int`, or `array` with keys "electric" and "gas"
    - Within each dictionary entry, the below rules for `array` and `float`/`int` are followed.
  - `array`
    - If an `array`, the units are assumed to be in power (i.e., kW, therm / hr, or cubic meter / hr).
    - The array is converted to kWh, therms, or cubic meters before being passed to `calculate_energy_cost`,
    but passed in the original units to `calculate_demand_cost`.
  - `float` or `int`
    - If a `float` or `int`, the units are assumed to be in energy (i.e., kWh, therms, or cubic meters).
    - The `consumption_estimate` is divided by the number of timesteps in the simulation to estimate the maximum consumption
    that is passed into `calculate_demand_cost`.

Here are examples of each method in code, assuming that tariff data has already been loaded as `tariff_df`:

.. code-block:: python

    from eeco import costs

    # load necessary data 
    start_dt = np.datetime64("2024-07-10")
    end_dt = np.datetime64("2024-07-11")
    charge_dict = costs.get_charge_dict(start_dt, end_dt, tariff_df)
    num_timesteps = 96

    # this is just a CVXPY variable, but a user would provide constraints to the optimization problem
    consumption_data_dict = {"electric": cp.Variable(num_timesteps), "gas": cp.Variable(num_timesteps)}

    # `consumptione_estimate` as a dict of floats and/or arrays
    total_monthly_bill, _ = costs.calculate_costs(
        charge_dict,
        consumption_data_dict,
        consumption_estimate={"gas": np.ones(num_timesteps), "electric": sum(np.ones(num_timesteps) * 100)}
    )

    # `consumptione_estimate` as an array
    total_monthly_bill, _ = costs.calculate_costs(
        charge_dict,
        consumption_data_dict,
        consumption_estimate=np.ones(num_timesteps),
        desired_utility="gas"
    )

    # `consumptione_estimate` as a float or int
    total_monthly_bill, _ = costs.calculate_costs(
        charge_dict,
        consumption_data_dict,
        consumption_estimate=sum(np.ones(num_timesteps) * 100),
        desired_utility="electric"
    )

Note that consumption estimate is for the simulation period only since :ref:`prev-consumption` will take into account 
the previous consumption during this billing period in conjunction with `consumption_estimate` to estimate the charge tier.

.. _scale-demand:

How to Use `demand_scale_factor`
================================

By default `demand_scale_factor=1`, meaning that there will be no modifications applied to the demand or energy charges.
The purpose of the scale factor is to modify the demand charges proportional to energy charges when performing moving horizon optimization.

There are various heuristics that could be used to calculate the scale factor (see :ref:`why-scale-demand`), 
but for now let's assume that we just want to scale the demand charge down by the length of the horizon window proportional to billing period.

.. code-block:: python

    from eeco import costs
    
    # load necessary data 
    start_dt = np.datetime64("2024-07-10")
    end_dt = np.datetime64("2024-07-11")
    charge_dict = costs.get_charge_dict(start_dt, end_dt, tariff_df)
    num_timesteps_horizon = 96
    num_timesteps_billing = 96 * 31

    # this is just a CVXPY variable, but a user would provide constraints to the optimization problem
    consumption_data_dict = {"electric": cp.Variable(num_timesteps), "gas": cp.Variable(num_timesteps)}

    total_monthly_bill, _ = costs.calculate_costs(
        charge_dict,
        consumption_data_dict,
        demand_scale_factor=num_timesteps_horizon/num_timesteps_billing
    )

.. _varstr-alias:

How to Use `varstr_alias_func`
==============================

The software creates new variables when building the optimization problem.
At times, users want control over what variable names are assigned to the new variables.
By default, `default_varstr_alias_func` creates variable names of the following format:

.. code-block:: python 

    def default_varstr_alias_func(
        utility, charge_type, name, start_date, end_date, charge_limit
    ):
        return f"{utility}_{charge_type}_{name}_{start_date}_{end_date}_{charge_limit}"

However, users can pass in their own custom variable name function into `calculate_costs`. 
For example, to change "gas" to "ng" in all the variable names:

.. code-block:: python 

    from eeco import costs

    def custom_varstr_alias_func(
        utility, charge_type, name, start_date, end_date, charge_limit
    ):
        if utility == "gas":
            utility = "ng"
        return f"{utility}_{charge_type}_{name}_{start_date}_{end_date}_{charge_limit}"

    # load necessary data 
    start_dt = np.datetime64("2024-07-10")
    end_dt = np.datetime64("2024-07-11")
    charge_dict = costs.get_charge_dict(start_dt, end_dt, tariff_df)
    num_timesteps_horizon = 96

    # this is just a CVXPY variable, but a user would provide constraints to the optimization problem
    consumption_data_dict = {"electric": cp.Variable(num_timesteps), "gas": cp.Variable(num_timesteps)}

    total_monthly_bill, _ = costs.calculate_costs(
        charge_dict,
        consumption_data_dict,
        varstr_alias_func=custom_varstr_alias_func,
    )
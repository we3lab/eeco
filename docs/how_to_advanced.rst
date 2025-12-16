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

The `demand_scale_factor` parameter allows you to scale demand charges to reflect shorter optimization horizons or to prioritize demand differently across sequential optimization horizons.

By default, `demand_scale_factor=1.0`. Use values less than 1.0 when solving for a subset of the billing period, or to adjust demand charge weighting in sequential optimization.

When `demand_scale_factor < 1.0`, demand charges are proportionally reduced to reflect the shorter optimization horizon. This is useful for:
- Moving horizon optimization where you solve for sub-periods of the billing cycle
- Sequential optimization where you want to reduce demand charge weighting as time goes on in the month

.. code-block:: python

    from electric_emission_cost import costs
    
    # E.g. solving for 3 days out of a 30-day billing period
    demand_scale_factor = 3 / 30
    
    result, model = costs.calculate_cost(
        charge_dict,
        consumption_data,
        demand_scale_factor=demand_scale_factor
        # ...
    )

For more details on applying the sequential optimization strategy, see:

    Bolorinos, J., Mauter, M.S. & Rajagopal, R. Integrated Energy Flexibility Management at Wastewater Treatment Facilities. *Environ. Sci. Technol.* **57**, 46, 18362–18371 (2023). DOI: [10.1021/acs.est.3c00365](https://doi.org/10.1021/acs.est.3c00365)

In `bibtex` format:

.. code-block:: bibtex

   @Article{Bolorinos2023,
   author={Bolorinos, Jose
   and Mauter, Meagan S.
   and Rajagopal, Ram},
   title={Integrated Energy Flexibility Management at Wastewater Treatment Facilities},
   journal={Environmental Science & Technology},
   year={2023},
   month={Jun},
   day={16},
   volume={57},
   number={46},
   pages={18362--18371},
   doi={10.1021/acs.est.3c00365},
   url={https://doi.org/10.1021/acs.est.3c00365}
   }


.. _decompose-exports:

How to Use `decomposition_type`
===============================

The `decomposition_type` parameter allows you to decompose consumption data into positive (imports) and negative (exports) components. This is useful when you have export charges or credits in your rate structure.
Options include:

- Default `None`
- `"binary_variable"`: To be implemented
- `"absolute_value"`

.. code-block:: python

    from electric_emission_cost import costs
    
    # Example with export charges
    charge_dict = {
        "electric_export_0_2024-07-10_2024-07-10_0": np.ones(96) * 0.025,
    }
    
    consumption_data = {
        "electric": np.concatenate([np.ones(48) * 10, -np.ones(48) * 5]),
        "gas": np.ones(96),
    }
    
    # Decompose consumption into imports and exports
    result, model = costs.calculate_cost(
        charge_dict,
        consumption_data,
        decomposition_type="absolute_value"
    )

When decomposition_type is not None the function creates separate variables for positive consumption (imports) and negative consumption (exports)
and applies export charges only to the export component.
For Pyomo models, decomposition_type adds a constraint total_consumption = imports - exports


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
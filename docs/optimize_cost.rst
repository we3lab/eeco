.. contents::

.. _tutorial-cost:

**************************
Optimize Electricity Costs
**************************

This tutorial will walkthrough how to optimize electricity bill savings for a simple battery model in either :ref:`cvx-cost` or :ref:`pyo-cost` by going through the following steps:

  #. Load an electricity tariff spreadsheet

     - We publish a nationwide tariff dataset: https://github.com/we3lab/industrial-electricity-tariffs
     - In this case, we use `tariff.csv <https://github.com/we3lab/electric-emission-cost/blob/main/electric_emission_cost/data/tariff.csv>`_ from the EEC `data` folder
  #. Configure an optimization model of the electricity consumer with system constraints
  
     - Consider using flexibility metrics to encode system constraints (:ref:`tutorial-metrics`)!
     - The models are presented step-by-step to demonstrate the model building process, 
       but the complete models are available in the `examples` folder:

       - `pyomo_battery_model.py <https://github.com/we3lab/electric-emission-cost/blob/main/examples/pyomo_battery_model.py>`_
       - `cvxpy_battery_model.py <https://github.com/we3lab/electric-emission-cost/blob/main/examples/cvxpy_battery_model.py>`_
  #. Create an objective function of electricity costs based on this tariff sheet
  #. Minimize the electriciy costs of this consumer given the system constraints and base load consumption
  #. Display the results to validate that the optimization is correct

.. _cvx-cost:

CVXPY
=====

0. Import dependencies

.. code-block:: python
   
    import datetime
    import cvxpy as cp
    import pandas as pd
    from electric_emission_cost import costs 

1. Load an electricity tariff spreadsheet

.. code-block:: python
   
    path_to_tariffsheet = "electric_emission_cost/data/tariff.csv"
    rate_df = pd.read_csv(path_to_tariffsheet, sep=",")
   
    # get the charge dictionary
    charge_dict = costs.get_charge_dict(
        datetime.datetime(2023, 4, 9), datetime.datetime(2023, 4, 11), rate_df, resolution="1m"
    )

We are going to evaluate the electricity consumption from only April 9th to April 10th since that is where our 
synthetic data comes from (https://github.com/we3lab/electric-emission-cost/blob/main/electric_emission_cost/data/consumption.csv).
You will also see that it is in 1-minute intervals, hence `resolution="1m"`.
If you print `charge_dict`, then you should get the following:

.. code-block:: python

    {'electric_customer_0_2022-07-01_2022-07-31_0': array([666.65]),
    'electric_energy_0_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_1_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_2_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_3_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_4_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_5_2022-07-01_2022-07-31_0': array([0.0254538, 0.0254538, 0.0254538, ..., 0.       , 0.       ,
        0.       ], shape=(2976,)),
    'electric_energy_6_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_7_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_8_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_9_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_10_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_11_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_12_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_13_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_14_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_energy_15_2022-07-01_2022-07-31_0': array([0.       , 0.       , 0.       , ..., 0.0254538, 0.0254538,
        0.0254538], shape=(2976,)),
    'electric_energy_16_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_demand_0_2022-07-01_2022-07-31_0': array([19.79, 19.79, 19.79, ...,  0.  ,  0.  ,  0.  ], shape=(2976,)),
    'electric_demand_1_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_demand_2_2022-07-01_2022-07-31_0': array([0., 0., 0., ..., 0., 0., 0.], shape=(2976,)),
    'electric_demand_3_2022-07-01_2022-07-31_0': array([ 0.  ,  0.  ,  0.  , ..., 19.79, 19.79, 19.79], shape=(2976,))}

2. Configure an optimization model of the electricity consumer with system constraints

.. code-block:: python

    # load historical consumption data
    load_df = pd.read_csv("electric_emission_cost/data/consumption.csv", parse_dates=["Datetime"])

    # set battery parameters
    # create variables for battery total energy, max charge and discharge power, and SOC limits
    total_capacity = 10 # kWh
    min_soc = 0 
    max_soc = 1
    init_soc = 0.5
    fin_soc = 0.5
    max_discharge = 5 # kW
    max_charge = 5 # kW
    T = len(load_df["Datetime"])
    delta_t = ((load_df.iloc[-1]["Datetime"] - load_df.iloc[0]["Datetime"]) / T) / datetime.timedelta(hours=1)

    # initialize variables
    battery_output_kW = cp.Variable(T)
    battery_soc = cp.Variable(T+1)
    grid_demand_kW = cp.Variable(T)

    # set constraints
    constraints = [
        battery_output_kW >= -max_discharge,
        battery_output_kW <= max_charge,
        battery_soc >= min_soc,
        battery_soc <= max_soc,
        battery_soc[0] == init_soc,
        battery_soc[T] == fin_soc,
        grid_demand_kW >= 0
    ]
    for t in range(T):
        constraints += [
            battery_soc[t+1] == battery_soc[t] + (battery_output_kW[t] * delta_t) / total_capacity,
            grid_demand_kW[t] == load_df.iloc[t]["Load [kW]"] + battery_output_kW[t]
        ]

This is a standard battery model with energy (i.e., total charge) and power (i.e., discharge/charge rate) constraints.
The round-trip efficiency is 1.0 since there is no penalty applied when discharging the battery, 
but that's fine for these demonstration purposes.

3. Create an objective function of electricity costs based on this tariff sheet

.. code-block:: python

    # requires a consumption dictionary in case there is natural gas in addition to electricity
    consumption_data_dict = {"electric": grid_demand_kW}
    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    obj, _ = costs.calculate_cost(
        charge_dict,
        {"electric": grid_demand_kW},
        resolution="1m",
        consumption_estimate=load_df["Load [kW]"].sum(),
        desired_utility="electric",
    )

The charge and consumption dictionaries are relatively straightforward: 
`charge_dict` comes from the EEC package and `consumption_data_dict` is either an optimization variable or
numpy array (in the case of historical analysis).
The only caveat would be that an entry with key "gas" must be included to analzye natural gas consumption.

Carefully note that the function `calculate_cost` returns a tuple. 
The second entry of the tuple is for Pyomo, so it can be ignored since we are using CVXPY.

The `resolution` argument represents the temporal granularity of the data in string format. 
The default value is "15m" for 15-minute intervals, but our consumption data is on 1-minute intervals,
so we use `resolution="1m"` (just like with `charge_dict`).

For this simple example the `prev_demand_dict`, `prev_consumption_dict`, `demand_scale_factor`, `desired_charge_type`, 
and `varstr_alias_func` have not been used. More information on how to use those flags is available in :ref:`how-to-advanced`.

4. Minimize the electriciy costs of this consumer given the system constraints and base load consumption

.. code-block:: python

    # solve the CVX problem (objective variable should be named obj)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()

5. Display the results to validate that the optimization is correct

Always compute the ex-post cost using numpy due to the convex relaxations that we apply in our optimization code:

.. code-block:: python

    # requires a consumption dictionary in case there is natural gas in addition to electricity
    consumption_data_dict = {"electric": grid_demand_kW.value}
    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    result, _ = costs.calculate_cost(
        charge_dict,
        consumption_data_dict,
        resolution="1m",
        desired_utility="electric",
    )

Note that the `consumption_estimate` optional argument is not needed because the electricity consumption is a numpy array instead of an optimization variable.

Below are a few simple plots to validate our results

# TODO: plot results

.. _pyo-cost:

Pyomo
=====


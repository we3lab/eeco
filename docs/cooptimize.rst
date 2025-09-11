.. contents::

.. _tutorial-cooptimize:

*******************************
Co-optimize Costs and Emissions
*******************************

This tutorial will walk through how to co-optimize electricity bill savings and Scope 2 emissions for a simple battery model in either :ref:`cvx-coopt` or :ref:`pyo-coopt` by going through the following steps:

  #. Load an electricity tariff spreadsheet and Scope 2 emissions data
  #. Configure an optimization model of the electricity consumer with system constraints
  
     - Consider using flexibility metrics to encode system constraints (:ref:`tutorial-metrics`)!
     - The models are presented step-by-step to demonstrate the model building process, 
       but the complete models are available in the `examples` folder:

       - `pyomo_battery_model.py <https://github.com/we3lab/eeco/blob/main/examples/pyomo_battery_model.py>`_
       - `cvxpy_battery_model.py <https://github.com/we3lab/eeco/blob/main/examples/cvxpy_battery_model.py>`_
  #. Create an objective function of electricity costs and emissions with carbon cost
  #. Minimize the electricity costs and emissions of this consumer, given the system constraints, base load consumption, and carbon cost
  #. Display the results to validate that the optimization is correct


.. _cvx-coopt:

CVXPY
=====

0. Import dependencies

.. code-block:: python
   
    import datetime
    import cvxpy as cp
    import pandas as pd
    import matplotlib.pyplot as plt
    from eeco.units import u
    from eeco import emissions
    from eeco import costs

1. Load a electricity tariff and Scope 2 emissions spreadsheet

.. code-block:: python
   
    path_to_emissions_sheet = "eeco/data/emissions.csv"
    emission_df = pd.read_csv(path_to_emissions_sheet, sep=",")
   
    # get the carbon intensity
    carbon_intensity = emissions.get_carbon_intensity(
        datetime.datetime(2023, 4, 9), datetime.datetime(2023, 4, 11), emission_df, resolution="1m"
    )

    path_to_tariff_sheet = "eeco/data/tariff.csv"
    tariff_df = pd.read_csv(path_to_tariff_sheet, sep=",")
   
    # get the charge dictionary
    charge_dict = costs.get_charge_dict(
        datetime.datetime(2023, 4, 9), datetime.datetime(2023, 4, 11), tariff_df, resolution="1m"
    )

We are going to evaluate the electricity consumption from only April 9th to April 10th since that is where our 
synthetic data comes from (https://github.com/we3lab/eeco/blob/main/eeco/data/consumption.csv).
You will also see that it is in 1-minute intervals, hence `resolution="1m"`.

2. Configure an optimization model of the electricity consumer with system constraints

.. code-block:: python

    # load historical consumption data
    load_df = pd.read_csv("eeco/data/consumption.csv", parse_dates=["Datetime"])

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

3. Create an objective function of electricity costs and Scope 2 emissions with carbon cost

.. code-block:: python

    # dollars per kg CO2 - converted from $192/metric ton CO2-eq
    cost_of_carbon = 0.192

    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    emissions_obj, _ = emissions.calculate_grid_emissions(
        carbon_intensity,
        grid_demand_kW,
        resolution="1m",
        consumption_units=u.kW
    )
    # requires a consumption dictionary in case there is natural gas in addition to electricity
    consumption_data_dict = {"electric": grid_demand_kW}
    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    cost_obj, _ = costs.calculate_cost(
        charge_dict,
        {"electric": grid_demand_kW},
        resolution="1m",
        consumption_estimate=load_df["Load [kW]"].sum(),
        desired_utility="electric",
    )
    obj = cost_obj + emissions_obj * cost_of_carbon

4. Minimize the costs and emissions of this consumer, given the system constraints, base load consumption, and carbon cost

.. code-block:: python

    # solve the CVX problem (objective variable should be named obj)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()

5. Display the results to validate that the optimization is correct

Always compute the ex-post cost using numpy due to the convex relaxations that we apply in our optimization code:

.. code-block:: python

    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    baseline_electricity_emissions, _ = emissions.calculate_grid_emissions(
        carbon_intensity,
        load_df["Load [kW]"].values,
        resolution="1m",
        consumption_units=u.kW
    )
    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    optimized_electricity_emissions, _ = emissions.calculate_grid_emissions(
        carbon_intensity,
        grid_demand_kW.value,
        resolution="1m",
        consumption_units=u.kW
    )
    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    baseline_electricity_cost, _ = costs.calculate_cost(
        charge_dict,
        {"electric": load_df["Load [kW]"].values},
        resolution="1m",
        desired_utility="electric",
    )
    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    optimized_electricity_cost, _ = costs.calculate_cost(
        charge_dict,
        {"electric": grid_demand_kW.value},
        resolution="1m",
        desired_utility="electric",
    )

    total_baseline_cost = baseline_electricity_cost + cost_of_carbon * baseline_electricity_emissions.magnitude
    total_optimized_cost = optimized_electricity_cost + cost_of_carbon * optimized_electricity_emissions.magnitude

If we print our results, we confirm that the optimal electricity profile has daily emissions of
10.57 kg CO:sub:`2`-eq, 0.12 kg CO:sub:`2`-eq less than the baseline emissions of 10.69 kg CO:sub:`2`-eq.
It simultaneously lowers daily costs by $61.48 from $765.29 to $703.81.

.. code-block:: python

    >>>print(f"Baseline Scope 2 Emissions: {baseline_electricity_emissions:.2f} kg CO_2-eq")
    Baseline Scope 2 Emissions: 10.69 kilogram kg CO_2-eq
    >>>print(f"Optimized Scope 2 Emissions: {optimized_electricity_emissions:.2f} kg CO_2-eq")
    Optimized Scope 2 Emissions: 10.57 kilogram kg CO_2-eq
    >>>print(f"Baseline Electricity Costs: ${baseline_electricity_cost:.2f}")
    Baseline Electricity Costs: $765.29
    >>>print(f"Optimized Electricity Costs: ${optimized_electricity_cost:.2f}")
    Optimized Electricity Costs: $703.81
    >>>print(f"Total Baseline Cost w/ $192/ton CO2-eq: ${total_baseline_cost:.2f}")
    Total Baseline Cost w/ $192/ton CO2-eq: $767.34
    >>>print(f"Total Optimized Cost w/ $192/ton CO2-eq: ${total_optimized_cost:.2f}")
    Total Optimized Cost w/ $192/ton CO2-eq: $705.84
    
We could make similar plots to :ref:`tutorial-cost` and :ref:`tutorial-emit`, but we omitted them from these instructions for the sake of space.

.. _pyo-coopt:

Pyomo
=====

0. Import dependencies

.. code-block:: python
   
    import datetime
    import numpy as np
    import pandas as pd
    import pyomo.environ as pyo
    import matplotlib.pyplot as plt
    from eeco.units import u
    from eeco import costs
    from eeco import emissions
    from examples.pyomo_battery_model import BatteryPyomo

1. Load a electricity tariff and Scope 2 emissions spreadsheet

.. code-block:: python
   
    path_to_emissions_sheet = "eeco/data/emissions.csv"
    emission_df = pd.read_csv(path_to_emissions_sheet, sep=",")
   
    # get the carbon intensity
    carbon_intensity = emissions.get_carbon_intensity(
        datetime.datetime(2022, 7, 1), datetime.datetime(2022, 8, 1), emission_df, resolution="15m"
    )

    path_to_tariffsheet = "eeco/data/tariff.csv"
    tariff_df = pd.read_csv(path_to_tariffsheet, sep=",")
   
    # get the charge dictionary
    charge_dict = costs.get_charge_dict(
        datetime.datetime(2022, 7, 1), datetime.datetime(2022, 8, 1), tariff_df, resolution="15m"
    )

We are going to evaluate the electricity consumption for the entire month of July 2022.
Below we will create synthetic `baseload` data for this month with 15-minute resolution, so `resolution="15m"`.

2. Configure an optimization model of the electricity consumer with system constraints

We rely on the virtual battery model in `pyomo_battery_model.py <https://github.com/we3lab/eeco/blob/main/examples/pyomo_battery_model.py>`_.
We're going to stick to the electricity cost calculation details, but we encourage you to go check out the code to better understand the model.

.. code-block:: python

    # Define the parameters for the battery model
    battery_params = {
        "start_date": "2022-07-01 00:00:00",
        "end_date": "2022-08-01 00:00:00",
        "timestep": 0.25,   # 15 minutes defined in hours
        "rte": 0.86,
        "energycapacity": 100,
        "powercapacity": 50,
        "soc_min": 0.05,
        "soc_max": 0.95,
        "soc_init": 0.5,
    }

    # Create a sample baseload profile based on a sine wave
    baseload = np.sin(np.linspace(0, 4 * np.pi, 96))*100 + 1000 + np.random.normal(0, 10, 96)

    # Create an instance of the BatteryOpt class
    battery = BatteryPyomo(battery_params, baseload, baseload_repeat=True)

    # create the model on the instance battery
    battery.create_model()

The above code initializes the battery model with flexibility metrics like round-trip efficiency (RTE), 
power capacity, and energy capacity.

3. Create an objective function of electricity costs and Scope 2 emissions with carbon cost

.. code-block:: python

    # dollars per kg CO2 - converted from $192/metric ton CO2-eq
    cost_of_carbon = 0.192

    # compute Scope 2 emissions using Pyomo variable
    battery.model.emissions, battery.model = emissions.calculate_grid_emissions(
        carbon_intensity,
        battery.model.net_facility_load,
        resolution="15m",
        consumption_units=u.kW,
        model=battery.model
    )

    # monthly total consumption - divided by 4 because of 15-min resolution
    consumption_estimate = sum(baseload) / 4
    # this example tariff only has electric utility types so we do not pass the gas key
    consumption_data_dict = {"electric": battery.model.net_facility_load}

    # use the built-in helper function to add more terms to the objective
    battery.model = costs.build_pyomo_costing(
        charge_dict,
        consumption_data_dict,
        battery.model,
        resolution="15m",
        consumption_estimate=consumption_estimate,
        desired_utility="electric",
        additional_objective_terms=[cost_of_carbon * battery.model.emissions]
    )

4. Minimize the costs and emissions of this consumer, given the system constraints, base load consumption, and carbon cost

.. code-block:: python

    # use the glpk solver to solve the model - (any pyomo-supported LP solver will work here)
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(battery.model, tee=False) # turn tee=True to see solver output

5. Display the results to validate that the optimization is correct

Always compute the ex-post cost using numpy due to the convex relaxations that we apply in our optimization code:

.. code-block:: python

    # retrieve outputs from Pyomo model
    net_load = np.array([battery.model.net_facility_load[t].value for t in battery.model.t])
    baseload = np.array([battery.model.baseload[t] for t in battery.model.t])

    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    baseline_electricity_emissions, _ = emissions.calculate_grid_emissions(
        carbon_intensity,
        baseload,
        resolution="15m",
        consumption_units=u.kW
    )
    optimized_electricity_emissions = pyo.value(battery.model.emissions)

    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    baseline_electricity_cost, _ = costs.calculate_cost(
        charge_dict,
        {"electric": baseload},
        resolution="15m",
        desired_utility="electric",
    )
    # NOTE: second entry of the tuple can be ignored since it's for Pyomo
    optimized_electricity_cost, _ = costs.calculate_cost(
        charge_dict,
        {"electric": net_load},
        resolution="15m",
        desired_utility="electric",
    )

total_baseline_cost = baseline_electricity_cost + cost_of_carbon * baseline_electricity_emissions.magnitude
    total_optimized_cost = optimized_electricity_cost + cost_of_carbon * optimized_electricity_emissions

If we print our results, we confirm that the optimal electricity profile has monthly emissions of
276,142.32 kg CO:sub:`2`-eq, 111.82 kg CO:sub:`2`-eq less than the baseline emissions of 276,254.14 kg CO:sub:`2`-eq.
It simultaneously lowers monthly costs by $2,901.69 from $11,6269.08 to $11,3367.39.

.. code-block:: python

    >>>print(f"Baseline Scope 2 Emissions: {baseline_electricity_emissions:.2f} kg CO_2-eq")
    Baseline Scope 2 Emissions: 276254.14 kilogram kg CO_2-eq
    >>>print(f"Optimized Scope 2 Emissions: {optimized_electricity_emissions:.2f} kg CO_2-eq")
    Optimized Scope 2 Emissions: 276142.32 kilogram kg CO_2-eq
    >>>print(f"Baseline Electricity Costs: ${baseline_electricity_cost:.2f}")
    Baseline Electricity Costs: $116269.08
    >>>print(f"Optimized Electricity Costs: ${optimized_electricity_cost:.2f}")
    Optimized Electricity Costs: $113367.39
    >>>print(f"Total Baseline Cost w/ $192/ton CO2-eq: ${total_baseline_cost:.2f}")
    Total Baseline Cost w/ $192/ton CO2-eq: $169309.88
    >>>print(f"Total Optimized Cost w/ $192/ton CO2-eq: ${total_optimized_cost:.2f}")
    Total Optimized Cost w/ $192/ton CO2-eq: $166386.71
    
We could make similar plots to :ref:`tutorial-cost` and :ref:`tutorial-emit`, but we omitted them from these instructions for the sake of space.
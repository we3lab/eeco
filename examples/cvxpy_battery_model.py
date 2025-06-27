import os
import numpy as np
import cvxpy as cp
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
max_charge = 4 # kW
peak_price = 0.43 # $ / kWh
off_peak_price = 0.2 # $ / kWh

# initialize variables
battery_output_kW = cp.Variable(T)
battery_soc = cp.Variable(T+1)
grid_demand_kW = cp.Variable(T)

# TODO: create_charge_dict, etc etc.
# obj = 

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

# solve the CVX problem (objective variable should be named obj)
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve()
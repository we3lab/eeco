import os
import datetime
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
from electric_emission_cost import costs 

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load tariff data
path_to_tariffsheet = "electric_emission_cost/data/tariff.csv"
tariff_df = pd.read_csv(path_to_tariffsheet, sep=",")
   
# get the charge dictionary
charge_dict = costs.get_charge_dict(
    datetime.datetime(2023, 4, 9), datetime.datetime(2023, 4, 11), tariff_df, resolution="1m"
)

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

# requires a consumption dictionary in case there is natural gas in addition to electricity
consumption_data_dict = {"electric": grid_demand_kW}
# NOTE: second entry of the tuple can be ignored since it's for Pyomo
obj, _ = costs.calculate_cost(
    charge_dict,
    consumption_data_dict,
    resolution="1m",
    consumption_estimate=load_df["Load [kW]"].sum(),
    desired_utility="electric",
)
    
# solve the CVX problem (objective variable should be named obj)
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve()

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

print(f"Baseline Electricity Cost: ${baseline_electricity_cost:.2f}")
print(f"Optimized Electricity Cost: ${optimized_electricity_cost:.2f}")

# create a subset of the charge_df for energy and demand charges
charge_df = costs.get_charge_df(datetime.datetime(2023, 4, 9), datetime.datetime(2023, 4, 11), tariff_df, resolution="1m")
charge_df.head()

energy_charge_df = charge_df.filter(like="energy")
demand_charge_df = charge_df.filter(like="demand")

# sum across all energy charges
total_energy_charge = energy_charge_df.sum(axis=1)

# plot the model outputs
fig, ax= plt.subplots()
ax.step(charge_df["DateTime"], grid_demand_kW.value, color="C0", lw=2, label="Net Load")
ax.step(charge_df["DateTime"], load_df["Load [kW]"].values, color="k", lw=1, ls='--', label="Baseload")
ax.set(xlabel="DateTime", ylabel="Power (kW)", xlim=(datetime.datetime(2023, 4, 9), datetime.datetime(2023, 4, 11)))
plt.xticks(rotation=45)
fig.tight_layout()
plt.legend()
plt.savefig("cvx-model-out.png")

# plot the battery charge
fig, ax = plt.subplots()
ax.step(charge_df["DateTime"], battery_soc.value[1:], color="C1", lw=2, label="Battery SOC")
ax.set(xlabel="Time", ylabel="Battery SOC", ylim=[0,1], xlim=(datetime.datetime(2023, 4, 9), datetime.datetime(2023, 4, 11)))
plt.xticks(rotation=45)
fig.tight_layout()
plt.savefig("cvx-battery-soc.png")
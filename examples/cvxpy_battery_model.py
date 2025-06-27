import os
import timedelta
import cvxpy as cp
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load tariff data
path_to_tariffsheet = "electric_emission_cost/data/tariff.csv"
rate_df = pd.read_csv(path_to_tariffsheet, sep=",")
   
# get the charge dictionary
charge_dict = costs.get_charge_dict(
    battery.start_dt, battery.end_dt, rate_df, resolution="15m"
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
delta_t = ((df.iloc[-1]["Datetime"] - df.iloc[0]["Datetime"]) / T) / timedelta(hours=1)

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

consumption_data_dict = {"electric": grid_demand_kW}
obj, _ = costs.calculate_cost(
    charge_dict,
    ,
    resolution="15m",
    prev_demand_dict=None,
    consumption_estimate=sum(grid_demand_kW),
    desired_utility="electric",
    desired_charge_type=None,
)
    
# solve the CVX problem (objective variable should be named obj)
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve()

result, _ = costs.calculate_cost(
    charge_dict,
    consumption_data_dict,
    resolution=resolution,
    prev_demand_dict=prev_demand_dict,
    consumption_estimate=consumption_estimate,
    desired_utility=desired_utility,
    desired_charge_type=desired_charge_type,
)
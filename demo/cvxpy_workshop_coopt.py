from eeco import emissions
# get the carbon intensity
carbon_intensity = emissions.get_carbon_intensity(
    start_dt, end_dt, emission_df, resolution="1m"
)
emissions_obj, _ = emissions.calculate_grid_emissions(
    carbon_intensity,
    grid_demand_kW,
    resolution="1m",
    consumption_units=u.kW
)
cost_obj, _ = costs.calculate_cost(
    charge_dict,
    {"electric": grid_demand_kW},
    resolution="1m",
    consumption_estimate=load_df["Load [kW]"].sum(),
    desired_utility="electric",
)
obj = cost_obj + emissions_obj * cost_of_carbon
# solve the CVX problem 
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve()
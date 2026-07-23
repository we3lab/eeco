# requires a consumption dictionary in case there is natural gas
consumption_data_dict = {"electric": grid_demand_kW}
itemized_cost, _ = costs.calculate_itemized_cost(
    charge_dict,
    {"electric": grid_demand_kW},
    resolution="1m",
    consumption_estimate=load_df["Load [kW]"].sum(),
    desired_utility="electric",
    demand_scale_factor=2/30,
)
obj = itemized_cost["electric"]["energy"] + itemized_cost["electric"]["demand"]
# solve the CVX problem
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve()
# always compute the ex-post cost using NumPy due to the convex relaxations
cost_opt_elec_cost, _ = costs.calculate_itemized_cost(
    charge_dict,
    {"electric": grid_demand_kW.value},
    resolution="1m",
    desired_utility="electric",
    demand_scale_factor=2/30,
)
total_cost_opt = (
    cost_opt_elec_cost["electric"]["demand"] + cost_opt_elec_cost["electric"]["energy"]
)
print(f"Cost Optimized Electricity Cost: ${total_cost_opt:.2f}")


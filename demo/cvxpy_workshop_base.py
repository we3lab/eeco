from eeco import costs
# preliminaries
path_to_tariff_sheet = "tariff.csv"
start_dt = datetime.datetime(2023, 4, 9)
end_dt = datetime.datetime(2023, 4, 11)
# load a tariff spreadsheet
tariff_df = pd.read_csv(path_to_tariff_sheet, sep=",")
# get the charge dictionary
charge_dict = costs.get_charge_dict(start_dt, end_dt, tariff_df, resolution="1m")
# load historical consumption data
load_df = pd.read_csv("consumption.csv", parse_dates=["Datetime"])
# NOTE: second entry of the tuple can be ignored since it's for Pyomo
baseline_electricity_cost, _ = costs.calculate_itemized_cost(
    charge_dict,
    {"electric": load_df["Load [kW]"].values},
    resolution="1m",
    desired_utility="electric",
    demand_scale_factor=2/30,
)
total_baseline_cost = (
    baseline_electricity_cost["electric"]["demand"] + baseline_electricity_cost["electric"]["energy"]
)
print(f"Baseline Electricity Cost: ${total_baseline_cost:.2f}")


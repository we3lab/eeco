"""Functions to calculate costs from electricity consumption data."""

import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
import datetime as dt
import pyomo.environ as pyo
from itertools import compress

from . import utils as ut


def create_charge_array(charge, datetime, effective_start_date, effective_end_date):
    """Creates a single charge array based on the given parameters.

    Parameters
    ----------
    charge : pandas.DataFrame
        data for the charge including columns `month_start`, `month_end`,
        `weekday_start`, `weekday_end`, `hour_start`, `hour_end`, and `charge`

    datetime : pandas.DataFrame, pandas.Series, or numpy.ndarray
        If a pandas Series, it must be of type datetime
        If a DataFrame it must have a column "DateTime" and row for each timestep

    effective_start_date : datetime.datetime
        date on which this charge becomes effective

    effective_end_date : datetime.datetime
        date at which this charge is no longer in effect (i.e., this day is excluded)

    Raises
    ------
    TypeError
        When `datetime` is not pandas.DataFrame, pandas.Series, or numpy.ndarray

    Returns
    -------
    panda.Series
        timeseries of the cost of the charge with irrelevant timesteps zeroed out
    """
    if isinstance(datetime, np.ndarray):
        datetime = pd.to_datetime(datetime)
    elif isinstance(datetime, pd.DataFrame):
        datetime = datetime["DateTime"]
    if not isinstance(datetime, pd.Series):
        raise TypeError("'datetime' must be of type DataFrame, Series, or array")

    # calculate months, days, and hours
    weekdays = datetime.dt.weekday.values
    months = datetime.dt.month.values
    hours = datetime.dt.hour.astype(float).values
    # Make sure hours are being incremented by XX-minute increments
    minutes = datetime.dt.minute.astype(float).values
    hours += minutes / 60

    # create an empty charge array
    apply_charge = (
        (months >= charge["month_start"])
        & (months <= charge["month_end"])
        & (weekdays >= charge["weekday_start"])
        & (weekdays <= charge["weekday_end"])
        & (hours >= charge["hour_start"])
        & (hours < charge["hour_end"])
        & (datetime >= effective_start_date).values
        & (datetime < effective_end_date).values
    )

    try:
        charge_array = apply_charge * charge["charge (metric)"]
    except KeyError:
        warnings.warn(
            "Please switch to new 'charge (metric)' and 'charge (imperial)' format",
            DeprecationWarning,
        )
        charge_array = apply_charge * charge["charge"]
    return charge_array


def add_to_charge_array(charge_dict, key_str, charge_array):
    """Add to an existing charge array, or an arrya of all zeros if this charge
    array does not exist.

    This functionality is useful for noncontiguous charges that should be saved
    under the same key. For example, partial peak hours from 3-5 and 8-10 PM that
    are billing as a single demand period.

    Modifies `charge_dict` in-place, so nothing is returned.

    Parameters
    ----------
    charge_dict : dict of numpy.ndarray
        Dictionary of arrays with keys of the form
        `utility`_`charge_type`_`name`_`start_date`_`end_date`_`charge_limit`
        and values being the $ per kW (electric demand), kWh (electric energy/export),
        cubic meter / day (gas demand), cubic meter (gas energy),
        or $ / month (customer)

    key_str : str
        The key for the charge array we'd like to modify of the form
        `utility`_`charge_type`_`name`_`start_date`_`end_date`_`charge_limit`

    charge_array : numpy.ndarray
        Value of the charge to add in $ per kW (electric demand),
        kWh (electric energy/export), cubic meter / day (gas demand),
        cubic meter (gas energy), or $ / month (customer)
    """
    try:
        old_charge_array = charge_dict[key_str]
    except KeyError:
        old_charge_array = np.zeros(len(charge_array))

    charge_dict[key_str] = old_charge_array + charge_array


def get_charge_dict(start_dt, end_dt, rate_data, resolution="15m"):
    """Creates a dictionary where the values are charge arrays and keys are of the form
    `{utility}_{type}_{name}_{start_date}_{end_date}_{limit}`

    Parameters
    ----------
    start_dt : datetime.datetime
        first timestep to be included in the cost analysis

    end_dt : datetime.datetime
        last timestep to be included in the cost analysis

    rate_data : pandas.DataFrame
        tariff data with required columns `utility`, `type`, `basic_charge_limit`,
        `name`, `month_start`, `month_end`, `weekday_start`, `weekday_end`,
        `hour_start`, `hour_end`, and `charge` and optional columns `assessed`,
        `effective_start_date`, and `effective_end_date`

    resolution : str
        granularity of each timestep in string form with default value of "15m"

    Returns
    -------
    dict
        dictionary of charge arrays
    """
    charge_dict = {}

    # Get the number of timesteps in a day (according to charge resolution)
    res_binsize_minutes = ut.get_freq_binsize_minutes(resolution)
    if isinstance(start_dt, dt.datetime) or isinstance(end_dt, dt.datetime):
        ntsteps = int((end_dt - start_dt) / dt.timedelta(minutes=res_binsize_minutes))
        datetime = pd.DataFrame(
            np.array(
                [
                    start_dt + dt.timedelta(minutes=i * res_binsize_minutes)
                    for i in range(ntsteps)
                ]
            ),
            columns=["DateTime"],
        )
    else:
        ntsteps = int((end_dt - start_dt) / np.timedelta64(res_binsize_minutes, "m"))
        datetime = pd.DataFrame(
            np.array(
                [
                    start_dt + np.timedelta64(i * res_binsize_minutes, "m")
                    for i in range(ntsteps)
                ]
            ),
            columns=["DateTime"],
        )
    hours = datetime["DateTime"].dt.hour.astype(float).values
    # Make sure hours are being incremented by XX-minute increments
    minutes = datetime["DateTime"].dt.minute.astype(float).values
    hours += minutes / 60

    for utility in ["gas", "electric"]:
        for charge_type in ["customer", "energy", "demand", "export"]:
            charges = rate_data.loc[
                (rate_data["utility"] == utility) & (rate_data["type"] == charge_type),
                :,
            ]
            # if there are no charges of this type skip to the next iteration
            if charges.empty:
                continue

            try:
                effective_starts = pd.to_datetime(charges["effective_start_date"])
                effective_ends = pd.to_datetime(charges["effective_end_date"])
            except KeyError:
                # repeat start datetime for every charge
                effective_starts = pd.Series([start_dt]).repeat(len(charges))
                if isinstance(end_dt, dt.datetime):
                    effective_ends = pd.Series([end_dt - dt.timedelta(days=1)]).repeat(
                        len(charges)
                    )
                else:
                    effective_ends = pd.Series(
                        [end_dt - np.timedelta64(1, "D")]
                    ).repeat(len(charges))

            # numpy.unique does not work on datetimes so this is a workaround
            starts_ends = []
            for start, end in zip(effective_starts, effective_ends):
                if (start, end) not in starts_ends:
                    starts_ends.append((start, end))

            for start, end in starts_ends:
                # effective_end_date is meant to be inclusive, so add a day
                new_end = end + dt.timedelta(days=1)
                effective_charges = charges.loc[
                    (effective_starts == start).values & (effective_ends == end).values,
                    :,
                ]
                try:
                    charge_limits = effective_charges["basic_charge_limit (metric)"]
                except KeyError:
                    charge_limits = effective_charges["basic_charge_limit"]
                    warnings.warn(
                        "Please switch to new 'basic_charge_limit (metric)' "
                        "and 'basic_charge_limit (imperial)' format",
                        DeprecationWarning,
                    )
                for limit in np.unique(charge_limits):
                    if np.isnan(limit):
                        limit_charges = effective_charges.loc[
                            np.isnan(charge_limits), :
                        ]
                        limit = 0
                    else:
                        limit_charges = effective_charges.loc[charge_limits == limit, :]
                    for i, idx in enumerate(limit_charges.index):
                        charge = limit_charges.loc[idx, :]
                        try:
                            name = charge["name"]
                        except KeyError:
                            name = charge["period"]

                        # if no name was given just use the index to differentiate
                        if not (isinstance(name, str) and name != ""):
                            name = str(i)
                        # replace underscores with dashes for unique delimiter
                        name = name.replace("_", "-")

                        try:
                            assessed = charge["assessed"]
                        except KeyError:
                            assessed = "monthly"

                        if charge_type == "customer":
                            try:
                                charge_array = np.array([charge["charge (metric)"]])
                            except KeyError:
                                charge_array = np.array([charge["charge"]])
                                warnings.warn(
                                    "Please switch to new 'charge (metric)' "
                                    "and 'charge (imperial)' format",
                                    DeprecationWarning,
                                )
                            key_str = "_".join(
                                (
                                    utility,
                                    charge_type,
                                    name,
                                    start.strftime("%Y%m%d"),
                                    end.strftime("%Y%m%d"),
                                    str(int(limit)),
                                )
                            )
                            add_to_charge_array(charge_dict, key_str, charge_array)
                        elif charge_type == "demand" and assessed == "daily":
                            for day in range((end - start).days + 1):
                                new_start = start + dt.timedelta(days=day)
                                new_end = new_start + dt.timedelta(days=1)
                                charge_array = create_charge_array(
                                    charge, datetime, new_start, new_end
                                )
                                key_str = "_".join(
                                    (
                                        utility,
                                        charge_type,
                                        name,
                                        new_start.strftime("%Y%m%d"),
                                        new_start.strftime("%Y%m%d"),
                                        str(limit),
                                    )
                                )
                                add_to_charge_array(charge_dict, key_str, charge_array)
                        else:
                            charge_array = create_charge_array(
                                charge, datetime, start, new_end
                            )
                            key_str = "_".join(
                                (
                                    utility,
                                    charge_type,
                                    name,
                                    start.strftime("%Y%m%d"),
                                    end.strftime("%Y%m%d"),
                                    str(int(limit)),
                                )
                            )
                            add_to_charge_array(charge_dict, key_str, charge_array)
    return charge_dict


def get_charge_df(
    start_dt, end_dt, rate_data, resolution="15m", keep_fixed_charges=True
):
    """Creates a dictionary where the values are charge arrays and keys are of the form
    `{utility}_{type}_{name}_{start_date}_{end_date}_{limit}`

    Parameters
    ----------
    start_dt : datetime.datetime
        first timestep to be included in the cost analysis

    end_dt : datetime.datetime
        last timestep to be included in the cost analysis

    rate_data : pandas.DataFrame
        tariff data with required columns `utility`, `type`, `basic_charge_limit`,
        `name`, `month_start`, `month_end`, `weekday_start`, `weekday_end`,
        `hour_start`, `hour_end`, and `charge` and optional columns `assessed`,
        `effective_start_date`, and `effective_end_date`

    resolution : str
        granularity of each timestep in string form with default value of "15m"

    keep_fixed_charges : bool
        If True, fixed charges will be divided amongst all time steps and included.
        If False, fixed charges will be dropped from the output. Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame of charge arrays
    """
    charge_dict = get_charge_dict(start_dt, end_dt, rate_data, resolution=resolution)

    res_binsize_minutes = ut.get_freq_binsize_minutes(resolution)
    if isinstance(start_dt, dt.datetime) or isinstance(end_dt, dt.datetime):
        ntsteps = int((end_dt - start_dt) / dt.timedelta(minutes=res_binsize_minutes))
        datetime = pd.DataFrame(
            np.array(
                [
                    start_dt + dt.timedelta(minutes=i * res_binsize_minutes)
                    for i in range(ntsteps)
                ]
            ),
            columns=["DateTime"],
        )
    else:
        ntsteps = int((end_dt - start_dt) / np.timedelta64(res_binsize_minutes, "m"))
        datetime = pd.DataFrame(
            np.array(
                [
                    start_dt + np.timedelta64(i * res_binsize_minutes, "m")
                    for i in range(ntsteps)
                ]
            ),
            columns=["DateTime"],
        )

    # first find the value of the fixed charge
    fixed_charge_dict = {
        key: value
        for key, value in charge_dict.items()
        if any(k in key for k in ["electric_customer", "gas_customer"])
    }

    if keep_fixed_charges:
        # replace the fixed charge in charge_dict with its time-averaged value
        for key, value in fixed_charge_dict.items():
            charge_dict[key] = np.ones(ntsteps) * value / ntsteps

    else:
        # remove fixed charges from the charge_dict
        for key in fixed_charge_dict.keys():
            del charge_dict[key]

    charge_df = pd.DataFrame(charge_dict)

    charge_df = pd.concat([datetime, charge_df], axis=1)

    # remove all zero columns
    charge_df = charge_df.loc[:, (charge_df != 0).any(axis=0)]
    return charge_df


def get_next_limit(key_substr, current_limit, keys):
    """Finds the next charge limit for the charge represented by `key`

    Parameters
    ----------
    key_substr : str
        The beginnging of the key for which we want to get the next limit
        (i.e., `{utility}_{type}_{name}_{start_date}_{end_date}`)

    current_limit : int
        The limit for the current tier

    keys : list of str
        List of all the keys in the charge dictionary

    Returns
    -------
    float
        limit in the tier after `key`, which is `inf` if there is no higher tier
    """
    matching_keys = [key_substr in key for key in keys]
    limits = sorted(
        [float(key.split("_")[-1]) for key in compress(keys, matching_keys)]
    )
    try:
        matching_idx = limits.index(current_limit)
        return limits[matching_idx + 1]
    except IndexError:
        return float("inf")


def calculate_demand_cost(
    charge_array,
    consumption_data,
    limit=0,
    next_limit=float("inf"),
    prev_demand=0,
    prev_demand_cost=0,
    consumption_estimate=0,
    scale_factor=1,
    model=None,
    varstr=None,
):
    """Calculates the cost of given demand charges for the given billing rate structure,
    utility, and consumption information

    Parameters
    ----------
    charge_array : array
        Array of charge cost (in $/kW)

    consumption_data : numpy.ndarray or cvxpy.Expression
        Baseline electrical or gas usage data as an optimization variable object

    limit : float
        The total consumption, or limit, that this charge came into effect.
        Default is 0

    next_limit : float
        The total consumption, or limit, that the next charge comes into effect.
        Default is float('inf') indicating that there is no higher tier

    prev_demand : float
        The previous maximum demand for this charge during the same billing period.
        Only used for optimizing on a horizon shorter than the billing period,
        so the default is 0

    prev_demand_cost : float
        The previous cost for this demand charge during the same billing period.
        Only used for optimizing on a horizon shorter than the billing period,
        so the default is 0

    consumption_estimate : float
        Estimate of the total monthly demand or energy consumption from baseline data.
        Only used when `consumption_data` is cvxpy.Expression for convex relaxation
        of tiered charges, while numpy.ndarray `consumption_data` will use actual
        consumption and ignore the estimate.

    scale_factor : float
        Optional factor for scaling demand charges relative to energy charges
        when the optimization/simulation period is not a full billing cycle.
        Default is 1

    model : pyomo.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    varstr : str
        Name of the variable to be created if using a Pyomo `model`

    Returns
    -------
    cvxpy.Expression or float
        float or cvxpy Expression representing cost in USD for the given
        `charge_array` and `consumption_data`
    """
    if isinstance(consumption_data, np.ndarray):
        if (np.max(consumption_data) >= limit) or (
            (prev_demand >= limit) and (prev_demand <= next_limit)
        ):
            if np.max(consumption_data) >= next_limit:
                demand_charged, model = ut.multiply(next_limit - limit, charge_array)
            else:
                demand_charged, model = ut.multiply(
                    consumption_data - limit, charge_array
                )
        else:  # ignore if current and previous maxima outside of charge limit
            demand_charged = np.array([0])
    elif isinstance(consumption_data, (pyo.Param, pyo.Var)):
        if consumption_estimate >= limit:
            if consumption_estimate <= next_limit:
                model.add_component(
                    varstr + "_limit",
                    pyo.Var(model.t, initialize=0, bounds=(0, None)),
                )
                var = model.find_component(varstr + "_limit")

                def const_rule(model, t):
                    return var[t] == consumption_data[t] - limit

                constraint = pyo.Constraint(model.t, rule=const_rule)
                model.add_component(varstr + "_limit_constraint", constraint)

                demand_charged, model = ut.multiply(
                    var,
                    charge_array,
                    model=model,
                    varstr=varstr + "_multiply",
                )
            else:
                demand_charged, model = ut.multiply(
                    next_limit - limit,
                    charge_array,
                    model=model,
                    varstr=varstr + "_multiply",
                )
        else:
            demand_charged = np.array([0])
    elif isinstance(consumption_data, cp.Expression):
        if consumption_estimate >= limit:
            if consumption_estimate <= next_limit:
                demand_charged, model = ut.multiply(
                    consumption_data - limit,
                    charge_array,
                    model=model,
                    varstr=varstr + "_multiply",
                )
            else:
                demand_charged, model = ut.multiply(
                    next_limit - limit,
                    charge_array,
                    model=model,
                    varstr=varstr + "_multiply",
                )
        else:
            demand_charged = np.array([0])
    else:
        raise ValueError(
            "consumption_data must be of type numpy.ndarray or cvxpy.Expression"
        )
    if model is None:
        result, _ = ut.max(demand_charged)
        return ut.max_pos(result - prev_demand_cost) * scale_factor
    else:
        max_var, model = ut.max(demand_charged, model=model, varstr=varstr + "_max")
        return (
            ut.max_pos(
                max_var - prev_demand_cost, model=model, varstr=varstr + "_max_pos"
            )
            * scale_factor
        )


def calculate_energy_cost(
    charge_array,
    consumption_data,
    divisor,
    limit=0,
    next_limit=float("inf"),
    prev_consumption=0,
    consumption_estimate=0,
    model=None,
    varstr=None,
):
    """Calculates the cost of given energy charges for the given billing rate
    structure, utility, and consumption information.

    Parameters
    ----------
    charge_array : numpy.ndarray
        Array of the charges in $/kWh for electric and $/cubic meter for gas

    consumption_data : numpy.ndarray or cvxpy.Expression
        Baseline electrical or gas usage data as an optimization variable object

    divisor : int
        Divisor for the energy charges

    limit : float
        The total consumption, or limit, that this charge came into effect.
        Default is 0

    next_limit : float
        The total consumption, or limit, that the next charge comes into effect.
        Default is float('inf') indicating that there is no higher tier

    consumption_estimate : float
        Estimate of the total monthly demand or energy consumption from baseline data.
        Only used when `consumption_data` is cvxpy.Expression for convex relaxation
        of tiered charges, while numpy.ndarray `consumption_data` will use actual
        consumption and ignore the estimate.

    model : pyomo.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    varstr : str
        Name of the variable to be created if using a Pyomo `model`

    Raises
    ------
    ValueError
        When invalid `utility`, `charge_type`, or `assessed`
        is provided in `charge_arrays`

    Returns
    -------
    cvxpy.Expression, pyomo.Model, or float
        float or cvxpy Expression representing cost in USD for the given
        `charge_array` and `consumption_data`
    """
    cost = 0
    if model is None:
        n_steps = consumption_data.shape[0]
    else:  # Pyomo does not support shape attribute
        n_steps = len(consumption_data)

    if isinstance(consumption_data, np.ndarray):
        energy = prev_consumption
        # set the flag if we are starting with previous consumption that lands us
        # within the current tier of charge limits
        within_limit_flag = energy >= float(limit) and energy < float(next_limit)
        for i in range(len(consumption_data)):
            energy += consumption_data[i] / divisor
            # only add to charges if already within correct charge limits
            if within_limit_flag:
                # went over next charge limit on this iteration
                # set flag to false to avoid overcounting after this iteration
                if energy >= float(next_limit):
                    within_limit_flag = False
                    cost += (
                        float(next_limit) + consumption_data[i] / divisor - energy
                    ) * charge_array[i]
                else:
                    cost += consumption_data[i] / divisor * charge_array[i]
            # went over existing charge limit on this iteration
            elif energy >= float(limit) and energy < float(next_limit):
                within_limit_flag = True
                cost += (energy - float(limit)) * charge_array[i]
    elif isinstance(consumption_data, (cp.Expression, pyo.Var, pyo.Param)):
        charge_expr, model = ut.multiply(
            consumption_data, charge_array, model=model, varstr=varstr + "_multiply"
        )
        if next_limit == float("inf"):
            limit_to_subtract = float(limit) / n_steps
            sum_result, model = ut.sum(charge_expr, model=model, varstr=varstr + "_sum")
            cost, model = ut.max_pos(
                (sum_result / divisor - np.sum(charge_array * limit_to_subtract)),
                model=model,
                varstr=varstr,
            )
        else:
            if consumption_estimate < float(next_limit):
                prev_limit_expr, model = ut.multiply(
                    float(limit) / n_steps,
                    charge_array,
                    model=model,
                    varstr=varstr + "_prev",
                )
                sum_result, model = ut.sum(
                    charge_expr, model=model, varstr=varstr + "_sum"
                )
                cost, model = ut.max_pos(
                    sum_result / divisor - (np.sum(prev_limit_expr)),
                    model=model,
                    varstr=varstr,
                )
            else:
                cost = np.sum(
                    charge_array * (float(next_limit) - float(limit)) / n_steps
                )
    else:
        raise ValueError(
            "consumption_data must be of type numpy.ndarray or cvxpy.Expression"
        )

    return cost, model


def calculate_export_revenues(
    charge_array, export_data, divisor, model=None, varstr=None
):
    """Calculates the export revenues for the given billing rate structure,
    utility, and consumption information.

    Only flat rates for exports are supported (in $ / kWh).

    Parameters
    ----------
    charge_array : numpy.ndarray
        array with price per kWh sold back to the grid

    consumption_data : numpy.ndarray or cvxpy.Expression
        Baseline electrical or gas usage data as an optimization variable object

    divisor : int
        Divisor for the energy charges

    model : pyomo.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    varstr : str
        Name of the variable to be created if using a Pyomo `model`

    Returns
    -------
    cvxpy.Expression or float
        float or cvxpy Expression representing revenue in USD for the given
        `charge_array` and `export_data`
    """
    varstr_mul = varstr + "_multiply" if varstr is not None else None
    varstr_sum = varstr + "_sum" if varstr is not None else None
    result, model = ut.multiply(
        charge_array, export_data, model=model, varstr=varstr_mul
    )
    revenues, model = ut.sum(result, model=model, varstr=varstr_sum)
    return revenues / divisor, model


def calculate_cost(
    charge_dict,
    consumption_data_dict,
    resolution="15m",
    prev_demand_dict=None,
    prev_consumption_dict=None,
    consumption_estimate=0,
    desired_utility=None,
    desired_charge_type=None,
    demand_scale_factor=1,
    model=None,
):
    """Calculates the cost of given charges (demand or energy) for the given
    billing rate structure, utility, and consumption information as a
    cvxpy expression or numpy array

    Parameters
    ----------
    charge_dict : dict
        dictionary of arrays with keys of the form
        `utility`_`charge_type`_`name`_`start_date`_`end_date`_`charge_limit`
        and values being the $ per kW (electric demand), kWh (electric energy/export),
        cubic meter / day (gas demand), cubic meter (gas energy),
        or $ / month (customer)

    consumption_data_dict : dict of numpy.ndarray or cvxpy.Expression
        Baseline electrical and gas usage data as an optimization variable object
        with keys "electric" and "gas"

    resolution : str
        String of the form `[int][str]` giving the temporal resolution
        on which charges are assessed, the `str` portion corresponds to
        numpy.timedelta64 types for example '15m' specifying demand charges
        that are applied to 15-minute intervals of electricity consumption

    prev_demand_dict : dict
        Nested dictionary previous maximmum demand charges with an entry of the form
        {"cost" : float, "demand" : float} for each charge.
        Default is None, which results in an a prev_demand and prev_demand_cost
        of zero for all charges.

    prev_consumption_dict : dict
        Dictionary of previous total energy consumption with a key for each charge
        to be used when starting the cost calculation partway into a billing period
        (e.g., while using a moving horizon that is shorter than a month).
        Default is None, resulting in an a prev_consumption of zero for all charges.

    consumption_estimate : float
        Estimate of the total monthly demand or energy consumption from baseline data.
        Only used when `consumption_data` is cvxpy.Expression for convex relaxation
         of tiered charges, while numpy.ndarray `consumption_data` will use actual
         consumption and ignore the estimate.

    desired_charge_type : str
        Name of desired charge type for itemized costs.
        Either 'customer', 'energy', 'demand', or 'export'.
        Default is None, meaning that all costs will be summed together.

    desired_utility : str
        Name of desired utility for itemized costs. Either 'electric' or 'gas'
        Default is None, meaning that all costs will be summed together.

    demand_scale_factor : float
        Optional factor for scaling demand charges relative to energy charges
        when the optimization/simulation period is not a full billing cycle.
        Default is 1

    model : pyomo.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    Raises
    ------
    ValueError
        When invalid `utility`, `charge_type`, or `assessed`
        is provided in `charge_arrays`

    Returns
    -------
    numpy.Array, cvxpy.Expression, or pyomo.Model
        numpy array, cvxpy Expression representing cost in USD for the given
        `consumption_data`, `charge_type`, and `utility`
    """
    cost = 0
    n_per_hour = int(60 / ut.get_freq_binsize_minutes(resolution))
    n_per_day = n_per_hour * 24

    for key, charge_array in charge_dict.items():
        utility, charge_type, name, eff_start, eff_end, limit_str = key.split("_")
        # if we want itemized costs skip irrelvant portions of the bill
        if (desired_utility and utility != desired_utility) or (
            desired_charge_type and charge_type != desired_charge_type
        ):
            continue

        charge_limit = int(limit_str)
        key_substr = "_".join([utility, charge_type, name, eff_start, eff_end])
        next_limit = get_next_limit(key_substr, charge_limit, charge_dict.keys())
        consumption_data = consumption_data_dict[utility]

        # TODO: this assumes units of kW for electricity and meters cubed for gas
        if utility == "electric":
            divisor = n_per_hour
        elif utility == "gas":
            divisor = n_per_day
        else:
            raise ValueError("Invalid utility: " + utility)

        if charge_type == "demand":
            if prev_demand_dict is not None:
                prev_demand = prev_demand_dict[key]["demand"]
                prev_demand_cost = prev_demand_dict[key]["cost"]
            else:
                prev_demand = 0
                prev_demand_cost = 0
            new_cost, model = calculate_demand_cost(
                charge_array,
                consumption_data,
                limit=charge_limit,
                next_limit=next_limit,
                prev_demand=prev_demand,
                prev_demand_cost=prev_demand_cost,
                consumption_estimate=consumption_estimate,
                scale_factor=demand_scale_factor,
                model=model,
                varstr=key,
            )
            cost += new_cost
        elif charge_type == "energy":
            if prev_consumption_dict is not None:
                prev_consumption = prev_consumption_dict[key]
            else:
                prev_consumption = 0
            new_cost, model = calculate_energy_cost(
                charge_array,
                consumption_data,
                divisor,
                limit=charge_limit,
                next_limit=next_limit,
                prev_consumption=prev_consumption,
                consumption_estimate=consumption_estimate,
                model=model,
                varstr=key,
            )
            cost += new_cost
        elif charge_type == "export":
            new_cost, model = calculate_export_revenues(
                charge_array, consumption_data, divisor, model=model, varstr=key
            )
            cost -= new_cost
        elif charge_type == "customer":
            cost += charge_array.sum()
        else:
            raise ValueError("Invalid charge_type: " + charge_type)
    return cost, model


def calculate_itemized_cost(
    charge_dict,
    consumption_data,
    resolution="15m",
    prev_demand_dict=None,
    consumption_estimate=0,
    model=None,
):
    """Calculates itemized costs as a nested dictionary

    Parameters
    ----------
    charge_dict : dict
        dictionary of arrays with keys of the form
        `utility`_`charge_type`_`name`_`start_date`_`end_date`_`charge_limit`
        and values being the $ per kW (electric demand), kWh (electric energy/export),
        cubic meter / day (gas demand), cubic meter (gas energy),
        or $ / month (customer)

    consumption_data : numpy.ndarray or cvxpy.Expression
        Baseline electrical or gas usage data as an optimization variable object

    resolution : str
        granularity of each timestep in string form with default value of "15m"

     prev_demand_dict : dict
        Dictionary previous maximmum demand charges with a key for each charge.
        Default is None, which results in an a prev_demand of zero for all charges.

    consumption_estimate : float
        estimated total consumption up to this point in the bililng period to determine
        correct tier based on charge limits

    model : pyomo.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    Returns
    -------
    dict
        {

            "electric": {

                "customer": `float`

                "energy": `float`

                "demand": `float`

                "export": `float`

                "total": `float`

            }

            "gas": {

                "customer": `float`,

                "energy": `float`

                "demand": `float`

                "export": `float`

                "total": `float`

            }

            "total": `float`

        }

    """
    total_cost = 0
    results_dict = {}
    for utility in ["electric", "gas"]:
        results_dict[utility] = {}
        total_utility_cost = 0
        for charge_type in ["customer", "energy", "demand", "export"]:
            cost, model = calculate_cost(
                charge_dict,
                consumption_data,
                resolution=resolution,
                prev_demand_dict=prev_demand_dict,
                desired_utility=utility,
                desired_charge_type=charge_type,
                model=model,
            )

            results_dict[utility][charge_type] = cost
            total_utility_cost += cost

        results_dict[utility]["total"] = total_utility_cost
        total_cost += total_utility_cost

    results_dict["total"] = total_cost
    return results_dict, model

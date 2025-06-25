"""Functions to calculate costs from electricity consumption data."""

import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
import datetime as dt
import pyomo.environ as pyo
from itertools import compress

from . import utils as ut

# Column strings
HOUR_START = "hour_start"
HOUR_END = "hour_end"
MONTH_START = "month_start"
MONTH_END = "month_end"
WEEKDAY_START = "weekday_start"
WEEKDAY_END = "weekday_end"
CHARGE = "charge"
CHARGE_METRIC = "charge (metric)"
CHARGE_IMPERIAL = "charge (imperial)"
BASIC_CHARGE_LIMIT = "basic_charge_limit"
BASIC_CHARGE_LIMIT_METRIC = "basic_charge_limit (metric)"
BASIC_CHARGE_LIMIT_IMPERIAL = "basic_charge_limit (imperial)"
UTILITY = "utility"
TYPE = "type"
NAME = "name"
PERIOD = "period"
ASSESSED = "assessed"
EFFECTIVE_START_DATE = "effective_start_date"
EFFECTIVE_END_DATE = "effective_end_date"
DATETIME = "DateTime"

# Utility type strings
ELECTRIC = "electric"
GAS = "gas"

# Charge type strings
CUSTOMER = "customer"
ENERGY = "energy"
DEMAND = "demand"
EXPORT = "export"


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
        datetime = datetime[DATETIME]
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
        (months >= charge[MONTH_START])
        & (months <= charge[MONTH_END])
        & (weekdays >= charge[WEEKDAY_START])
        & (weekdays <= charge[WEEKDAY_END])
        & (hours >= charge[HOUR_START])
        & (hours < charge[HOUR_END])
        & (datetime >= effective_start_date).values
        & (datetime < effective_end_date).values
    )

    try:
        charge_array = apply_charge * charge[CHARGE_METRIC]
    except KeyError:
        warnings.warn(
            "Please switch to new 'charge (metric)' and 'charge (imperial)' format",
            DeprecationWarning,
        )
        charge_array = apply_charge * charge[CHARGE]
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
            columns=[DATETIME],
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
            columns=[DATETIME],
        )
    hours = datetime[DATETIME].dt.hour.astype(float).values
    # Make sure hours are being incremented by XX-minute increments
    minutes = datetime[DATETIME].dt.minute.astype(float).values
    hours += minutes / 60

    for utility in [GAS, ELECTRIC]:
        for charge_type in [CUSTOMER, ENERGY, DEMAND, EXPORT]:
            charges = rate_data.loc[
                (rate_data[UTILITY] == utility) & (rate_data[TYPE] == charge_type),
                :,
            ]
            # if there are no charges of this type skip to the next iteration
            if charges.empty:
                continue

            try:
                effective_starts = pd.to_datetime(charges[EFFECTIVE_START_DATE])
                effective_ends = pd.to_datetime(charges[EFFECTIVE_END_DATE])
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
                    charge_limits = effective_charges[BASIC_CHARGE_LIMIT_METRIC]
                except KeyError:
                    charge_limits = effective_charges[BASIC_CHARGE_LIMIT]
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
                            name = charge[NAME]
                        except KeyError:
                            name = charge[PERIOD]

                        # if no name was given just use the index to differentiate
                        if not (isinstance(name, str) and name != ""):
                            name = str(i)
                        # replace underscores with dashes for unique delimiter
                        name = name.replace("_", "-")

                        try:
                            assessed = charge[ASSESSED]
                        except KeyError:
                            assessed = "monthly"

                        if charge_type == CUSTOMER:
                            try:
                                charge_array = np.array([charge[CHARGE_METRIC]])
                            except KeyError:
                                charge_array = np.array([charge[CHARGE]])
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
                        elif charge_type == DEMAND and assessed == "daily":
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
            columns=[DATETIME],
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
            columns=[DATETIME],
        )

    # first find the value of the fixed charge
    fixed_charge_dict = {
        key: value
        for key, value in charge_dict.items()
        if any(
            k in key for k in [f"{utility}_{CUSTOMER}" for utility in [ELECTRIC, GAS]]
        )
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
        if utility == ELECTRIC:
            divisor = n_per_hour
        elif utility == GAS:
            divisor = n_per_day
        else:
            raise ValueError("Invalid utility: " + utility)

        if charge_type == DEMAND:
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
        elif charge_type == ENERGY:
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
        elif charge_type == EXPORT:
            new_cost, model = calculate_export_revenues(
                charge_array, consumption_data, divisor, model=model, varstr=key
            )
            cost -= new_cost
        elif charge_type == CUSTOMER:
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
    for utility in [ELECTRIC, GAS]:
        results_dict[utility] = {}
        total_utility_cost = 0
        for charge_type in [CUSTOMER, ENERGY, DEMAND, EXPORT]:
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


def parametrize_rate_data(
    rate_data,
    peak_demand_ratio=1.0,
    peak_energy_ratio=1.0,
    avg_demand_ratio=1.0,
    avg_energy_ratio=1.0,
    peak_window_expand_hours=0.0,
    name=None,
):
    """Takes in rate data csv and creates parametric variations.

    Parameters
    ----------
    rate_data : pandas.DataFrame
        Tariff data df with columns: utility, type, basic_charge_limit, name,
        month_start, month_end, weekday_start, weekday_end, hour_start,
        hour_end, charge or charge (metric) and charge (imperial)
    peak_demand_ratio: float
        Float to scale peak demand charges. Default 1.0
    peak_energy_ratio: float
        Float to scale peak energy charges. Default 1.0
    avg_demand_ratio: float
        Float to scale average demand charges. Default 1.0
    avg_energy_ratio: float
        Float to scale average energy charges. Default 1.0
    peak_window_expand_hours: int
        Int to expand peak window width. Will be divided evenly on both sides
        of peak period (e.g. 1/2 hr before and after for value 1).
        Must be in hours. Default 0

    Returns
    -------
    df
        Updated rate_data dataframe for variant
    """
    variant_data = rate_data.copy(deep=True)

    # Convert hour_start and hour_end to float64
    variant_data[HOUR_START] = variant_data[HOUR_START].astype(float)
    variant_data[HOUR_END] = variant_data[HOUR_END].astype(float)

    # Get charge columns (based on whether tariff data has metric/imperial)
    if CHARGE_METRIC in variant_data.columns:
        charge_cols = [CHARGE_METRIC, CHARGE_IMPERIAL]
    else:
        charge_cols = [CHARGE]

    peak_ratios = {ENERGY: peak_energy_ratio, DEMAND: peak_demand_ratio}
    avg_ratios = {ENERGY: avg_energy_ratio, DEMAND: avg_demand_ratio}

    # Predefine masks to locate charges with different utilities, types, and durations
    electric_mask = variant_data[UTILITY] == ELECTRIC
    type_masks = {
        ENERGY: ((variant_data[TYPE] == ENERGY)),
        DEMAND: ((variant_data[TYPE] == DEMAND)),
    }
    full_day_mask = (variant_data[HOUR_END] - variant_data[HOUR_START]) == 24

    window_expand_hours = round(peak_window_expand_hours, 0) / 2

    # Get unique combinations of months and weekdays in the rate data
    month_combos = variant_data[[MONTH_START, MONTH_END]].drop_duplicates().values
    for month_start, month_end in month_combos:
        month_combo_mask = (variant_data[MONTH_START] == month_start) & (
            variant_data[MONTH_END] == month_end
        )
        weekday_combos = (
            variant_data[month_combo_mask][[WEEKDAY_START, WEEKDAY_END]]
            .drop_duplicates()
            .values
        )

        # Get monthly full-day charges for each type
        monthly_full_day_charges = {ENERGY: {}, DEMAND: {}}
        for type in [ENERGY, DEMAND]:
            type_data = variant_data[
                month_combo_mask & electric_mask & type_masks[type] & full_day_mask
            ]
            for charge_col in charge_cols:
                monthly_full_day_charges[type][charge_col] = (
                    type_data[charge_col].iloc[0] if not type_data.empty else None
                )

        for weekday_start, weekday_end in weekday_combos:
            # Filter data for current month/weekday combination
            month_weekday_combo_mask = (
                month_combo_mask
                & (variant_data[WEEKDAY_START] == weekday_start)
                & (variant_data[WEEKDAY_END] == weekday_end)
            )

            for type in [ENERGY, DEMAND]:
                # SCALE AVERAGE AND PEAK PERIODS

                # Get all charges for this type and month-weekday combination
                type_data = variant_data[
                    month_weekday_combo_mask & electric_mask & type_masks[type]
                ]

                if len(type_data) == 1:
                    # Only one charge of this type - treat as average
                    for charge_col in charge_cols:
                        variant_data.loc[type_data.index, charge_col] *= avg_ratios[
                            type
                        ]
                    continue

                # Multiple charges of this type
                # Check if ANY 24-hour charge applies to this month-weekday combo
                overlapping_full_day_mask = (
                    variant_data[HOUR_END] - variant_data[HOUR_START]
                ) == 24

                # Find 24-hour charges that overlap with current month-weekday period
                overlapping_full_day_charges = variant_data[
                    electric_mask
                    & type_masks[type]
                    & overlapping_full_day_mask
                    & (
                        # Does 24-hour charge's month range overlap current month?
                        (
                            (variant_data[MONTH_START] <= month_start)
                            & (variant_data[MONTH_END] >= month_start)
                        )
                        | (
                            (variant_data[MONTH_START] <= month_end)
                            & (variant_data[MONTH_END] >= month_end)
                        )
                        | (
                            (variant_data[MONTH_START] >= month_start)
                            & (variant_data[MONTH_END] <= month_end)
                        )
                    )
                    & (
                        # Does 24-hour charge's weekday range overlap current weekday?
                        (
                            (variant_data[WEEKDAY_START] <= weekday_start)
                            & (variant_data[WEEKDAY_END] >= weekday_start)
                        )
                        | (
                            (variant_data[WEEKDAY_START] <= weekday_end)
                            & (variant_data[WEEKDAY_END] >= weekday_end)
                        )
                        | (
                            (variant_data[WEEKDAY_START] >= weekday_start)
                            & (variant_data[WEEKDAY_END] <= weekday_end)
                        )
                    )
                ]
                # Align full_day_mask index with type_data for exact matches
                full_day_mask_aligned = (
                    full_day_mask.loc[type_data.index]
                    if hasattr(full_day_mask, "loc")
                    else full_day_mask
                )
                full_day_charges = type_data[full_day_mask_aligned]

                if not (full_day_charges.empty and overlapping_full_day_charges.empty):
                    # There is a 24-hour charge - scale charges directly by their ratios
                    for charge_col in charge_cols:
                        for idx in type_data.index:
                            charge = variant_data.loc[idx, charge_col]
                            if (
                                variant_data.loc[idx, HOUR_END]
                                - variant_data.loc[idx, HOUR_START]
                            ) == 24:
                                # 24-hour charge gets average scaling
                                variant_data.loc[idx, charge_col] = (
                                    charge * avg_ratios[type]
                                )
                            else:
                                # Non-24-hour charges get peak scaling
                                variant_data.loc[idx, charge_col] = (
                                    charge * peak_ratios[type]
                                )

                else:
                    # No 24-hour charge. Define average charge as
                    #   1) a monthly full-day charge if it exists
                    #   2) else the minimum daily charge.
                    #  Scale the difference between peak and average charge
                    for charge_col in charge_cols:
                        avg_charge = monthly_full_day_charges[type][charge_col]

                        if avg_charge is not None:
                            # Monthly full-day charge exists on another weekday
                            # Scale all charges based on difference from average
                            for idx in type_data.index:
                                charge = variant_data.loc[idx, charge_col]
                                diff = charge - avg_charge
                                variant_data.loc[idx, charge_col] = (
                                    avg_charge * avg_ratios[type]
                                    + (diff * peak_ratios[type])
                                )
                        else:
                            # No monthly full-day charge exists.
                            # Take minimum charge for the weekday-month combo as average
                            min_charge = type_data[charge_col].min()
                            min_idxs = type_data[
                                type_data[charge_col] == min_charge
                            ].index
                            peak_idxs = type_data[
                                type_data[charge_col] != min_charge
                            ].index
                            # Scale minimum periods by average ratio only
                            for idx in min_idxs:
                                variant_data.loc[idx, charge_col] = (
                                    min_charge * avg_ratios[type]
                                )
                            # New charge = min_charge * avg_ratio + (diff * peak_ratio)
                            for idx in peak_idxs:
                                charge = variant_data.loc[idx, charge_col]
                                diff = charge - min_charge
                                new_val = min_charge * avg_ratios[type] + (
                                    diff * peak_ratios[type]
                                )
                                variant_data.loc[idx, charge_col] = new_val

                # SHIFT WINDOWS
                peak_periods = variant_data[
                    month_weekday_combo_mask
                    & electric_mask
                    & type_masks[type]
                    & ~full_day_mask
                ]

                if not peak_periods.empty:
                    # Find highest charge period
                    highest_period = peak_periods.sort_values(
                        charge_cols[0], ascending=False
                    ).iloc[0]

                    orig_peak_start = highest_period[HOUR_START]
                    orig_peak_end = highest_period[HOUR_END]

                    new_peak_start = max(0, orig_peak_start - window_expand_hours)
                    new_peak_end = min(24, orig_peak_end + window_expand_hours)

                    variant_data.loc[highest_period.name, HOUR_START] = new_peak_start
                    variant_data.loc[highest_period.name, HOUR_END] = new_peak_end

                    # Left side: abut each period to the next period to the right
                    left_periods = peak_periods[
                        peak_periods[HOUR_END] <= orig_peak_start
                    ].sort_values(HOUR_END, ascending=False)
                    left_start = new_peak_start
                    for idx, row in left_periods.iterrows():
                        duration = row[HOUR_END] - row[HOUR_START]
                        variant_data.loc[idx, HOUR_END] = left_start
                        variant_data.loc[idx, HOUR_START] = max(
                            0, left_start - duration
                        )
                        left_start = variant_data.loc[idx, HOUR_START]

                    # Right side: abut each period to the next period to the left
                    right_periods = peak_periods[
                        peak_periods[HOUR_START] >= orig_peak_end
                    ].sort_values(HOUR_START, ascending=True)
                    right_start = new_peak_end
                    for idx, row in right_periods.iterrows():
                        duration = row[HOUR_END] - row[HOUR_START]
                        variant_data.loc[idx, HOUR_START] = right_start
                        variant_data.loc[idx, HOUR_END] = min(
                            24, right_start + duration
                        )
                        right_start = variant_data.loc[idx, HOUR_END]

    return variant_data


def parametrize_charge_dict(start_dt, end_dt, rate_data, variants=None):
    """
    Takes in an existing charge_dict and varies it parametrically to create
    alternative rate structures. Calls parametrize_rate_data to parametrize the
    billing csv file, then calls it on the dates specified.

    Parameters
    ----------
    start_dt : datetime.datetime
        first timestep to be included in the cost analysis
    end_dt : datetime.datetime
        last timestep to be included in the cost analysis
    rate_data : pandas.DataFrame
        tariff data with required columns
    variants : list[dict]
        List of dictionaries containing variation parameters with keys:
        - peak_demand_ratio: float to scale peak demand charges
        - peak_energy_ratio: float to scale peak energy charges
        - avg_demand_ratio: float to scale average demand charges
        - avg_energy_ratio: float to scale average energy charges
        - peak_window_expand_hours: float to expand peak window width
        - name: str (optional) variant name. (Default 'variant_{i}')

    Returns
    -------
    dict
        dictionary of charge_dicts with different variations
    """

    # Initialize rate data dictionary with copy of original data
    billing_data_variants = {"original": rate_data.copy(deep=True)}

    # Initialize dictionary of charge dicts for given start/end dates with variants
    charge_dicts = {"original": get_charge_dict(start_dt, end_dt, rate_data)}

    for i, variant in enumerate(variants):
        variant_key = f"variant_{i}"
        # Use name if specified, or name variant_{i}
        variant_key = variant.get("name", f"variant_{i}")
        variant_data = parametrize_rate_data(rate_data.copy(deep=True), **variants[i])

        billing_data_variants[f"variant_{i}"] = variant_data.copy(deep=True)
        charge_dicts[variant_key] = get_charge_dict(start_dt, end_dt, variant_data)

    return charge_dicts

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
        If a pandas Series, it must be of type datetime.
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
    """Add to an existing charge array, or an array of all zeros if this charge
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


def default_varstr_alias_func(
    utility, charge_type, name, start_date, end_date, charge_limit
):
    """Default function for creating the variable name strings for each charge
    in the tariff sheet. Can be overwritten in the function call to `calculate_cost`
    to customize variable names.

    Parameters
    ----------
    utility : str
        Name of the utility ('electric' or 'gas')

    charge_type : str
        Name of the `charge_type` ('demand', 'energy', or 'customer')

    name : str
        The name of the period for this charge (e.g., 'all-day' or 'on-peak')

    start_date
        The inclusive start date for this charge

    end_date : str
        The exclusive end date for this charge

    charge_limit : str
        The consumption limit for this tier of charges converted to a string

    Returns
    -------
    str
        Variable name of the form
        `utility`_`charge_type`_`name`_`start_date`_`end_date`_`charge_limit`
    """
    return f"{utility}_{charge_type}_{name}_{start_date}_{end_date}_{charge_limit}"


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
        Applied to monthly charges where end_date - start_date > 1 day.
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
        max_var, _ = ut.max(demand_charged)
        max_pos_val, max_pos_model = ut.max_pos(max_var - prev_demand_cost)
        return max_pos_val * scale_factor, max_pos_model
    else:
        max_var, model = ut.max(demand_charged, model=model, varstr=varstr + "_max")
        max_pos_val, max_pos_model = ut.max_pos(
            max_var - prev_demand_cost, model=model, varstr=varstr + "_max_pos"
        )
        return max_pos_val * scale_factor, max_pos_model


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


def get_charge_array_duration(key):
    """Parse a charge array key to determine the duration of the charge period.

    Parameters
    ----------
    key : str
        Charge key of form `utility_charge_type_name_start_date_end_date_charge_limit`
        where start_date and end_date are in YYYYMMDD or YYYY-MM-DD format

    Returns
    -------
    int
        Duration of the charge period in days

    Raises
    ------
    ValueError
        If the key format is invalid or dates cannot be parsed
    """
    parts = key.split("_")
    if len(parts) < 6:
        raise ValueError(f"Invalid charge key format: {key}")

    start_date_str = parts[-3]
    end_date_str = parts[-2]

    # Allow 2 date formats
    date_formats = ["%Y%m%d", "%Y-%m-%d"]

    for date_format in date_formats:
        try:
            start_date = dt.datetime.strptime(start_date_str, date_format)
            end_date = dt.datetime.strptime(end_date_str, date_format)
            return (end_date - start_date).days
        except ValueError:
            continue

    raise ValueError(
        f"Invalid date format in charge key {key}:"
        f"cannot parse dates '{start_date_str}' and '{end_date_str}'"
    )


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
    varstr_alias_func=default_varstr_alias_func,
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
        Applied to monthly charges where end_date - start_date > 1 day.
        Default is 1

    model : pyomo.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    varstr_alias_func: function
        Function to generate variable name for pyomo,
        should take in a 6 inputs and generate a string output.
        The function will receive following six inputs:

        - utility: str
        - charge_type: str
        - name: str
        - start_date: str
        - end_date: str
        - charge_limit: str

        Examples of functions:
            f_no_dates=lambda utility,
            charge_type, name,
            start_date, end_date,
            charge_limit:
            f"{utility}_{charge_type}_{name}_{charge_limit}"

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
        var_str = ut.sanitize_varstr(
            varstr_alias_func(utility, charge_type, name, eff_start, eff_end, limit_str)
        )

        # if we want itemized costs skip irrelvant portions of the bill
        if (desired_utility and utility != desired_utility) or (
            desired_charge_type and charge_type != desired_charge_type
        ):
            continue

        charge_limit = int(limit_str)
        key_substr = "_".join([utility, charge_type, name, eff_start, eff_end])
        next_limit = get_next_limit(key_substr, charge_limit, charge_dict.keys())
        consumption_data = consumption_data_dict[utility]

        # Only apply demand_scale_factor if charge spans more than one day
        charge_duration_days = get_charge_array_duration(key)
        effective_scale_factor = demand_scale_factor if charge_duration_days > 1 else 1

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
                scale_factor=effective_scale_factor,
                model=model,
                varstr=var_str,
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
                varstr=var_str,
            )
            cost += new_cost
        elif charge_type == EXPORT:
            new_cost, model = calculate_export_revenues(
                charge_array, consumption_data, divisor, model=model, varstr=var_str
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


def get_avg_value(subset_rate_data, charge_col, find_avg_by="frequency"):
    """Get the average charge value based on the specified method.

    Parameters
    ----------
    subset_rate_data : pandas.DataFrame
        Subset of rate data for a specific month/weekday/type combination
    charge_col : str
        Column name for the charge values (e.g., 'charge (metric)')
    find_avg_by : str
        Method to find average value: "minimum" or "frequency"

    Returns
    -------
    float
        The average charge value
    """
    if find_avg_by == "frequency":
        # Calculate hours covered by each charge value
        charge_hours = {}
        for _, row in subset_rate_data.iterrows():
            charge_val = row[charge_col]
            hours = row[HOUR_END] - row[HOUR_START]
            charge_hours[charge_val] = charge_hours.get(charge_val, 0) + hours

        # Find charge value that covers the most hours
        if charge_hours:
            return max(charge_hours.items(), key=lambda x: x[1])[0]
        else:
            return 0.0
    elif find_avg_by == "minimum":
        # Return minimum charge value, including 0 for hours with no charge
        if not subset_rate_data.empty:
            return subset_rate_data[charge_col].min()
        else:
            return 0.0
    else:
        raise ValueError("find_avg_by must be 'minimum' or 'frequency'")


def check_spans_full_day(day_type_rows, resolution_minutes=15):
    """Check if charges span all hours 0-24 by examining each time slot independently.

    Parameters
    ----------
    day_type_rows : pandas.DataFrame
        Subset of rate data for a specific month/weekday/type combination
    resolution_minutes : int
        Time resolution in minutes (default 15 for 15-minute intervals)

    Returns
    -------
    bool
        True if charges span all 24 hours, False otherwise
    """
    if day_type_rows.empty:
        return False

    # Create time slots from 0 to 24 hours with the specified resolution
    time_slots = np.arange(0, 24, resolution_minutes / 60)

    # Check if each time slot is covered by at least one charge
    for slot_start in time_slots:
        slot_end = slot_start + resolution_minutes / 60
        slot_covered = False

        for _, row in day_type_rows.iterrows():
            # Check if this time slot overlaps with the charge period
            charge_start = row[HOUR_START]
            charge_end = row[HOUR_END]

            # Check for overlap: slot starts before charge ends
            # AND slot ends after charge starts
            if slot_start < charge_end and slot_end > charge_start:
                slot_covered = True
                break

        if not slot_covered:
            return False

    return True


def has_any_overlap(day_type_rows, resolution_minutes=30):
    # Create time slots from 0 to 24 hours
    time_slots = np.arange(0, 24, resolution_minutes / 60)
    coverage = np.zeros_like(time_slots, dtype=int)
    for _, row in day_type_rows.iterrows():
        start = row[HOUR_START]
        end = row[HOUR_END]
        # Mark all slots covered by this charge
        covered = (time_slots >= start) & (time_slots < end)
        coverage += covered.astype(int)
    # If any slot is covered by >1 charge, there is overlap
    return np.any(coverage > 1)


def parametrize_rate_data(
    rate_data,
    peak_demand_ratio=1.0,
    peak_energy_ratio=1.0,
    avg_demand_ratio=1.0,
    avg_energy_ratio=1.0,
    peak_window_expand_hours=0,
    find_avg_by="minimum",
    resolution_minutes=30,
    variant_name=None,
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
        Int to expand or contract peak window width, to be divided evenly on both sides
        of peak period (e.g. 1/2 hr before and after for value 1, -1/2 hr for value -1).
        Positive values expand the window, negative values contract it.
        Must be in hours. Default 0
    find_avg_by: str
        Method to find average value: "minimum" or "frequency". Default "frequency"
    resolution_minutes: int
        Time resolution in minutes for checking full day coverage. Default 30
    name: str, optional
        Name for this variant. Default None

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

    # Initialize dict for each row noting whether it has already been scaled and shifted
    scaled_rows = {idx: False for idx in variant_data.index}
    shifted_rows = {idx: False for idx in variant_data.index}

    # Predefine masks
    electric_mask = variant_data[UTILITY] == ELECTRIC
    type_masks = {
        ENERGY: (variant_data[TYPE] == ENERGY),
        DEMAND: (variant_data[TYPE] == DEMAND),
    }

    # Calculate window expansion/contraction
    peak_shift_hours = round(peak_window_expand_hours, 0) / 2

    for charge_type in [ENERGY, DEMAND]:
        for month in range(1, 13):
            for weekday in range(7):
                # Get original charge rows relevant for this month & weekday
                month_weekday_type_mask = (
                    (variant_data[MONTH_START] <= month)
                    & (variant_data[MONTH_END] >= month)
                    & (variant_data[WEEKDAY_START] <= weekday)
                    & (variant_data[WEEKDAY_END] >= weekday)
                    & electric_mask
                    & type_masks[charge_type]
                )

                day_type_rows = variant_data[month_weekday_type_mask]

                if day_type_rows.empty:
                    continue

                # Check if charges span all hours 0-24 independently
                spans_full_day = check_spans_full_day(day_type_rows, resolution_minutes)

                # Check for any overlapping charges
                has_overlapping = has_any_overlap(day_type_rows, resolution_minutes)

                # SCALING LOGIC
                for charge_col in charge_cols:
                    if spans_full_day:
                        # Charges span 24 hours
                        avg_value = get_avg_value(
                            day_type_rows, charge_col, find_avg_by
                        )
                        avg_increase = (avg_ratios[charge_type] - 1) * avg_value
                        new_avg_value = avg_value + avg_increase
                        if has_overlapping:
                            # Scale average and peak charges independently
                            avg_value = get_avg_value(
                                day_type_rows, charge_col, "frequency"
                            )
                            avg_increase = (avg_ratios[charge_type] - 1) * avg_value
                            new_avg_value = avg_value + avg_increase

                            for idx in day_type_rows.index:
                                if not scaled_rows[idx]:
                                    original_charge = variant_data.loc[idx, charge_col]
                                    if np.isclose(original_charge, avg_value):
                                        variant_data.loc[idx, charge_col] = (
                                            new_avg_value
                                        )
                                    else:
                                        variant_data.loc[idx, charge_col] = (
                                            original_charge * peak_ratios[charge_type]
                                        )
                                    scaled_rows[idx] = True
                        else:
                            # Scale peak charges relative to average charge
                            for idx in day_type_rows.index:
                                if not scaled_rows[idx]:
                                    original_charge = variant_data.loc[idx, charge_col]
                                    new_value = new_avg_value + peak_ratios[
                                        charge_type
                                    ] * (original_charge - avg_value)
                                    variant_data.loc[idx, charge_col] = new_value
                                    scaled_rows[idx] = True
                    else:
                        # Charges span <24 hours
                        if has_overlapping:
                            avg_value = get_avg_value(
                                day_type_rows, charge_col, "frequency"
                            )
                            avg_increase = (avg_ratios[charge_type] - 1) * avg_value
                            new_avg_value = avg_value + avg_increase

                            for idx in day_type_rows.index:
                                if not scaled_rows[idx]:
                                    original_charge = variant_data.loc[idx, charge_col]
                                    new_peak_value = new_avg_value + peak_ratios[
                                        charge_type
                                    ] * (original_charge - avg_value)
                                    variant_data.loc[idx, charge_col] = new_peak_value
                                    scaled_rows[idx] = True
                        else:
                            avg_value = 0.0

                            for idx in day_type_rows.index:
                                if not scaled_rows[idx]:
                                    original_charge = variant_data.loc[idx, charge_col]
                                    variant_data.loc[idx, charge_col] = (
                                        original_charge * peak_ratios[charge_type]
                                    )
                                    scaled_rows[idx] = True

                # WINDOW SHIFTING LOGIC
                if peak_shift_hours != 0:
                    # Get peak periods (non-24-hour charges)
                    # for this month/weekday/type combination
                    full_day_mask = (
                        day_type_rows[HOUR_END] - day_type_rows[HOUR_START]
                    ) == 24
                    peak_periods = day_type_rows[~full_day_mask]

                    if not peak_periods.empty:
                        # Find highest charge period
                        highest_period = peak_periods.sort_values(
                            charge_cols[0], ascending=False
                        ).iloc[0]

                        # Only shift if this period hasn't been shifted before
                        if not shifted_rows[highest_period.name]:
                            orig_peak_start = highest_period[HOUR_START]
                            orig_peak_end = highest_period[HOUR_END]

                            new_peak_start = max(0, orig_peak_start - peak_shift_hours)
                            new_peak_end = min(24, orig_peak_end + peak_shift_hours)

                            # Ensure the window doesn't become invalid (start >= end)
                            if new_peak_start < new_peak_end:
                                variant_data.loc[highest_period.name, HOUR_START] = (
                                    new_peak_start
                                )
                                variant_data.loc[highest_period.name, HOUR_END] = (
                                    new_peak_end
                                )
                                shifted_rows[highest_period.name] = True

                                # Shift the end time of the average period
                                # directly before the earliest peak period
                                adjacent_avg_mask = (
                                    (
                                        variant_data[HOUR_END].apply(
                                            lambda x: np.isclose(x, orig_peak_start)
                                        )
                                    )
                                    & (variant_data[TYPE] == charge_type)
                                    & (variant_data[UTILITY] == ELECTRIC)
                                    & (variant_data[MONTH_START] <= month)
                                    & (variant_data[MONTH_END] >= month)
                                    & (variant_data[WEEKDAY_START] <= weekday)
                                    & (variant_data[WEEKDAY_END] >= weekday)
                                    & (variant_data.index != highest_period.name)
                                )
                                adjacent_avg_indices = variant_data[
                                    adjacent_avg_mask
                                ].index
                                for idx in adjacent_avg_indices:
                                    if not shifted_rows[idx]:
                                        variant_data.loc[idx, HOUR_END] = new_peak_start
                                        shifted_rows[idx] = True

                                # Shift the start time of the average period
                                # directly after the latest peak period
                                adjacent_avg_mask_after = (
                                    (
                                        variant_data[HOUR_START].apply(
                                            lambda x: np.isclose(x, orig_peak_end)
                                        )
                                    )
                                    & (variant_data[TYPE] == charge_type)
                                    & (variant_data[UTILITY] == ELECTRIC)
                                    & (variant_data[MONTH_START] <= month)
                                    & (variant_data[MONTH_END] >= month)
                                    & (variant_data[WEEKDAY_START] <= weekday)
                                    & (variant_data[WEEKDAY_END] >= weekday)
                                    & (variant_data.index != highest_period.name)
                                )
                                adjacent_avg_indices_after = variant_data[
                                    adjacent_avg_mask_after
                                ].index
                                for idx in adjacent_avg_indices_after:
                                    if not shifted_rows[idx]:
                                        variant_data.loc[idx, HOUR_START] = new_peak_end
                                        shifted_rows[idx] = True

    return variant_data


def parametrize_charge_dict(
    start_dt, end_dt, rate_data, variants=None, find_avg_by="minimum"
):
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
        - peak_window_expand_hours: int to expand or contract peak window width
          (positive values expand, negative values contract)
        - find_avg_by: str (optional) method to find average value:
          ("minimum" which looks for lowest charge value or
          "frequency which looks for the most frequent charge value")
        - resolution_minutes: int (optional) time resolution in minutes
          for checking full day coverage
        - variant_name: str (optional) variant name. (Default 'variant_{i}')
    find_avg_by : str
        Method to find average value: "minimum" or "frequency". Default "frequency"
        Used for all variants unless overridden in a variant dict.

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
        # Use variant_name if specified, otherwise use name or default 'variant_{i}'
        variant_key = (
            variant.get("variant_name") or variant.get("variant_name") or f"variant_{i}"
        )
        # Use find_avg_by from variant if present, otherwise use the top-level one
        variant_find_avg_by = variant.get("find_avg_by", find_avg_by)
        # Remove find_avg_by from variant dict to avoid duplicate argument error
        variant_no_avg = {
            k: v for k, v in variant.items() if k not in ["find_avg_by", "variant_name"]
        }
        variant_data = parametrize_rate_data(
            rate_data.copy(deep=True), **variant_no_avg, find_avg_by=variant_find_avg_by
        )

        billing_data_variants[variant_key] = variant_data.copy(deep=True)
        charge_dicts[variant_key] = get_charge_dict(start_dt, end_dt, variant_data)

    return charge_dicts

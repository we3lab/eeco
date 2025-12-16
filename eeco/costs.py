"""Functions to calculate costs from electricity consumption data."""

import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
import datetime as dt
import pyomo.environ as pyo
from itertools import compress

from .units import u
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

# Charge tier strings
PEAK = "peak"
HALF_PEAK = "half_peak"
SUPER_OFF_PEAK = "super_off_peak"
OFF_PEAK = "off_peak"


def get_charge_name(charge, index=None):
    """
    Get a unique row name for each row of charge df.

    Parameters
    ----------
    charge : dict or pandas.Series
        The charge row data containing NAME and PERIOD fields
    index : int, optional
        Index to use if name is empty or None

    Returns
    -------
    str
        A unique name with underscores converted to dashes
    """
    try:
        name = charge[NAME]
    except KeyError:
        name = charge[PERIOD]
        warnings.warn(
            "Please update billing file to use 'name' column rather than 'period'",
            DeprecationWarning,
        )

    # If no name was given, use the index to differentiate
    if not (isinstance(name, str) and name != ""):
        name = str(index) if index is not None else ""

    # Replace underscores with dashes for unique delimiter
    return name.replace("_", "-")


def create_charge_array(
    charge, datetime, effective_start_date, effective_end_date, utility="electric"
):
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
            "Please switch to new 'charge (metric)' and 'charge (imperial)' format. "
            "Current behavior assumes units of therms (imperial) "
            "and converts to cubic meters (metric) when unspecified.",
            DeprecationWarning,
        )
        # if not specified, assume the old format (imperial units)
        if utility == GAS:
            # convert from therms to default units of cubic meters
            charge[CHARGE] /= 2.83168  # therm per cubic meter of CH4
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
        cubic meter / hour (gas demand), cubic meter (gas energy),
        or $ / month (customer)

    key_str : str
        The key for the charge array we'd like to modify of the form
        `utility`_`charge_type`_`name`_`start_date`_`end_date`_`charge_limit`

    charge_array : numpy.ndarray
        Value of the charge to add in $ per kW (electric demand),
        kWh (electric energy/export), cubic meter / hour (gas demand),
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
                        "and 'basic_charge_limit (imperial)' format. "
                        "Current behavior assumes units of therms (imperial) "
                        "and converts to cubic meters (metric) when unspecified.",
                        DeprecationWarning,
                    )
                    if utility == GAS:
                        # convert from therms to default units of cubic meters
                        charge_limits *= 2.83168  # cubic meters per therm of CH4
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
                        name = get_charge_name(charge, i)

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
                                    "and 'charge (imperial)' format. Current behavior "
                                    "assumes units of therms (imperial) and converts "
                                    "to cubic meters (metric) when unspecified.",
                                    DeprecationWarning,
                                )
                            key_str = default_varstr_alias_func(
                                utility,
                                charge_type,
                                name,
                                start.strftime("%Y%m%d"),
                                end.strftime("%Y%m%d"),
                                str(int(limit)),
                            )
                            add_to_charge_array(charge_dict, key_str, charge_array)
                        elif charge_type == DEMAND and assessed == "daily":
                            for day in range((end - start).days + 1):
                                new_start = start + dt.timedelta(days=day)
                                new_end = new_start + dt.timedelta(days=1)
                                charge_array = create_charge_array(
                                    charge,
                                    datetime,
                                    new_start,
                                    new_end,
                                    utility=utility,
                                )
                                key_str = default_varstr_alias_func(
                                    utility,
                                    charge_type,
                                    name,
                                    new_start.strftime("%Y%m%d"),
                                    new_start.strftime("%Y%m%d"),
                                    str(limit),
                                )
                                add_to_charge_array(charge_dict, key_str, charge_array)
                        else:
                            charge_array = create_charge_array(
                                charge, datetime, start, new_end, utility=utility
                            )
                            key_str = default_varstr_alias_func(
                                utility,
                                charge_type,
                                name,
                                start.strftime("%Y%m%d"),
                                end.strftime("%Y%m%d"),
                                str(int(limit)),
                            )
                            add_to_charge_array(charge_dict, key_str, charge_array)
    return charge_dict


def get_charge_df(
    start_dt,
    end_dt,
    rate_data,
    resolution="15m",
    keep_fixed_charges=True,
    scale_fixed_charges=True,
    scale_demand_charges=False,
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
        If True, fixed charges will included.
        If False, fixed charges will be dropped from the output. Default is False.

    scale_fixed_charges : bool
        If True, fixed charges will be divided amongst all time steps and scaled
        by timesteps in the month. If False, they will not be scaled
        but included in the first timestep only. Default is True.

    scale_demand_charges : bool
        If True, demand charges will be scaled by the number of timesteps in the month.
        If False, they will not be scaled. Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame of charge arrays
    """
    # get the number of timesteps in a day (according to charge resolution)
    res_binsize_minutes = ut.get_freq_binsize_minutes(resolution)

    # get the charge dictionary
    charge_dict = get_charge_dict(start_dt, end_dt, rate_data, resolution=resolution)

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

    if scale_fixed_charges or scale_demand_charges:
        # calculate the scale factor
        month = start_dt.month
        year = end_dt.year
        mins_in_month = (
            (dt.date(year, month + 1, 1) - dt.date(year, month, 1)).days * 24 * 60
        )
        bins_in_month = mins_in_month / res_binsize_minutes
        scale_factor = ntsteps / bins_in_month
    else:
        scale_factor = 1.0

    if keep_fixed_charges:
        # replace the fixed charge in charge_dict with its time-averaged value
        for key, value in fixed_charge_dict.items():
            if scale_fixed_charges:
                arr = np.ones(ntsteps) * value[0] * scale_factor / ntsteps
            else:
                arr = np.zeros(ntsteps)
                arr[0] = value[0]

            charge_dict[key] = arr
    else:
        # remove fixed charges from the charge_dict
        for key in fixed_charge_dict.keys():
            del charge_dict[key]

    if scale_demand_charges:
        demand_charge_dict = {
            key: value for key, value in charge_dict.items() if "demand" in key
        }
        for key, value in demand_charge_dict.items():
            charge_dict[key] = value * scale_factor

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
    varstr="",
):
    """Calculates the cost of given demand charges for the given billing rate structure,
    utility, and consumption information

    Parameters
    ----------
    charge_array : array
        Array of charge cost (in $/kW)

    consumption_data : numpy.ndarray, cvxpy.Expression, or pyomo.environ.Var
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

    consumption_estimate : float, array
        Estimate of the total consumption from baseline data in kW, therms/hr OR m3/hr.
        Only used when `consumption_data` is cvxpy.Expression or pyomo.environ.Var
        for convex relaxation of tiered charges, while numpy.ndarray `consumption_data`
        will use actual consumption and ignore the estimate.

    scale_factor : float
        Optional factor for scaling demand charges relative to energy charges
        when the optimization/simulation period is not a full billing cycle.
        Applied to monthly charges where end_date - start_date > 1 day.
        Default is 1.

    model : pyomo.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    varstr : str
        Name of the variable to be created if using a Pyomo `model`

    Returns
    -------
    (cvxpy.Expression, pyomo.environ.Var, or float), pyomo.Model
        tuple with the first entry being a float,
        cvxpy Expression, or pyomo Var representing demand charge costs
        in USD for the given `charge_array` and `consumption_data`
        and the second entry being the pyomo model object (or None)
    """
    if isinstance(consumption_estimate, (float, int)):
        consumption_max = max(float(consumption_estimate), prev_demand)
    else:
        consumption_max = max(max(consumption_estimate), prev_demand)

    if isinstance(consumption_data, np.ndarray):
        if np.any(consumption_data < 0):
            warnings.warn(
                "Demand calculation includes negative values. Pass in "
                "positive values or run calculate_cost with a decomposition_type",
                UserWarning,
            )
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
        if consumption_max >= limit:
            if consumption_max <= next_limit:
                model.add_component(
                    varstr + "_limit",
                    pyo.Var(model.t, initialize=0, bounds=(None, None)),
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
        if consumption_max >= limit:
            if consumption_max <= next_limit:
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
            "consumption_data must be of type numpy.ndarray, "
            "cvxpy.Expression, or pyomo.environ.Var"
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
    n_per_hour,
    limit=0,
    next_limit=float("inf"),
    prev_consumption=0,
    consumption_estimate=0,
    model=None,
    varstr="",
):
    """Calculates the cost of given energy charges for the given billing rate
    structure, utility, and consumption information.

    Parameters
    ----------
    charge_array : numpy.ndarray
        Array of the charges in $/kWh for electric and $/cubic meter for gas

    consumption_data : numpy.ndarray cvxpy.Expression, or pyomo.environ.Var
        Baseline electrical or gas usage data as an optimization variable object

    n_per_hour : int
        Divisor for the indexed consumption data elements,
        based on the timeseries resolution

    limit : float
        The total consumption, or limit, that this charge came into effect.
        Default is 0

    next_limit : float
        The total consumption, or limit, that the next charge comes into effect.
        Default is float('inf') indicating that there is no higher tier

    prev_consumption : float
        Consumption from within this billing period but outside the horizon window
        (e.g., previously in the month). Necessary for moving-horizon optimization.
        Default is 0

    consumption_estimate : float, array
        Estimate of the total monthly energy consumption from baseline data
        in kWh, therms OR cubic meters.
        Only used when `consumption_data` is cvxpy.Expression or pyomo.environ.Var
        for convex relaxation of tiered charges, while numpy.ndarray `consumption_data`
        will use actual consumption and ignore the estimate.

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
    (cvxpy.Expression, pyomo.environ.Var, or float), pyomo.Model
        tuple with the first entry being a float,
        cvxpy Expression, or pyomo Var representing energy charge costs
        in USD for the given `charge_array` and `consumption_data`
        and the second entry being the pyomo model object (or None)
    """
    cost = 0
    if model is None:
        n_steps = consumption_data.shape[0]
    else:  # Pyomo does not support shape attribute
        n_steps = len(consumption_data)

    if isinstance(consumption_data, np.ndarray):
        if np.any(consumption_data < 0):
            warnings.warn(
                "Energy calculation includes negative values. Pass in "
                "positive values or run calculate_cost with a decomposition_type",
                UserWarning,
            )

        energy = prev_consumption
        # set the flag if we are starting with previous consumption that lands us
        # within the current tier of charge limits
        within_limit_flag = energy >= float(limit) and energy < float(next_limit)
        for i in range(n_steps):
            energy += consumption_data[i] / n_per_hour
            # only add to charges if already within correct charge limits
            if within_limit_flag:
                # went over next charge limit on this iteration
                # set flag to false to avoid overcounting after this iteration
                if energy >= float(next_limit):
                    within_limit_flag = False
                    cost += (
                        max(
                            float(next_limit)
                            + consumption_data[i] / n_per_hour
                            - energy,
                            0,
                        )
                        * charge_array[i]
                    )
                else:
                    cost += consumption_data[i] / n_per_hour * charge_array[i]
            # went over existing charge limit on this iteration
            elif energy >= float(limit) and energy < float(next_limit):
                within_limit_flag = True
                cost += max(energy - float(limit), 0) * charge_array[i]

    elif isinstance(consumption_data, (cp.Expression, pyo.Var, pyo.Param)):
        # For tiered charges, approximate extimated consumption being split evenly
        # Only necessary if we have a finite next_limit OR if current limit > 0
        # NOTE: this convex approximation breaks global optimality guarantees
        if not np.isinf(next_limit) or (not np.isinf(limit) and limit > 0):
            if isinstance(consumption_estimate, (float, int)):
                consumption_per_timestep = consumption_estimate / n_steps
                consumption_estimate = np.ones(n_steps) * consumption_per_timestep

            cumulative_consumption = np.cumsum(consumption_estimate) + prev_consumption
            total_consumption = cumulative_consumption[-1]

            start_idx = np.argmax(cumulative_consumption > float(limit))
            # if not found argmax returns 0, but whole charge array should be zeroed
            if (start_idx == 0) and (total_consumption <= float(limit)):
                charge_array[:] = 0
            else:
                charge_array[:start_idx] = 0  # 0 for charge array before start index
            end_idx = np.argmax(cumulative_consumption > float(next_limit))
            # before applying end_idx ensure (1) that next limit exists
            # and (2) that it is higher than total consumption
            if not (np.isinf(next_limit) or (total_consumption <= float(next_limit))):
                charge_array[end_idx:] = 0  # 0 for charge array after end index

        charge_expr, model = ut.multiply(
            consumption_data, charge_array, model=model, varstr=varstr + "_multiply"
        )
        sum_result, model = ut.sum(charge_expr, model=model, varstr=varstr + "_sum")
        cost, model = ut.max_pos(
            sum_result / n_per_hour,
            model=model,
            varstr=varstr,
        )
    else:
        raise ValueError(
            "consumption_data must be of type numpy.ndarray, "
            "cvxpy.Expression, or pyomo.environ.Var"
        )
    return cost, model


def calculate_export_revenue(
    charge_array, consumption_data, n_per_hour, model=None, varstr=""
):
    """Calculates the export revenues for the given billing rate structure,
    utility, and consumption information.

    Only flat rates for exports are supported (in $ / kWh).
    Returns positive revenue values that should be subtracted from total cost.

    Parameters
    ----------
    charge_array : numpy.ndarray
        array with price per kWh sold back to the grid

    consumption_data : numpy.ndarray, cvxpy.Expression, or pyomo.environ.Var
        Magnitude of exported electrical or gas usage data
        as an optimization variable object.
        Should be positive values.

    n_per_hour : int
        Divisor for the indexed consumption data elements,
        based on the timeseries resolution

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
    (cvxpy.Expression, pyomo.environ.Var, or float), pyomo.Model
        tuple with the first entry being a float,
        cvxpy Expression, or pyomo Var representing export revenues
        in USD for the given `charge_array` and `export_data`
        and the second entry being the pyomo model object (or None)
    """
    if isinstance(consumption_data, np.ndarray):
        if np.any(consumption_data < 0):
            warnings.warn(
                "Export revenue calculation includes negative values. Pass in "
                "positive values or run calculate_cost with a decomposition_type",
                UserWarning,
            )
        return np.sum(consumption_data * charge_array) / n_per_hour, model

    elif isinstance(consumption_data, (cp.Expression, pyo.Var, pyo.Param)):
        cost_expr, model = ut.multiply(
            consumption_data,
            charge_array,
            model=model,
            varstr=varstr + "_multiply",
        )
        export_revenue, model = ut.sum(cost_expr, model=model, varstr=varstr + "_sum")

        return export_revenue / n_per_hour, model
    else:
        raise ValueError(
            "consumption_data must be of type numpy.ndarray, "
            "cvxpy.Expression, or pyomo.environ.Var"
        )


def get_conversion_factors(electric_consumption_units, gas_consumption_units):
    """Calculate conversion factors for electric and gas utilities.

    Parameters
    ----------
    electric_consumption_units : pint.Unit
        Units for the electricity consumption data
    gas_consumption_units : pint.Unit
        Units for the gas consumption data

    Returns
    -------
    dict
        Dictionary with conversion factors for each utility type
    """
    conversion_factors = {}
    conversion_factors[ELECTRIC] = (1 * electric_consumption_units).to(u.kW).magnitude
    conversion_factors[GAS] = (
        (1 * gas_consumption_units).to(u.meter**3 / u.hour).magnitude
    )
    return conversion_factors


def get_converted_consumption_data(
    consumption_data_dict, conversion_factors, decomposition_type, model=None
):
    """Ensure consumption_data_dict has imports/exports structure
    with proper unit conversion.

    Parameters
    ----------
    consumption_data_dict : dict
        Dictionary with utility keys and consumption data values
    conversion_factors : dict
        Dictionary with conversion factors for each utility type
    decomposition_type : str or None
        Type of decomposition to apply (e.g., "absolute_value")
    model : pyomo.Model, optional
        Pyomo model object for optimization variables

    Returns
    -------
    dict
        Updated consumption_data_dict with imports/exports structure
    """
    for utility in consumption_data_dict.keys():
        conversion_factor = conversion_factors[utility]

        # Check if this utility already has imports/exports structure
        if (
            isinstance(consumption_data_dict[utility], dict)
            and "imports" in consumption_data_dict[utility]
            and "exports" in consumption_data_dict[utility]
        ):
            # Apply conversion if conversion factor is nonzero
            if conversion_factor != 1.0:
                # Apply conversion if not already done
                imports_varstr = utility + "_imports_converted"
                if model is None or not hasattr(model, imports_varstr):
                    consumption_data_dict[utility]["imports"], model = ut.multiply(
                        consumption_data_dict[utility]["imports"],
                        conversion_factor,
                        model=model,
                        varstr=imports_varstr,
                    )
                    consumption_data_dict[utility]["exports"], model = ut.multiply(
                        consumption_data_dict[utility]["exports"],
                        conversion_factor,
                        model=model,
                        varstr=utility + "_exports_converted",
                    )
            continue
        else:  # create imports/exports
            # Convert if not already done
            converted_varstr = utility + "_converted"
            if model is None or not hasattr(model, converted_varstr):
                converted_consumption, model = ut.multiply(
                    consumption_data_dict[utility],
                    conversion_factor,
                    model=model,
                    varstr=converted_varstr,
                )

            if decomposition_type == "absolute_value":
                # Decompose consumption data into positive and negative components
                # with constraint that total = positive - negative
                # (where negative is stored as positive magnitude)
                pos_name, neg_name = ut.get_decomposed_var_names(utility)

                # Decompose if not already done
                if model is None or not hasattr(model, pos_name):
                    imports, exports, model = ut.decompose_consumption(
                        converted_consumption,
                        model=model,
                        varstr=utility,
                        decomposition_type="absolute_value",
                    )

                consumption_data_dict[utility] = {
                    "imports": imports,
                    "exports": exports,
                }
            elif decomposition_type is None:
                consumption_data_dict[utility] = {
                    "imports": converted_consumption,
                    "exports": converted_consumption,
                }
            else:
                raise NotImplementedError

    return consumption_data_dict


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
    electric_consumption_units=u.kW,
    gas_consumption_units=u.meters**3 / u.hour,
    resolution="15m",
    prev_demand_dict=None,
    prev_consumption_dict=None,
    consumption_estimate=0,
    desired_utility=None,
    desired_charge_type=None,
    demand_scale_factor=1,
    model=None,
    decomposition_type=None,
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
        cubic meter / hour (gas demand), cubic meter (gas energy),
        or $ / month (customer)

    consumption_data_dict : dict
        Baseline electrical and gas usage data as an optimization variable object
        with keys "electric" and "gas". Supports two formats:

        Default format:
            Values are cumulative consumption data
            Example: {"electric": np.array([]), "gas": np.array([])}

        Extended format for imports/exports:
            Values are dictionaries with decomposed "imports" and "exports" keys
            Example: {"electric": {"imports": np.array([]), "exports": np.array([])}}

        Values must be of type numpy.ndarray, cvxpy.Expression, or pyomo.environ.Var.

    electric_consumption_units : pint.Unit
        Units for the electricity consumption data. Default is kW

    gas_consumption_units : pint.Unit
        Units for the natural gas consumption data. Default is cubic meters / hour

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

    consumption_estimate : float, array, dict
        Estimate of the total monthly demand or energy consumption from baseline data.
        Only used when `consumption_data` is cvxpy.Expression or pyomo.environ.Var
        for convex relaxation of tiered charges, while numpy.ndarray `consumption_data`
        will use actual consumption and ignore the estimate. If consumption estimate is
        a float, int, it is assumed to carry units of kWh, therms OR m3. If consumption
        estimate is array-like it is assumed to carry units of kW, therms/hr OR m3/hr.
        If consumption estimate is a dict, it should have keys matching the charge dict.

    desired_charge_type : list or str
        Name of desired charge type for itemized costs.
        Either 'customer', 'energy', 'demand', 'export', or a list of charge types.
        Default is None, meaning that all costs will be summed together.

    desired_utility : list or str
        Name of desired utility for itemized costs.
        Either 'electric', 'gas', or a list of utilities (e.g., ['electric', 'gas']).
        Default is None, meaning that all costs will be summed together.

    demand_scale_factor : float
        Optional factor for scaling demand charges relative to energy charges
        when the optimization/simulation period is not a full billing cycle.
        Applied to monthly charges where end_date - start_date > 1 day.
        Default is 1

    model : pyomo.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    decomposition_type : str or None
        Type of decomposition to use for consumption data.
        - "absolute_value": Linear problem using absolute value
        - "binary_variable": `NotImplementedError`
        - None (default): No decomposition, treats all consumption as imports

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
    (numpy.Array, cvxpy.Expression, or pyomo.environ.Var),  pyomo.Model
        tuple with the first entry being a float,
        cvxpy Expression, or pyomo Var representing energy charge costs
        in USD for the given `charge_array` and `consumption_data`
        and the second entry being the pyomo model object (or None)
    """
    cost = 0
    n_per_hour = int(60 / ut.get_freq_binsize_minutes(resolution))

    if consumption_estimate is None:
        consumption_estimate = 0

    conversion_factors = get_conversion_factors(
        electric_consumption_units, gas_consumption_units
    )

    consumption_data_dict = get_converted_consumption_data(
        consumption_data_dict, conversion_factors, decomposition_type, model
    )

    for key, charge_array in charge_dict.items():
        utility, charge_type, name, eff_start, eff_end, limit_str = key.split("_")
        varstr = ut.sanitize_varstr(
            varstr_alias_func(
                utility, charge_type, name, eff_start, eff_end, limit_str
            )  # noqa: E501
        )

        # if we want itemized costs skip irrelvant portions of the bill
        if (desired_utility and utility not in desired_utility) or (
            desired_charge_type and charge_type not in desired_charge_type
        ):
            continue

        charge_limit = int(limit_str)
        key_substr = "_".join([utility, charge_type, name, eff_start, eff_end])
        next_limit = get_next_limit(key_substr, charge_limit, charge_dict.keys())

        # Only apply demand_scale_factor if charge spans more than one day
        charge_duration_days = get_charge_array_duration(key)
        effective_scale_factor = demand_scale_factor if charge_duration_days > 1 else 1

        if charge_type == DEMAND:
            if prev_demand_dict is not None:
                prev_demand = prev_demand_dict[key][DEMAND]
                prev_demand_cost = prev_demand_dict[key]["cost"]
            else:
                prev_demand = 0
                prev_demand_cost = 0

            if isinstance(consumption_estimate, (float, int)):
                # convert single kWh to the equivalent kW per timestep
                demand_consumption_estimate = (
                    consumption_estimate * n_per_hour / len(charge_array)
                )
            elif isinstance(consumption_estimate, (dict)):
                demand_consumption_estimate = consumption_estimate[utility]
                if isinstance(demand_consumption_estimate, (float, int)):
                    demand_consumption_estimate = (
                        demand_consumption_estimate * n_per_hour / len(charge_array)
                    )
            else:
                demand_consumption_estimate = consumption_estimate

            new_cost, model = calculate_demand_cost(
                charge_array,
                consumption_data_dict[utility]["imports"],
                limit=charge_limit,
                next_limit=next_limit,
                prev_demand=prev_demand,
                prev_demand_cost=prev_demand_cost,
                consumption_estimate=demand_consumption_estimate,
                scale_factor=effective_scale_factor,
                model=model,
                varstr=varstr,
            )
            cost += new_cost
        elif charge_type == ENERGY:
            if prev_consumption_dict is not None:
                prev_consumption = prev_consumption_dict[key]
            else:
                prev_consumption = 0

            if isinstance(consumption_estimate, (float, int)):
                # assumed to be kWh or therms/m3 per month
                energy_consumption_estimate = consumption_estimate
            elif isinstance(consumption_estimate, (dict)):
                energy_consumption_estimate = consumption_estimate[utility]
                if not isinstance(
                    energy_consumption_estimate, (float, int)
                ):  # array-like
                    energy_consumption_estimate = (
                        energy_consumption_estimate / n_per_hour
                    )
            else:
                energy_consumption_estimate = consumption_estimate / n_per_hour

            new_cost, model = calculate_energy_cost(
                charge_array,
                consumption_data_dict[utility]["imports"],
                n_per_hour,
                limit=charge_limit,
                next_limit=next_limit,
                prev_consumption=prev_consumption,
                consumption_estimate=energy_consumption_estimate,
                model=model,
                varstr=varstr,
            )
            cost += new_cost
        elif charge_type == EXPORT:
            new_cost, model = calculate_export_revenue(
                charge_array,
                consumption_data_dict[utility]["exports"],
                n_per_hour,
                model=model,
                varstr=varstr,
            )
            cost -= new_cost
        elif charge_type == CUSTOMER:
            cost += charge_array.sum()
        else:
            raise ValueError("Invalid charge_type: " + charge_type)

    return cost, model


def build_pyomo_costing(
    charge_dict,
    consumption_data_dict,
    model,
    electric_consumption_units=u.kW,
    gas_consumption_units=u.meters**3 / u.hour,
    resolution="15m",
    prev_demand_dict=None,
    prev_consumption_dict=None,
    consumption_estimate=0,
    desired_utility=None,
    desired_charge_type=None,
    demand_scale_factor=1,
    additional_objective_terms=None,
    decomposition_type=None,
    varstr_alias_func=default_varstr_alias_func,
):
    """
    Wrapper for `calculate_cost` to build the cost components into a Pyomo model.
    An objective function will be created in `model.objective`.

    Parameters
    ----------
    charge_dict : dict
        dictionary of arrays with keys of the form
        `utility`_`charge_type`_`name`_`start_date`_`end_date`_`charge_limit`
        and values being the $ per kW (electric demand), kWh (electric energy/export),
        cubic meter / hour (gas demand), cubic meter (gas energy),
        or $ / month (customer)

    consumption_data_dict : dict
        Baseline electrical and gas usage data as an optimization variable object
        with keys "electric" and "gas". Values of the dictionary must be of type
        numpy.ndarray, cvxpy.Expression, or pyomo.environ.Var

    model : pyomo.Model
        The model object associated with the problem.

    electric_consumption_units : pint.Unit
        Units for the electricity consumption data. Default is kW

    gas_consumption_units : pint.Unit
        Units for the natural gas consumption data. Default is cubic meters / hour

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
        Only used when `consumption_data` is cvxpy.Expression or pyomo.environ.Var
        for convex relaxation of tiered charges, while numpy.ndarray `consumption_data`
        will use actual consumption and ignore the estimate.

    desired_charge_type : list or str
        Name of desired charge type for itemized costs.
        Either 'customer', 'energy', 'demand', 'export', or a list of charge types.
        Default is None, meaning that all costs will be summed together.

    desired_utility : list or str
        Name of desired utility for itemized costs.
        Either 'electric', 'gas', or a list of utilities (e.g., ['electric', 'gas']).
        Default is None, meaning that all costs will be summed together.

    demand_scale_factor : float
        Optional factor for scaling demand charges relative to energy charges
        when the optimization/simulation period is not a full billing cycle.
        Applied to monthly charges where end_date - start_date > 1 day.
        Default is 1

    additional_objective_terms : pyomo.Expression, list
        Additional terms to be added to the objective function.
        Can be a single pyomo Expression or a list of pyomo Expressions.
        Only used in the case of Pyomo, so `None` by default.

    decomposition_type : str or None
        Type of decomposition to use for consumption data.
        - "absolute_value": Linear problem using absolute value
        - "binary_variable": `NotImplementedError`
        - None (default): No decomposition, treats all consumption as imports

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
    pyomo.Model
        The model object associated with the problem with costing components added.
    """
    model.electricity_cost, model = calculate_cost(
        charge_dict=charge_dict,
        consumption_data_dict=consumption_data_dict,
        electric_consumption_units=electric_consumption_units,
        gas_consumption_units=gas_consumption_units,
        resolution=resolution,
        prev_demand_dict=prev_demand_dict,
        prev_consumption_dict=prev_consumption_dict,
        consumption_estimate=consumption_estimate,
        desired_utility=desired_utility,
        desired_charge_type=desired_charge_type,
        demand_scale_factor=demand_scale_factor,
        model=model,
        decomposition_type=decomposition_type,
        varstr_alias_func=varstr_alias_func,
    )

    model.objective = pyo.Objective(expr=model.electricity_cost, sense=pyo.minimize)
    if additional_objective_terms is not None:
        for term in additional_objective_terms:
            model.objective.expr += term
    return model


def calculate_itemized_cost(
    charge_dict,
    consumption_data_dict,
    electric_consumption_units=u.kW,
    gas_consumption_units=u.meters**3 / u.hour,
    resolution="15m",
    prev_demand_dict=None,
    prev_consumption_dict=None,
    consumption_estimate=0,
    desired_utility=None,
    demand_scale_factor=1,
    model=None,
    decomposition_type=None,
    varstr_alias_func=default_varstr_alias_func,
):
    """Calculates itemized costs as a nested dictionary

    Parameters
    ----------
    charge_dict : dict
        dictionary of arrays with keys of the form
        `utility`_`charge_type`_`name`_`start_date`_`end_date`_`charge_limit`
        and values being the $ per kW (electric demand), kWh (electric energy/export),
        cubic meter / hour (gas demand), cubic meter (gas energy),
        or $ / month (customer)

    consumption_data_dict : dict
        Baseline electrical and gas usage data as an optimization variable object
        with keys "electric" and "gas". Values of the dictionary must be of type
        numpy.ndarray, cvxpy.Expression, or pyomo.environ.Var
        Positive values represent energy imports (consumption from the grid)
        Negative values represent energy exports (generation sent to the grid)

    electric_consumption_units : pint.Unit
        Units for the electricity consumption data. Default is kW

    gas_consumption_units : pint.Unit
        Units for the natural gas consumption data. Default is cubic meters / hour

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

    demand_scale_factor : float
        Optional factor for scaling demand charges relative to energy charges
        when the optimization/simulation period is not a full billing cycle.
        Applied to monthly charges where end_date - start_date > 1 day.
        Default is 1

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
    # Check if decomposition_type is used with CVXPY objects
    # (not yet supported because imports - exports creates non-DCP issues)
    if decomposition_type is not None:
        for utility in consumption_data_dict.keys():
            if isinstance(consumption_data_dict[utility], cp.Variable):
                raise NotImplementedError(
                    "Decomposition types are not supported with CVXPY objects. "
                    "Use Pyomo instead for problems requiring decomposition_type."
                )

    conversion_factors = get_conversion_factors(
        electric_consumption_units, gas_consumption_units
    )

    consumption_data_dict = get_converted_consumption_data(
        consumption_data_dict, conversion_factors, decomposition_type, model
    )

    total_cost = 0
    results_dict = {}

    if desired_utility is None:
        for utility in [ELECTRIC, GAS]:
            results_dict[utility] = {}
            total_utility_cost = 0
            for charge_type in [CUSTOMER, ENERGY, DEMAND, EXPORT]:
                cost, model = calculate_cost(
                    charge_dict,
                    consumption_data_dict,
                    resolution=resolution,
                    prev_demand_dict=prev_demand_dict,
                    consumption_estimate=consumption_estimate,
                    desired_utility=utility,
                    desired_charge_type=charge_type,
                    demand_scale_factor=demand_scale_factor,
                    model=model,
                    decomposition_type=decomposition_type,
                    varstr_alias_func=varstr_alias_func,
                )

                results_dict[utility][charge_type] = cost
                total_utility_cost += cost

            results_dict[utility]["total"] = total_utility_cost
            total_cost += total_utility_cost
    else:
        results_dict[desired_utility] = {}
        total_utility_cost = 0
        for charge_type in [CUSTOMER, ENERGY, DEMAND, EXPORT]:
            cost, model = calculate_cost(
                charge_dict,
                consumption_data_dict,
                resolution=resolution,
                prev_demand_dict=prev_demand_dict,
                prev_consumption_dict=prev_consumption_dict,
                consumption_estimate=consumption_estimate,
                desired_utility=desired_utility,
                desired_charge_type=charge_type,
                demand_scale_factor=demand_scale_factor,
                model=model,
                decomposition_type=decomposition_type,
                varstr_alias_func=varstr_alias_func,
            )

            results_dict[desired_utility][charge_type] = cost
            total_utility_cost += cost

        results_dict[desired_utility]["total"] = total_utility_cost
        total_cost += total_utility_cost

    results_dict["total"] = total_cost
    return results_dict, model


def detect_charge_periods(
    rate_data, charge_type, month, weekday, resolution_minutes=30
):
    """Categorize charges into charge periods
    (peak, half-peak, off-peak, super-off-peak)
    for a particular month, weekday and charge_type.

    Parameters
    ----------
    rate_data : pandas.DataFrame
        Tariff data with required columns
    charge_type : str
        Type of charge (ENERGY or DEMAND)
    month : int
        Month (1-12)
    weekday : int
        Weekday (0-6)
    resolution_minutes : int
        Time resolution in minutes for checking full day coverage. Default 30

    Returns
    -------
    tuple with:
        - period_categories: dict with period classifications for each row index:
          - PEAK: highest charge periods
          - HALF_PEAK: periods adjacent to peak
          - OFF_PEAK: average charge periods
          - SUPER_OFF_PEAK: below average charge periods
        - base_charge: the most frequent charge value
        - has_overlapping: the most frequent charge value
    """

    # Get charge columns for the tariff csv
    if CHARGE_METRIC in rate_data.columns:
        charge_col = CHARGE_METRIC  # if csv includes metric and imperial
    else:
        charge_col = CHARGE  # if csv units are unspecified

    # Get rows for this month, weekday, and type
    electric_mask = rate_data[UTILITY] == ELECTRIC
    type_mask = rate_data[TYPE] == charge_type
    month_weekday_mask = (
        (rate_data[MONTH_START] <= month)
        & (rate_data[MONTH_END] >= month)
        & (rate_data[WEEKDAY_START] <= weekday)
        & (rate_data[WEEKDAY_END] >= weekday)
        & electric_mask
        & type_mask
    )
    day_type_rows = rate_data[month_weekday_mask]

    if day_type_rows.empty:
        return {}, None, False

    # Check for any overlapping charges
    has_overlapping = False
    time_slots = np.arange(0, 24, resolution_minutes / 60)
    if len(day_type_rows) > 1:
        # Check for overlap using same time slots
        coverage = np.zeros_like(time_slots, dtype=int)
        for _, row in day_type_rows.iterrows():
            start, end = row[HOUR_START], row[HOUR_END]
            covered = (time_slots >= start) & (time_slots < end)
            coverage += covered.astype(int)
            if np.any(coverage > 1):
                has_overlapping = True
                break

    # Calculate average charge (most frequently occurring)
    avg_charge = 0.0
    charge_hours = {}
    for _, row in day_type_rows.iterrows():
        charge_val = row[charge_col]
        hours = row[HOUR_END] - row[HOUR_START]
        charge_hours[charge_val] = charge_hours.get(charge_val, 0) + hours
    if charge_hours:
        avg_charge = max(charge_hours.items(), key=lambda x: x[1])[0]

    # Get unique charge values and their frequencies
    charge_values = day_type_rows[charge_col].values
    unique_charges, counts = np.unique(charge_values, return_counts=True)

    # Sort charges in descending order by charge value
    sorted_indices = np.argsort(unique_charges)[::-1]
    unique_charges = unique_charges[sorted_indices]
    counts = counts[sorted_indices]

    # Classify periods
    period_categories = {}
    for idx in day_type_rows.index:
        charge_value = rate_data.loc[idx, charge_col]
        if has_overlapping:  # Overlapping charges
            # Check if this is a 24-hour charge (off-peak charge)
            hour_span = rate_data.loc[idx, HOUR_END] - rate_data.loc[idx, HOUR_START]
            if hour_span == 24:
                period_categories[idx] = OFF_PEAK
            else:  # This is an additional charge on top of the base
                non_24h_charges = [
                    rate_data.loc[other_idx, charge_col]
                    for other_idx in day_type_rows.index
                    if not np.isclose(
                        rate_data.loc[other_idx, HOUR_END]
                        - rate_data.loc[other_idx, HOUR_START],
                        24,
                    )
                ]
                if non_24h_charges:  # if peaks exist, PEAK is max
                    max_non_24h = max(non_24h_charges)
                    period_categories[idx] = (
                        PEAK if np.isclose(charge_value, max_non_24h) else HALF_PEAK
                    )
                else:
                    period_categories[idx] = HALF_PEAK
        else:  # Non-overlapping charges
            if charge_value > avg_charge:
                # Above "average" includes peak and half-peak
                if charge_value == unique_charges[0]:  # Maximum
                    period_categories[idx] = PEAK
                else:
                    period_categories[idx] = HALF_PEAK
            elif charge_value < avg_charge:
                # Below "average" includes super off-peak
                period_categories[idx] = SUPER_OFF_PEAK
            else:
                period_categories[idx] = OFF_PEAK

    return period_categories, avg_charge, has_overlapping


def parametrize_rate_data(
    rate_data,
    percent_change_dict={},
    shift_peak_hours_before=0,
    shift_peak_hours_after=0,
    variant_name=None,
):
    """Parametrize rate data by charge periods
    (peak, half-peak, off-peak, super-off-peak) or exact charge keys.
    Applies scaling and window shifting to create
    alternative rate structures.

    Parameters
    ----------
    rate_data : pandas.DataFrame
        Tariff data with required columns
    percent_change_dict : dict, optional
        Dictionary for charge scaling. Can be one of three formats:

        Format 1 - Nested dictionary with structure for charge scaling:

            {

                'demand': {

                    'peak': float,
                    'half_peak': float,
                    'off_peak': float,
                    'super_off_peak': float

                },

                'energy': {

                    'peak': float,
                    'half_peak': float,
                    'off_peak': float,
                    'super_off_peak': float

                }

            }

        Format 2 - Dictionary with exact charge key prefixes based on csv:

            {

                'electric_demand_peak-summer': float,
                'electric_energy_0': float,
                'electric_demand_all-day': float,
                ...

            }

        Format 3 - Global scaling for all charges of each type:

            {

                'demand': float,  # scales all demand charges
                'energy': float,   # scales all energy charges

            }

        If None, all ratios default to 1.0

    shift_peak_hours_before : float, optional
        Hours to shift peak window start
        (negative=earlier, positive=later).
        Must be multiple of 0.25 hours. Default 0

    shift_peak_hours_after : float, optional
        Hours to shift peak window end
        (negative=earlier, positive=later).
        Must be multiple of 0.25 hours. Default 0

    variant_name : str, optional
        Name for this variant. Default None

    Returns
    -------
    pandas.DataFrame
        Updated rate_data dataframe with
        parametrized charges and windows

    Raises
    ------
    ValueError
        If percent_change_dict contains both period-based scaling and individual charge
        scaling for the same charge type
    UserWarning
        If percent_change_dict contains exact charge keys that are not found in the data
    """
    variant_data = rate_data.copy(deep=True)  # deep copy required for variants
    variant_data[HOUR_START] = variant_data[HOUR_START].astype(float)
    variant_data[HOUR_END] = variant_data[HOUR_END].astype(float)

    charge_cols = (
        [CHARGE_METRIC, CHARGE_IMPERIAL]
        if CHARGE_METRIC in variant_data.columns
        else [CHARGE]
    )

    # Determine which format percent_change_dict was passed in
    has_exact_keys = len(percent_change_dict) > 0 and any(
        isinstance(k, str) and ("electric_" in k or "gas_" in k)
        for k in percent_change_dict.keys()
    )

    has_global_scaling = len(percent_change_dict) > 0 and any(
        k in [DEMAND, ENERGY] and isinstance(v, (int, float))
        for k, v in percent_change_dict.items()
    )

    has_period_scaling = len(percent_change_dict) > 0 and any(
        isinstance(v, dict) and k in [DEMAND, ENERGY]
        for k, v in percent_change_dict.items()
    )

    # Check for conflicts between period/global scaling and exact keys
    if has_exact_keys and (has_period_scaling or has_global_scaling):
        raise ValueError(
            "percent_change_dict cannot contain both exact charge keys"
            " and global/period-based scaling"
        )

    # Cache original charges and track processed/shifted rows
    original_charges = {
        col: variant_data[col].copy()
        for col in charge_cols
        if col in variant_data.columns
    }
    processed_rows = set()
    missing_keys = set()
    shifted_rows = {ENERGY: set(), DEMAND: set()}

    # Process each month, weekday, charge_type
    for charge_type in [ENERGY, DEMAND]:
        if (
            has_global_scaling
            and charge_type in percent_change_dict
            and isinstance(percent_change_dict[charge_type], (int, float))
        ):
            # Format 3: Global scaling for all charges of this type
            scale_factor = percent_change_dict[charge_type]
            charge_ratios = {
                PEAK: scale_factor,
                HALF_PEAK: scale_factor,
                OFF_PEAK: scale_factor,
                SUPER_OFF_PEAK: scale_factor,
            }
        # Format 2: Exact charge keys
        elif has_exact_keys:
            charge_ratios = percent_change_dict
        # Format 1: Period-based scaling
        elif has_period_scaling and charge_type in percent_change_dict:
            charge_ratios = percent_change_dict[charge_type]
        else:  # No scaling - window shifting only
            charge_ratios = {
                PEAK: 1.0,
                HALF_PEAK: 1.0,
                OFF_PEAK: 1.0,
                SUPER_OFF_PEAK: 1.0,
            }

        # Loop through each combination of month & weekday
        for month in range(1, 13):
            for weekday in range(7):
                period_categories, base_charge, has_overlapping = detect_charge_periods(
                    variant_data, charge_type, month, weekday, 30
                )

                if not period_categories:  # No charges for this month/weekday combo
                    continue

                # SCALING LOGIC
                ratio = 1.0  # default if no scaling
                for row_idx, period in period_categories.items():
                    if row_idx in processed_rows:  # Skip if already processed
                        continue

                    row = variant_data.loc[row_idx]

                    if has_exact_keys:  # Format 2: Exact charge key matching
                        row_name = get_charge_name(row, row_idx)
                        for key_prefix, key_ratio in charge_ratios.items():
                            if key_prefix.startswith(
                                f"{row[UTILITY]}_{row[TYPE]}_{row_name}"
                            ):
                                ratio = key_ratio
                                # Remove dashes in the df when key is found
                                variant_data.loc[row_idx, NAME] = row_name
                                break
                        else:
                            missing_keys.add(f"{row[UTILITY]}_{row[TYPE]}_{row_name}")
                    elif charge_ratios:  # Format 1: Period-based approach
                        ratio = charge_ratios[period]

                    for col in charge_cols:
                        if col in variant_data.columns:
                            current_charge = original_charges[col][row_idx]
                            if has_overlapping:  # Scale each charge independently
                                variant_data.loc[row_idx, col] = current_charge * ratio
                            else:  # Scale relative to average charge
                                if np.isclose(
                                    current_charge, base_charge
                                ):  # If average
                                    variant_data.loc[row_idx, col] = (
                                        current_charge * ratio
                                    )
                                elif current_charge > base_charge:  # If peak period
                                    adder = current_charge - base_charge
                                    variant_data.loc[row_idx, col] = (
                                        base_charge + adder * ratio
                                    )
                                else:  # If super off-peak period (below average)
                                    discount = base_charge - current_charge
                                    variant_data.loc[row_idx, col] = (
                                        base_charge - discount * (1 / ratio)
                                    )
                    processed_rows.add(row_idx)

                # WINDOW SHIFTING LOGIC
                if shift_peak_hours_before != 0 or shift_peak_hours_after != 0:
                    peak_period_indices = [
                        idx
                        for idx, period in period_categories.items()
                        if period == PEAK
                    ]

                    if peak_period_indices:
                        peak_period_idx = peak_period_indices[
                            0
                        ]  # Assuming one peak period
                        if (
                            peak_period_idx not in shifted_rows[charge_type]
                        ):  # Not yet shifted
                            peak_period = variant_data.loc[peak_period_idx]
                            orig_peak_start, orig_peak_end = (
                                peak_period[HOUR_START],
                                peak_period[HOUR_END],
                            )
                            new_peak_start = max(
                                0, orig_peak_start + shift_peak_hours_before
                            )
                            new_peak_end = min(
                                24, orig_peak_end + shift_peak_hours_after
                            )
                            if (
                                new_peak_start >= new_peak_end
                            ):  # Invalid window with zero duration
                                variant_data = variant_data.drop(
                                    peak_period_idx
                                )  # Drop the row
                                warnings.warn(
                                    f"Peak window was shifted to zero width for"
                                    f"{charge_type} in month {month} weekday {weekday}",
                                    UserWarning,
                                )
                            else:
                                variant_data.loc[peak_period_idx, HOUR_START] = (
                                    new_peak_start
                                )
                                variant_data.loc[peak_period_idx, HOUR_END] = (
                                    new_peak_end
                                )
                                shifted_rows[charge_type].add(peak_period_idx)

                                # Shift half-peaks, assuming they are adjacent to peak
                                half_peak_indices = [
                                    idx
                                    for idx, period in period_categories.items()
                                    if period == HALF_PEAK
                                ]
                                for idx in half_peak_indices:
                                    if idx not in shifted_rows[charge_type]:
                                        row = variant_data.loc[idx]
                                        if np.isclose(row[HOUR_END], orig_peak_start):
                                            variant_data.loc[idx, HOUR_END] = (
                                                new_peak_start
                                            )
                                            shifted_rows[charge_type].add(idx)
                                        elif np.isclose(row[HOUR_START], orig_peak_end):
                                            variant_data.loc[idx, HOUR_START] = (
                                                new_peak_end
                                            )
                                            shifted_rows[charge_type].add(idx)

                                # Handle adjacent periods
                                if not has_overlapping:
                                    # Find all non-peak periods that might be affected
                                    non_peak_indices = [
                                        idx
                                        for idx, period in period_categories.items()
                                        if period != PEAK
                                    ]

                                    for idx in non_peak_indices:
                                        if idx not in shifted_rows[charge_type]:
                                            row = variant_data.loc[idx]
                                            period_start, period_end = (
                                                row[HOUR_START],
                                                row[HOUR_END],
                                            )
                                            if (
                                                period_start < new_peak_end
                                                and period_end > new_peak_start
                                            ):
                                                # Period starts before peak: truncate it
                                                if period_start < new_peak_start:
                                                    variant_data.loc[idx, HOUR_END] = (
                                                        new_peak_start
                                                    )
                                                # Period ends after peak: adjust start
                                                elif period_end > new_peak_end:
                                                    variant_data.loc[
                                                        idx, HOUR_START
                                                    ] = new_peak_end
                                                # Period is within peak: remove it
                                                else:
                                                    variant_data = variant_data.drop(
                                                        idx
                                                    )

                                                shifted_rows[charge_type].add(idx)

    if missing_keys and has_exact_keys:
        warnings.warn(
            f"The following charge keys were not found in percent_change_dict and "
            f"will use default ratio of 1.0: {sorted(list(missing_keys))}",
            UserWarning,
        )

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
        - percent_change_dict: dict for charge scaling (see parametrize_rate_data)
        - shift_peak_hours_before: float to shift peak start, in hours
        - shift_peak_hours_after: float to shift peak end, in hours
        - variant_name: str (optional) variant name

    Returns
    -------
    dict
        dictionary of charge_dicts with different variations

    Raises
    ------
    UserWarning
        If global scaling overrides individual ratios in any variant
        (inherited from parametrize_rate_data)
    """

    # Initialize charge dicts for given start/end dates
    charge_dicts = {"original": get_charge_dict(start_dt, end_dt, rate_data)}

    if variants is None:
        return charge_dicts

    for i, variant in enumerate(variants):
        variant_key = variant.get("variant_name", f"variant_{i}")  # Default to index
        variant_data = parametrize_rate_data(rate_data.copy(deep=True), **variant)
        charge_dicts[variant_key] = get_charge_dict(start_dt, end_dt, variant_data)

    return charge_dicts

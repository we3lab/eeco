"""Functions to calculate emissions from electricity consumption data."""

import datetime
import calendar
import numpy as np
import pandas as pd
import cvxpy as cp
import pint

from .units import u
from . import utils as ut

HOUR_VARNAME = "hour"
DAY_VARNAME = "day"
MONTH_VARNAME = "month"
DT_VARNAME = "DateTime"
DT_LOCAL_VARNAME = "datetime_local"
EI_VARNAME = "co2_eq_kg_per_MWh"


def calculate_grid_emissions(
    carbon_intensity,
    consumption_data,
    net_demand_varname,
    emission_units=u.kg / u.MWh,
    consumption_units=u.kW,
    resolution="15m",
    ei_varname=EI_VARNAME,
):
    """Calculates the emissions for the given consumption information and
    carbon intensity of electricity generation structure as DataFrames.

    Parameters
    ----------
    carbon_intensity : DataFrame
        Pandas DataFrame with kg of CO2 per kWh by hour and month

    consumption_data : DataFrame
        Baseline electrical or gas usage data as a Pandas DataFrame

    emissions_units : pint.Unit
        Units for the emissions data. Default is kg / kWh

    consumption_units : pint.Unit
        Units for the electricity consumption data. Default is kW

    resolution : str
        granularity of each timestep in string form with default value of "15m"

    Returns
    -------
    pint.Quantity
        total emissions due to grid electricity generation in kilograms CO2
    """
    df = consumption_data.copy()
    df[MONTH_VARNAME] = df[DT_VARNAME].dt.month
    df[HOUR_VARNAME] = df[DT_VARNAME].dt.hour
    # convert the resolution to hourly
    n_per_hour = int(60 / ut.get_freq_binsize_minutes(resolution))
    if isinstance(carbon_intensity, pint.Quantity):
        emission_units = carbon_intensity.units
        carbon_intensity = carbon_intensity.magnitude
    df = df.merge(carbon_intensity, on=[MONTH_VARNAME, HOUR_VARNAME])
    emissions = (
        np.sum(df[net_demand_varname] * df[ei_varname])
        * consumption_units
        * emission_units
        * u.hour
        / n_per_hour
    )
    return emissions.to(u.kg)


def calculate_grid_emissions_cvx(
    carbon_intensity,
    consumption_data,
    emission_units=u.kg / u.MWh,
    consumption_units=u.kW,
    resolution="15m",
):
    """Calculates the emissions for the given consumption information as a cvxpy object
    carbon intensity of electricity generation structure as a DataFrame.

    Parameters
    ----------
    carbon_intensity : array
        numpy array with kg of CO2 per kWh

    consumption_data : Variable
        Baseline electrical or gas usage data as a CVXPY Variable

    emissions_units : pint.Unit
        Units for the emissions data. Default is kg / kWh

    consumption_units : pint.Unit
        Units for the electricity consumption data. Default is kW

    resolution : str
        granularity of each timestep in string form with default value of "15m"

    Returns
    -------
    Expression
        cvxpy Expression representing emissions in kg of CO2 for the given
        `consumption_data` and `carbon_intensity`
    """
    n_per_hour = int(60 / ut.get_freq_binsize_minutes(resolution))
    if isinstance(carbon_intensity, pint.Quantity):
        emission_units = carbon_intensity.units
        carbon_intensity = carbon_intensity.magnitude
    conversion_factor = (consumption_units * emission_units / u.hour).magnitude
    emissions = cp.sum(cp.multiply(consumption_data, carbon_intensity)) / n_per_hour
    return emissions * conversion_factor


def get_carbon_intensity(
    start_dt,
    end_dt,
    emissions_data,
    emissions_units=u.kg / u.MWh,
    resolution="15m",
    ei_varname=EI_VARNAME,
):
    """Computes the emissions (as kilograms of CO2) of a horizon of data

    Parameters
    ----------
    start_dt : datetime.datetime or numpy.datetime64
        Start datetime to gather rate information

    end_dt : datetime.datetime or numpy.datetime64
        End datetime to gather rate information

    emissions_data : DataFrame
        Electric grid emissions information.
        Only one of `datetime_local` and `month`/`day`/`hour` are required.

        ==================  =================================================
        datetime_local      local datetime to estimate the marginal emissions
        month               month for which the emissions data was averaged
        day                 day for which the emissions data was averaged
        hour                hour for which the emissions data was averaged
        co2_eq_kg_per_MWh   emissions in kg of CO2 per MWh of grid electricity
        ==================  ==================================================

    emissions_units : pint.Unit
        Units for the emissions data. Default is kg / MWh

    resolution : str
        a string of the form `[int][str]` giving the temporal resolution
        on which charges are assessed. The `str` portion corresponds to numpy
        timedelta64 types. For example '15m' specifying demand charges
        that are applied to 15-minute intervals of electricity consumption

    Returns
    -------
    pint.Quantity
        emissions from `start_dt` to `end_dt` in kg CO2 / kWh
    """
    # Get the number of timesteps in a day (according to charge resolution)
    res_binsize_minutes = ut.get_freq_binsize_minutes(resolution)
    n_per_hour = int(60 / res_binsize_minutes)
    if isinstance(start_dt, datetime.datetime) or isinstance(end_dt, datetime.datetime):
        ntsteps = int(
            (end_dt - start_dt) / datetime.timedelta(minutes=res_binsize_minutes)
        )
        # make end_dt non-inclusive to align with Python syntax
        end_dt = end_dt - datetime.timedelta(minutes=res_binsize_minutes)
    else:
        ntsteps = int((end_dt - start_dt) / np.timedelta64(res_binsize_minutes, "m"))
        start_dt = start_dt.astype(datetime.datetime)
        # make end_dt non-inclusive to align with Python syntax
        end_dt = (end_dt - np.timedelta64(res_binsize_minutes, "m")).astype(
            datetime.datetime
        )

    start_day = start_dt.day
    start_month = start_dt.month
    start_year = start_dt.year
    start_hour = start_dt.hour
    start_minute = start_dt.minute
    end_day = end_dt.day
    end_month = end_dt.month
    end_hour = end_dt.hour

    # add date variables to emissions data
    no_day_var = False
    try:
        emissions_data[DT_LOCAL_VARNAME] = pd.to_datetime(
            emissions_data[DT_LOCAL_VARNAME]
        )
        emissions_data[MONTH_VARNAME] = emissions_data.datetime_local.dt.month
        emissions_data[DAY_VARNAME] = emissions_data.datetime_local.dt.day
        emissions_data[HOUR_VARNAME] = emissions_data.datetime_local.dt.hour
    except KeyError:
        # if datetime_local is not found then assume month and hour
        # are already accounted for but day is not for backwards compatability
        if DAY_VARNAME not in emissions_data.columns:
            no_day_var = True

    # adjust carbon_intensity to match the index of net_demand_kW
    # this gets the number of timesteps to include the first hour's carbon intensity
    # since it is not necessarily a full hour (which corresponds to 4 timesteps)
    num_first_min = start_minute % res_binsize_minutes
    if num_first_min == 0:
        num_first_min = n_per_hour
    if no_day_var:
        start_index = (
            emissions_data.loc[
                (emissions_data[HOUR_VARNAME] == start_hour)
                & (emissions_data[MONTH_VARNAME] == start_month)
            ]
            .idxmax()
            .iloc[0]
        )
    else:
        start_index = (
            emissions_data.loc[
                (emissions_data[HOUR_VARNAME] == start_hour)
                & (emissions_data[DAY_VARNAME] == start_day)
                & (emissions_data[MONTH_VARNAME] == start_month)
            ]
            .idxmax()
            .iloc[0]
        )
    carbon_intensity_adj = np.full(
        (num_first_min, 1), emissions_data.iloc[start_index][ei_varname]
    )

    rollover = (
        end_month != start_month
    )  # should we extend this for multiple months/years?
    if rollover:
        old_end_day = end_day
        end_day = calendar.monthrange(start_year, start_month)[1]
        old_end_hour = end_hour
        end_hour = 24

    # increment index since first hour was included
    start_index += 1
    start_hour += 1
    for i in range(end_day + 1 - start_day):
        if i == int(end_day - start_day):
            # don't include last hour as it may not be a full hour
            for j in range(0, int(end_hour) - int(start_hour)):
                carbon_intensity_adj = np.append(
                    carbon_intensity_adj,
                    np.full(
                        (n_per_hour, 1),
                        emissions_data.iloc[start_index + j][ei_varname],
                    ),
                )
        else:
            for j in range(0, 24 - int(start_hour)):
                carbon_intensity_adj = np.append(
                    carbon_intensity_adj,
                    np.full(
                        (n_per_hour, 1),
                        emissions_data.iloc[start_index + j][ei_varname],
                    ),
                )
        # reset start index to first hour of the month
        start_hour = 0
        if no_day_var:
            start_index = (
                emissions_data.loc[
                    (emissions_data[HOUR_VARNAME] == start_hour)
                    & (emissions_data[MONTH_VARNAME] == start_month)
                ]
                .idxmax()
                .iloc[0]
            )
        else:
            start_index = (
                emissions_data.loc[
                    (emissions_data[HOUR_VARNAME] == start_hour)
                    & (emissions_data[DAY_VARNAME] == start_day)
                    & (emissions_data[MONTH_VARNAME] == start_month)
                ]
                .idxmax()
                .iloc[0]
            )

    # if we went into a new month, process that second month
    if rollover:
        end_hour = old_end_hour
        for i in range(old_end_day):
            # reset start index to first hour of the month
            # can assume start hour is zero since we are rolling over
            start_hour = 0
            if no_day_var:
                start_index = emissions_data.loc[
                    (emissions_data[HOUR_VARNAME] == start_hour)
                    & (emissions_data[MONTH_VARNAME] == end_month)
                ].idxmax()[0]
            else:
                start_index = emissions_data.loc[
                    (emissions_data[HOUR_VARNAME] == start_hour)
                    & (emissions_data[DAY_VARNAME] == start_day)
                    & (emissions_data[MONTH_VARNAME] == end_month)
                ].idxmax()[0]
            if i == int(old_end_day - 1):
                # don't include last hour as it may not be a full hour
                for j in range(0, int(end_hour)):
                    carbon_intensity_adj = np.append(
                        carbon_intensity_adj,
                        np.full(
                            (n_per_hour, 1),
                            emissions_data.iloc[start_index + j][ei_varname],
                        ),
                    )
            else:
                for j in range(0, 24):
                    carbon_intensity_adj = np.append(
                        carbon_intensity_adj,
                        np.full(
                            (n_per_hour, 1),
                            emissions_data.iloc[start_index + j][ei_varname],
                        ),
                    )

    # add in remainder of last hour
    leftover = (ntsteps - num_first_min) % n_per_hour
    if leftover == 0:
        leftover = n_per_hour
    if no_day_var:
        end_index = (
            emissions_data.loc[
                (emissions_data[HOUR_VARNAME] == end_hour)
                & (emissions_data[MONTH_VARNAME] == end_month)
            ]
            .idxmax()
            .iloc[0]
        )
    else:
        end_index = (
            emissions_data.loc[
                (emissions_data[HOUR_VARNAME] == end_hour)
                & (emissions_data[DAY_VARNAME] == end_day)
                & (emissions_data[MONTH_VARNAME] == end_month)
            ]
            .idxmax()
            .iloc[0]
        )

    carbon_intensity_adj = np.append(
        carbon_intensity_adj,
        np.full((leftover, 1), emissions_data.iloc[end_index][ei_varname]),
    )

    # convert from kg/MWh to kg/kWh
    carbon_intensity_adj = (carbon_intensity_adj * emissions_units).to(u.kg / u.kWh)

    return carbon_intensity_adj

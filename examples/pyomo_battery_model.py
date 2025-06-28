############################################################
#
# This file contains an example of a time-discretized
# Pyomo model for a battery energy storage system
# operating on the site of an industrial power consumer.
#
############################################################

# Import necessary libraries
import numpy as np
import pandas as pd
import calendar, math
from datetime import datetime, timedelta
from pyomo.environ import (
    SolverFactory,
    ConcreteModel,
    Block,
    Var,
    Param,
    Set,
    Objective,
    Constraint,
    Expression,
    exp,
    units as pyunits,
    value,
    minimize,
)


# create a class for the battery optimization model
class BatteryPyomo:
    """
    A Pyomo-based model for simulating and optimizing the operation of a battery energy storage system.

    This class models the behavior of a battery system operating on the site of an industrial power consumer.
    It uses time-discretized simulation to optimize battery usage based on input parameters and baseline load data.

    Parameters
    -----------
    params (dict): A dictionary of model parameters. Expected keys include:
            - 'start_date' (str): The start date of the simulation in ISO format (e.g., 'YYYY-MM-DDTHH:MM:SS').
            - 'end_date' (str): The end date of the simulation in ISO format.
            - 'timestep' (float): The time step for the simulation in hours.
            (Additional keys may be required depending on the specific model configuration.)

    baseload (array-like): The baseline load profile for the site, provided as a time series.

    baseload_repeat (bool, optional): If True, the baseline load profile is repeated to match the simulation period.
            Defaults to False.
    """

    def __init__(self, params, baseload, baseload_repeat=False):

        # assign all items in params to attributes of the class
        for key, value in params.items():
            setattr(self, key, value)

        # set the parameters for the battery model
        self.baseload = baseload
        self.baseload_repeat = baseload_repeat

        # set up timing for the model
        self.start_dt = datetime.fromisoformat(self.start_date)
        self.end_dt = datetime.fromisoformat(self.end_date)
        self.time_delta = timedelta(hours=self.timestep)
        self.datetimerange = datetime.fromisoformat(
            self.end_date
        ) - datetime.fromisoformat(self.start_date)

        self.t = np.arange(
            self.start_dt, self.end_dt, timedelta(hours=self.timestep)
        ).astype(datetime)
        # number of time steps
        self.num_steps = int(len(self.t))
        # length of time step in seconds
        self.dt_seconds = self.timestep * 3600
        # total sim time in seconds
        self.sim_time_seconds = self.datetimerange.total_seconds()
        # number of days
        self.days = self.sim_time_seconds / (3600 * 24)
        # number of hours
        self.hours = self.days * 24
        # find the number of days in the month associated with the start date
        days_in_month = calendar.monthrange(
            month=self.start_dt.month, year=self.start_dt.year
        )[1]
        # number of months
        self.months = self.days / days_in_month
        # time step in hours
        self.h_scale = self.timestep
        # time step in months
        self.m_scale = self.timestep / (24 * days_in_month)

        # repeat baseline pattern over full simulation period
        if self.baseload_repeat == True:
            baseload = np.repeat(
                self.baseload, int(1 / self.timestep)
            )  # match timestep resolution
            self.baseload = np.tile(baseload, math.ceil(self.days))[
                : self.num_steps
            ]  # repeat pattern over simulation
        return

    def create_model(self):
        # create a concrete model

        model = ConcreteModel()

        # create timing parameters on the model
        model.t = Set(initialize=range(self.num_steps))
        model.dt = Param(initialize=self.timestep, mutable=False)
        model.h_scale = Param(initialize=self.h_scale)
        model.m_scale = Param(initialize=self.m_scale)
        model.months = Param(initialize=self.months)

        # define the characteristics of the battery
        model.eta_charging = Param(initialize=np.sqrt(self.rte))
        model.eta_discharging = Param(initialize=1 / np.sqrt(self.rte))
        model.e_capacity = Param(initialize=self.energycapacity)
        model.p_capacity = Param(initialize=self.powercapacity)

        # add the baseload to the model
        def init_baseload(model, t, data=self.baseload):
            return data[t]

        model.baseload = Param(model.t, initialize=init_baseload)

        # add battery dynamics
        # create variables
        model.soc = Var(
            model.t,
            bounds=(self.soc_min, self.soc_max),
            initialize=self.soc_init,
            doc="State of charge",
        )
        model.energy = Var(
            model.t,
            bounds=(0, self.energycapacity),
            initialize=self.soc_init * self.energycapacity,
            doc="Energy stored",
        )
        model.power = Var(
            model.t,
            bounds=(-self.powercapacity, self.powercapacity),
            initialize=0,
            doc="Power into the battery",
        )
        model.power_C = Var(
            model.t, bounds=(0, self.powercapacity), initialize=0, doc="Charging power"
        )
        model.power_D = Var(
            model.t,
            bounds=(0, self.powercapacity),
            initialize=0,
            doc="Discharging power",
        )

        model.net_facility_load = Var(
            model.t,
            bounds=(None, None),
            initialize=0,
            doc="Net facility load including battery",
        )

        # create model constraints
        @model.Constraint(model.t)
        def state_of_charge_constraint(b, t):
            return b.soc[t] == b.energy[t] / b.e_capacity

        @model.Constraint(model.t)
        def energy_balance_constraint(b, t):
            if t == 0:
                return Constraint.Skip
            else:
                return b.energy[t] == b.energy[t - 1] + b.power[t - 1] * b.dt

        @model.Constraint(model.t)
        def power_balance_constraint(b, t):
            return (
                b.power[t]
                == b.power_C[t] * b.eta_charging - b.power_D[t] * b.eta_discharging
            )

        @model.Constraint(model.t)
        def net_facility_load_constraint(b, t):
            return b.net_facility_load[t] == b.baseload[t] + b.power[t]

            # set boundary conditions

        @model.Constraint()
        def initial_soc_constraint(b):
            return b.soc[0] == self.soc_init

        @model.Constraint()
        def final_soc_constraint(b):
            return b.soc[self.num_steps - 1] == self.soc_init

        # assign the model as an attribute of the class
        self.model = model
        return model

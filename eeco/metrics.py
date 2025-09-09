"""Functions to estimate flexibility metrics from power consumption trajectories."""

import warnings
import numpy as np


def roundtrip_efficiency(baseline_kW, flexible_kW):
    """Calculate the round-trip efficiency of a flexibly operating power trajectory
    relative to a baseline.

    Parameters
    ----------
    baseline_kW : list or np.ndarray
        power consumption data of the baseline system in units of kW

    flexible_kW : list or np.ndarray
        power consumption data of the flexibly operating or cost-optimized system
        in units of kW.

    Raises
    ------
    TypeError
        When `baseline_kW` and `flexible_kW` are not an acceptable type
        (e.g., list vs. np.ndarray).

    ValueError
        When `baseline_kW` and `flexible_kW` are not of the same length

    Warnings
        When `baseline_kW` and `flexible_kW` contain negative values,
        which may indicate an error in the data.

    ValueError
        When `baseline_kW` and `flexible_kW` contain missing values.

    Warnings
        When rte is calculated to be greater than 1.
        This may indicate an error in the assumptions behind the data.

    Returns
    -------
    float
        The round-trip efficiency [0,1] of the flexible power trajectory
        relative to the baseline.
    """
    # Check if inputs are lists or numpy arrays
    if not isinstance(baseline_kW, (list, np.ndarray)):
        raise TypeError("baseline_kW must be a list or numpy array.")
    if not isinstance(flexible_kW, (list, np.ndarray)):
        raise TypeError("flexible_kW must be a list or numpy array.")

    # Check if inputs are of the same length
    if len(baseline_kW) != len(flexible_kW):
        raise ValueError("baseline_kW and flexible_kW must have the same length.")

    # Convert inputs to numpy arrays for easier calculations
    baseline_kW = np.array(baseline_kW)
    flexible_kW = np.asarray(flexible_kW)

    # Check for negative or missing values
    if np.any(baseline_kW < 0) or np.any(flexible_kW < 0):
        warnings.warn(
            "Negative values detected in baseline_kW or flexible_kW. "
            "This may indicate an error in the data."
        )

    if np.any(np.isnan(baseline_kW)) or np.any(np.isnan(flexible_kW)):
        raise ValueError(
            "Missing values detected in baseline_kW or flexible_kW. "
            "This may indicate an error in the data."
        )

    # Calculate the round-trip efficiency
    baseline_energy = np.sum(baseline_kW)
    flexible_energy = np.sum(flexible_kW)
    if flexible_energy == 0:
        raise ValueError("The sum of flexible_kW is zero, cannot compute rte.")

    rte_value = baseline_energy / flexible_energy
    if rte_value > 1:
        warnings.warn(
            "RTE calculated to be greater than 1. "
            "This may indicate an error in the assumptions behind the data."
        )

    return rte_value


def power_capacity(
    baseline_kW, flexible_kW, timestep=0.25, pc_type="average", relative=True
):
    """Calculate the power capacity of a virtual battery system.
    This approach implicitly assumes the system has completed a round-trip.

    Parameters
    ----------
    baseline_kW : array-like
        The baseline power consumption of the facility in kW.

    flexible_kW : array-like
        The flexible power consumption of the facility in kW.

    timestep : float
        The time step of the data in hours. Default is 0.25 hours (15 minutes).

    pc_type : str
        The type of power capacity to calculate.
        Options are 'average', 'charging', 'discharging', 'maximum'

    relative : bool
        If True, return the fractional power capacity.
        If False, return the absolute power capacity.

    Raises
    ------
    ValueError
        If `pc_type` is not one of the expected values
        ('average', 'charging', 'discharging', 'maximum').

    Returns
    -------
    float
        The power capacity of the virtual battery system in either relative
        or absolute terms.
    """
    # calculate the effective battery power (diff)
    diff_kW = flexible_kW - baseline_kW
    # charging is positive, discharging is negative
    charging = np.where(diff_kW > 0, diff_kW, 0)
    discharging = np.where(diff_kW < 0, -diff_kW, 0)

    # calculate the power capacity
    if pc_type == "average":
        power_capacity = (np.sum(charging) + np.sum(discharging)) / (len(diff_kW))
    elif pc_type == "charging":
        power_capacity = np.sum(charging) / (len(charging))
    elif pc_type == "discharging":
        power_capacity = np.sum(discharging) / (len(discharging))
    elif pc_type == "maximum":
        power_capacity = np.max(np.abs(diff_kW))
    else:
        raise ValueError(
            "Invalid power capacity type. Must be 'average', "
            "'charging', 'discharging', or 'maximum'."
        )

    if relative:
        # normalize by the max baseline power
        return power_capacity / np.max(baseline_kW)
    else:
        return power_capacity


def energy_capacity(
    baseline_kW, flexible_kW, timestep=0.25, ec_type="discharging", relative=True
):
    """Calculate the energy capacity of a virtual battery system.
    This approach implicitly assumes the system has completed a round-trip.

    Parameters
    ----------
    baseline_kW : array-like
        The baseline power consumption of the facility in kW.

    flexible_kW : array-like
        The flexible power consumption of the facility in kW.

    timestep : float
        The time step of the data in hours. Default is 0.25 hours (15 minutes).

    ec_type : str
        The type of energy capacity to calculate.
        Options are 'average', 'charging', 'discharging'

    relative : bool
        If True, return the fractional energy capacity.
        If False, return the absolute energy capacity.

    Raises
    ------
    ValueError
        If `ec_type` is not one of the expected values
        ('average', 'charging', 'discharging').

    Returns
    -------
    float
        The energy capacity of the virtual battery system in either relative
        or absolute terms.
    """
    # calculate the effective battery power (diff)
    diff_kW = flexible_kW - baseline_kW
    # charging is positive, discharging is negative
    charging = np.where(diff_kW > 0, diff_kW, 0)
    discharging = np.where(diff_kW < 0, -diff_kW, 0)

    # calculate the energy capacity
    if ec_type == "average":
        energy_capacity = (np.sum(charging) + np.sum(discharging)) * timestep
    elif ec_type == "charging":
        energy_capacity = np.sum(charging) * timestep
    elif ec_type == "discharging":
        energy_capacity = np.sum(discharging) * timestep
    else:
        raise ValueError(
            "Invalid energy capacity type. "
            "Must be 'average', 'charging', or 'discharging'."
        )

    if relative:
        # normalize by the total baseline power
        return energy_capacity / (
            np.sum(baseline_kW) * timestep + 1e-12
        )  # add small value to avoid division by zero
    else:
        return energy_capacity


def net_present_value(
    capital_cost=0,
    electricity_savings=0,
    maintenance_diff=0,
    ancillary_service_benefit=0,
    service_curtailment=0,
    service_price=1.0,
    timestep=0.25,
    simulation_years=1,
    upgrade_lifetime=30,
    interest_rate=0.03,
):
    """
    Calculate the net present value of flexibility of a virtual battery system.

    Parameters
    ----------
    capital_cost : float
        The capital cost of the virtual battery system in $.

    electricity_savings : float
        The electricity savings from the flexible operation in $.

    maintenance_diff : float
        The difference in maintenance costs between the baseline
        and flexible operation in $.

    ancillary_service_benefit : float
        The benefit from providing ancillary services in $.

    service_curtailment : float
        The amount of service curtailment. If the virtual battery system produces
        a product, this may be in units of volume or mass (e.g., m^3 or kg).

    service_price : float
        The marginal price of curtailed service $/amount.
        Amount here may refer to units of volume or mass (e.g., $/m^3 or $/kg).

    timestep : float
        The time step of the data in hours. Default is 0.25 hours (15 minutes).

    simulation_years : int
        The number of years in which the electricity savings or
        ancillary service benefits are calculated for. Default is 1 year.

    upgrade_lifetime : int
        The number of years of operation left for the upgrade. Default is 30 years.

    interest_rate : float
        The interest rate used to discount future cash flows. Default is 0.03.

    Raises
    ------
    Warning
        If the capital cost is less than 0

    ValueError
        If the upgrade lifetime is less than or equal to 0

    ValueError
        If the interest rate is less than 0.

    ValueError
        if the timestep is less than or equal to 0.

    Returns
    -------
    float
        The net present value benefit of the virtual battery system in $.
    """
    # check if capital cost is negative
    if capital_cost < 0:
        warnings.warn(
            "Capital cost is negative. This may indicate an error in the data."
        )
    # check if upgrade lifetime is valid
    if upgrade_lifetime <= 0:
        raise ValueError("Upgrade lifetime must be greater than 0 years.")
    # check if interest rate is valid
    if interest_rate < 0:
        raise ValueError("Interest rate must be greater than or equal to 0.")
    # check if timestep is valid
    if timestep <= 0:
        raise ValueError("Timestep must be greater than 0 hours.")

    # calculate the total cash flow
    benefit = electricity_savings + ancillary_service_benefit
    cost = maintenance_diff + service_curtailment * service_price
    cash_flow = benefit - cost

    # calculate the net discount factor
    discount = sum([1 / ((1 + interest_rate) ** n) for n in range(1, upgrade_lifetime)])

    # calculate the net present value
    npv = discount * (cash_flow / simulation_years) - capital_cost

    return npv

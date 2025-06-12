import re
import pytz
import datetime
import numpy as np
import cvxpy as cp
import pyomo.environ as pyo
from pyomo.core.expr.numeric_expr import (
    SumExpression,
    LinearExpression,
    MonomialTermExpression,
)
from pyomo.core.base.var import ScalarVar
from pyomo.core.base.expression import IndexedExpression

# Dictionary mapping region types to timezone strings
TIMEZONE_DICT = {
    "iso_rto_code": {
        "CAISO": "America/Los_Angeles",  # Pacific Time (PT)
        "ERCOT": "America/Chicago",  # Central Time (CT)
        "ISONE": "America/New_York",  # Eastern Time (ET)
        "MISO": "America/Chicago",  # Central Time (CT)
        "NYISO": "America/New_York",  # Eastern Time (ET)
        "OTHER": "America/New_York",  # Default to Eastern Time (ET)
        "PJM": "America/New_York",  # Eastern Time (ET)
        "SPP": "America/Chicago",  # Central Time (CT)
    },
    "nerc_region": {
        "MRO": "America/Chicago",  # Central Time (CT)
        "NPCC": "America/New_York",  # Eastern Time (ET)
        "RFC": "America/New_York",  # Eastern Time (ET)
        "SERC": "America/New_York",  # Eastern Time (ET)
        "SPP": "America/Chicago",  # Central Time (CT)
        "TRE": "America/Chicago",  # Central Time (CT)
        "WECC": "America/Denver",  # Mountain Time (MT) / Pacific Time (PT)
    },
    "state": {
        "AK": "America/Anchorage",  # Alaska Time (AKT)
        "AL": "America/Chicago",  # Central Time (CT)
        "AR": "America/Chicago",  # Central Time (CT)
        "AZ": "America/Phoenix",  # Mountain Standard Time (MST, no DST)
        "CA": "America/Los_Angeles",  # Pacific Time (PT)
        "CO": "America/Denver",  # Mountain Time (MT)
        "CT": "America/New_York",  # Eastern Time (ET)
        "DE": "America/New_York",  # Eastern Time (ET)
        "FL": "America/New_York",  # Eastern Time (ET)
        "GA": "America/New_York",  # Eastern Time (ET)
        "IA": "America/Chicago",  # Central Time (CT)
        "ID": "America/Boise",  # Mountain Time (MT)
        "IL": "America/Chicago",  # Central Time (CT)
        "IN": "America/Indiana/Indianapolis",  # Mostly Eastern Time (ET)
        "KS": "America/Chicago",  # Central Time (CT)
        "KY": "America/New_York",  # Eastern Time (ET)
        "LA": "America/Chicago",  # Central Time (CT)
        "MA": "America/New_York",  # Eastern Time (ET)
        "MD": "America/New_York",  # Eastern Time (ET)
        "ME": "America/New_York",  # Eastern Time (ET)
        "MI": "America/Detroit",  # Mostly Eastern Time (ET)
        "MN": "America/Chicago",  # Central Time (CT)
        "MO": "America/Chicago",  # Central Time (CT)
        "MS": "America/Chicago",  # Central Time (CT)
        "MT": "America/Denver",  # Mountain Time (MT)
        "NC": "America/New_York",  # Eastern Time (ET)
        "ND": "America/Denver",  # Central Time (CT)
        "NE": "America/Chicago",  # Mountain Time (MT)
        "NH": "America/New_York",  # Eastern Time (ET)
        "NJ": "America/New_York",  # Eastern Time (ET)
        "NM": "America/Denver",  # Mountain Time (MT)
        "NV": "America/Denver",  # Mountain Time (MT)
        "NY": "America/New_York",  # Eastern Time (ET)
        "OH": "America/New_York",  # Eastern Time (ET)
        "OK": "America/Chicago",  # Central Time (CT)
        "OR": "America/Los_Angeles",  # Pacific Time (PT)
        "PA": "America/New_York",  # Eastern Time (ET)
        "RI": "America/New_York",  # Eastern Time (ET)
        "SC": "America/New_York",  # Eastern Time (ET)
        "SD": "America/Denver",  # Mountain Time (MT)
        "TN": "America/Chicago",  # Central Time (CT)
        "TX": "America/Chicago",  # Central Time (CT)
        "UT": "America/Denver",  # Mountain Time (MT)
        "VA": "America/New_York",  # Eastern Time (ET)
        "VT": "America/New_York",  # Eastern Time (ET)
        "WA": "America/Los_Angeles",  # Pacific Time (PT)
        "WI": "America/Chicago",  # Central Time (CT)
        "WV": "America/New_York",  # Eastern Time (ET)
        "WY": "America/Denver",  # Mountain Time (MT)
    },
    "egrid_subregions": {
        "AKGD": "America/Anchorage",  # Alaska Time (AKT)
        "AZNM": "America/Phoenix",  # Mountain Standard Time (MST, no DST)
        "CAMX": "America/Los_Angeles",  # Pacific Time (PT)
        "ERCT": "America/Chicago",  # Central Time (CT)
        "FRCC": "America/New_York",  # Eastern Time (ET)
        "MROE": "America/Chicago",  # Central Time (CT)
        "MROW": "America/Chicago",  # Central Time (CT)
        "NEWE": "America/New_York",  # Eastern Time (ET)
        "NWPP": "America/Los_Angeles",  # Pacific Time (PT)
        "NYCW": "America/New_York",  # Eastern Time (ET)
        "NYLI": "America/New_York",  # Eastern Time (ET)
        "NYUP": "America/New_York",  # Eastern Time (ET)
        "RFCE": "America/New_York",  # Eastern Time (ET)
        "RFCM": "America/Chicago",  # Central Time (CT)
        "RFCW": "America/Chicago",  # Central Time (CT)
        "RMPA": "America/Denver",  # Mountain Time (MT)
        "SPNO": "America/Chicago",  # Central Time (CT)
        "SPSO": "America/Chicago",  # Central Time (CT)
        "SRMV": "America/New_York",  # Eastern Time (ET)
        "SRMW": "America/Chicago",  # Central Time (CT)
        "SRSO": "America/New_York",  # Eastern Time (ET)
        "SRTV": "America/New_York",  # Eastern Time (ET)
        "SRVC": "America/New_York",  # Eastern Time (ET)
    },
}


def idxparam_value(idx_param):
    """Returns the parameter value at the given index.

    Parameters
    ----------
    idx_param : pyomo.environ.Param or pyomo.environ.Var
        The Pyomo parameter or variable to be converted

    Returns
    -------
    numpy.ndarray
        Indexed variable or parameter as a numpy array
    """
    return np.array([idx_param[i].value for i in range(len(idx_param))])


def max(expression, model=None, varstr=None):
    """Elementwise maximum of an expression or array

    Parameters
    ----------
    expression : [
        numpy.Array,
        cvxpy.Expression,
        pyomo.core.expr.numeric_expr.NumericExpression,
        pyomo.core.expr.numeric_expr.NumericNDArray,
        pyomo.core.expr.numeric_expr.NumericNDArray,
        pyomo.environ.Param,
        pyomo.environ.Var
    ]
        The expression to find the maximum of

    model : pyomo.environ.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    varstr : str
        Name of the variable to be created if using a Pyomo `model`

    Raises
    ------
    TypeError
        When `expression` is not of type `numpy.Array`, `cvxpy.Expression`,
        `pyomo.core.expr.numeric_expr.NumericExpression`,
        `pyomo.core.expr.numeric_expr.NumericNDArray`,
        `pyomo.environ.Param`, or `pyomo.environ.Var`

    Returns
    -------
    ([numpy.Array, cvxpy.Expression, pyomo.environ.Var], pyomo.environ.Model)
        Expression representing max of `expression`
    """
    if isinstance(
        expression, (LinearExpression, SumExpression, MonomialTermExpression, ScalarVar)
    ):
        model.add_component(varstr, pyo.Var())
        var = model.find_component(varstr)

        def const_rule(model):
            return var >= expression

        constraint = pyo.Constraint(rule=const_rule)
        model.add_component(varstr + "_constraint", constraint)
        return (var, model)
    elif isinstance(expression, (IndexedExpression, pyo.Param, pyo.Var)):
        model.add_component(varstr, pyo.Var())
        var = model.find_component(varstr)

        def const_rule(model, t):
            return var >= expression[t]

        constraint = pyo.Constraint(model.t, rule=const_rule)
        model.add_component(varstr + "_constraint", constraint)
        return (var, model)
    elif isinstance(
        expression, (int, float, np.int32, np.int64, np.float32, np.float64, np.ndarray)
    ):
        return (np.max(expression), model)
    elif isinstance(expression, cp.Expression):
        return (cp.max(expression), None)
    else:
        raise TypeError(
            "Only CVXPY or Pyomo variables and NumPy arrays are currently supported."
        )


def sum(expression, axis=0, model=None, varstr=None):
    """Elementwise maximum of an expression or array

    Parameters
    ----------
    expression : [
        numpy.Array,
        cvxpy.Expression,
        pyomo.core.expr.numeric_expr.NumericExpression,
        pyomo.core.expr.numeric_expr.NumericNDArray,
        pyomo.environ.Param,
        pyomo.environ.Var
    ]
        Expression representing a matrix to sum

    axis: int
        Optional axis along which to compute sum. Default is 0

    model : pyomo.environ.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    varstr : str
        Name of the variable to be created if using a Pyomo `model`

    Raises
    ------
    TypeError
        When `expression` is not of type `numpy.Array`, `cvxpy.Expression`,
        `pyomo.core.expr.numeric_expr.NumericExpression`,
        `pyomo.core.expr.numeric_expr.NumericNDArray` `pyomo.environ.Param`,
        or `pyomo.environ.Var`

    Returns
    -------
    [numpy.Array, cvxpy.Expression, pyomo.environ.Expression]
        Expression representing sum of `expression` along `axis`
    """
    if isinstance(expression, (SumExpression, IndexedExpression, pyo.Param, pyo.Var)):
        model.add_component(varstr, pyo.Var())
        var = model.find_component(varstr)

        def const_rule(model):
            total = 0
            for i in range(len(expression)):
                total += expression[i]
            return var == total

        constraint = pyo.Constraint(rule=const_rule)
        model.add_component(varstr + "_constraint", constraint)
        return (var, model)
    elif isinstance(
        expression, (int, float, np.int32, np.int64, np.float32, np.float64, np.ndarray)
    ):
        return (np.sum(expression, axis=axis), model)
    elif isinstance(expression, cp.Expression):
        return (cp.sum(expression, axis=axis), None)
    else:
        raise TypeError(
            "Only CVXPY or Pyomo variables and NumPy arrays are currently supported."
        )


def max_pos(expression, model=None, varstr=None):
    """Returns the maximum positive scalar value of an expression.
    I.e., max([x, 0]) where x is any element of the expression (if a matrix)

    Parameters
    ----------
    expression : [
        numpy.Array,
        cvxpy.Expression,
        pyomo.core.expr.numeric_expr.NumericExpression,
        pyomo.core.expr.numeric_expr.NumericNDArray,
        pyomo.environ.Param,
        pyomo.environ.Var
    ]
        Expression representing a matrix, vector, or scalar

    model : pyomo.environ.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    varstr : str
        Name of the variable to be created if using a Pyomo `model`

    Raises
    ------
    TypeError
        When `expression` is not of type `numpy.Array`, `cvxpy.Expression`,
        `pyomo.core.expr.numeric_expr.NumericNDArray`,
        `pyomo.core.expr.numeric_expr.NumericExpression`,
        `pyomo.environ.Param`,  or `pyomo.environ.Var`

    Returns
    -------
    (
        [numpy.float, numpy.int, numpy.Array, cvxpy.Expression, or pyomo.environ.Var],
        pyomo.environ.Model
    )
        Expression representing maximum positive scalar value of `expression`
    """
    if isinstance(
        expression, (LinearExpression, SumExpression, MonomialTermExpression, ScalarVar)
    ):
        model.add_component(varstr, pyo.Var(initialize=0, bounds=(0, None)))
        var = model.find_component(varstr)

        def const_rule(model):
            return var >= expression

        constraint = pyo.Constraint(rule=const_rule)
        model.add_component(varstr + "_constraint", constraint)
        return (var, model)
    elif isinstance(expression, (IndexedExpression, pyo.Param, pyo.Var)):
        model.add_component(varstr, pyo.Var(bounds=(0, None)))
        var = model.find_component(varstr)

        def const_rule(model, t):
            return var >= expression[t]

        constraint = pyo.Constraint(model.t, rule=const_rule)
        model.add_component(varstr + "_constraint", constraint)
        return (var, model)
    elif isinstance(
        expression, (int, float, np.int32, np.int64, np.float32, np.float64, np.ndarray)
    ):
        return (np.max(expression), model) if np.max(expression) > 0 else (0, model)
    elif isinstance(expression, cp.Expression):
        return cp.max(cp.vstack([expression, 0])), None
    else:
        raise TypeError(
            "Only CVXPY or Pyomo variables and NumPy arrays are currently supported."
        )


def multiply(expression1, expression2, model=None, varstr=None):
    """Implements elementwise multiplication operation on two optimization expressions

    Parameters
    ----------
    expression1 : [
        numpy.Array,
        cvxpy.Expression,
        pyomo.core.expr.numeric_expr.NumericExpression,
        pyomo.core.expr.numeric_expr.NumericNDArray,
        pyomo.environ.Param,
        pyomo.environ.Var
    ]
        LHS of multiply operation

    expression2 : [
        numpy.Array,
        cvxpy.Expression,
        pyomo.core.expr.numeric_expr.NumericExpression,
        pyomo.core.expr.numeric_expr.NumericNDArray,
        pyomo.environ.Param,
        pyomo.environ.Var
    ]
        RHS of multiply operation

    model : pyomo.environ.Model
        The model object associated with the problem.
        Only used in the case of Pyomo, so `None` by default.

    varstr : str
        Name of the variable to be created if using a Pyomo `model`

    Raises
    ------
    TypeError
        When `expression` is not of type `numpy.Array`, `cvxpy.Expression`,
        `pyomo.core.expr.numeric_expr.NumericNDArray`,
        `pyomo.core.expr.numeric_expr.NumericExpression`,
        `pyomo.environ.Param`, or `pyomo.environ.Var`

    Returns
    -------
    [
        numpy.Array,
        cvxpy.Expression,
        pyomo.core.expr.numeric_expr.NumericNDArray,
        pyomo.environ.Expression
    ]
        result from elementwise multiplication of `expression1` and `expression2`
    """
    if isinstance(expression1, cp.Expression) or isinstance(expression2, cp.Expression):
        return (cp.multiply(expression1, expression2), None)
    elif isinstance(
        expression1, (SumExpression, IndexedExpression, pyo.Param, pyo.Var)
    ) or isinstance(
        expression2, (SumExpression, IndexedExpression, pyo.Param, pyo.Var)
    ):
        if len(expression1) > 1:
            # TODO: replace model.t with better way to get dimensions
            model.add_component(varstr, pyo.Var(model.t))
            var = model.find_component(varstr)

            def const_rule(model, t):
                return var[t] == expression1[t] * expression2[t]

            constraint = pyo.Constraint(model.t, rule=const_rule)
            model.add_component(varstr + "_constraint", constraint)
            return (var, model)
        else:
            return (expression1 * expression2, model)
    elif isinstance(
        expression1,
        (int, float, np.int32, np.int64, np.float32, np.float64, np.ndarray),
    ) and isinstance(
        expression2,
        (int, float, np.int32, np.int64, np.float32, np.float64, np.ndarray),
    ):
        return (np.multiply(expression1, expression2), model)
    else:
        raise TypeError(
            "Only CVXPY or Pyomo variables and NumPy arrays are currently supported."
        )


def parse_freq(freq):
    """Parses a time frequency code string, returning its type and its freq_binsize

    Parameters
    ----------
    freq: str
        a string of the form [type][freq_binsize], where type corresponds to a
        numpy.timedelta64 encoding and freq binsize is an integer giving the number
        of increments of `type` of one binned increment of our time variable
        (for example '6h' means the data are grouped into increments of 6 hours)

    Returns
    -------
    tuple
        tuple of the form (`int`,`str`) giving the binsize and units (freq_type)
    """
    freq_type = re.sub("[0-9]", "", freq)
    freq_binsize = int(re.sub("[^0-9]", "", freq))
    return freq_binsize, freq_type


def get_freq_binsize_minutes(freq):
    """Gets size of a given time frequency expressed in units of minutes

    Parameters
    ----------
    freq: str
        a string of the form [type][freq_binsize], where type corresponds to a
        numpy.timedelta64 encoding and freq binsize is an integer giving the number
        of increments of `type` of one binned increment of our time variable
        (for example '6h' means the data are grouped into increments of 6 hours)

    Raises
    ------
    ValueError
        when resolution is not minute, hourly, or daily

    Returns
    -------
    int
        integer giving the number of minutes in the given time frequency unit
    """
    freq_binsize, freq_type = parse_freq(freq)
    if freq_type == "m":
        multiplier = 1
    elif freq_type == "h":
        multiplier = 60
    elif freq_type in ["D", "d"]:
        multiplier = 60 * 24
    else:
        raise ValueError(
            "Cannot deal with data that are not in minute, hourly, or daily resolution"
        )
    return multiplier * freq_binsize


def convert_utc_to_timezone(utc_hour, timezone_str):
    """
    Convert UTC hour (0-23) to the corresponding hour in a specified timezone.

    Parameters:
    utc_hour (int): Hour in UTC (0-23).
    timezone_str (str): Timezone string, e.g., 'America/New_York'.

    Returns:
    int: Corresponding hour in the specified timezone.
    """
    # Ensure the UTC hour is within the valid range
    if not (0 <= utc_hour <= 23):
        raise ValueError("UTC hour must be between 0 and 23.")

    # Create a UTC datetime object with the specified hour
    utc_time = datetime.datetime.utcnow().replace(
        hour=utc_hour, minute=0, second=0, microsecond=0, tzinfo=pytz.utc
    )

    # Convert to the specified timezone
    target_timezone = pytz.timezone(timezone_str)
    local_time = utc_time.astimezone(target_timezone)

    return local_time.hour

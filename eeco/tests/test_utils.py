import os
import pytest
import numpy as np
import pyomo.environ as pyo
import cvxpy as cp

from eeco import utils as ut
from eeco.tests.test_costs import setup_pyo_vars_constraints

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
skip_all_tests = False


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize("freq, expected", [("15m", (15, "m")), ("1h", (1, "h"))])
def test_parse_freq(freq, expected):
    assert ut.parse_freq(freq) == expected


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, varstr, expected",
    [
        ({"electric": np.ones(96) * 100, "gas": np.ones(96)}, "electric", 9600),
    ],
)
def test_sum_pyo(consumption_data, varstr, expected):
    model = pyo.ConcreteModel()
    model.T = len(consumption_data["electric"])
    model.t = range(model.T)
    pyo_vars = {}
    for key, val in consumption_data.items():
        var = pyo.Var(range(len(val)), initialize=np.zeros(len(val)), bounds=(0, None))
        model.add_component(key, var)
        pyo_vars[key] = var

    @model.Constraint(model.t)
    def electric_constraint(m, t):
        return consumption_data["electric"][t] == m.electric[t]

    @model.Constraint(model.t)
    def gas_constraint(m, t):
        return consumption_data["gas"][t] == m.gas[t]

    var = getattr(model, varstr)
    result, model = ut.sum(var, model=model, varstr="test")
    model.objective = pyo.Objective(expr=0)
    solver = pyo.SolverFactory("gurobi")
    solver.solve(model)
    assert pyo.value(result) == expected
    assert model is not None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, varstr1, varstr2, expected",
    [
        (
            {"electric": np.ones(96) * 100, "gas": np.ones(96)},
            "electric",
            "gas",
            np.ones(96) * 100,
        ),
    ],
)
def test_multiply_pyo(consumption_data, varstr1, varstr2, expected):
    model = pyo.ConcreteModel()
    model.T = len(consumption_data["electric"])
    model.t = range(model.T)
    pyo_vars = {}
    for key, val in consumption_data.items():
        var = pyo.Var(model.t, initialize=np.zeros(len(val)), bounds=(0, None))
        model.add_component(key, var)
        pyo_vars[key] = var

    @model.Constraint(model.t)
    def electric_constraint(m, t):
        return consumption_data["electric"][t] == m.electric[t]

    @model.Constraint(model.t)
    def gas_constraint(m, t):
        return consumption_data["gas"][t] == m.gas[t]

    var1 = getattr(model, varstr1)
    var2 = getattr(model, varstr2)
    result, model = ut.multiply(var1, var2, model=model, varstr="test")
    model.objective = pyo.Objective(expr=0)
    solver = pyo.SolverFactory("ipopt")
    solver.solve(model)
    assert np.allclose([pyo.value(result[i]) for i in range(len(result))], expected)
    assert model is not None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, varstr, expected",
    [
        ({"electric": np.ones(96) * 100, "gas": np.ones(96)}, "electric", 100),
        ({"electric": np.arange(96), "gas": np.ones(96)}, "electric", 95),
        ({"electric": np.arange(96), "gas": np.ones(96)}, "gas", 1),
    ],
)
def test_max_pyo(consumption_data, varstr, expected):
    model = pyo.ConcreteModel()
    model.T = len(consumption_data["electric"])
    model.t = range(model.T)
    pyo_vars = {}
    for key, val in consumption_data.items():
        var = pyo.Var(model.t, initialize=np.zeros(len(val)), bounds=(0, None))
        model.add_component(key, var)
        pyo_vars[key] = var

    @model.Constraint(model.t)
    def electric_constraint(m, t):
        return consumption_data["electric"][t] == m.electric[t]

    @model.Constraint(model.t)
    def gas_constraint(m, t):
        return consumption_data["gas"][t] == m.gas[t]

    var = getattr(model, varstr)
    result, model = ut.max(var, model=model, varstr="test")
    model.objective = pyo.Objective(expr=0)
    solver = pyo.SolverFactory("gurobi")
    solver.solve(model)
    assert pyo.value(result) == expected
    assert model is not None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, varstr, expected",
    [
        (
            {"electric": np.ones(96) * 45, "gas": np.ones(96) * -1},
            "electric",
            np.ones(96) * 45,
        ),
        ({"electric": np.ones(96) * 100, "gas": np.ones(96) * -1}, "gas", np.zeros(96)),
    ],
)
def test_max_pos_pyo(consumption_data, varstr, expected):
    model = pyo.ConcreteModel()
    model.T = len(consumption_data["electric"])
    model.t = range(model.T)
    pyo_vars = {}
    for key, val in consumption_data.items():
        var = pyo.Var(model.t, initialize=np.zeros(len(val)))
        model.add_component(key, var)
        pyo_vars[key] = var

    @model.Constraint(model.t)
    def electric_constraint(m, t):
        return consumption_data["electric"][t] == m.electric[t]

    @model.Constraint(model.t)
    def gas_constraint(m, t):
        return consumption_data["gas"][t] == m.gas[t]

    var = getattr(model, varstr)
    result, model = ut.max_pos(var, model=model, varstr="test")
    model.objective = pyo.Objective(expr=0)
    solver = pyo.SolverFactory("gurobi")
    solver.solve(model)

    # Check each element in returned vector
    for t in result.index_set():
        expected_element = expected[t]
        assert pyo.value(result[t]) == expected_element

    # TODO: add scalar test

    assert model is not None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, expected_positive, expected_negative",
    [
        (
            np.array([1, -2, 3, -4, 0]),
            np.array([1, 0, 3, 0, 0]),
            np.array([0, 2, 0, 4, 0]),
        ),
        (
            np.array([5, 0, -3, 7, -1]),
            np.array([5, 0, 0, 7, 0]),
            np.array([0, 0, 3, 0, 1]),
        ),
        (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])),
        (np.array([-10, -5, -1]), np.array([0, 0, 0]), np.array([10, 5, 1])),
        (np.array([10, 5, 1]), np.array([10, 5, 1]), np.array([0, 0, 0])),
    ],
)
def test_decompose_consumption_np(
    consumption_data, expected_positive, expected_negative
):
    """Test decompose_consumption with numpy arrays."""
    positive_values, negative_values, model = ut.decompose_consumption(consumption_data)

    assert np.array_equal(positive_values, expected_positive)
    assert np.array_equal(negative_values, expected_negative)
    assert model is None
    assert np.array_equal(consumption_data, positive_values - negative_values)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_decompose_consumption_cvx():
    """Test decompose_consumption with cvxpy expressions."""
    x = cp.Variable(5)
    positive_values, negative_values, model = ut.decompose_consumption(x)
    assert isinstance(positive_values, cp.Expression)
    assert isinstance(negative_values, cp.Expression)
    # TODO: add value checks


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, expected_positive_sum, expected_negative_sum",
    [
        (np.array([1, -2, 3, -4, 0]), 4, 6),  # positive: 1+3=4, negative: 2+4=6
        (np.array([0, 0, 0]), 0, 0),
        (np.array([-10, -5, -1]), 0, 16),  # positive: 0, negative: 10+5+1=16
        (np.array([10, 5, 1]), 16, 0),  # positive: 10+5+1=16, negative: 0
    ],
)
def test_decompose_consumption_pyo(
    consumption_data, expected_positive_sum, expected_negative_sum
):
    consumption_data_dict = {
        "electric": consumption_data,
        "gas": np.zeros_like(consumption_data),
    }
    model, pyo_vars = setup_pyo_vars_constraints(consumption_data_dict)
    positive_var, negative_var, model = ut.decompose_consumption(
        pyo_vars["electric"], model=model, varstr="electric"
    )

    # Check that variables exist and have the correct length
    assert hasattr(model, "electric_positive")
    assert hasattr(model, "electric_negative")
    assert hasattr(model, "electric_decomposition_constraint")
    assert hasattr(model, "electric_magnitude_constraint")
    assert len(positive_var) == len(consumption_data)
    assert len(negative_var) == len(consumption_data)
    # Testing of values handled after solving problem in test_costs.py

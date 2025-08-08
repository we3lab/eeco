import os
import pytest
import numpy as np
import pyomo.environ as pyo
import cvxpy as cp

from electric_emission_cost import utils as ut

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
    model.obj = pyo.Objective(expr=0)
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
    model.obj = pyo.Objective(expr=0)
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
    model.obj = pyo.Objective(expr=0)
    solver = pyo.SolverFactory("gurobi")
    solver.solve(model)
    assert pyo.value(result) == expected
    assert model is not None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, varstr, expected",
    [
        ({"electric": np.ones(96) * 45, "gas": np.ones(96) * -1}, "electric", 45),
        ({"electric": np.ones(96) * 100, "gas": np.ones(96) * -1}, "gas", 0),
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
    model.obj = pyo.Objective(expr=0)
    solver = pyo.SolverFactory("gurobi")
    solver.solve(model)
    assert pyo.value(result) == expected
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
        (np.array([5, 0, -3, 7, -1]), 12, 4),  # positive: 5+7=12, negative: 3+1=4
        (np.array([0, 0, 0]), 0, 0),
        (np.array([-10, -5, -1]), 0, 16),  # positive: 0, negative: 10+5+1=16
        (np.array([10, 5, 1]), 16, 0),  # positive: 10+5+1=16, negative: 0
    ],
)
def test_decompose_consumption_pyo(
    consumption_data, expected_positive_sum, expected_negative_sum
):
    """Test decompose_consumption with pyomo variables."""
    model = pyo.ConcreteModel()
    model.T = len(consumption_data)
    model.t = range(1, model.T + 1)  # Pyomo uses 1-indexed

    model.electric_consumption = pyo.Var(model.t, initialize=0)
    for t in model.t:
        model.electric_consumption[t].value = consumption_data[t - 1]

    positive_var, negative_var, model = ut.decompose_consumption(
        model.electric_consumption, model=model, varstr="electric"
    )

    init_consumption_data = {
        "electric": consumption_data,
    }
    ut.initialize_decomposed_pyo_vars(init_consumption_data, model, None)

    # Verify the expected sums from the initialized values
    assert (
        abs(sum(pyo.value(positive_var[t]) for t in model.t) - expected_positive_sum)
        < 1e-6
    )
    assert (
        abs(sum(pyo.value(negative_var[t]) for t in model.t) - expected_negative_sum)
        < 1e-6
    )

    # Verify the decomposition constraint is satisfied
    for t in model.t:
        assert (
            abs(
                pyo.value(model.electric_consumption[t])
                - (pyo.value(positive_var[t]) - pyo.value(negative_var[t]))
            )
            < 1e-6
        )

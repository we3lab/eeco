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
    "consumption_data, varstr, expected, expect_error",
    [
        (
            {"electric": np.ones(96) * 45, "gas": np.ones(96) * -1},
            "electric",
            np.ones(96) * 45,
            False,
        ),
        (
            {"electric": np.ones(96) * 100, "gas": np.ones(96) * -1},
            "gas",
            np.zeros(96),
            False,
        ),
        ({"electric": 45.0, "gas": -10.0}, "electric", 45.0, False),
        ({"electric": 100.0, "gas": -5.0}, "gas", 0.0, False),
        ([1, 2, 3], None, None, True),  # invalid type
    ],
)
def test_max_pos_pyo(consumption_data, varstr, expected, expect_error):
    if expect_error:
        with pytest.raises(TypeError):
            ut.max_pos(consumption_data)
        return

    model = pyo.ConcreteModel()

    if isinstance(consumption_data["electric"], (int, float)):
        # LinearExpression case
        pyo_vars = {}
        for key, val in consumption_data.items():
            var = pyo.Var(initialize=0)
            model.add_component(key, var)
            pyo_vars[key] = var

        @model.Constraint()
        def electric_constraint(m):
            return consumption_data["electric"] == m.electric

        @model.Constraint()
        def gas_constraint(m):
            return consumption_data["gas"] == m.gas

        var = getattr(model, varstr)
        expr = var - 0  # like max_var - prev_demand_cost
        result, model = ut.max_pos(expr, model=model, varstr="test")
        model.objective = pyo.Objective(expr=0)
        solver = pyo.SolverFactory("gurobi")
        solver.solve(model)

        assert pyo.value(result) == expected
    else:
        # Vector case
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

    assert model is not None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, expected_positive, expected_negative, expect_error",
    [
        (
            np.array([1, -2, 3, -4, 0]),
            np.array([1, 0, 3, 0, 0]),
            np.array([0, 2, 0, 4, 0]),
            False,
        ),
        (
            np.array([5, 0, -3, 7, -1]),
            np.array([5, 0, 0, 7, 0]),
            np.array([0, 0, 3, 0, 1]),
            False,
        ),
        (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), False),
        (np.array([-10, -5, -1]), np.array([0, 0, 0]), np.array([10, 5, 1]), False),
        (np.array([10, 5, 1]), np.array([10, 5, 1]), np.array([0, 0, 0]), False),
        ([1, 2, 3], None, None, True),  # invalid type
    ],
)
def test_decompose_consumption_np(
    consumption_data, expected_positive, expected_negative, expect_error
):
    """Test decompose_consumption with numpy arrays."""
    if expect_error:
        with pytest.raises(TypeError):
            ut.decompose_consumption(consumption_data)
    else:
        positive_values, negative_values, model = ut.decompose_consumption(
            consumption_data
        )

        assert np.array_equal(positive_values, expected_positive)
        assert np.array_equal(negative_values, expected_negative)
        assert model is None
        assert np.array_equal(consumption_data, positive_values - negative_values)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, expected_positive, expected_negative, "
    "decomposition_type, expect_error",
    [
        # absolute_value not DCP-compliant, raises NotImplementedError
        (np.array([1, -2, 3]), None, None, "absolute_value", True),
        (np.array([1, -2, 3]), None, None, None, True),  # default is absolute_value
        # binary_big_M works with MIP solver
        (
            np.array([10, -5, 3, -2, 0]),
            np.array([10, 0, 3, 0, 0]),
            np.array([0, 5, 0, 2, 0]),
            "binary_big_M",
            False,
        ),
        (
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            "binary_big_M",
            False,
        ),
        (
            np.array([-10, -5, -1]),
            np.array([0, 0, 0]),
            np.array([10, 5, 1]),
            "binary_big_M",
            False,
        ),
    ],
)
def test_decompose_consumption_cvx(
    consumption_data,
    expected_positive,
    expected_negative,
    decomposition_type,
    expect_error,
):
    """Test decompose_consumption with CVXPY expressions."""
    x = cp.Variable(len(consumption_data))

    if expect_error:
        with pytest.raises(NotImplementedError):
            if decomposition_type is None:
                ut.decompose_consumption(x)
            else:
                ut.decompose_consumption(x, decomposition_type=decomposition_type)
    else:
        positive_var, negative_var, constraints = ut.decompose_consumption(
            x, decomposition_type=decomposition_type
        )
        assert isinstance(positive_var, cp.Variable)
        assert isinstance(negative_var, cp.Variable)
        assert isinstance(constraints, list)
        assert len(constraints) == 3  # decomposition + 2 Big-M constraints

        # Solve with Gurobi and verify values
        constraints.append(x == consumption_data)
        prob = cp.Problem(cp.Minimize(0), constraints)
        prob.solve(solver=cp.GUROBI)

        np.testing.assert_array_almost_equal(positive_var.value, expected_positive)
        np.testing.assert_array_almost_equal(negative_var.value, expected_negative)


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, expected_positive_sum, expected_negative_sum, "
    "decomposition_type, expect_warning",
    [
        (np.array([1, -2, 3, -4, 0]), 4, 6, "absolute_value", False),
        (np.array([0, 0, 0]), 0, 0, "absolute_value", False),
        (np.array([-10, -5, -1]), 0, 16, "absolute_value", False),
        (np.array([10, 5, 1]), 16, 0, "absolute_value", False),
        (np.array([1, -2, 3]), 4, 2, "binary_big_M", False),
        (np.array([-10, -5, -1]), 0, 16, "binary_big_M", False),
        (np.array([10, 5, 1]), 16, 0, "binary_big_M", False),
        (np.array([1, -2, 3]), 4, 2, "unsupported_type", True),
    ],
)
def test_decompose_consumption_pyo(
    consumption_data,
    expected_positive_sum,
    expected_negative_sum,
    decomposition_type,
    expect_warning,
):
    consumption_data_dict = {
        "electric": consumption_data,
        "gas": np.zeros_like(consumption_data),
    }
    model, pyo_vars = setup_pyo_vars_constraints(consumption_data_dict)

    if expect_warning:
        with pytest.warns(UserWarning):
            positive_var, negative_var, model = ut.decompose_consumption(
                pyo_vars["electric"],
                model=model,
                varstr="electric",
                decomposition_type=decomposition_type,
            )
        assert positive_var is None
        assert negative_var is None
    else:
        positive_var, negative_var, model = ut.decompose_consumption(
            pyo_vars["electric"],
            model=model,
            varstr="electric",
            decomposition_type=decomposition_type,
        )
        # Check that variables exist and have the correct length
        assert hasattr(model, "electric_positive")
        assert hasattr(model, "electric_negative")
        assert hasattr(model, "electric_decomposition_constraint")
        if decomposition_type == "absolute_value":
            assert hasattr(model, "electric_magnitude_constraint")
        elif decomposition_type == "binary_big_M":
            assert hasattr(model, "electric_is_importing")
            assert hasattr(model, "electric_import_bigm_constraint")
            assert hasattr(model, "electric_export_bigm_constraint")
        assert len(positive_var) == len(consumption_data)
        assert len(negative_var) == len(consumption_data)
        # Testing of values handled after solving problem in test_costs.py

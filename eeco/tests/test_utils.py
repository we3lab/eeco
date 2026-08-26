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
    "freq, expected",
    [
        ("15m", 15),
        ("1m", 1),
        ("1h", 60),
        ("6h", 360),
        ("1D", 1440),
        ("1d", 1440),
        ("2d", 2880),
    ],
)
def test_get_freq_binsize_minutes(freq, expected):
    assert ut.get_freq_binsize_minutes(freq) == expected


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_get_freq_binsize_minutes_invalid_type_raises():
    with pytest.raises(ValueError):
        ut.get_freq_binsize_minutes("1y")


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
    ut.create_pyomo_model_index_ref(model, var)
    result, model = ut.sum(var, model=model, varstr="test")
    model.objective = pyo.Objective(expr=0)
    solver = pyo.SolverFactory("scip")
    solver.solve(model)
    assert pyo.value(result) == expected
    assert model is not None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, varstr1, varstr2, time_set, expected",
    [
        # two Pyo variables
        (
            {"electric": np.ones(96) * 100, "gas": np.ones(96)},
            "electric",
            "gas",
            None,
            np.ones(96) * 100,
        ),
        # Pyo variable * numpy charge array on non-standard index
        (
            {"electric": np.array([100.0, 200.0, 300.0, 400.0])},
            "electric",
            np.array([0.1, 0.2, 0.3, 0.4]),
            [0.0, 900.0, 1800.0, 2700.0],
            np.array([10.0, 40.0, 90.0, 160.0]),
        ),
        # Pyo variable * numpy charge array on irregular (non-uniform) index
        (
            {"electric": np.array([100.0, 200.0, 300.0, 400.0])},
            "electric",
            np.array([0.1, 0.2, 0.3, 0.4]),
            [2.0, 4.0, 5.0, 8.0],
            np.array([10.0, 40.0, 90.0, 160.0]),
        ),
    ],
)
def test_multiply_pyo(consumption_data, varstr1, varstr2, time_set, expected):
    model = pyo.ConcreteModel()
    model.T = len(consumption_data[varstr1])
    model.t = range(model.T) if time_set is None else time_set
    pos = {t: i for i, t in enumerate(model.t)}
    for key, val in consumption_data.items():
        var = pyo.Var(model.t, bounds=(None, None))
        model.add_component(key, var)
        for t in model.t:
            var[t].fix(float(val[pos[t]]))

    var1 = getattr(model, varstr1)
    var2 = getattr(model, varstr2) if isinstance(varstr2, str) else varstr2
    ut.create_pyomo_model_index_ref(model, var1)
    result, model = ut.multiply(var1, var2, model=model, varstr="test")
    model.objective = pyo.Objective(expr=0)
    solver = pyo.SolverFactory("ipopt")
    solver.solve(model)
    assert np.allclose([pyo.value(result[t]) for t in model.t], expected)
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

    ut.create_pyomo_model_index_ref(model, var)
    result, model = ut.max(var, model=model, varstr="test")

    model.objective = pyo.Objective(expr=0)
    solver = pyo.SolverFactory("scip")
    solver.solve(model)
    assert pyo.value(result) == expected
    assert model is not None


@pytest.mark.parametrize(
    "dict_type", ["pyovar", "normal", "nested", "multi_input", "empty"]
)
def test_create_pyomo_model_index_ref_from_dict(dict_type):
    model = pyo.ConcreteModel()
    model.e = pyo.Var(range(10), initialize=1)
    if dict_type == "normal":
        input_dict = {"electric": model.e}
    elif dict_type == "pyovar":
        input_dict = model.e
    elif dict_type == "nested":
        input_dict = {"nest_test": {"electric": model.e}}
    elif dict_type == "multi_input":
        input_dict = {
            "nest_test": {"electric": model.e, "another_entery": np.arange(10)},
            "electric": np.arange(10),
        }
    elif dict_type == "empty":
        input_dict = {"nest_test": {"electric": np.arange(10)}}

    if dict_type == "empty":
        with pytest.raises(TypeError):
            ut.createa_pyomo_model_index_from_dict(model, input_dict)
        assert not hasattr(model, "_var_index_ref")
        assert not hasattr(model, "_var_index")
    else:
        ut.createa_pyomo_model_index_from_dict(model, input_dict)
        assert hasattr(model, "_var_index_ref")
        assert hasattr(model, "_var_index")
        assert model._var_index_ref == {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
        }
        assert model._var_index == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


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
        ut.create_pyomo_model_index_ref(model, var)

        expr = var - 0  # like max_var - prev_demand_cost
        result, model = ut.max_pos(expr, model=model, varstr="test")
        model.objective = pyo.Objective(expr=0)
        solver = pyo.SolverFactory("scip")
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

        ut.create_pyomo_model_index_ref(model, var)
        result, model = ut.max_pos(var, model=model, varstr="test")
        model.objective = pyo.Objective(expr=0)
        solver = pyo.SolverFactory("scip")
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
def test_decompose_consumption_cvx():
    """Test decompose_consumption with cvxpy expressions."""
    x = cp.Variable(5)
    positive_values, negative_values, model = ut.decompose_consumption(x)
    assert isinstance(positive_values, cp.Expression)
    assert isinstance(negative_values, cp.Expression)

    # Test warning for unimplemented decomposition_type
    with pytest.warns(UserWarning):
        positive_values, negative_values, model = ut.decompose_consumption(
            x, decomposition_type="unimplemented"
        )
    assert positive_values is None
    assert negative_values is None


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "consumption_data, expected_positive_sum, expected_negative_sum, "
    "decomposition_type, expect_warning",
    [
        (np.array([1, -2, 3, -4, 0]), 4, 6, "absolute_value", False),
        (np.array([0, 0, 0]), 0, 0, "absolute_value", False),
        (np.array([-10, -5, -1]), 0, 16, "absolute_value", False),
        (np.array([10, 5, 1]), 16, 0, "absolute_value", False),
        (np.array([1, -2, 3]), 4, 2, "binary_variable", True),
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
    ut.createa_pyomo_model_index_from_dict(model, pyo_vars)
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
        assert hasattr(model, "electric_magnitude_constraint")
        assert len(positive_var) == len(consumption_data)
        assert len(negative_var) == len(consumption_data)
        # Testing of values handled after solving problem in test_costs.py


def test_pyomo_type():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1)
    m.x_i = pyo.Var([1, 2], initialize=1)
    m.e = pyo.Expression(expr=m.x_i[1] + m.x_i[2])
    m.e_i = pyo.Expression([1, 2], rule=lambda m, i: m.x_i[i] + 1)
    m.p_i = pyo.Param([1, 2], mutable=False, initialize={1: 1, 2: 2})
    m.p = pyo.Param(initialize=1)

    assert ut.check_indexed_pyomo_type(m.x) is False
    assert ut.check_nonindexed_pyomo_type(m.x)
    assert ut.check_indexed_pyomo_type(m.x_i)
    assert ut.check_indexed_pyomo_type(m.x_i[1]) is False
    assert ut.check_nonindexed_pyomo_type(m.x_i[1])

    assert ut.check_indexed_pyomo_type(m.e) is False
    assert ut.check_nonindexed_pyomo_type(m.e)
    assert ut.check_indexed_pyomo_type(m.e_i)
    assert ut.check_indexed_pyomo_type(m.e_i[1]) is False
    assert ut.check_nonindexed_pyomo_type(m.e_i[1])

    assert ut.check_indexed_pyomo_type(m.p) is False
    assert ut.check_nonindexed_pyomo_type(m.p)
    assert ut.check_indexed_pyomo_type(m.p_i[1]) is False
    # this will be false as pyomo returns an int for params
    assert ut.check_nonindexed_pyomo_type(m.p_i[1]) is False

    assert ut.check_cvx_type(m.x) is False
    assert ut.check_cvx_type(m.x_i) is False
    assert ut.check_cvx_type(m.x_i[1]) is False
    assert ut.check_cvx_type(m.e) is False
    assert ut.check_cvx_type(m.e_i) is False
    assert ut.check_cvx_type(m.e_i[1]) is False
    assert ut.check_cvx_type(m.p) is False
    assert ut.check_cvx_type(m.p_i[1]) is False


def test_cvx_type():

    cv = cp.Variable()
    ce = cv + 1

    assert ut.check_cvx_type(ce)
    assert ut.check_cvx_type(cv)

    assert ut.check_nonindexed_pyomo_type(ce) is False
    assert ut.check_nonindexed_pyomo_type(cv) is False

    assert ut.check_indexed_pyomo_type(ce) is False
    assert ut.check_indexed_pyomo_type(cv) is False


def test_python_types():
    x = 1
    y = 1.1
    z = np.array([1, 2, 3])
    ls = [1, 2, 3, 4]
    t = (1, 2, 3, 4)
    assert ut.check_nonindexed_python_type(x)
    assert ut.check_nonindexed_python_type(y)
    assert ut.check_indexed_np_array(z)

    # We do not support lists or tuples
    assert ut.check_indexed_np_array(ls) is False
    assert ut.check_nonindexed_python_type(ls) is False

    assert ut.check_indexed_np_array(t) is False
    assert ut.check_nonindexed_python_type(t) is False

    assert ut.check_nonindexed_python_type(z) is False
    assert ut.check_nonindexed_python_type(z[0])

    assert ut.check_cvx_type(x) is False
    assert ut.check_cvx_type(y) is False
    assert ut.check_cvx_type(z) is False
    assert ut.check_cvx_type(z[0]) is False

    assert ut.check_indexed_pyomo_type(x) is False
    assert ut.check_indexed_pyomo_type(y) is False
    assert ut.check_indexed_pyomo_type(z) is False
    assert ut.check_indexed_pyomo_type(z[0]) is False

    assert ut.check_nonindexed_pyomo_type(x) is False
    assert ut.check_nonindexed_pyomo_type(y) is False
    assert ut.check_nonindexed_pyomo_type(z) is False
    assert ut.check_nonindexed_pyomo_type(z[0]) is False

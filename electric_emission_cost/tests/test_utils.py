import os
import pytest
import numpy as np
import pyomo.environ as pyo

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
        setattr(model, key, var)
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
        setattr(model, key, var)
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
    solver = pyo.SolverFactory("ipopt", executable="/content/ipopt")
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
        setattr(model, key, var)
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
        setattr(model, key, var)
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

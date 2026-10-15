import pytest
import pyomo.environ as pyo


def assert_constraint_satisfied(constraint, abs_tol=1e-6):
    """Asserts a pyomo constraint (indexed or scalar) holds at its current,
    fully-valued variables, without needing to actually run a solver."""
    constraints = constraint.values() if constraint.is_indexed() else [constraint]
    for c in constraints:
        body_val = pyo.value(c.body)
        if c.equality:
            assert body_val == pytest.approx(pyo.value(c.lower), abs=abs_tol)
        else:
            if c.lower is not None:
                assert body_val >= pyo.value(c.lower) - abs_tol
            if c.upper is not None:
                assert body_val <= pyo.value(c.upper) + abs_tol

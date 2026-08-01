import numpy as np
import pytest

from fit_second_order_actuator import model_bounds, refine_bounded


def test_model_bounds_allows_high_bandwidth_0729_fits():
    assert model_bounds("second_order", fn_min_hz=0.4, fn_max_hz=20.0, delay_max_s=0.14) == [
        (0.4, 20.0),
        (0.03, 2.5),
        (0.0, 0.14),
    ]
    assert model_bounds("lead_lag", fn_min_hz=0.4, fn_max_hz=20.0, delay_max_s=0.20) == [
        (0.4, 20.0),
        (0.03, 2.5),
        (0.0, 0.20),
        (0.0, 0.16),
    ]


def test_refine_bounded_does_not_leave_delay_bounds():
    pytest.importorskip("scipy")
    bounds = [(0.4, 20.0), (0.03, 2.5), (0.0, 0.14)]

    def objective(params):
        # The unconstrained minimum is at a negative delay. The local refinement
        # must respect the physical delay bound instead of returning it.
        return float((params[0] - 2.0) ** 2 + (params[1] - 0.5) ** 2 + (params[2] + 0.02) ** 2)

    result = refine_bounded(objective, np.array([2.0, 0.5, 0.01]), bounds)

    assert result.x[2] >= 0.0
    assert result.x[2] <= 0.14

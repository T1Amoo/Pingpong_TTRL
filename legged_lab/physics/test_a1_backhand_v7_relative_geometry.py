from __future__ import annotations

import math

from legged_lab.physics import a1_backhand_v3_contract as v3
from legged_lab.physics import a1_backhand_v7_contract as v7


def _assert_pair_close(left, right, *, atol=1.0e-12):
    assert len(left) == len(right)
    for actual, expected in zip(left, right):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol)


def test_v7_translates_the_complete_v6_strike_box_by_ready_paddle_fk_delta():
    delta = tuple(
        new - old
        for old, new in zip(v7.V6_READY_PADDLE_WORLD_M, v7.V7_READY_PADDLE_WORLD_M)
    )
    _assert_pair_close(v7.READY_PADDLE_TRANSLATION_M, delta)

    assert math.isclose(
        v7.HIT_PLANE_X,
        v7.V6_HIT_PLANE_X + delta[0],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    _assert_pair_close(v7.HIT_TARGET_Y_RANGE, tuple(value + delta[1] for value in v3.HIT_TARGET_Y_RANGE))
    _assert_pair_close(v7.HIT_TARGET_Z_RANGE, tuple(value + delta[2] for value in v3.HIT_TARGET_Z_RANGE))
    _assert_pair_close(
        v7.PREFLIGHT_HIT_Z_RANGE,
        tuple(value + delta[2] for value in v3.PREFLIGHT_HIT_Z_RANGE),
    )


def test_v7_preserves_every_ready_paddle_relative_plane_and_window_offset():
    assert math.isclose(
        v7.V6_PLANE_FORWARD_OFFSET_M,
        v7.V7_PLANE_FORWARD_OFFSET_M,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    _assert_pair_close(v7.V6_HIT_Y_OFFSETS_M, v7.V7_HIT_Y_OFFSETS_M)
    _assert_pair_close(v7.V6_HIT_Z_OFFSETS_M, v7.V7_HIT_Z_OFFSETS_M)


def test_v7_compensated_serve_proposal_stays_on_the_physical_table():
    for pair in (
        v7.EASY_BOUNCE_X_RANGE,
        v7.HARD_BOUNCE_X_RANGE,
        v7.TAIL_BOUNCE_X_RANGE_HARD,
    ):
        assert -1.18 <= pair[0] < pair[1] <= -0.04
    assert v7.EASY_BOUNCE_VZ_RANGE[0] > 0.0
    assert v7.HARD_BOUNCE_VZ_RANGE[0] > 0.0
    assert v7.TAIL_BOUNCE_VZ_RANGE_HARD[0] > 0.0
    assert v7.HIT_TARGET_Y_RANGE[0] <= v7.FALLBACK_BOUNCE_Y <= v7.HIT_TARGET_Y_RANGE[1]
    assert v7.CURRICULUM_EASY_ITERS == v3.CURRICULUM_EASY_ITERS
    assert v7.CURRICULUM_RAMP_ITERS == v3.CURRICULUM_RAMP_ITERS
    assert v7.CURRICULUM_HOLD_ITERS == v3.CURRICULUM_HOLD_ITERS

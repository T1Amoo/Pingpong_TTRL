from __future__ import annotations

import pytest

from legged_lab.physics import a1_backhand_v5_contract as v5
from legged_lab.physics import a1_backhand_v6_contract as v6


DT = 0.02


def test_v6_inherits_v5_physics_outcomes_and_spin_without_redefinition():
    assert v6.MAX_ITERATIONS == v5.MAX_ITERATIONS == 20_000
    for inherited_name in (
        "CONTACT_EVENT_REWARD_WEIGHT",
        "TABLE_SUCCESS_REWARD_WEIGHT",
        "ACTUAL_TABLE_CENTER_REWARD_WEIGHT",
        "WEAK_TOPSPIN_EASY_RANGE_RAD_S",
        "WEAK_TOPSPIN_HARD_RANGE_RAD_S",
        "WEAK_TOPSPIN_TILT_DEG",
    ):
        assert not hasattr(v6, inherited_name)


def test_v6_contact_speed_and_drawdown_have_bounded_dt_budget():
    assert v6.FORWARD_SPEED_QUALITY_MIN_MPS == pytest.approx(0.40)
    assert v6.FORWARD_SPEED_QUALITY_FULL_MPS == pytest.approx(1.60)
    assert v6.FORWARD_SPEED_QUALITY_REWARD_WEIGHT * DT == pytest.approx(2.0)

    assert v6.PRECONTACT_DRAWDOWN_FREE_M == pytest.approx(0.04)
    assert v6.PRECONTACT_DRAWDOWN_FULL_M == pytest.approx(0.12)
    assert v6.PRECONTACT_DRAWDOWN_PENALTY_WEIGHT * DT == pytest.approx(-0.8)
    assert abs(v6.PRECONTACT_DRAWDOWN_PENALTY_WEIGHT) < (
        v6.FORWARD_SPEED_QUALITY_REWARD_WEIGHT
    )


def test_v6_dense_guides_stay_below_one_shot_contact_speed_quality():
    max_swing_per_tick = (
        v6.SWING_THROUGH_REWARD_WEIGHT * v6.SWING_THROUGH_CAP_MPS * DT
    )
    max_speed_quality_event = v6.FORWARD_SPEED_QUALITY_REWARD_WEIGHT * DT
    assert v6.FUTURE_YZ_REWARD_WEIGHT == pytest.approx(0.5)
    assert max_swing_per_tick == pytest.approx(0.024)
    assert max_swing_per_tick < max_speed_quality_event


def test_v6_disables_route_and_predicted_miss_penalties():
    assert v6.EARLY_FORWARD_PENALTY_WEIGHT == 0.0
    assert v6.LATE_BACKTRACK_PENALTY_WEIGHT == 0.0
    assert v6.CONTACT_LATERAL_SPEED_PENALTY_WEIGHT == 0.0
    assert v6.LANDING_OUTSIDE_PENALTY_WEIGHT == 0.0

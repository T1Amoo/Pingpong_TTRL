from __future__ import annotations

from legged_lab.physics import a1_backhand_v4_contract as v4
from legged_lab.physics import a1_backhand_v5_contract as v5


def test_v5_keeps_v4_translational_serve_contract():
    assert v5.MAX_ITERATIONS == v4.MAX_ITERATIONS == 20_000
    # V5 deliberately defines no replacement launch box.  The environment
    # inherits all of these values directly from A1TableTennisBackhandV4EnvCfg.
    assert not hasattr(v5, "EASY_BOUNCE_X_RANGE")
    assert not hasattr(v5, "HARD_BOUNCE_X_RANGE")
    assert not hasattr(v5, "TAIL_BOUNCE_X_RANGE_HARD")


def test_v5_sparse_outcome_budget_has_the_requested_positive_hierarchy():
    dt = 0.02
    contact = v5.CONTACT_EVENT_REWARD_WEIGHT * dt
    best_proxy = (
        v5.PREDICTED_LANDING_REWARD_WEIGHT + v5.PASS_NET_REWARD_WEIGHT
    ) * dt
    table = v5.TABLE_SUCCESS_REWARD_WEIGHT * dt
    center = v5.ACTUAL_TABLE_CENTER_REWARD_WEIGHT * dt

    assert 0.0 < contact
    assert contact < contact + table
    assert contact + best_proxy < contact + best_proxy + table
    assert contact + best_proxy + table < contact + best_proxy + table + center
    # A physical bounce is worth more than every predicted return proxy at its
    # simultaneous maximum, preventing proxy farming from beating real return.
    assert table > best_proxy

    # Even a best-case off-table sparse outcome stays below the worst physical
    # table event after bounding the approach bonus. The minimum center score
    # is evaluated at the farthest opponent-table corner.
    contact_quality = v5.CONTACT_QUALITY_REWARD_WEIGHT * dt
    sweet = v5.SWEET_CONTACT_REWARD_WEIGHT * dt
    approach = (
        v5.APPROACH_VELOCITY_REWARD_WEIGHT
        * v5.APPROACH_VELOCITY_CAP_MPS
        * dt
    )
    direction = v5.HIT_DIRECTION_REWARD_WEIGHT * dt
    best_off_table = contact + contact_quality + sweet + approach + direction + best_proxy
    farthest_corner_distance_sq = (0.0 - v5.LANDING_TARGET_X) ** 2 + (
        0.7625 - v5.LANDING_TARGET_Y
    ) ** 2
    corner_center_quality = 2.0 ** (
        -farthest_corner_distance_sq
        / (v5.LANDING_TARGET_HALF_REWARD_RADIUS_M**2)
    )
    worst_table = contact + table + center * corner_center_quality
    assert worst_table > best_off_table


def test_v5_removes_new_v4_constraints_but_keeps_legacy_high_guard():
    assert v5.EARLY_FORWARD_PENALTY_WEIGHT < 0.0
    assert v5.LATE_BACKTRACK_PENALTY_WEIGHT == 0.0
    assert v5.CONTACT_LATERAL_SPEED_PENALTY_WEIGHT == 0.0
    assert v5.LANDING_OUTSIDE_PENALTY_WEIGHT == 0.0
    assert v5.PADDLE_ABOVE_TARGET_PENALTY_WEIGHT == -4.0


def test_v5_spin_is_weak_topspin_only():
    assert v5.WEAK_TOPSPIN_EASY_RANGE_RAD_S == (2.0, 4.0)
    assert v5.WEAK_TOPSPIN_HARD_RANGE_RAD_S == (2.0, 8.0)
    assert 0.0 < v5.WEAK_TOPSPIN_TILT_DEG <= 10.0

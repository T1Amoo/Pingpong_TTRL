from __future__ import annotations

import math

import torch

from legged_lab.physics import a1_backhand_v7_contract as v7
from legged_lab.physics.a1_backhand_v7_safety import (
    minimum_vertical_capsule_clearance,
    normalized_clearance_penalty,
    safe_postimpact_retraction_reward,
)


def test_v7_ready_clearance_leaves_model_and_assembly_margin():
    assert v7.ENABLE_SELF_COLLISIONS
    assert v7.AUDITED_READY_MESH_CLEARANCE_M > v7.AUDITED_READY_PROXY_CLEARANCE_M
    assert v7.AUDITED_READY_PROXY_CLEARANCE_M > v7.POSTIMPACT_RETRACTION_FULL_REWARD_CLEARANCE_M
    assert v7.POSTIMPACT_RETRACTION_FULL_REWARD_CLEARANCE_M > v7.PADDLE_BODY_SAFE_CLEARANCE_M
    assert v7.PADDLE_BODY_SAFE_CLEARANCE_M > v7.PADDLE_BODY_TERMINATION_CLEARANCE_M
    assert (
        v7.AUDITED_READY_PROXY_CLEARANCE_M - v7.PADDLE_BODY_SAFE_CLEARANCE_M
        >= 0.03
    )


def test_vertical_capsule_clearance_uses_the_nearest_paddle_sample():
    points = torch.tensor(
        [
            [[0.30, 0.0, 1.0], [0.20, 0.0, 1.0]],
            [[0.04, 0.0, 1.60], [0.04, 0.0, 1.50]],
        ],
        dtype=torch.float64,
    )
    clearance = minimum_vertical_capsule_clearance(
        points,
        center_xy_m=(0.04, 0.0),
        z_range_m=(0.65, 1.40),
        radius_m=0.12,
    )
    torch.testing.assert_close(clearance, torch.tensor([0.04, -0.02], dtype=torch.float64))


def test_clearance_penalty_is_zero_at_soft_boundary_and_full_at_hard_boundary():
    clearance = torch.tensor([0.10, 0.065, 0.050, 0.035, 0.0])
    penalty = normalized_clearance_penalty(
        clearance,
        safe_clearance_m=v7.PADDLE_BODY_SAFE_CLEARANCE_M,
        full_penalty_clearance_m=v7.PADDLE_BODY_FULL_PENALTY_CLEARANCE_M,
    )
    torch.testing.assert_close(
        penalty,
        torch.tensor([0.0, 0.0, 0.25, 1.0, 1.0]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_postimpact_retraction_caps_before_ready_and_never_rewards_safety_margin():
    start = v7.POSTIMPACT_RETRACTION_START_X_M
    target = v7.POSTIMPACT_RETRACTION_TARGET_X_M
    midpoint = 0.5 * (start + target)
    paddle_x = torch.tensor([start, midpoint, target, target - 0.10, target])
    clearance = torch.tensor([0.10, 0.10, 0.10, 0.10, 0.065])
    active = torch.tensor([True, True, True, True, True])
    reward = safe_postimpact_retraction_reward(
        paddle_x,
        clearance,
        active,
        start_x_m=start,
        target_x_m=target,
        safe_clearance_m=v7.PADDLE_BODY_SAFE_CLEARANCE_M,
        full_reward_clearance_m=v7.POSTIMPACT_RETRACTION_FULL_REWARD_CLEARANCE_M,
    )
    torch.testing.assert_close(
        reward,
        torch.tensor([0.0, 0.5, 1.0, 1.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert math.isclose(
        target - v7.V7_READY_PADDLE_WORLD_M[0], 0.04, rel_tol=0.0, abs_tol=1.0e-12
    )
    assert 0.15 < start - target < 0.17

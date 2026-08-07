from __future__ import annotations

import torch

from legged_lab.physics.swing_timing import (
    curriculum_lerp,
    early_hold_gate,
    excess_speed_penalty,
    late_swing_gate,
    linear_curriculum_progress,
    phase_open_gate,
    staged_intercept_reward,
    weighted_yz_distance,
    x_position_progress,
    x_tracking_gain,
)


def test_x_tracking_is_disabled_early_and_full_at_contact():
    times = torch.tensor([0.60, 0.45, 0.225, 0.0])
    gain = x_tracking_gain(times, full_tracking_window_s=0.45, floor=0.0)
    torch.testing.assert_close(gain, torch.tensor([0.0, 0.0, 0.5, 1.0]))


def test_early_hold_releases_smoothly_before_swing_window():
    times = torch.tensor([0.60, 0.525, 0.45, 0.30])
    gate = early_hold_gate(times, release_s=0.45, ramp_s=0.15)
    torch.testing.assert_close(gate, torch.tensor([1.0, 0.5, 0.0, 0.0]))


def test_late_swing_gate_only_grows_toward_contact():
    times = torch.tensor([0.60, 0.45, 0.225, 0.0])
    gate = late_swing_gate(times, window_s=0.45)
    torch.testing.assert_close(gate, torch.tensor([0.0, 0.0, 0.5, 1.0]))


def test_excess_speed_penalty_preserves_contact_deadband():
    penalty = excess_speed_penalty(
        torch.tensor([-0.83, -0.38, 0.30, 1.20]),
        deadband_mps=0.30,
        ramp_mps=0.70,
    )
    torch.testing.assert_close(
        penalty,
        torch.tensor([(0.53 / 0.70) ** 2, (0.08 / 0.70) ** 2, 0.0, 1.0]),
    )


def test_phase_gate_is_closed_early_and_opens_over_two_control_ticks():
    times = torch.tensor([0.70, 0.60, 0.58, 0.56, 0.30])
    gate = phase_open_gate(times, open_time_s=0.60, transition_s=0.04)
    torch.testing.assert_close(gate, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    assert torch.all(gate[:-1] <= gate[1:])


def test_opening_x_phase_never_reduces_tracking_reward():
    yz_distance = torch.tensor([0.08, 0.20, 0.40])
    x_progress = torch.tensor([1.0, 0.5, 0.0])
    closed = staged_intercept_reward(
        yz_distance,
        x_progress,
        torch.zeros(3),
        std_ee=0.5,
        threshold=0.08,
    )
    half = staged_intercept_reward(
        yz_distance,
        x_progress,
        torch.full((3,), 0.5),
        std_ee=0.5,
        threshold=0.08,
    )
    opened = staged_intercept_reward(
        yz_distance,
        x_progress,
        torch.ones(3),
        std_ee=0.5,
        threshold=0.08,
    )
    assert torch.all(closed <= half)
    assert torch.all(half <= opened)


def test_timing_curriculum_is_resume_safe_and_eval_can_select_final_value():
    start = 1_000 * 240
    ramp = 3_000 * 240
    assert linear_curriculum_progress(
        start - 1,
        start_raw_step=start,
        ramp_raw_steps=ramp,
    ) == 0.0
    assert linear_curriculum_progress(
        start + ramp // 2,
        start_raw_step=start,
        ramp_raw_steps=ramp,
    ) == 0.5
    assert linear_curriculum_progress(
        start + ramp,
        start_raw_step=start,
        ramp_raw_steps=ramp,
    ) == 1.0
    assert linear_curriculum_progress(
        0,
        start_raw_step=0,
        ramp_raw_steps=0,
    ) == 1.0
    assert curriculum_lerp(0.80, 0.60, 0.5) == 0.70


def test_intercept_distance_and_reward_have_finite_gradient_at_exact_target():
    diff = torch.zeros((2, 3), requires_grad=True)
    yz_distance = weighted_yz_distance(diff, z_weight=2.5)
    x_progress = x_position_progress(
        diff[:, 0],
        zero_reward_error_m=0.22,
        full_reward_error_m=0.02,
    )
    reward = staged_intercept_reward(
        yz_distance,
        x_progress,
        torch.ones(2),
        std_ee=0.5,
        threshold=0.08,
    )
    reward.sum().backward()
    assert torch.isfinite(diff.grad).all()
    torch.testing.assert_close(diff.grad, torch.zeros_like(diff.grad))


def test_x_progress_starts_outside_ready_and_strictly_grows_inward():
    errors = torch.tensor([0.30, 0.22, 0.197, 0.12, 0.02, 0.00])
    progress = x_position_progress(
        errors,
        zero_reward_error_m=0.22,
        full_reward_error_m=0.02,
    )
    torch.testing.assert_close(
        progress,
        torch.tensor([0.0, 0.0, 0.013225, 0.25, 1.0, 1.0]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.all(progress[:-1] <= progress[1:])


def test_closed_x_phase_is_independent_of_x_error():
    x_progress = x_position_progress(
        torch.tensor([0.20, 0.05, 0.00]),
        zero_reward_error_m=0.22,
        full_reward_error_m=0.02,
    )
    reward = staged_intercept_reward(
        torch.full((3,), 0.10),
        x_progress,
        torch.zeros(3),
        std_ee=0.5,
        threshold=0.08,
    )
    torch.testing.assert_close(reward, reward[0].expand_as(reward))

from __future__ import annotations

import torch

from legged_lab.physics.swing_timing import (
    early_hold_gate,
    excess_speed_penalty,
    late_swing_gate,
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

from __future__ import annotations

import torch

from legged_lab.physics.return_flight import (
    project_height_to_forward_x_plane,
    smooth_clearance_gate,
)


def test_forward_plane_projection_matches_drag_inverse_time():
    z_at_net, time_s, valid = project_height_to_forward_x_plane(
        torch.tensor([-1.243]),
        torch.tensor([1.10]),
        torch.tensor([4.0]),
        torch.tensor([2.0]),
        target_x=0.0,
        horizontal_drag_accel_k=0.09910893224020438,
    )
    expected_t = torch.expm1(torch.tensor(0.09910893224020438 * 1.243)) / (
        0.09910893224020438 * 4.0
    )
    torch.testing.assert_close(time_s, expected_t.reshape(1))
    torch.testing.assert_close(
        z_at_net,
        torch.tensor([1.10]) + 2.0 * expected_t - 0.5 * 9.81 * expected_t.square(),
    )
    assert bool(valid[0])


def test_forward_plane_projection_rejects_wrong_way_and_stays_finite():
    z_at_net, time_s, valid = project_height_to_forward_x_plane(
        torch.tensor([-1.0, 0.1]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([-2.0, 2.0]),
        torch.tensor([0.0, 0.0]),
        target_x=0.0,
        horizontal_drag_accel_k=0.1,
    )
    torch.testing.assert_close(valid, torch.tensor([False, False]))
    torch.testing.assert_close(time_s, torch.zeros(2))
    torch.testing.assert_close(z_at_net, torch.tensor([1.0, 2.0]))


def test_clearance_gate_is_zero_below_net_and_smoothly_opens_above():
    gate = smooth_clearance_gate(
        torch.tensor([0.90, 0.945, 0.960, 0.975, 1.10]),
        min_height=0.945,
        ramp_m=0.030,
    )
    torch.testing.assert_close(gate, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))

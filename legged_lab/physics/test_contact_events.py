from __future__ import annotations

import torch

from legged_lab.physics.contact_events import (
    horizontal_alignment_squared,
    post_impact_event_mask,
)


def test_post_impact_waits_for_outgoing_state_but_times_out_failed_return():
    emit = post_impact_event_mask(
        torch.tensor([True, True, True, False]),
        torch.tensor([0.02, 0.02, 0.06, 0.10]),
        torch.tensor([-2.0, 0.20, -0.30, 1.0]),
        min_outgoing_vx_mps=0.05,
        timeout_s=0.06,
    )
    torch.testing.assert_close(emit, torch.tensor([False, True, True, False]))


def test_horizontal_alignment_uses_latched_xy_and_ignores_vertical_component():
    score = horizontal_alignment_squared(
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        torch.tensor([[-2.0, 0.0, 20.0], [-2.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(score, torch.tensor([1.0, 0.0]))

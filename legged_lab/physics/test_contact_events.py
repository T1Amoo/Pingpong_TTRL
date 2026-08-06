from __future__ import annotations

import torch

from legged_lab.physics.contact_events import (
    first_latched_event_mask,
    horizontal_alignment_squared,
    post_impact_event_mask,
    update_table_bounce_latches,
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


def test_first_latched_event_fires_once_and_ignores_inactive_samples():
    event = first_latched_event_mask(
        torch.tensor([False, True, True, False]),
        torch.tensor([False, False, True, True]),
    )
    torch.testing.assert_close(event, torch.tensor([False, True, False, False]))


def test_table_bounce_requires_post_hit_descent_then_fires_once_on_rise():
    descending_seen = torch.tensor([False])
    bounce_seen = torch.tensor([False])
    events = []
    for has_hit, in_band, vz in (
        (False, True, -2.0),
        (True, False, -1.0),
        (True, True, -0.2),
        (True, True, 0.0),
        (True, True, 0.4),
        (True, True, 0.8),
    ):
        event, descending_seen, bounce_seen = update_table_bounce_latches(
            torch.tensor([has_hit]),
            torch.tensor([in_band]),
            torch.tensor([vz]),
            descending_seen,
            bounce_seen,
        )
        events.append(bool(event[0]))
    assert events == [False, False, False, False, True, False]

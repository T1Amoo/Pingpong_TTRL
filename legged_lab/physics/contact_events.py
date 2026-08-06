"""Simulator-independent contact event helpers."""

from __future__ import annotations

import torch


def post_impact_event_mask(
    pending: torch.Tensor,
    pending_age_s: torch.Tensor,
    outgoing_vx_mps: torch.Tensor,
    *,
    min_outgoing_vx_mps: float,
    timeout_s: float,
) -> torch.Tensor:
    """Emit once an outgoing return is observed or the contact wait expires."""

    outgoing = outgoing_vx_mps >= float(min_outgoing_vx_mps)
    timed_out = pending_age_s >= float(timeout_s)
    return pending.bool() & (outgoing | timed_out)


def first_latched_event_mask(
    active_now: torch.Tensor,
    active_seen: torch.Tensor,
) -> torch.Tensor:
    """Return only the first active sample of a latched episode event.

    ``active_seen`` is intentionally supplied by the caller rather than
    mutated here so this helper remains simulator-independent and easy to
    regression-test.
    """

    return active_now.bool() & (~active_seen.bool())


def update_table_bounce_latches(
    has_hit: torch.Tensor,
    in_opponent_table_band: torch.Tensor,
    vertical_velocity: torch.Tensor,
    descending_seen: torch.Tensor,
    bounce_seen: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect the first post-hit table bounce from a down-to-up transition.

    The caller invokes this at physics-substep rate. Entering the broad table
    geometry band while still descending is not success; a positive vertical
    velocity in the same band is required after descent has been observed.
    """

    active = has_hit.bool() & in_opponent_table_band.bool()
    descending_seen_next = descending_seen.bool() | (
        active & (vertical_velocity < 0.0)
    )
    bounced_now = active & descending_seen_next & (vertical_velocity > 0.0)
    event = first_latched_event_mask(bounced_now, bounce_seen)
    bounce_seen_next = bounce_seen.bool() | bounced_now
    return event, descending_seen_next, bounce_seen_next


def horizontal_alignment_squared(
    paddle_normal: torch.Tensor,
    incoming_velocity: torch.Tensor,
) -> torch.Tensor:
    """Squared absolute alignment in the table plane, ignoring vertical speed."""

    normal_xy = torch.nn.functional.normalize(paddle_normal[..., :2], dim=-1, eps=1.0e-6)
    incoming_xy = torch.nn.functional.normalize(
        incoming_velocity[..., :2], dim=-1, eps=1.0e-6
    )
    return torch.abs((normal_xy * incoming_xy).sum(dim=-1)).clamp(max=1.0).square()

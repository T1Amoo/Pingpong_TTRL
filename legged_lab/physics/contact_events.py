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

"""Small, simulator-independent gates for table-tennis swing timing rewards."""

from __future__ import annotations

import torch


def x_tracking_gain(
    time_to_hit_s: torch.Tensor,
    *,
    full_tracking_window_s: float,
    floor: float = 0.0,
) -> torch.Tensor:
    """Ramp x-position tracking from ``floor`` to one near contact.

    A zero/negative window preserves the historical always-on behavior.
    """

    if full_tracking_window_s <= 0.0:
        return torch.ones_like(time_to_hit_s)
    return torch.clamp(
        1.0 - time_to_hit_s / float(full_tracking_window_s),
        min=float(floor),
        max=1.0,
    )


def early_hold_gate(
    time_to_hit_s: torch.Tensor,
    *,
    release_s: float,
    ramp_s: float,
) -> torch.Tensor:
    """Return one well before the hit and fade to zero at ``release_s``."""

    if ramp_s <= 0.0:
        return (time_to_hit_s > float(release_s)).to(time_to_hit_s.dtype)
    return torch.clamp(
        (time_to_hit_s - float(release_s)) / float(ramp_s),
        min=0.0,
        max=1.0,
    )


def late_swing_gate(
    time_to_hit_s: torch.Tensor,
    *,
    window_s: float,
) -> torch.Tensor:
    """Ramp from zero at the swing-window boundary to one at contact."""

    if window_s <= 0.0:
        return torch.zeros_like(time_to_hit_s)
    return torch.clamp(
        1.0 - time_to_hit_s / float(window_s),
        min=0.0,
        max=1.0,
    )


def excess_speed_penalty(
    speed_mps: torch.Tensor,
    *,
    deadband_mps: float,
    ramp_mps: float,
) -> torch.Tensor:
    """Squared unit penalty above an allowed speed deadband."""

    excess = torch.clamp(torch.abs(speed_mps) - float(deadband_mps), min=0.0)
    scaled = torch.clamp(excess / max(float(ramp_mps), 1.0e-6), max=1.0)
    return torch.square(scaled)

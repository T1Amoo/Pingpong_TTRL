"""Simulator-independent outgoing-ball projection helpers."""

from __future__ import annotations

import torch


def project_height_to_forward_x_plane(
    x: torch.Tensor,
    z: torch.Tensor,
    vx: torch.Tensor,
    vz: torch.Tensor,
    *,
    target_x: float,
    horizontal_drag_accel_k: float,
    gravity: float = 9.81,
    min_forward_speed_mps: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project height/time to a forward x-plane with horizontal quadratic drag.

    This matches the established landing approximation: x follows
    ``dv/dt=-k*|v|*v`` component-wise while z remains ballistic. The function
    returns ``(z_at_plane, time_s, valid)`` and emits finite placeholders for
    samples that are not moving toward the requested plane.
    """

    dx = float(target_x) - x
    valid = (dx >= 0.0) & (vx > float(min_forward_speed_mps))
    distance = torch.where(valid, dx, torch.zeros_like(dx))
    safe_vx = torch.where(valid, vx, torch.ones_like(vx))
    k = max(float(horizontal_drag_accel_k), 0.0)
    if k <= 1.0e-8:
        time_s = distance / safe_vx
    else:
        time_s = torch.expm1(k * distance) / (k * safe_vx)
    time_s = torch.where(valid, time_s, torch.zeros_like(time_s))
    z_at_plane = z + vz * time_s - 0.5 * float(gravity) * torch.square(time_s)
    return z_at_plane, time_s, valid


def smooth_clearance_gate(
    height: torch.Tensor,
    *,
    min_height: float,
    ramp_m: float,
) -> torch.Tensor:
    """Zero below clearance, then cubic-smoothly open to one over ``ramp_m``."""

    width = max(float(ramp_m), 1.0e-6)
    phase = torch.clamp((height - float(min_height)) / width, min=0.0, max=1.0)
    return torch.square(phase) * (3.0 - 2.0 * phase)

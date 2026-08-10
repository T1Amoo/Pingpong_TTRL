"""Pure tensor helpers for the A1 backhand-v7 paddle/body safety contract."""

from __future__ import annotations

import torch


def minimum_vertical_capsule_clearance(
    points_base: torch.Tensor,
    *,
    center_xy_m: tuple[float, float],
    z_range_m: tuple[float, float],
    radius_m: float,
) -> torch.Tensor:
    """Return minimum sample-point clearance to a vertical torso capsule.

    ``points_base`` has shape ``(..., samples, 3)`` and is expressed in the
    robot-root frame.  The returned value is positive outside the capsule and
    negative inside it.
    """

    if points_base.ndim < 2 or points_base.shape[-1] != 3:
        raise ValueError("points_base must have shape (..., samples, 3)")
    if z_range_m[1] <= z_range_m[0]:
        raise ValueError("z_range_m must be strictly increasing")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")

    closest_z = torch.clamp(
        points_base[..., 2], min=float(z_range_m[0]), max=float(z_range_m[1])
    )
    dx = points_base[..., 0] - float(center_xy_m[0])
    dy = points_base[..., 1] - float(center_xy_m[1])
    dz = points_base[..., 2] - closest_z
    distance = torch.sqrt(torch.clamp(dx * dx + dy * dy + dz * dz, min=0.0))
    clearance = torch.amin(distance, dim=-1) - float(radius_m)
    return torch.nan_to_num(clearance, nan=-float(radius_m))


def normalized_clearance_penalty(
    clearance_m: torch.Tensor,
    *,
    safe_clearance_m: float,
    full_penalty_clearance_m: float,
) -> torch.Tensor:
    """Squared penalty: zero at the soft boundary, one at the hard boundary."""

    span = float(safe_clearance_m) - float(full_penalty_clearance_m)
    if span <= 0.0:
        raise ValueError("safe_clearance_m must exceed full_penalty_clearance_m")
    normalized = torch.clamp(
        (float(safe_clearance_m) - clearance_m) / span,
        min=0.0,
        max=1.0,
    )
    return torch.square(torch.nan_to_num(normalized, nan=1.0))


def safe_postimpact_retraction_reward(
    paddle_x_m: torch.Tensor,
    clearance_m: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    start_x_m: float,
    target_x_m: float,
    safe_clearance_m: float,
    full_reward_clearance_m: float,
) -> torch.Tensor:
    """Bounded post-impact retraction progress with an independent safety gate.

    X increases toward the table.  Reward therefore grows while the paddle
    moves from ``start_x_m`` backward to ``target_x_m`` and then saturates.
    Moving still farther toward the chassis cannot increase it.  The smooth
    clearance gate removes reward at the soft safety boundary, so the term can
    never pay for using the safety margin.
    """

    retraction_span = float(start_x_m) - float(target_x_m)
    clearance_span = float(full_reward_clearance_m) - float(safe_clearance_m)
    if retraction_span <= 0.0:
        raise ValueError("start_x_m must be greater than target_x_m")
    if clearance_span <= 0.0:
        raise ValueError("full_reward_clearance_m must exceed safe_clearance_m")

    progress = torch.clamp(
        (float(start_x_m) - paddle_x_m) / retraction_span,
        min=0.0,
        max=1.0,
    )
    clearance_gate = torch.clamp(
        (clearance_m - float(safe_clearance_m)) / clearance_span,
        min=0.0,
        max=1.0,
    )
    # Smoothstep avoids a discontinuous reward gradient at either clearance
    # boundary while retaining exact zero/one endpoints.
    clearance_gate = clearance_gate * clearance_gate * (3.0 - 2.0 * clearance_gate)
    reward = progress * clearance_gate * active_mask.float()
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

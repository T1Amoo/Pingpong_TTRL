"""Simulator-independent scoring for opponent-table landing quality."""

from __future__ import annotations

import math

import torch


def landing_target_quality(
    pred_x: torch.Tensor,
    pred_y: torch.Tensor,
    *,
    target_x: float,
    target_y: float,
    half_reward_radius_m: float,
) -> torch.Tensor:
    """Return a smooth Gaussian quality score around the target.

    ``half_reward_radius_m`` is the half-reward contour, not a zero-reward
    boundary. A safe return 50 cm from the opponent-table center therefore
    remains positively rewarded. Off-table predictions are handled by the
    separate rectangular penalty below instead of shifting the whole kernel.
    """

    radius = max(float(half_reward_radius_m), 1.0e-6)
    distance_sq = torch.square(pred_x - float(target_x)) + torch.square(
        pred_y - float(target_y)
    )
    return torch.exp(-math.log(2.0) * distance_sq / (radius * radius))


def landing_rectangle_outside_penalty(
    pred_x: torch.Tensor,
    pred_y: torch.Tensor,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    half_penalty_distance_m: float,
) -> torch.Tensor:
    """Return a smooth zero-inside, one-far-outside rectangle penalty.

    Distance is Euclidean to the nearest point on the opponent-table
    rectangle. The score is zero everywhere inside, 0.5 at the configured
    outside distance and approaches one for a clear miss. Its zero derivative
    at the edge avoids a hard reward discontinuity from projection noise.
    """

    x_lo, x_hi = sorted((float(x_range[0]), float(x_range[1])))
    y_lo, y_hi = sorted((float(y_range[0]), float(y_range[1])))
    dx = torch.clamp(x_lo - pred_x, min=0.0) + torch.clamp(pred_x - x_hi, min=0.0)
    dy = torch.clamp(y_lo - pred_y, min=0.0) + torch.clamp(pred_y - y_hi, min=0.0)
    outside_distance_sq = torch.square(dx) + torch.square(dy)
    radius = max(float(half_penalty_distance_m), 1.0e-6)
    return 1.0 - torch.exp(-math.log(2.0) * outside_distance_sq / (radius * radius))

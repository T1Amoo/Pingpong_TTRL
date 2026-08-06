"""Simulator-independent scoring for a safe opponent-table landing target."""

from __future__ import annotations

import math

import torch


def landing_target_score(
    pred_x: torch.Tensor,
    pred_y: torch.Tensor,
    *,
    target_x: float,
    target_y: float,
    radius_m: float,
    outside_floor: float = -1.0,
) -> torch.Tensor:
    """Return a smooth shifted-Gaussian score around the target.

    ``radius_m`` is the meaningful zero contour rather than a hard cutoff:
    the score is +1 at the target, exactly zero at the requested radius and
    approaches -1 smoothly farther away.  This keeps a useful gradient on
    both sides of the 50 cm safe-return boundary without paying positive
    reward for arbitrarily distant or off-table predictions.
    """

    radius = max(float(radius_m), 1.0e-6)
    distance = torch.sqrt(
        torch.square(pred_x - float(target_x)) + torch.square(pred_y - float(target_y))
    )
    score = 2.0 * torch.exp(-math.log(2.0) * torch.square(distance / radius)) - 1.0
    return torch.clamp(score, min=float(outside_floor), max=1.0)

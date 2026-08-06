from __future__ import annotations

import torch

from legged_lab.physics.landing_target import (
    landing_rectangle_outside_penalty,
    landing_target_quality,
)


def test_landing_target_quality_uses_half_reward_radius():
    score = landing_target_quality(
        torch.tensor([0.70, 1.20, 1.70, 2.20]),
        torch.zeros(4),
        target_x=0.70,
        target_y=0.0,
        half_reward_radius_m=0.50,
    )
    torch.testing.assert_close(
        score,
        torch.tensor([1.0, 0.5, 0.0625, 0.001953125]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_landing_target_quality_treats_lateral_error_symmetrically():
    score = landing_target_quality(
        torch.full((2,), 0.70),
        torch.tensor([-0.25, 0.25]),
        target_x=0.70,
        target_y=0.0,
        half_reward_radius_m=0.50,
    )
    expected = torch.pow(torch.tensor(2.0), torch.tensor(-0.25))
    torch.testing.assert_close(score, expected.repeat(2))


def test_landing_target_quality_decreases_continuously_with_distance():
    distance = torch.linspace(0.0, 1.5, 301)
    score = landing_target_quality(
        0.70 + distance,
        torch.zeros_like(distance),
        target_x=0.70,
        target_y=0.0,
        half_reward_radius_m=0.50,
    )
    assert torch.all(score[:-1] >= score[1:])
    assert torch.isclose(score[100], torch.tensor(0.5), atol=1.0e-6)


def test_rectangle_outside_penalty_is_zero_inside_and_smooth_outside():
    penalty = landing_rectangle_outside_penalty(
        torch.tensor([0.70, 1.35, 1.40, 1.50, 1.65]),
        torch.tensor([0.00, 0.70, 0.00, 0.00, 0.00]),
        x_range=(0.0, 1.35),
        y_range=(-0.7625, 0.7625),
        half_penalty_distance_m=0.15,
    )
    expected = torch.tensor(
        [
            0.0,
            0.0,
            1.0 - 2.0 ** (-(0.05 / 0.15) ** 2),
            0.5,
            1.0 - 2.0 ** -4.0,
        ]
    )
    torch.testing.assert_close(penalty, expected, atol=1.0e-6, rtol=1.0e-6)


def test_landing_scores_have_finite_zero_gradient_at_center_and_table_edge():
    pred_x = torch.tensor([0.70, 1.35], requires_grad=True)
    pred_y = torch.zeros(2, requires_grad=True)
    score = landing_target_quality(
        pred_x,
        pred_y,
        target_x=0.70,
        target_y=0.0,
        half_reward_radius_m=0.50,
    )
    penalty = landing_rectangle_outside_penalty(
        pred_x,
        pred_y,
        x_range=(0.0, 1.35),
        y_range=(-0.7625, 0.7625),
        half_penalty_distance_m=0.15,
    )
    (score.sum() + penalty.sum()).backward()
    assert torch.isfinite(pred_x.grad).all()
    assert torch.isfinite(pred_y.grad).all()
    assert torch.isclose(pred_x.grad[0], torch.tensor(0.0), atol=1.0e-7)

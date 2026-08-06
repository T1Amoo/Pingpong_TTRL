from __future__ import annotations

import torch

from legged_lab.physics.landing_target import landing_target_score


def test_landing_target_score_has_safe_radius_and_bounded_outside_penalty():
    score = landing_target_score(
        torch.tensor([0.70, 1.20, 1.70, 2.20]),
        torch.zeros(4),
        target_x=0.70,
        target_y=0.0,
        radius_m=0.50,
        outside_floor=-1.0,
    )
    torch.testing.assert_close(
        score,
        torch.tensor([1.0, 0.0, -0.875, -0.99609375]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_landing_target_score_penalizes_lateral_error_symmetrically():
    score = landing_target_score(
        torch.full((2,), 0.70),
        torch.tensor([-0.25, 0.25]),
        target_x=0.70,
        target_y=0.0,
        radius_m=0.50,
    )
    expected = 2.0 * torch.pow(torch.tensor(2.0), torch.tensor(-0.25)) - 1.0
    torch.testing.assert_close(score, expected.repeat(2))


def test_landing_target_score_decreases_continuously_with_distance():
    distance = torch.linspace(0.0, 1.5, 301)
    score = landing_target_score(
        0.70 + distance,
        torch.zeros_like(distance),
        target_x=0.70,
        target_y=0.0,
        radius_m=0.50,
    )
    assert torch.all(score[:-1] >= score[1:])
    assert torch.isclose(score[100], torch.tensor(0.0), atol=1.0e-6)

from __future__ import annotations

import pytest
import torch

from legged_lab.physics.a1_backhand_spin_prior import (
    sample_serve_spin_prior,
    sample_weak_topspin_prior,
)


def test_spin_prior_is_correlated_bounded_and_curriculum_scaled():
    torch.manual_seed(20260806)
    hard = sample_serve_spin_prior(
        20_000,
        device="cpu",
        dtype=torch.float64,
        curriculum=1.0,
        magnitude_jitter=0.0,
    )
    torch.manual_seed(20260806)
    easy = sample_serve_spin_prior(
        20_000,
        device="cpu",
        dtype=torch.float64,
        curriculum=0.0,
        magnitude_jitter=0.0,
    )
    torch.testing.assert_close(easy, 0.5 * hard)
    norm = torch.linalg.norm(hard, dim=1)
    assert 15.0 < float(torch.quantile(norm, 0.50)) < 30.0
    assert 40.0 < float(torch.quantile(norm, 0.95)) < 70.0
    # Full vectors come from an empirical codebook; they are not a Cartesian
    # product of independently sampled per-axis bins.
    assert torch.unique(hard, dim=0).shape[0] <= 32


def test_spin_prior_handles_empty_batch():
    value = sample_serve_spin_prior(
        0,
        device="cpu",
        dtype=torch.float32,
        curriculum=1.0,
    )
    assert value.shape == (0, 3)


def test_weak_topspin_has_correct_sign_and_bounded_small_tilt():
    torch.manual_seed(20260807)
    spin = sample_weak_topspin_prior(
        50_000,
        device="cpu",
        dtype=torch.float64,
        curriculum=1.0,
        easy_range_rad_s=(2.0, 4.0),
        hard_range_rad_s=(2.0, 8.0),
        tilt_deg=10.0,
    )
    norm = torch.linalg.norm(spin, dim=1)
    assert torch.all(spin[:, 0] == 0.0)
    assert torch.all(spin[:, 1] < 0.0)
    assert 4.9 < float(torch.quantile(norm, 0.50)) < 5.1
    assert 7.6 < float(torch.quantile(norm, 0.95)) < 7.8
    assert float(torch.max(torch.abs(spin[:, 2]))) < 1.40


def test_weak_topspin_curriculum_only_widens_upper_magnitude():
    torch.manual_seed(7)
    easy = sample_weak_topspin_prior(
        20_000,
        device="cpu",
        dtype=torch.float32,
        curriculum=0.0,
    )
    easy_norm = torch.linalg.norm(easy, dim=1)
    assert 2.0 <= float(easy_norm.min())
    assert float(easy_norm.max()) <= 4.0


def test_weak_topspin_handles_empty_batch():
    value = sample_weak_topspin_prior(
        0,
        device="cpu",
        dtype=torch.float32,
        curriculum=1.0,
    )
    assert value.shape == (0, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"easy_range_rad_s": (-1.0, 4.0)},
        {"hard_range_rad_s": (8.0, 2.0)},
        {"tilt_deg": 90.0},
        {"tilt_deg": float("nan")},
    ],
)
def test_weak_topspin_rejects_unphysical_configuration(kwargs):
    with pytest.raises(ValueError):
        sample_weak_topspin_prior(
            4,
            device="cpu",
            dtype=torch.float32,
            curriculum=1.0,
            **kwargs,
        )

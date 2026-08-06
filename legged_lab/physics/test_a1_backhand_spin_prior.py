from __future__ import annotations

import torch

from legged_lab.physics.a1_backhand_spin_prior import sample_serve_spin_prior


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

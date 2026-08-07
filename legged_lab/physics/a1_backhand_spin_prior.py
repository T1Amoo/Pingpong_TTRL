"""Compact correlated spin prior for A1 backhand training.

The weighted codebook was fitted (KMeans, seed 20260806) to the 1--99 percent
core of the 945 serve trajectories that pass the A1 workspace filter in
``google-deepmind/competitive_robot_table_tennis``.  Coordinates are already
rotated into the A1 table frame.  Keeping full 3-D vectors avoids the
unphysical combinations produced by independent per-axis uniform sampling.

The source dataset is CC-BY-4.0.  See ``docs/a1_backhand_v4_20260806.md`` for
the filtering contract and attribution.

Source JSON SHA256 prefixes: serves ``258179a5f873d99c`` and rallies
``300f4bf19556dd8e``.  Only the filtered serve subset contributes to this
first-stage contact prior; rally spin is intentionally held for a later
spin-aware flight/target experiment.
"""

from __future__ import annotations

import math

import torch


# (omega_x, omega_y, omega_z, empirical cluster probability)
SERVE_SPIN_CODEBOOK = (
    (-1.9171, +2.0419, -4.8019, 0.101622),
    (+3.2210, +0.0310, +4.6091, 0.083243),
    (+7.0442, -9.3347, -3.1295, 0.062703),
    (+5.9819, +10.2096, +5.8992, 0.064865),
    (-7.9553, -10.4372, -7.8187, 0.041081),
    (-11.4530, +11.0351, -4.0900, 0.061622),
    (-8.3050, +4.5297, -16.7053, 0.060541),
    (+2.0288, -4.2028, +19.1702, 0.032432),
    (-16.7562, -14.4607, +8.5367, 0.018378),
    (+2.5927, -10.7403, -20.9958, 0.061622),
    (-19.3768, -1.2366, -13.8300, 0.031351),
    (+16.1390, +12.5515, -12.8934, 0.009730),
    (-0.3331, +24.3825, +4.7166, 0.024865),
    (+17.2927, +4.8095, +18.0095, 0.033514),
    (+6.0573, -27.4883, -0.6267, 0.034595),
    (-1.4118, -25.5227, -15.4309, 0.032432),
    (-14.9129, -17.0730, -19.8609, 0.024865),
    (+22.5219, -18.8819, -10.5693, 0.029189),
    (-10.4502, -32.9302, +6.1742, 0.007568),
    (+10.7532, -26.7980, +20.6104, 0.022703),
    (-2.4634, -1.3342, -36.6000, 0.017297),
    (-21.9239, +28.8549, -12.8712, 0.010811),
    (+24.2207, +23.5538, +20.4825, 0.014054),
    (+19.0332, -30.8918, -16.7199, 0.028108),
    (-23.2598, +7.7807, -32.1621, 0.019459),
    (+13.8041, -1.3574, +38.4505, 0.019459),
    (-39.7241, +6.8807, -13.2198, 0.016216),
    (-27.0516, +19.7254, +31.8951, 0.007568),
    (+46.2418, +1.3526, -7.1885, 0.008649),
    (-21.2634, -27.4819, -37.9947, 0.008649),
    (+47.7174, +5.2783, +27.7189, 0.007568),
    (-9.3878, -60.4301, -13.3336, 0.003243),
)


def sample_serve_spin_prior(
    count: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    curriculum: float,
    easy_scale: float = 0.50,
    hard_scale: float = 1.00,
    magnitude_jitter: float = 0.15,
) -> torch.Tensor:
    """Sample correlated serve spin and ramp its magnitude with curriculum."""

    if count <= 0:
        return torch.empty((0, 3), device=device, dtype=dtype)
    codebook = torch.tensor(SERVE_SPIN_CODEBOOK, device=device, dtype=dtype)
    weights = torch.clamp(codebook[:, 3], min=0.0)
    indices = torch.multinomial(weights, count, replacement=True)
    spin = codebook[indices, :3]
    c = min(max(float(curriculum), 0.0), 1.0)
    scale = float(easy_scale) + c * (float(hard_scale) - float(easy_scale))
    jitter = abs(float(magnitude_jitter))
    if jitter > 0.0:
        scale_per_sample = torch.empty((count, 1), device=device, dtype=dtype).uniform_(
            max(0.0, 1.0 - jitter), 1.0 + jitter
        )
        spin = spin * scale_per_sample
    return spin * scale


def sample_weak_topspin_prior(
    count: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    curriculum: float,
    easy_range_rad_s: tuple[float, float] = (2.0, 4.0),
    hard_range_rad_s: tuple[float, float] = (2.0, 8.0),
    tilt_deg: float = 10.0,
) -> torch.Tensor:
    """Sample weak topspin for an incoming A1 ball travelling along ``-x``.

    Topspin has negative ``omega_y`` in the A1 table frame.  A small random
    tilt toward ``omega_z`` supplies bounded side-spin robustness without
    reintroducing v4's large mixed side/backspin clusters.  ``omega_x`` stays
    zero because it does not model the user's hand-fed serve mechanism.
    """

    for name, value_range in (
        ("easy_range_rad_s", easy_range_rad_s),
        ("hard_range_rad_s", hard_range_rad_s),
    ):
        if len(value_range) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        lo_value, hi_value = map(float, value_range)
        if not (math.isfinite(lo_value) and math.isfinite(hi_value)):
            raise ValueError(f"{name} must be finite")
        if lo_value < 0.0 or hi_value < lo_value:
            raise ValueError(f"{name} must satisfy 0 <= low <= high")
    if not math.isfinite(float(tilt_deg)) or not 0.0 <= float(tilt_deg) < 90.0:
        raise ValueError("tilt_deg must satisfy 0 <= tilt_deg < 90")

    if count <= 0:
        return torch.empty((0, 3), device=device, dtype=dtype)

    c = min(max(float(curriculum), 0.0), 1.0)
    magnitude_lo = float(easy_range_rad_s[0]) + c * (
        float(hard_range_rad_s[0]) - float(easy_range_rad_s[0])
    )
    magnitude_hi = float(easy_range_rad_s[1]) + c * (
        float(hard_range_rad_s[1]) - float(easy_range_rad_s[1])
    )
    magnitude = torch.empty((count,), device=device, dtype=dtype).uniform_(
        magnitude_lo,
        magnitude_hi,
    )

    max_tilt_rad = math.radians(abs(float(tilt_deg)))
    tilt = torch.empty((count,), device=device, dtype=dtype).uniform_(
        -max_tilt_rad,
        max_tilt_rad,
    )
    spin = torch.zeros((count, 3), device=device, dtype=dtype)
    spin[:, 1] = -magnitude * torch.cos(tilt)
    spin[:, 2] = magnitude * torch.sin(tilt)
    return spin

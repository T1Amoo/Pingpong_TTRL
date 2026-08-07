"""Small, simulator-independent gates for table-tennis swing timing rewards."""

from __future__ import annotations

import torch


def linear_curriculum_progress(
    raw_step: int,
    *,
    start_raw_step: int,
    ramp_raw_steps: int,
) -> float:
    """Return a clamped scalar curriculum progress in ``[0, 1]``.

    ``TT_SIM_STEP_OFFSET`` seeds the environment's raw-step clock on resume,
    so this helper is safe for watchdog restarts.  A non-positive ramp selects
    the final value, which is useful for deterministic eval configs.
    """

    if ramp_raw_steps <= 0:
        return 1.0
    return min(
        max((int(raw_step) - int(start_raw_step)) / float(ramp_raw_steps), 0.0),
        1.0,
    )


def curriculum_lerp(
    start_value: float,
    final_value: float,
    progress: float,
) -> float:
    """Linearly interpolate a scalar after clamping curriculum progress."""

    value = min(max(float(progress), 0.0), 1.0)
    return float(start_value) + value * (float(final_value) - float(start_value))


def phase_open_gate(
    time_to_hit_s: torch.Tensor,
    *,
    open_time_s: float,
    transition_s: float,
) -> torch.Tensor:
    """Open a late-swing reward phase at a remaining-time boundary.

    The gate is zero while ``time_to_hit_s >= open_time_s`` and reaches one
    once ``transition_s`` more seconds have elapsed.  Smoothstep gives a short
    differentiable hand-off without continuously rescaling the x error during
    the entire incoming flight.
    """

    if transition_s <= 0.0:
        return (time_to_hit_s < float(open_time_s)).to(time_to_hit_s.dtype)
    progress = torch.clamp(
        (float(open_time_s) - time_to_hit_s) / float(transition_s),
        min=0.0,
        max=1.0,
    )
    return progress * progress * (3.0 - 2.0 * progress)


def staged_intercept_reward(
    yz_distance: torch.Tensor,
    x_progress: torch.Tensor,
    x_gate: torch.Tensor,
    *,
    std_ee: float,
    threshold: float,
) -> torch.Tensor:
    """Positive-only y/z guidance plus a gated forward-progress bonus.

    The x bonus is conditioned on y/z quality and starts from zero at its
    configurable shaping boundary. Opening the phase can therefore never make
    an otherwise identical state lose reward; the squared ramp leaves only a
    negligible bonus at the measured ready pose and grows toward the hit plane.
    """

    denominator = float(std_ee) * float(std_ee) + 1.0e-12
    yz_quality = torch.exp(-torch.clamp(yz_distance, min=float(threshold)) / denominator)
    gated_progress = (
        torch.clamp(x_gate, min=0.0, max=1.0)
        * torch.clamp(x_progress, min=0.0, max=1.0)
    )
    return yz_quality * (1.0 + gated_progress)


def weighted_yz_distance(
    diff_xyz: torch.Tensor,
    *,
    z_weight: float,
) -> torch.Tensor:
    """Return a zero-gradient-safe weighted y/z intercept distance."""

    weighted_z = diff_xyz[..., 2] * float(z_weight)
    yz_components = torch.stack((diff_xyz[..., 1], weighted_z), dim=-1)
    return torch.linalg.vector_norm(yz_components, dim=-1)


def bounded_forward_speed_quality(
    forward_speed_mps: torch.Tensor,
    *,
    min_speed_mps: float,
    full_speed_mps: float,
) -> torch.Tensor:
    """Return a smooth, bounded quality for a forward-speed target.

    Quality is zero through ``min_speed_mps``, rises to one at
    ``full_speed_mps``, then stays saturated.  Smoothstep has zero slope at both
    ends, which avoids making a one-shot contact reward numerically sensitive
    to a few millimetres per second of velocity noise and never rewards speeds
    above the configured target.
    """

    minimum = float(min_speed_mps)
    full = float(full_speed_mps)
    if minimum < 0.0:
        raise ValueError("min_speed_mps must be non-negative")
    if full <= minimum:
        raise ValueError("full_speed_mps must be > min_speed_mps")

    progress = torch.clamp(
        (forward_speed_mps - minimum) / (full - minimum),
        min=0.0,
        max=1.0,
    )
    return progress * progress * (3.0 - 2.0 * progress)


def max_drawdown_excess_penalty(
    max_drawdown_m: torch.Tensor,
    *,
    free_drawdown_m: float,
    full_penalty_drawdown_m: float,
) -> torch.Tensor:
    """Return a squared unit penalty for pre-contact drawdown above a free band."""

    free = float(free_drawdown_m)
    full = float(full_penalty_drawdown_m)
    if free < 0.0:
        raise ValueError("free_drawdown_m must be non-negative")
    if full <= free:
        raise ValueError("full_penalty_drawdown_m must be > free_drawdown_m")
    scaled = torch.clamp((max_drawdown_m - free) / (full - free), min=0.0, max=1.0)
    return torch.square(scaled)


def latch_precontact_max_drawdown(
    max_drawdown_m: torch.Tensor,
    armed: torch.Tensor,
) -> torch.Tensor:
    """Latch drawdown at contact, mapping an unarmed trajectory to full penalty.

    An unarmed contact means the paddle never returned behind the configured
    arming boundary during this ball.  Treating that case as zero drawdown would
    let a parked-forward strategy bypass the path constraint, so it is encoded
    as infinity and subsequently saturates the bounded drawdown penalty.
    """

    return torch.where(
        armed.bool(),
        max_drawdown_m,
        torch.full_like(max_drawdown_m, float("inf")),
    )


def update_precontact_max_drawdown(
    current_x_m: torch.Tensor,
    peak_x_m: torch.Tensor,
    max_drawdown_m: torch.Tensor,
    armed: torch.Tensor,
    tracking_mask: torch.Tensor,
    arm_allowed_mask: torch.Tensor,
    *,
    arm_x_max_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance a per-ball maximum-drawdown state without crossing resets.

    Tracking arms only after the paddle reaches the configured retracted side
    of the hit plane.  This prevents the previous ball's forward finish and the
    following normal return-to-ready motion from being charged to the new ball.
    Once armed, the maximum records the worst ``peak_x - current_x`` even if the
    paddle has recovered all of that drawdown by the eventual contact tick.
    """

    tracking = tracking_mask.bool()
    arm_allowed = arm_allowed_mask.bool()
    armed_now = armed.bool()
    newly_armed = (
        tracking
        & arm_allowed
        & (~armed_now)
        & (current_x_m <= float(arm_x_max_m))
    )
    next_armed = armed_now | newly_armed
    seeded_peak = torch.where(newly_armed, current_x_m, peak_x_m)
    active = tracking & next_armed
    next_peak = torch.where(active, torch.maximum(seeded_peak, current_x_m), seeded_peak)
    current_drawdown = torch.clamp(next_peak - current_x_m, min=0.0)
    next_max_drawdown = torch.where(
        active,
        torch.maximum(max_drawdown_m, current_drawdown),
        max_drawdown_m,
    )
    return next_peak, next_max_drawdown, next_armed


def x_position_progress(
    x_error_m: torch.Tensor,
    *,
    zero_reward_error_m: float,
    full_reward_error_m: float,
) -> torch.Tensor:
    """Return squared forward progress from a retracted boundary to target."""

    zero_error = float(zero_reward_error_m)
    full_error = float(full_reward_error_m)
    if not 0.0 <= full_error < zero_error:
        raise ValueError(
            "x progress requires 0 <= full_reward_error_m < zero_reward_error_m"
        )
    progress = torch.clamp(
        (zero_error - torch.abs(x_error_m)) / (zero_error - full_error),
        min=0.0,
        max=1.0,
    )
    return torch.square(progress)


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

"""Single source of truth for the real-serve-aligned A1 backhand-v3 contract."""

from __future__ import annotations

from legged_lab.physics.a1_backhand_v2_contract import (
    BALL_DRAG_COEFF,
    BALL_MASS_KG,
    BALL_RADIUS,
    DRAG_ACCEL_K,
    NET_CENTER_Z_MIN,
    NET_PREDICTION_MARGIN,
    NET_TOP_Z,
    PHYSICAL_BOUNCE_X_RANGE,
    TABLE_DYNAMIC_FRICTION,
    TABLE_RESTITUTION,
)


# 2026-08-04 raw-camera, gravity-aware hit-plane fit (x=-1.243 m), 142
# observed crossings.  Robust physical subset (n=133):
#   y p05/p50/p95 = -0.183 / 0.063 / 0.217 m
#   z p05/p50/p95 =  0.907 / 1.097 / 1.166 m
# |vx| p05/p50/p95 = 1.197 / 2.352 / 3.260 m/s
# The hard envelope keeps measured-tail margin without making high/slow balls
# the dominant mode.  The easy distribution is centered on the empirical
# median speed/height rather than v2's faster bootstrap.
REAL_SAMPLE_COUNT = 142
REAL_HIT_Y_P05_P50_P95 = (-0.183, 0.063, 0.217)
REAL_HIT_Z_P05_P50_P95 = (0.907, 1.097, 1.166)
REAL_HIT_ABS_VX_P05_P50_P95 = (1.197, 2.352, 3.260)

EASY_BOUNCE_X_RANGE = (-1.10, -0.50)
EASY_BOUNCE_VZ_RANGE = (1.50, 3.00)
EASY_Y_CENTER = 0.041
EASY_Y_HALF = 0.040

HARD_BOUNCE_X_RANGE = (-1.10, -0.08)
HARD_BOUNCE_VZ_RANGE = (1.20, 3.60)
HARD_Y_CENTER = 0.020
HARD_Y_HALF = 0.180

HIT_TARGET_Y_RANGE = (-0.22, 0.26)
HIT_TARGET_Z_RANGE = (0.84, 1.32)
# Preflight-to-Isaac contact can differ by about 4--5 cm at the lowest edge.
PREFLIGHT_HIT_Z_RANGE = (0.88, 1.32)
HIT_ARRIVAL_ABS_VX_RANGE = (1.00, 5.80)

CURRICULUM_EASY_ITERS = 5_000
CURRICULUM_RAMP_ITERS = 10_000
CURRICULUM_HOLD_ITERS = 5_000
MAX_ITERATIONS = CURRICULUM_EASY_ITERS + CURRICULUM_RAMP_ITERS + CURRICULUM_HOLD_ITERS

# Independent per-joint response DR.  r4 is deliberately broader because the
# natural-swing traces show 2--5 Hz gain/phase and near-8 Nm behavior outside
# the old globally-correlated linear DR family.  Well-aligned r7 stays narrow.
RESPONSE_FN_SCALE_RANGES = (
    (0.70, 1.20),
    (0.85, 1.15),
    (0.85, 1.15),
    (0.75, 1.25),
    (0.90, 1.10),
    (0.90, 1.10),
    (0.95, 1.05),
)
RESPONSE_ZETA_SCALE_RANGES = (
    (0.55, 1.25),
    (0.80, 1.25),
    (0.80, 1.25),
    (0.45, 1.35),
    (0.85, 1.15),
    (0.85, 1.15),
    (0.90, 1.10),
)
RESPONSE_GAIN_SCALE_RANGES = (
    (0.95, 1.08),
    (0.98, 1.02),
    (0.98, 1.02),
    (0.92, 1.12),
    (0.98, 1.02),
    (0.98, 1.02),
    (0.99, 1.01),
)
RESPONSE_DELAY_S = (0.010, 0.040, 0.040, 0.040, 0.040, 0.030, 0.040)
RESPONSE_DELAY_JITTER_S = (0.005, 0.010, 0.010, 0.015, 0.008, 0.008, 0.006)

# Output-space closed-loop acceleration envelope derived from 90 ms Savitzky-
# Golay estimates of the natural policy traces.  These bounds mainly prevent
# broad r1/r4 DR draws from creating unrealistically unlimited response; they
# are not a deployment q_des slew limiter.
RESPONSE_ACCEL_LIMIT_RAD_S2 = (60.0, 60.0, 95.0, 90.0, 160.0, 175.0, 240.0)
RESPONSE_ACCEL_LIMIT_SCALE_RANGES = (
    (0.85, 1.20),
    (0.90, 1.15),
    (0.90, 1.15),
    (0.75, 1.25),
    (0.90, 1.15),
    (0.90, 1.15),
    (0.95, 1.10),
)

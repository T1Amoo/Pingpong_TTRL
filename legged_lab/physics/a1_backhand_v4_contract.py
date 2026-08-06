"""A1 backhand-v4 swing-timing and safe-return reward contract.

The serve, camera, contact, and actuator contracts remain inherited from v3.
Only reward semantics change, so no unverified physical-domain parameter is
silently mixed into the run.
"""

from __future__ import annotations

from legged_lab.physics.a1_backhand_v3_contract import (
    BALL_DRAG_COEFF,
    BALL_MASS_KG,
    BALL_RADIUS,
    DRAG_ACCEL_K,
    HIT_ARRIVAL_ABS_VX_RANGE,
    HIT_TARGET_Y_RANGE,
    HIT_TARGET_Z_RANGE,
    MAX_ITERATIONS,
    NET_CENTER_Z_MIN,
    NET_PREDICTION_MARGIN,
    NET_TOP_Z,
    PHYSICAL_BOUNCE_X_RANGE,
    PREFLIGHT_HIT_Z_RANGE,
    REAL_HIT_ABS_VX_P05_P50_P95,
    REAL_HIT_Y_P05_P50_P95,
    REAL_HIT_Z_P05_P50_P95,
    TABLE_DYNAMIC_FRICTION,
    TABLE_RESTITUTION,
)


# A single wide uniform box made high/slow or high/fast tails visually
# dominant. The two-component hard proposal below is calibrated against 142
# real hit-plane crossings. Rejection turns the 55/45 candidate mix into
# about 89/11 accepted main/slow-tail samples in the analytic preflight.
EASY_BOUNCE_X_RANGE = (-0.80, -0.45)
EASY_BOUNCE_VZ_RANGE = (1.60, 2.60)
EASY_Y_CENTER = 0.050
EASY_Y_HALF = 0.050

HARD_BOUNCE_X_RANGE = (-0.95, -0.75)
HARD_BOUNCE_VZ_RANGE = (1.20, 2.30)
HARD_Y_CENTER = 0.020
HARD_Y_HALF = 0.180

TAIL_CANDIDATE_WEIGHT_HARD = 0.45
TAIL_BOUNCE_X_RANGE_HARD = (-0.75, -0.08)
TAIL_BOUNCE_VZ_RANGE_HARD = (3.00, 3.80)

# CPU preflight regression targets for the final hard proposal. These are not
# physical-arrival targets: compliant PhysX contact produces a lower speed
# median. A 500-crossing Isaac probe measured y=-.164/.022/.217,
# z=1.030/1.086/1.176 and |vx|=1.546/2.286/3.268 (p5/p50/p95), close to the
# real 142-crossing medians/tails. The CPU verifier protects sampler drift;
# the Isaac TT_SERVE_PROBE is the authoritative physical check.
CPU_HARD_HIT_Z_P50 = 1.073
CPU_HARD_HIT_Z_P95 = 1.160
CPU_HARD_ABS_VX_P50 = 3.010
CPU_HARD_ABS_VX_P95 = 3.903
CPU_HARD_ACCEPTED_TAIL = 0.105


# Keep x reach disabled until the last 450 ms, while y/z tracking remains
# active for wide/high balls.  The early hold fades over the preceding 150 ms.
X_TRACKING_WINDOW_S = 0.45
X_TRACKING_FLOOR = 0.10
EARLY_HOLD_RELEASE_S = 0.45
EARLY_HOLD_RAMP_S = 0.15

# Ready paddle x is about -1.44 m and the hit plane is -1.243 m.  Allow about
# 8 cm of harmless preparation but keep at least 12 cm of forward stroke.
EARLY_MIN_RETRACTION_M = 0.12
EARLY_MAX_EXCESS_M = 0.18
EARLY_FORWARD_PENALTY_WEIGHT = -2.0

# Once the swing window opens, discourage only negative-x blade velocity.
# The bounded unit-scale term is tiny relative to contact/pass/table outcomes.
LATE_BACKTRACK_WINDOW_S = 0.45
LATE_BACKTRACK_SPEED_SCALE_MPS = 1.0
LATE_BACKTRACK_PENALTY_WEIGHT = -1.0

# v2 real contacts frequently left sideways/high even though the legacy
# landing reward stayed strongly positive.  Center the return on the opponent
# half (x spans 0..1.37 m), make the requested 50 cm safe radius meaningful,
# and stop paying a large positive score for out-of-bounds predictions.
LANDING_TARGET_X = 0.70
LANDING_TARGET_Y = 0.0
LANDING_TARGET_RADIUS_M = 0.50
LANDING_OUTSIDE_FLOOR = -1.0
LANDING_REWARD_WEIGHT = 100.0
PASS_NET_HEIGHT_STD_M = 0.20

# The real v2 paddle swept laterally at about -0.83 m/s at contact while the
# same policy in MuJoCo was about -0.38 m/s.  Penalize either lateral direction
# only on first contact; y pre-positioning before contact remains unrestricted.
CONTACT_LATERAL_SPEED_DEADBAND_MPS = 0.30
CONTACT_LATERAL_SPEED_RAMP_MPS = 0.70
CONTACT_LATERAL_SPEED_PENALTY_WEIGHT = -20.0

# The source trajectory prior is spin-aware, but the current reset preflight
# and fixed-plane target are not.  Inject a correlated real-serve spin only
# after the own-table bounce and keep Magnus disabled for this version.  This
# isolates paddle-contact robustness without corrupting target geometry.
POST_BOUNCE_SPIN_EASY_SCALE = 0.50
POST_BOUNCE_SPIN_HARD_SCALE = 1.00
POST_BOUNCE_SPIN_MAGNITUDE_JITTER = 0.15

# Geometric proximity can lead physical collision by one or two control ticks.
# Wait for a positive-x outgoing state, with a short timeout that still emits a
# negative outcome for a glancing/non-returning contact.
POST_IMPACT_MIN_OUTGOING_VX_MPS = 0.05
POST_IMPACT_TIMEOUT_S = 0.06

# First-contact paddle-normal alignment was capped at only 8 reward points in
# v2/v3, while the measured blade normal remained almost vertical.  This term
# cannot be farmed between contacts, so a moderate increase is low risk.
HIT_DIRECTION_REWARD_WEIGHT = 20.0

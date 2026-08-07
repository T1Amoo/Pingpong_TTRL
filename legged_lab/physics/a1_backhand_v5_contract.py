"""A1 backhand-v5 positive outcome ladder and weak-topspin contract.

V5 inherits the complete v4 serve, camera, actuator, contact-material and
post-impact contracts.  It changes only reward semantics, swing-phase timing,
and the post-bounce spin prior.  In particular, the translational serve
distribution and Magnus coefficient remain unchanged.
"""

from __future__ import annotations

from legged_lab.physics.a1_backhand_v4_contract import MAX_ITERATIONS


# At 50 Hz, one PPO iteration contains 24 control steps and every control step
# contains ten 500 Hz physics substeps.  TTEnv.sim_step_counter uses the latter.
RAW_STEPS_PER_ITER = 240

# Let a scratch policy discover contact with a generous x phase first, then
# tighten to the final timing before the easy-serve stage ends at iter 5000.
TIMING_CURRICULUM_START_ITER = 1_000
TIMING_CURRICULUM_END_ITER = 4_000
TIMING_CURRICULUM_START_RAW_STEP = (
    TIMING_CURRICULUM_START_ITER * RAW_STEPS_PER_ITER
)
TIMING_CURRICULUM_RAMP_RAW_STEPS = (
    (TIMING_CURRICULUM_END_ITER - TIMING_CURRICULUM_START_ITER)
    * RAW_STEPS_PER_ITER
)

# X is a true late phase: y/z guidance is always positive, while forward
# progress inside the retracted boundary opens only when stable reset-time
# t_hit crosses this boundary. The 40 ms smoothstep is two policy ticks.
X_PHASE_OPEN_START_S = 0.80
X_PHASE_OPEN_FINAL_S = 0.60
X_PHASE_TRANSITION_S = 0.04

# Before the x phase, permit roughly 11.7 cm of preparation from the measured
# ready blade x ~= -1.44 m, but retain about 8 cm of final forward stroke to
# hit_plane_x=-1.243 m.  This is the only v5 paddle-motion penalty.
EARLY_HOLD_RAMP_S = 0.10
EARLY_MIN_RETRACTION_M = 0.08
EARLY_MAX_EXCESS_M = 0.20
EARLY_FORWARD_PENALTY_WEIGHT = -0.50
# Start the gated progress just outside the measured ready error (~0.197 m).
# Using the 8 cm early-hold boundary here would leave an 11.7 cm zero-gradient
# gap and recreate the wait-at-ready optimum after the phase opens.
X_PROGRESS_ZERO_REWARD_ERROR_M = 0.22
X_PROGRESS_FULL_REWARD_ERROR_M = 0.02

# Positive outcome ladder.  Isaac's RewardManager multiplies these raw weights
# by dt=0.02 s. Thus every valid contact receives a guaranteed +5 event; a
# physical table bounce adds +11; and the actual center adds up to another +8.
CONTACT_EVENT_REWARD_WEIGHT = 250.0
CONTACT_QUALITY_REWARD_WEIGHT = 50.0
SWEET_CONTACT_REWARD_WEIGHT = 50.0
APPROACH_VELOCITY_REWARD_WEIGHT = 3.0
APPROACH_VELOCITY_CAP_MPS = 3.0
HIT_DIRECTION_REWARD_WEIGHT = 8.0
PREDICTED_LANDING_REWARD_WEIGHT = 200.0
PASS_NET_REWARD_WEIGHT = 180.0
TABLE_SUCCESS_REWARD_WEIGHT = 550.0
ACTUAL_TABLE_CENTER_REWARD_WEIGHT = 400.0

LANDING_TARGET_X = 0.70
LANDING_TARGET_Y = 0.0
LANDING_TARGET_HALF_REWARD_RADIUS_M = 0.50

# Explicitly disabled v4 constraints. Generic actuator/action regularizers,
# the anti-forearm-contact safety term, and v3's relative high-paddle guard
# remain inherited; these are not the new v4 swing trajectory prescriptions.
LATE_BACKTRACK_PENALTY_WEIGHT = 0.0
CONTACT_LATERAL_SPEED_PENALTY_WEIGHT = 0.0
LANDING_OUTSIDE_PENALTY_WEIGHT = 0.0
PADDLE_ABOVE_TARGET_PENALTY_WEIGHT = -4.0

# The current A1 incoming direction is -x, so topspin is negative omega_y.
# This is deliberately much weaker than v4's empirical mixed-spin codebook
# (roughly 21 rad/s median and 44 rad/s P95 at full scale).
WEAK_TOPSPIN_EASY_RANGE_RAD_S = (2.0, 4.0)
WEAK_TOPSPIN_HARD_RANGE_RAD_S = (2.0, 8.0)
WEAK_TOPSPIN_TILT_DEG = 10.0

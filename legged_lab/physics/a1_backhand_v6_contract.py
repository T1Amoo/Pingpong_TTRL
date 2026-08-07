"""A1 backhand-v6 contact-speed and pre-contact drawdown contract.

V6 inherits the complete v5 serve, camera, actuator, contact-material,
post-impact outcome, landing-ladder, and weak-topspin contracts.  It changes
only pre-contact reward semantics: intercept shaping is y/z-only, forward
contact speed receives a bounded quality reward, and losing already-achieved
forward paddle progress receives a bounded drawdown penalty.
"""

from __future__ import annotations

from legged_lab.physics import a1_backhand_v5_contract as v5


MAX_ITERATIONS = v5.MAX_ITERATIONS

# RewardManager multiplies configured weights by the 50 Hz control dt=0.02 s.
# The normalized contact-speed quality is zero at/below 0.4 m/s and one at/above
# 1.6 m/s, so its one-shot contribution is bounded in [0, +2].
FORWARD_SPEED_QUALITY_REWARD_WEIGHT = 100.0
FORWARD_SPEED_QUALITY_MIN_MPS = 0.40
FORWARD_SPEED_QUALITY_FULL_MPS = 1.60

# Track the maximum forward paddle x reached during the current incoming ball.
# A retreat of <=4 cm is free; >=12 cm receives the full one-shot -0.8 cost.
PRECONTACT_DRAWDOWN_FREE_M = 0.04
PRECONTACT_DRAWDOWN_FULL_M = 0.12
PRECONTACT_DRAWDOWN_PENALTY_WEIGHT = -40.0

# Keep early target guidance only in y/z.  X motion is discovered through the
# contact-speed outcome and the bounded swing-through term, not a prescribed
# position path.
FUTURE_YZ_REWARD_WEIGHT = 0.50
FUTURE_YZ_STD_EE = 0.50
FUTURE_YZ_THRESHOLD_M = 0.08
FUTURE_YZ_Z_WEIGHT = 2.50

# Dense forward speed remains available only near the predicted y/z target and
# is capped so it cannot dominate the sparse contact/table outcome ladder.
SWING_THROUGH_REWARD_WEIGHT = 0.75
SWING_THROUGH_CAP_MPS = 1.60
SWING_THROUGH_NEAR_DIST_M = 0.30
SWING_THROUGH_TARGET_YZ_GATE_M = 0.18

# V6 adds no hand-authored paddle-route or predicted-miss penalties.
EARLY_FORWARD_PENALTY_WEIGHT = 0.0
LATE_BACKTRACK_PENALTY_WEIGHT = 0.0
CONTACT_LATERAL_SPEED_PENALTY_WEIGHT = 0.0
LANDING_OUTSIDE_PENALTY_WEIGHT = 0.0

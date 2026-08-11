"""A1 backhand-v8 reward-side timing and wrist-stability contract.

V8 inherits v7 geometry, V2 fixed-sj asset, serve distribution, camera model,
actuator response and collision safety unchanged.  It keeps the 39 x 5 = 195
actor observation interface.  ``ball_future_t`` is used only by rewards to
remove the long-ETA early-forward/drawdown exploit observed in v7 model_9200.
"""

from __future__ import annotations

from legged_lab.physics import a1_backhand_v7_contract as v7


MAX_ITERATIONS = v7.MAX_ITERATIONS

# Real v7 trial: gate-time ETA p50 ~=0.356 s and p95 ~=0.630 s, while actual
# paddle onset is about 0.10 s.  A 0.60 s boundary preserves immediate motion
# for genuinely urgent balls but makes long-ETA balls wait.  The 0.10 s fade
# avoids a hard reward discontinuity around the boundary.
SWING_WINDOW_S = 0.60
SWING_TRANSITION_S = 0.10

# Before the swing window, allow the ready blade to prepare by about 12 cm but
# retain the final 8 cm stroke into the translated v7 hit plane.  This is a
# deliberately small cost compared with contact/return outcomes.
EARLY_MIN_RETRACTION_M = 0.08
EARLY_MAX_EXCESS_M = 0.20
EARLY_FORWARD_PENALTY_WEIGHT = -0.50

# The real and MuJoCo traces both show actor-target 3--5 Hz content on r5/r7.
# Penalize only those two raw action-rate channels.  Inside the swing window
# retain 25% strength so necessary wrist motion remains available.
WRIST_ACTION_RATE_JOINT_WEIGHTS = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
WRIST_ACTION_RATE_SWING_FLOOR = 0.25
WRIST_ACTION_RATE_PENALTY_WEIGHT = -0.08

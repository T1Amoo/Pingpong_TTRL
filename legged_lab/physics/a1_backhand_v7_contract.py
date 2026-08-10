"""A1 backhand-v7 ready-pose-relative strike and serve contract.

V7 keeps the complete v6 policy/reward/actuator contract, but the real ready
pose and the fixed r1 height move the nominal paddle center.  The strike plane
and arrival window therefore move by the same FK displacement, preserving the
v6 paddle-relative geometry instead of preserving stale world coordinates.

The table and net remain fixed in world coordinates.  The bounce proposal and
launch pose below were calibrated with the repository RK4/Coulomb serve probe
so the translated easy/mid/hard arrival y/z, speed, and time quantiles remain
close to v6 after the physical table bounce.
"""

from __future__ import annotations

from legged_lab.physics import a1_backhand_v3_contract as v3


# Nominal Link_r_paddle FK in world coordinates.  Both poses use base
# (-1.8, 0.0, 0.0282); v6 uses the V1_3 asset and v7 uses the 1.08 m V2 asset.
V6_READY_PADDLE_WORLD_M = (
    -1.442117529908,
    0.041044966312,
    0.947251525625,
)
V7_READY_PADDLE_WORLD_M = (
    -1.534800068603,
    0.133844799241,
    1.094850772611,
)
READY_PADDLE_TRANSLATION_M = tuple(
    new - old for old, new in zip(V6_READY_PADDLE_WORLD_M, V7_READY_PADDLE_WORLD_M)
)


def _translate_pair(values: tuple[float, float], offset: float) -> tuple[float, float]:
    return (values[0] + offset, values[1] + offset)


# Preserve the complete v6 paddle-relative strike box.
V6_HIT_PLANE_X = -1.243
HIT_PLANE_X = V6_HIT_PLANE_X + READY_PADDLE_TRANSLATION_M[0]
HIT_TARGET_Y_RANGE = _translate_pair(v3.HIT_TARGET_Y_RANGE, READY_PADDLE_TRANSLATION_M[1])
HIT_TARGET_Z_RANGE = _translate_pair(v3.HIT_TARGET_Z_RANGE, READY_PADDLE_TRANSLATION_M[2])
PREFLIGHT_HIT_Z_RANGE = _translate_pair(
    v3.PREFLIGHT_HIT_Z_RANGE,
    READY_PADDLE_TRANSLATION_M[2],
)

# Keep the v6 relative offsets explicit and auditable.
V6_PLANE_FORWARD_OFFSET_M = V6_HIT_PLANE_X - V6_READY_PADDLE_WORLD_M[0]
V7_PLANE_FORWARD_OFFSET_M = HIT_PLANE_X - V7_READY_PADDLE_WORLD_M[0]
V6_HIT_Y_OFFSETS_M = tuple(value - V6_READY_PADDLE_WORLD_M[1] for value in v3.HIT_TARGET_Y_RANGE)
V7_HIT_Y_OFFSETS_M = tuple(value - V7_READY_PADDLE_WORLD_M[1] for value in HIT_TARGET_Y_RANGE)
V6_HIT_Z_OFFSETS_M = tuple(value - V6_READY_PADDLE_WORLD_M[2] for value in v3.HIT_TARGET_Z_RANGE)
V7_HIT_Z_OFFSETS_M = tuple(value - V7_READY_PADDLE_WORLD_M[2] for value in HIT_TARGET_Z_RANGE)

# Fixed-world table/net compensation.  These are proposal parameters, not a
# second target box: accepted serves are still rejected against the translated
# arrival ranges above.  The higher launch and lower vertical command recover
# the +14.76 cm post-bounce arrival shift without moving the table.
BALL_LAUNCH_POS = (1.55, READY_PADDLE_TRANSLATION_M[1], 1.33)
EASY_BOUNCE_X_RANGE = (-0.80, -0.45)
EASY_BOUNCE_VZ_RANGE = (1.10, 2.10)
EASY_Y_CENTER = 0.050 + READY_PADDLE_TRANSLATION_M[1]
EASY_Y_HALF = 0.050

HARD_BOUNCE_X_RANGE = (-1.03, -0.83)
HARD_BOUNCE_VZ_RANGE = (0.85, 1.95)
HARD_Y_CENTER = 0.020 + READY_PADDLE_TRANSLATION_M[1]
HARD_Y_HALF = 0.180

TAIL_CANDIDATE_WEIGHT_HARD = 0.36
TAIL_BOUNCE_X_RANGE_HARD = (-0.83, -0.16)
TAIL_BOUNCE_VZ_RANGE_HARD = (2.65, 3.45)

FALLBACK_BOUNCE_X = -0.70
FALLBACK_BOUNCE_Y = EASY_Y_CENTER
FALLBACK_BOUNCE_VZ = 1.60

# The curriculum duration and arrival-speed envelope intentionally remain v6.
CURRICULUM_EASY_ITERS = v3.CURRICULUM_EASY_ITERS
CURRICULUM_RAMP_ITERS = v3.CURRICULUM_RAMP_ITERS
CURRICULUM_HOLD_ITERS = v3.CURRICULUM_HOLD_ITERS
MAX_ITERATIONS = v3.MAX_ITERATIONS
HIT_ARRIVAL_ABS_VX_RANGE = v3.HIT_ARRIVAL_ABS_VX_RANGE

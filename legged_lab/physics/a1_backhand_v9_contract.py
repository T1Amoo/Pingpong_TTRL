"""A1 backhand-v9 R4 response and MIT torque-projection contract.

V9 inherits v8 geometry, serve/camera/contact physics, reward timing and the
195-D actor interface unchanged.  It promotes only the independently held-out
R4 low-load response measured at the v7/v8 ready pose and replaces the old
90 rad/s^2 response-acceleration proxy with an 8 Nm MIT torque-observer command
projection on R4.  The response remains in :class:`TTEnv`; the high-bandwidth
implicit drive remains an ideal response-target tracker, so a second motor
dynamic is not introduced.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from legged_lab.physics import a1_backhand_v8_contract as v8


MAX_ITERATIONS = v8.MAX_ITERATIONS

R4_INDEX = 3
R4_RESPONSE_FN_HZ = 4.6540465907137785
R4_RESPONSE_ZETA = 0.1218426552876445
R4_RESPONSE_DELAY_S = 0.02893988654476473
R4_RESPONSE_GAIN = 1.0
R4_RESPONSE_BIAS_RAD = 0.009078698834346307

# DM4310 peak effort.  The observer projects a delayed command only when its
# identified MIT demand exceeds this hard boundary.
R4_EFFORT_LIMIT_NM = 8.0

# Robot-clock torque observer validated on the independent low-load holdout.
R4_TORQUE_KP = 120.0
R4_TORQUE_KD = 1.0
R4_TORQUE_SCALE = 0.9932751315748902
R4_TORQUE_OFFSET_NM = -0.08238003677908433
R4_TORQUE_OBSERVER_RMSE_NM = 0.046254692002191095

# V3--v8 used 90 rad/s^2 as a torque-envelope proxy.  V9 replaces that proxy
# on R4 with the torque projection above; retaining both would double-limit
# the same nonlinearity.  Other joints keep their inherited acceleration caps.
R4_RESPONSE_ACCEL_LIMIT_RAD_S2 = float("inf")

# Independent robot-clock holdout evidence (2026-08-11).  These values are
# documentation/contract gates, not optimization inputs at runtime.
HOLDOUT_Q_RMSE_RAD = 0.000826645
OLD_RECENTERED_Q_RMSE_RAD = 0.002348791
HOLDOUT_IMPROVEMENT_FRACTION = 0.6481
HOLDOUT_TORQUE_PEAK_NM = 1.809523582
SATURATION_HOLDOUT_TORQUE_PEAK_NM = 8.007326126098633


_T = TypeVar("_T")


def replace_r4(values: Sequence[_T], value: _T) -> tuple[_T, ...]:
    """Return a seven-joint tuple with only R4 replaced."""

    result = tuple(values)
    if len(result) != 7:
        raise ValueError(f"expected seven right-arm values, got {len(result)}")
    return result[:R4_INDEX] + (value,) + result[R4_INDEX + 1 :]


def project_r4_command(
    command: float, position: float, velocity: float
) -> tuple[float, float, float]:
    """Project one R4 command to the fitted MIT 8 Nm observer boundary.

    Returns ``(projected_command, raw_torque, projected_torque)``.  Runtime
    applies the vectorized equivalent in ``TTEnv``.
    """

    raw_torque = R4_TORQUE_SCALE * (
        R4_TORQUE_KP * (command - position) - R4_TORQUE_KD * velocity
    ) + R4_TORQUE_OFFSET_NM
    projected_torque = max(
        -R4_EFFORT_LIMIT_NM,
        min(R4_EFFORT_LIMIT_NM, raw_torque),
    )
    if projected_torque == raw_torque:
        return command, raw_torque, projected_torque
    projected_command = position + (
        (projected_torque - R4_TORQUE_OFFSET_NM) / R4_TORQUE_SCALE
        + R4_TORQUE_KD * velocity
    ) / R4_TORQUE_KP
    return projected_command, raw_torque, projected_torque

"""A1 table-tennis serve sampler for MuJoCo sim2sim viewing."""

from __future__ import annotations

import numpy as np

from a1_scene import SERVE_BOUNCE_VZ_RANGE, SERVE_BOUNCE_X_RANGE, SERVE_Y_CENTER as PROFILE_SERVE_Y_CENTER, SERVE_Y_HALF


SERVE_LAUNCH = np.array([1.35, 0.0, 1.03], dtype=np.float64)
G = 9.81
Z_BOUNCE = 0.78
BALL_RADIUS = 0.02
BALL_MASS = 0.0034
AIR_DENSITY = 1.225
DRAG_COEFF = 0.4378
BALL_DRAG_K = 0.5 * AIR_DENSITY * DRAG_COEFF * (np.pi * BALL_RADIUS ** 2) / BALL_MASS
BOUNCE_X = SERVE_BOUNCE_X_RANGE
BOUNCE_VZ = SERVE_BOUNCE_VZ_RANGE
SERVE_Y_CENTER = PROFILE_SERVE_Y_CENTER


def horizontal_velocity_for_drag_displacement(
    displacement: float | np.ndarray,
    duration: float | np.ndarray,
    drag_k: float = BALL_DRAG_K,
) -> np.ndarray:
    """Invert s(T)=sign(v)/k*log(1+k*|v|*T), matching Isaac training serves."""
    displacement = np.asarray(displacement, dtype=np.float64)
    duration = np.maximum(np.asarray(duration, dtype=np.float64), 1.0e-6)
    no_drag = displacement / duration
    if drag_k <= 1.0e-8:
        return no_drag
    return np.sign(displacement) * np.expm1(np.clip(drag_k * np.abs(displacement), None, 20.0)) / (
        drag_k * duration
    )


class Serve:
    def __init__(self, rng: np.random.Generator | None = None, interval_steps: int = 250):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.interval_steps = int(interval_steps)

    def due(self, control_step: int) -> bool:
        return control_step > 0 and control_step % self.interval_steps == 0

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        x_b = self.rng.uniform(*BOUNCE_X)
        y_b = SERVE_Y_CENTER + self.rng.uniform(-SERVE_Y_HALF, SERVE_Y_HALF)
        vz = self.rng.uniform(*BOUNCE_VZ)
        t_b = (vz + np.sqrt(vz * vz + 2.0 * G * (SERVE_LAUNCH[2] - Z_BOUNCE))) / G
        vx = horizontal_velocity_for_drag_displacement(x_b - SERVE_LAUNCH[0], t_b)
        vy = horizontal_velocity_for_drag_displacement(y_b - SERVE_LAUNCH[1], t_b)
        return SERVE_LAUNCH.copy(), np.array([vx, vy, vz], dtype=np.float64)

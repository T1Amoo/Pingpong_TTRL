"""Table-tennis ball validity gate for sim2sim/deploy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy_io import HIT_PLANE_X
from serve import BALL_DRAG_K, G, Z_BOUNCE


@dataclass
class BallGateConfig:
    own_x_lo: float = -1.37
    own_x_hi: float = 0.0
    y_abs: float = 1.2
    z_min: float = 0.75
    z_max: float = 2.6
    x_max: float = 1.65
    speed_max: float = 15.0
    vx_away: float = -0.05
    behind_margin: float = 0.05
    bounce_vz_down: float = 0.30
    bounce_vz_up: float = 0.05
    bounce_near_table: float = 0.15
    confirm_frames: int = 1
    coast_frames: int = 1


@dataclass
class BallGateOutput:
    live: bool
    engaged: bool
    reason: str
    first_bounce_x: float
    own_bounces: int
    hit_seen: bool


class BallValidityGate:
    """Hard gate matching the training/deploy invalid-ball contract.

    Invalid balls make the actor see the fixed home sentinel instead of raw ball
    or predictor output. The rules mirror the current A1 training mask and add a
    predictive first-bounce check so out/volley serves are ignored immediately.
    """

    def __init__(self, cfg: BallGateConfig | None = None):
        self.cfg = cfg if cfg is not None else BallGateConfig()
        self.reset()

    def reset(self) -> None:
        self._prev_vz: float | None = None
        self._own_bounces = 0
        self._hit_seen = False
        self._live_run = 0
        self._dead_run = 0
        self._engaged = False
        self.last = BallGateOutput(False, False, "reset", float("nan"), 0, False)

    def register_paddle_hit(self) -> None:
        self._hit_seen = True

    def update(self, ball_pos: np.ndarray, ball_vel: np.ndarray) -> BallGateOutput:
        pos = np.asarray(ball_pos, dtype=np.float64).reshape(3)
        vel = np.asarray(ball_vel, dtype=np.float64).reshape(3)
        self._update_bounces(pos, vel)
        first_bounce_x = self._predicted_first_bounce_x(pos, vel)
        live, reason = self._classify(pos, vel, first_bounce_x)

        if live:
            self._live_run += 1
            self._dead_run = 0
        else:
            self._dead_run += 1
            self._live_run = 0

        if not self._engaged and self._live_run >= max(1, self.cfg.confirm_frames):
            self._engaged = True
        if self._engaged and self._dead_run >= max(1, self.cfg.coast_frames):
            self._engaged = False

        self._prev_vz = float(vel[2]) if np.isfinite(vel[2]) else None
        self.last = BallGateOutput(
            live=live,
            engaged=self._engaged,
            reason=reason,
            first_bounce_x=first_bounce_x,
            own_bounces=self._own_bounces,
            hit_seen=self._hit_seen,
        )
        return self.last

    def _classify(self, pos: np.ndarray, vel: np.ndarray, first_bounce_x: float) -> tuple[bool, str]:
        cfg = self.cfg
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            return False, "nonfinite"
        if self._hit_seen:
            return False, "paddle_hit"
        speed = float(np.linalg.norm(vel))
        if speed > cfg.speed_max:
            return False, "speed"
        if abs(float(pos[1])) > cfg.y_abs or pos[2] > cfg.z_max or pos[0] > cfg.x_max:
            return False, "out_volume"
        if pos[0] < HIT_PLANE_X - cfg.behind_margin:
            return False, "behind_hit_plane"
        if vel[0] > cfg.vx_away:
            return False, "moving_away"
        if pos[2] < cfg.z_min:
            return False, "low_or_dead"
        if self._own_bounces >= 2:
            return False, "double_bounce"
        if self._own_bounces == 0:
            if not np.isfinite(first_bounce_x):
                return False, "no_table_bounce"
            if not (cfg.own_x_lo <= first_bounce_x <= cfg.own_x_hi):
                return False, "first_bounce_out"
        return True, "valid"

    def _update_bounces(self, pos: np.ndarray, vel: np.ndarray) -> None:
        if self._prev_vz is None or not np.isfinite(pos).all() or not np.isfinite(vel).all():
            return
        cfg = self.cfg
        was_descending = self._prev_vz < -cfg.bounce_vz_down
        ascending_now = vel[2] > cfg.bounce_vz_up
        near_table = pos[2] < Z_BOUNCE + cfg.bounce_near_table
        own_half = cfg.own_x_lo <= pos[0] <= cfg.own_x_hi
        if was_descending and ascending_now and near_table and own_half:
            self._own_bounces += 1

    @staticmethod
    def _predicted_first_bounce_x(pos: np.ndarray, vel: np.ndarray) -> float:
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            return float("nan")
        a = -0.5 * G
        b = float(vel[2])
        c = float(pos[2] - Z_BOUNCE)
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return float("nan")
        root = float(np.sqrt(disc))
        denom = 2.0 * a
        roots = [(-b - root) / denom, (-b + root) / denom]
        positive = [t for t in roots if t > 1e-4]
        if not positive:
            return float("nan")
        t = max(positive)
        if BALL_DRAG_K > 1.0e-8:
            dx = np.sign(vel[0]) * np.log1p(BALL_DRAG_K * abs(vel[0]) * t) / BALL_DRAG_K
        else:
            dx = vel[0] * t
        return float(pos[0] + dx)

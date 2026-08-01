"""Policy observation/action glue for A1/H1 table-tennis sim2sim."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from a1_scene import (
    DEFAULT_QPOS,
    HIT_BODY_HEIGHT,
    HIT_PLANE_X,
    HIT_TARGET_Y_RANGE,
    HIT_TARGET_Z_RANGE,
    HOME_Y,
    PADDLE_Y_OFFSET,
    PADDLE_OFFSET,
    PRED_SENTINEL,
    RIGHT_ARM_JOINTS,
    ROBOT_TABLE_POS,
    body_id,
    joint_dof_addr,
    joint_qpos_addr,
)


def guess_lgy_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Pingpong_TTRL").exists() and (parent / "unitree_rl_lab").exists():
            return parent
    return Path(__file__).resolve().parents[4]


LGY_ROOT = guess_lgy_root()
DEFAULT_POLICY = (
    LGY_ROOT
    / "Pingpong_TTRL/logs/a1_tt_v13/2026-07-08_12-40-15/exported/policy.onnx"
)
DEFAULT_PREDICTOR = DEFAULT_POLICY.with_name("predictor.onnx")

ACTION_SCALE = 0.25
CLIP_ACTIONS = 10.0
CLIP_OBS = 100.0
SOFT_JOINT_LIMIT_FACTOR = 0.95
FRAME_SIZE = 39
HISTORY = 5
OBS_SIZE = FRAME_SIZE * HISTORY

KP = np.array([300.0, 300.0, 300.0, 120.0, 120.0, 120.0, 60.0], dtype=np.float64)
KD = np.array([3.5, 3.5, 3.5, 1.0, 1.0, 1.0, 0.5], dtype=np.float64)
EFFORT = np.array([28.0, 28.0, 28.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float64)
VEL_LIMIT = np.array([10.0, 10.0, 10.0, 30.0, 30.0, 30.0, 30.0], dtype=np.float64)
# h1_pingpong/src/armcontrol/src/inference_arm_control_node.cpp initializes the
# A1 right arm as r1-r3 DM4340_48V and r4-r7 DM4310_48V. These are DAMIAO
# command/feedback ranges, not a calibrated motor dynamics model.
DAMIAO_EFFORT = np.array([28.0, 28.0, 28.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float64)
DAMIAO_DQ_LIMIT = np.array([10.0, 10.0, 10.0, 30.0, 30.0, 30.0, 30.0], dtype=np.float64)
# Actual Isaac play velocity envelope measured under model_800 with the 28/8Nm
# effort limits. This is a deploy-visual servo cap, not a motor specification.
ISAAC_PLAY_SERVO_VEL = np.array([1.0, 1.2, 1.8, 1.6, 4.0, 3.2, 8.0], dtype=np.float64)
BRIDGE_MAX_DELTA_PER_TICK = np.array([0.020, 0.024, 0.036, 0.032, 0.080, 0.064, 0.160], dtype=np.float64)
REAL_RESPONSE_MAX_DELTA_PER_TICK = np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10], dtype=np.float64)
# Fitted from mentor's 2026-07-29 backhand real-robot chirp data and validated
# against the matching step data. The old FixStand/0714 response was too fast
# for r5/r7 and used an artificial 80 ms deterministic delay in sim2sim.
REAL_RESPONSE_U_MEAN = np.array(
    [1.7700659393737623, -0.7609315193334025, -1.862263833851564, 1.4460611991561492,
     0.20699718244975998, -0.8260044098093582, 1.0440311471658599],
    dtype=np.float64,
)
REAL_RESPONSE_FN_HZ = np.array(
    [3.081469882952914, 3.110360258230149, 3.5084746761876504, 3.938078474900925,
     8.393387809118405, 6.081603320508635, 6.2167388827512085],
    dtype=np.float64,
)
REAL_RESPONSE_ZETA = np.array(
    [0.29001527380439157, 0.45573251210433835, 0.5090613481505404, 0.4438206941118423,
     0.6423265474666591, 0.5161814994035029, 0.5346367478112269],
    dtype=np.float64,
)
REAL_RESPONSE_DELAY_S = np.array(
    [3.4518349565248606e-09, 2.5026306975070193e-13, 1.5374720497906557e-11,
     0.006037572727162875, 0.01121360625256748, 0.00642196457222213, 0.021843507909910763],
    dtype=np.float64,
)
REAL_RESPONSE_GAIN = np.array(
    [0.8151195546881129, 1.0022953758506237, 0.9527639884101302, 1.0525053363155508,
     1.0103665491921643, 0.9716496094155513, 0.9455371267034796],
    dtype=np.float64,
)
REAL_RESPONSE_BIAS_RAD = np.array(
    [-0.022742565653490976, -0.001145840134567977, -0.01281695940028782,
     0.001689992587619038, -0.006230753796553995, 0.00217495010862645, -0.0004602019305564031],
    dtype=np.float64,
)
DEFAULT_RIGHT_Q = np.array([DEFAULT_QPOS[j] for j in RIGHT_ARM_JOINTS], dtype=np.float64)

# Training-matched DamiaoMIT constants (A1_TT_REAL_TORQUE_ONLY_CFG).
# r1-r3 = DM4340_48V (idx<3), r4-r7 = DM4310_48V (idx>=3).
DAMIAO_MIT_KP = np.array([300.0, 300.0, 300.0, 120.0, 120.0, 120.0, 60.0], dtype=np.float64)
DAMIAO_MIT_KD = np.array([3.5, 3.5, 3.5, 1.0, 1.0, 1.0, 0.5], dtype=np.float64)
# effort r1-3=28, r4-7=8 (training _EFFORT: idx<3 -> 28 else 8).
DAMIAO_MIT_EFFORT = np.array([28.0, 28.0, 28.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float64)
# 0729 backhand real logs: step max actual_dq was about
# [2.8, 2.4, 3.4, 3.9, 5.2, 3.9, 6.5] rad/s. Keep margin for policy
# transients while preventing the old 20 rad/s distal no-load limit from
# producing high-frequency MuJoCo swings.
DAMIAO_MIT_VEL = np.array([4.0, 4.0, 5.0, 6.0, 8.0, 6.0, 9.0], dtype=np.float64)
DAMIAO_MIT_BRAKE_EFFORT = DAMIAO_MIT_EFFORT.copy()


def project_prediction_to_target_geometry(pred: np.ndarray) -> np.ndarray:
    """Match IsaacLab's actor-facing hit-target projection for sim2sim."""
    out = np.asarray(pred, dtype=np.float32).reshape(3).copy()
    out[0] = np.float32(HIT_PLANE_X)
    out[1] = np.clip(out[1], HIT_TARGET_Y_RANGE[0], HIT_TARGET_Y_RANGE[1])
    out[2] = np.clip(out[2], HIT_TARGET_Z_RANGE[0], HIT_TARGET_Z_RANGE[1])
    return out


def damiao_clip_effort(tau, dq, vel_limit, effort_limit, brake_effort_limit):
    """Mirror DamiaoMIT._clip_effort (torque_speed_limit_enable=True)."""
    vl = np.maximum(vel_limit, 1.0e-6)
    speed_scale = np.clip(1.0 - np.abs(dq) / vl, 0.0, 1.0)
    accel_limit = effort_limit * speed_scale
    brake_limit = np.maximum(brake_effort_limit, effort_limit)
    tau_max = np.where(dq > 0.0, accel_limit, brake_limit)
    neg_abs_limit = np.where(dq < 0.0, accel_limit, brake_limit)
    tau_min = -neg_abs_limit
    return np.clip(tau, tau_min, tau_max)


def damiao_slew(cmd, q_des, vel_limit, dt):
    """Mirror DamiaoMIT._apply_command_slew (response/lead/delay are identity in v7)."""
    max_delta = vel_limit * dt
    delta = q_des - cmd
    return cmd + np.clip(delta, -max_delta, max_delta)


class OnnxPolicy:
    def __init__(self, policy_path: Path | str = DEFAULT_POLICY):
        self.path = Path(policy_path)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        shape = self.session.get_inputs()[0].shape
        if int(shape[1]) != OBS_SIZE:
            raise ValueError(f"policy input must be {OBS_SIZE}, got {shape}")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, OBS_SIZE)
        out = self.session.run([self.output_name], {self.input_name: obs})[0]
        return np.asarray(out[0], dtype=np.float32)


class OnnxBallPredictor:
    def __init__(self, predictor_path: Path | str = DEFAULT_PREDICTOR, history_len: int = 5):
        self.path = Path(predictor_path)
        self.history_len = int(history_len)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.history: deque[np.ndarray] = deque(maxlen=self.history_len)

    def clear(self) -> None:
        self.history.clear()

    def __call__(self, ball_pos: np.ndarray, valid_ball: bool) -> np.ndarray | None:
        if not valid_ball or not np.isfinite(ball_pos).all():
            self.clear()
            return None
        b = np.asarray(ball_pos, dtype=np.float32).reshape(3)
        self.history.append(b)
        rows = list(self.history)
        if len(rows) < self.history_len:
            return None
        x = np.concatenate(rows).reshape(1, 3 * self.history_len).astype(np.float32)
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        return np.asarray(out[0], dtype=np.float32)


class FittedSecondOrderActionResponse:
    """Seven-joint closed-loop response fitted from A1 real-arm logs."""

    def __init__(self, dt: float, initial_joint_pos: np.ndarray):
        self.dt = float(dt)
        self.u_mean = REAL_RESPONSE_U_MEAN.copy()
        self.fn_hz = REAL_RESPONSE_FN_HZ.copy()
        self.zeta = REAL_RESPONSE_ZETA.copy()
        self.delay_s = REAL_RESPONSE_DELAY_S.copy()
        self.gain = REAL_RESPONSE_GAIN.copy()
        self.bias = REAL_RESPONSE_BIAS_RAD.copy()
        self.omega = 2.0 * np.pi * np.maximum(self.fn_hz, 1e-6)
        self.delay_steps = np.maximum(np.rint(self.delay_s / self.dt).astype(np.int64), 0)
        self.delay_buffer = np.zeros((int(np.max(self.delay_steps)) + 1, 7), dtype=np.float64)
        self.delay_index = 0
        self.x_rel = np.zeros(7, dtype=np.float64)
        self.v_rel = np.zeros(7, dtype=np.float64)
        self.response = np.asarray(initial_joint_pos, dtype=np.float64).reshape(7).copy()
        self.response_velocity = np.zeros(7, dtype=np.float64)
        self.reset(initial_joint_pos)

    def reset(self, joint_pos: np.ndarray) -> np.ndarray:
        joint_pos = np.asarray(joint_pos, dtype=np.float64).reshape(7)
        gain = np.maximum(self.gain, 1e-6)
        self.x_rel = (joint_pos - self.u_mean - self.bias) / gain
        self.v_rel.fill(0.0)
        steady_raw_command = self.u_mean + self.x_rel
        self.delay_buffer[:] = steady_raw_command
        self.delay_index = 0
        self.response = joint_pos.copy()
        self.response_velocity.fill(0.0)
        return self.response.copy()

    def _delayed_command(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float64).reshape(7)
        write_idx = self.delay_index
        self.delay_buffer[write_idx] = command
        delayed = np.empty(7, dtype=np.float64)
        for i, delay_step in enumerate(self.delay_steps):
            read_idx = (write_idx - int(delay_step)) % len(self.delay_buffer)
            delayed[i] = self.delay_buffer[read_idx, i]
        self.delay_index = (write_idx + 1) % len(self.delay_buffer)
        return delayed

    def step(self, command: np.ndarray, dt: float | None = None) -> np.ndarray:
        dt = self.dt if dt is None else float(dt)
        delayed_command = self._delayed_command(command)
        u_rel = delayed_command - self.u_mean
        accel = (
            np.square(self.omega) * (u_rel - self.x_rel)
            - 2.0 * self.zeta * self.omega * self.v_rel
        )
        self.v_rel += accel * dt
        self.x_rel += self.v_rel * dt
        self.response = self.u_mean + self.bias + self.gain * self.x_rel
        self.response_velocity = self.gain * self.v_rel
        return self.response.copy()


class A1PolicyIO:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        predictor_path: Path | str = DEFAULT_PREDICTOR,
        use_predictor: bool = True,
    ):
        self.model = model
        self.data = data
        self.qpos_addr = np.array([joint_qpos_addr(model, j) for j in RIGHT_ARM_JOINTS], dtype=np.int32)
        self.dof_addr = np.array([joint_dof_addr(model, j) for j in RIGHT_ARM_JOINTS], dtype=np.int32)
        self.q_min, self.q_max = self._soft_joint_ranges()
        self.hold_joints = [
            name for name in DEFAULT_QPOS
            if name not in RIGHT_ARM_JOINTS and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        ]
        self.hold_qpos_addr = np.array([joint_qpos_addr(model, j) for j in self.hold_joints], dtype=np.int32)
        self.hold_dof_addr = np.array([joint_dof_addr(model, j) for j in self.hold_joints], dtype=np.int32)
        self.hold_q = np.array([DEFAULT_QPOS[j] for j in self.hold_joints], dtype=np.float64)
        self.last_action = np.zeros(7, dtype=np.float32)
        self.raw_q_des = DEFAULT_RIGHT_Q.copy()
        self.q_des = DEFAULT_RIGHT_Q.copy()
        self.motor_q_des = DEFAULT_RIGHT_Q.copy()
        self.predictor = (
            OnnxBallPredictor(predictor_path)
            if use_predictor and Path(predictor_path).exists()
            else None
        )
        self.history = deque(maxlen=HISTORY)
        self.servo_q = self.right_q()
        self.damiao_cmd = self.right_q()
        self.servo_dq = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float64)
        self.last_ball_pred = PRED_SENTINEL.copy()
        frame = self.compute_frame(np.zeros(3), np.zeros(3), valid_ball=False)
        for _ in range(HISTORY):
            self.history.append(frame)

    def reset_ball_history(self) -> None:
        if self.predictor is not None:
            self.predictor.clear()

    def right_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_addr].copy()

    def right_dq(self) -> np.ndarray:
        return self.data.qvel[self.dof_addr].copy()

    def set_right_state(
        self,
        q: np.ndarray,
        dq: np.ndarray | None = None,
        *,
        reset_targets: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).reshape(7)
        dq = np.zeros(7, dtype=np.float64) if dq is None else np.asarray(dq, dtype=np.float64).reshape(7)
        q = np.clip(q, self.q_min, self.q_max)
        self.data.qpos[self.qpos_addr] = q
        self.data.qvel[self.dof_addr] = dq
        self.servo_q = q.copy()
        self.damiao_cmd = q.copy()
        self.servo_dq = dq.copy()
        if reset_targets:
            self.last_action.fill(0.0)
            self.raw_q_des = q.copy()
            self.q_des = q.copy()
            self.motor_q_des = q.copy()
        mujoco.mj_forward(self.model, self.data)
        return q.copy(), dq.copy()

    def reset_policy_history(self, ball_pos: np.ndarray, ball_vel: np.ndarray, *, valid_ball: bool = False) -> None:
        self.reset_ball_history()
        self.history.clear()
        frame = self.compute_frame(ball_pos, ball_vel, valid_ball=valid_ball)
        for _ in range(HISTORY):
            self.history.append(frame.copy())

    def analytic_prediction(self, ball_pos: np.ndarray, ball_vel: np.ndarray, valid_ball: bool) -> np.ndarray:
        if not valid_ball or not np.isfinite(ball_pos).all() or ball_vel[0] >= -0.05:
            return PRED_SENTINEL.copy()
        t = (HIT_PLANE_X - ball_pos[0]) / ball_vel[0]
        if t <= 0.0 or t > 2.0:
            return PRED_SENTINEL.copy()
        pred = ball_pos + ball_vel * t
        pred[2] += -0.5 * 9.81 * t * t
        if pred[2] < 0.2 or pred[2] > 2.0:
            return PRED_SENTINEL.copy()
        return project_prediction_to_target_geometry(pred)

    def plausible_prediction(self, pred: np.ndarray) -> bool:
        y_center = HOME_Y + PADDLE_Y_OFFSET
        return bool(
            np.isfinite(pred).all()
            and HIT_PLANE_X - 0.50 < pred[0] < HIT_PLANE_X + 0.30
            and abs(float(pred[1] - y_center)) < 0.45
            and 0.85 < pred[2] < 1.55
        )

    def predict_ball(self, ball_pos: np.ndarray, ball_vel: np.ndarray, valid_ball: bool) -> np.ndarray:
        if self.predictor is not None:
            pred = self.predictor(ball_pos, valid_ball)
            if pred is not None and self.plausible_prediction(pred):
                return project_prediction_to_target_geometry(pred)
            return PRED_SENTINEL.copy()
        return self.analytic_prediction(ball_pos, ball_vel, valid_ball)

    def compute_frame(self, ball_pos: np.ndarray, ball_vel: np.ndarray, valid_ball: bool = True) -> np.ndarray:
        q = self.right_q()
        dq = self.right_dq()
        joint_pos_rel = q - DEFAULT_RIGHT_Q
        robot_pos = ROBOT_TABLE_POS.astype(np.float32)
        ball_pred = self.predict_ball(ball_pos, ball_vel, valid_ball)
        self.last_ball_pred = ball_pred.copy()
        if not valid_ball:
            ball_obs = PRED_SENTINEL.copy()
        else:
            ball_obs = ball_pos.astype(np.float32)
        perception = np.concatenate([ball_obs, robot_pos]).astype(np.float32)
        rel_target_xy = np.array(
            [
                (ball_pred[0] - 0.1) - robot_pos[0],
                (ball_pred[1] - PADDLE_Y_OFFSET) - robot_pos[1],
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                np.zeros(3, dtype=np.float32),               # root angular velocity
                np.array([0.0, 0.0, -1.0], dtype=np.float32), # projected gravity
                joint_pos_rel.astype(np.float32),
                dq.astype(np.float32),
                self.last_action.astype(np.float32),
                perception,
                ball_pred.astype(np.float32),
                rel_target_xy,
                np.array([0.0], dtype=np.float32),           # heading
            ]
        ).astype(np.float32)

    def observe(self, ball_pos: np.ndarray, ball_vel: np.ndarray, valid_ball: bool = True) -> np.ndarray:
        frame = self.compute_frame(ball_pos, ball_vel, valid_ball)
        if frame.shape != (FRAME_SIZE,):
            raise RuntimeError(f"bad frame shape {frame.shape}")
        self.history.append(frame)
        obs = np.concatenate(list(self.history), dtype=np.float32)
        return np.clip(obs, -CLIP_OBS, CLIP_OBS)

    @staticmethod
    def limit_q_des_delta(
        q_des: np.ndarray,
        previous_q_des: np.ndarray,
        max_delta_per_tick: np.ndarray | None,
    ) -> np.ndarray:
        q_des = np.asarray(q_des, dtype=np.float64).reshape(7)
        previous_q_des = np.asarray(previous_q_des, dtype=np.float64).reshape(7)
        if max_delta_per_tick is None:
            return q_des.copy()
        max_delta = np.asarray(max_delta_per_tick, dtype=np.float64).reshape(7)
        if np.all(max_delta <= 0.0):
            return q_des.copy()
        delta = np.clip(q_des - previous_q_des, -max_delta, max_delta)
        return previous_q_des + delta

    def update_action(
        self,
        raw_action: np.ndarray,
        max_delta_per_tick: np.ndarray | None = None,
        update_motor_target: bool = True,
    ) -> np.ndarray:
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(7)
        self.last_action = raw_action
        clipped = np.clip(raw_action, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float64)
        q_des = clipped * ACTION_SCALE + DEFAULT_RIGHT_Q
        self.raw_q_des = np.clip(q_des, self.q_min, self.q_max)
        self.q_des = self.limit_q_des_delta(self.raw_q_des, self.q_des, max_delta_per_tick)
        self.q_des = np.clip(self.q_des, self.q_min, self.q_max)
        if update_motor_target:
            self.motor_q_des = self.q_des.copy()
        return self.q_des.copy()

    def set_motor_q_des(self, q_des: np.ndarray) -> np.ndarray:
        self.motor_q_des = np.asarray(q_des, dtype=np.float64).reshape(7).copy()
        return self.motor_q_des.copy()

    def estimate_mit_tau_raw(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_addr]
        dq = self.data.qvel[self.dof_addr]
        return KP * (self.motor_q_des - q) + KD * (0.0 - dq) + self.data.qfrc_bias[self.dof_addr]

    def estimate_mit_tau(self, effort_limit: np.ndarray | None = None) -> np.ndarray:
        tau = self.estimate_mit_tau_raw()
        limit = EFFORT if effort_limit is None else effort_limit
        return np.clip(tau, -limit, limit)

    def apply_mit_pd(self, effort_limit: np.ndarray | None = None) -> np.ndarray:
        tau = self.estimate_mit_tau(effort_limit)
        self.data.qfrc_applied[self.dof_addr] += tau
        return tau

    def apply_damiao_mit(self, dt: float, desired_vel: np.ndarray | None = None) -> np.ndarray:
        """Faithful mujoco replica of the training DamiaoMIT torque chain.

        Mirrors DamiaoMIT.compute for A1_TT_REAL_TORQUE_ONLY_CFG:
        motor_q_des -> slew(_VEL) -> MIT-PD -> torque-speed clip.
        """
        q = self.data.qpos[self.qpos_addr]
        dq = self.data.qvel[self.dof_addr]
        desired_vel = (
            np.zeros(7, dtype=np.float64)
            if desired_vel is None
            else np.asarray(desired_vel, dtype=np.float64).reshape(7)
        )
        self.damiao_cmd = damiao_slew(self.damiao_cmd, self.motor_q_des, DAMIAO_MIT_VEL, dt)
        tau = DAMIAO_MIT_KP * (self.damiao_cmd - q) + DAMIAO_MIT_KD * (desired_vel - dq)
        tau = damiao_clip_effort(tau, dq, DAMIAO_MIT_VEL, DAMIAO_MIT_EFFORT, DAMIAO_MIT_BRAKE_EFFORT)
        self.data.qfrc_applied[self.dof_addr] += tau
        return tau

    def apply_position_servo(
        self,
        dt: float,
        velocity_limit: np.ndarray | None = None,
        tracking_tau: float = 0.25,
    ) -> None:
        if dt > 0.0:
            prev = self.servo_q.copy()
            limit = ISAAC_PLAY_SERVO_VEL if velocity_limit is None else velocity_limit
            desired_dq = (self.motor_q_des - self.servo_q) / max(float(tracking_tau), 1e-6)
            dq_cmd = np.clip(desired_dq, -limit, limit)
            self.servo_q = self.servo_q + dq_cmd * dt
            self.servo_q = np.clip(self.servo_q, self.q_min, self.q_max)
            self.servo_dq = (self.servo_q - prev) / dt
        self.data.qpos[self.qpos_addr] = self.servo_q
        self.data.qvel[self.dof_addr] = self.servo_dq

    def apply_direct_response(self, dt: float) -> None:
        prev = self.data.qpos[self.qpos_addr].copy()
        q = np.clip(self.motor_q_des, self.q_min, self.q_max)
        if dt > 0.0:
            dq = (q - prev) / dt
        else:
            dq = self.data.qvel[self.dof_addr].copy()
        self.servo_q = q.copy()
        self.servo_dq = dq.copy()
        self.data.qpos[self.qpos_addr] = q
        self.data.qvel[self.dof_addr] = dq

    def enforce_motor_limits(self, velocity_limit: np.ndarray | None = VEL_LIMIT) -> None:
        q = np.clip(self.data.qpos[self.qpos_addr], self.q_min, self.q_max)
        dq = self.data.qvel[self.dof_addr].copy()
        if velocity_limit is not None:
            dq = np.clip(dq, -velocity_limit, velocity_limit)
            eps = 1e-5
            dq = np.where((q <= self.q_min + eps) & (dq < 0.0), 0.0, dq)
            dq = np.where((q >= self.q_max - eps) & (dq > 0.0), 0.0, dq)
        self.data.qpos[self.qpos_addr] = q
        self.data.qvel[self.dof_addr] = dq
        self.servo_q = np.clip(self.servo_q, self.q_min, self.q_max)

    def enforce_hold_joints(self) -> None:
        if len(self.hold_joints) == 0:
            return
        self.data.qpos[self.hold_qpos_addr] = self.hold_q
        self.data.qvel[self.hold_dof_addr] = 0.0

    def paddle_touch_point(self) -> np.ndarray:
        paddle_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Link_r_paddle")
        if paddle_bid >= 0:
            return (
                self.data.xpos[paddle_bid].copy()
                + self.data.xmat[paddle_bid].reshape(3, 3) @ PADDLE_OFFSET
            )
        # MuJoCo collapses the fixed r_paddle joint into Link_r7 when compiling the
        # URDF. The fixed joint is xyz=(0,0,0.172), rpy=(pi,0,0); profile local
        # paddle offsets therefore need to be expressed in the parent frame here.
        wrist_bid = body_id(self.model, "Link_r7")
        parent_offset = np.array([PADDLE_OFFSET[0], -PADDLE_OFFSET[1], -PADDLE_OFFSET[2]], dtype=np.float64)
        return (
            self.data.xpos[wrist_bid].copy()
            + self.data.xmat[wrist_bid].reshape(3, 3) @ (np.array([0.0, 0.0, 0.172], dtype=np.float64) + parent_offset)
        )

    def _soft_joint_ranges(self) -> tuple[np.ndarray, np.ndarray]:
        q_min = np.full(len(RIGHT_ARM_JOINTS), -np.inf, dtype=np.float64)
        q_max = np.full(len(RIGHT_ARM_JOINTS), np.inf, dtype=np.float64)
        for i, name in enumerate(RIGHT_ARM_JOINTS):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0 or not self.model.jnt_limited[jid]:
                continue
            lo, hi = self.model.jnt_range[jid]
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * SOFT_JOINT_LIMIT_FACTOR
            q_min[i] = mid - half
            q_max[i] = mid + half
        return q_min, q_max

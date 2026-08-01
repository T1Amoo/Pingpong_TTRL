#!/usr/bin/env python3
"""Run the A1/H1 table-tennis policy in MuJoCo.

This is the sim2sim half of the deployment path:
policy.onnx -> action -> q_des -> MIT-PD torque, matching the real DAMIAO SDK
call shape used later by sim2real:

    control_mit(motor, kp, kd, q_des, dq_des, tau_ff)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mujoco
import numpy as np

from a1_scene import ACTIVE_PROFILE, BALL_RADIUS, PHYSICS_DT, ball_addresses, body_id, load_scene, set_ball_state
from ball_gate import BallGateConfig, BallGateOutput, BallValidityGate
from policy_io import (
    A1PolicyIO,
    BRIDGE_MAX_DELTA_PER_TICK,
    DEFAULT_POLICY,
    DEFAULT_RIGHT_Q,
    DAMIAO_DQ_LIMIT,
    DAMIAO_EFFORT,
    DAMIAO_MIT_VEL,
    EFFORT,
    FittedSecondOrderActionResponse,
    HIT_PLANE_X,
    ISAAC_PLAY_SERVO_VEL,
    REAL_RESPONSE_MAX_DELTA_PER_TICK,
    VEL_LIMIT,
    OnnxPolicy,
)
from serve import Serve


DECIMATION = 10
CONTROL_DT = PHYSICS_DT * DECIMATION
IDENTITY_MAT = np.eye(3, dtype=np.float64).ravel()
PARKED_BALL_POS = np.array([1.75, 1.35, 0.20], dtype=np.float64)
PARKED_BALL_VEL = np.zeros(3, dtype=np.float64)


def velocity_clip_limit_for_mode(actuator_mode: str, no_qvel_clip: bool) -> np.ndarray | None:
    if no_qvel_clip:
        return None
    if actuator_mode == "real_deploy_preview":
        return DAMIAO_DQ_LIMIT
    if actuator_mode == "damiao_mit":
        return DAMIAO_MIT_VEL
    return VEL_LIMIT


def action_for_gate(
    policy: OnnxPolicy | None,
    obs: np.ndarray,
    gate_out: BallGateOutput,
    *,
    invalid_ball_action: str,
) -> np.ndarray:
    """Return the action sent to the controller after ball-validity gating."""
    if policy is None:
        return np.zeros(7, dtype=np.float32)
    policy_action = policy(obs)
    if invalid_ball_action == "policy" or gate_out.engaged:
        return policy_action
    if invalid_ball_action == "zero":
        return np.zeros(7, dtype=np.float32)
    raise ValueError(f"unknown invalid_ball_action={invalid_ball_action!r}")


def _load_initial_joint_state(csv_path: Path | str, time_s: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    q_cols = [f"q{i}" for i in range(1, 8)]
    dq_cols = [f"dq{i}" for i in range(1, 8)]
    required = {"time_s", *q_cols}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    q_values = np.asarray([[float(row[col]) for col in q_cols] for row in rows], dtype=np.float64)
    if all(col in rows[0] for col in dq_cols):
        dq_values = np.asarray([[float(row[col]) for col in dq_cols] for row in rows], dtype=np.float64)
    else:
        dq_values = np.zeros_like(q_values)
    if time_s is None:
        return q_values[0].copy(), dq_values[0].copy(), float(times[0])
    t = float(np.clip(float(time_s), times[0], times[-1]))
    q = np.asarray([np.interp(t, times, q_values[:, i]) for i in range(7)], dtype=np.float64)
    dq = np.asarray([np.interp(t, times, dq_values[:, i]) for i in range(7)], dtype=np.float64)
    return q, dq, t


class BallTrajectoryReplay:
    """Kinematic replay of a recorded real ball trajectory."""

    def __init__(self, csv_path: Path | str):
        self.path = Path(csv_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        times: list[float] = []
        pos: list[list[float]] = []
        vel: list[list[float]] = []
        source_ball: list[int] = []
        with self.path.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"segment_time_s", "x", "y", "z", "vx", "vy", "vz"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.path} missing columns: {sorted(missing)}")
            has_source_ball = "source_ball" in (reader.fieldnames or [])
            for row in reader:
                try:
                    t = float(row["segment_time_s"])
                    p = [float(row["x"]), float(row["y"]), float(row["z"])]
                    v = [float(row["vx"]), float(row["vy"]), float(row["vz"])]
                except (TypeError, ValueError):
                    continue
                if not np.isfinite([t, *p, *v]).all():
                    continue
                if times and t <= times[-1]:
                    t = times[-1] + 1e-6
                times.append(t)
                pos.append(p)
                vel.append(v)
                if has_source_ball:
                    try:
                        source_ball.append(int(float(row.get("source_ball", 1))))
                    except (TypeError, ValueError):
                        source_ball.append(1)
                else:
                    source_ball.append(1)
        if len(times) < 2:
            raise ValueError(f"{self.path} must contain at least two finite samples")
        self.time = np.asarray(times, dtype=np.float64)
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.asarray(vel, dtype=np.float64)
        self.source_ball = np.asarray(source_ball, dtype=np.int32)
        self.duration = float(self.time[-1])

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, int]:
        t = float(np.clip(t, self.time[0], self.time[-1]))
        idx = int(np.searchsorted(self.time, t, side="right") - 1)
        idx = int(np.clip(idx, 0, len(self.source_ball) - 1))
        source_ball = int(self.source_ball[idx])
        if idx + 1 < len(self.time) and int(self.source_ball[idx + 1]) == source_ball:
            dt = float(self.time[idx + 1] - self.time[idx])
            alpha = 0.0 if dt <= 1e-9 else float(np.clip((t - self.time[idx]) / dt, 0.0, 1.0))
            pos = (1.0 - alpha) * self.pos[idx] + alpha * self.pos[idx + 1]
            vel = (1.0 - alpha) * self.vel[idx] + alpha * self.vel[idx + 1]
        else:
            pos = self.pos[idx].copy()
            vel = self.vel[idx].copy()
        return pos, vel, source_ball


def _ball_paddle_contact_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    ball_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    paddle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_blade")
    if ball_gid < 0 or paddle_gid < 0:
        return None
    for i in range(data.ncon):
        con = data.contact[i]
        if {int(con.geom1), int(con.geom2)} == {ball_gid, paddle_gid}:
            return np.asarray(con.pos, dtype=np.float64).copy()
    return None


def _add_debug_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        IDENTITY_MAT,
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _update_hit_point_visuals(
    scene: mujoco.MjvScene | None,
    ball_pos: np.ndarray,
    hit_point: np.ndarray,
    gate_out: BallGateOutput,
) -> None:
    if scene is None:
        return
    scene.ngeom = 0
    valid_rgba = np.array([0.0, 0.9, 0.25, 0.95], dtype=np.float32)
    invalid_rgba = np.array([0.55, 0.55, 0.55, 0.45], dtype=np.float32)
    ball_rgba = np.array([1.0, 0.2, 0.05, 0.75], dtype=np.float32)
    rgba = valid_rgba if gate_out.engaged else invalid_rgba
    _add_debug_sphere(scene, hit_point, 0.035 if gate_out.engaged else 0.025, rgba)
    _add_debug_sphere(scene, ball_pos, 0.024, ball_rgba)


def _apply_ball_drag(model: mujoco.MjModel, data: mujoco.MjData, ball_vadr: int) -> None:
    v = data.qvel[ball_vadr : ball_vadr + 3]
    speed = float(np.linalg.norm(v))
    if speed <= 1e-9:
        return
    force = (-0.5 * 1.225 * (np.pi * BALL_RADIUS ** 2) * 0.4378 * speed) * v
    data.qfrc_applied[ball_vadr : ball_vadr + 3] += force


def _maybe_manual_hit(io: A1PolicyIO, ball_pos: np.ndarray, ball_vel: np.ndarray) -> bool:
    """Fallback contact model for fast balls if mesh contact misses.

    It is intentionally conservative: only flips a ball that is near the training
    paddle touch point and still traveling toward the robot.
    """

    if ball_vel[0] >= -0.05:
        return False
    paddle_pos = io.paddle_touch_point()
    dist = np.linalg.norm(ball_pos - paddle_pos) - BALL_RADIUS
    if dist > 0.045:
        return False
    ball_vel[0] = max(2.0, abs(ball_vel[0]) * 0.75)
    ball_vel[1] *= 0.3
    ball_vel[2] = max(0.8, ball_vel[2] * 0.3 + 1.0)
    return True


def _dead_ball(ball_pos: np.ndarray, ball_vel: np.ndarray) -> bool:
    return bool(
        not np.isfinite(ball_pos).all()
        or ball_pos[2] < 0.08
        or ball_pos[0] < HIT_PLANE_X - 0.20
        or ball_pos[0] > 1.65
        or abs(float(ball_pos[1])) > 1.2
    )


def next_serve_step_after_inactive(control_step: int, pause_steps: int) -> int:
    """Return the first control step allowed to start a new serve after a dead ball."""
    return int(control_step) + max(1, int(pause_steps))


def run(args: argparse.Namespace) -> dict:
    if args.dynamic_pd and args.actuator_mode == "isaac_approx":
        args.actuator_mode = "torque_chain"
    model, data, xml_path = load_scene()
    predictor_path = Path(args.policy).with_name("predictor.onnx")
    io = A1PolicyIO(model, data, predictor_path=predictor_path)
    initial_state_time_s = float("nan")
    if args.initial_state_csv is not None:
        q0, dq0, initial_state_time_s = _load_initial_joint_state(
            args.initial_state_csv,
            args.initial_state_time_s,
        )
        q0, dq0 = io.set_right_state(q0, dq0, reset_targets=True)
        print(
            "[a1_sim2sim] initial_state "
            f"path={args.initial_state_csv} time_s={initial_state_time_s:.6f} "
            f"q={np.round(q0, 4).tolist()} dq={np.round(dq0, 4).tolist()}",
            flush=True,
        )
    if args.max_delta_per_tick is None:
        max_delta_per_tick = (
            REAL_RESPONSE_MAX_DELTA_PER_TICK.copy()
            if args.real_response_model
            else BRIDGE_MAX_DELTA_PER_TICK.copy()
        )
    else:
        max_delta_per_tick = np.asarray(args.max_delta_per_tick, dtype=np.float64).reshape(7)
    response_model = (
        FittedSecondOrderActionResponse(PHYSICS_DT, io.right_q())
        if args.real_response_model
        else None
    )
    if response_model is not None:
        io.set_motor_q_des(response_model.response)
    if io.predictor is not None:
        print(f"[a1_sim2sim] predictor: {predictor_path}")
    else:
        print(f"[a1_sim2sim] predictor: none ({predictor_path} not found)")
    policy = None if args.no_policy else OnnxPolicy(args.policy)
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    serve = Serve(rng=rng, interval_steps=args.serve_interval)
    trajectory = BallTrajectoryReplay(args.ball_trajectory_csv) if args.ball_trajectory_csv is not None else None
    trajectory_interval = (
        max(float(args.trajectory_loop_interval), trajectory.duration)
        if trajectory is not None and args.trajectory_loop_interval > 0.0
        else (trajectory.duration + float(args.trajectory_pause_s) if trajectory is not None else 0.0)
    )
    gate = BallValidityGate(
        BallGateConfig(
            confirm_frames=args.gate_confirm_frames,
            coast_frames=args.gate_coast_frames,
        )
    )
    ball_qadr, ball_vadr = ball_addresses(model)
    ball_bid = body_id(model, "ball")

    ball_active = False
    hit_counted = False
    next_serve_step: int | None = None
    static_ball = None if args.static_ball is None else np.asarray(args.static_ball, dtype=np.float64)
    trajectory_cycle = -1
    trajectory_source_ball = -1
    trajectory_phase_s = float("nan")
    trajectory_replay_active = False
    io.update_action(
        np.zeros(7, dtype=np.float32),
        max_delta_per_tick=max_delta_per_tick if args.real_response_model else None,
        update_motor_target=response_model is None,
    )
    if args.initial_state_csv is not None:
        io.set_right_state(io.right_q(), io.right_dq(), reset_targets=True)
        if response_model is not None:
            response_model.reset(io.right_q())
            io.set_motor_q_des(response_model.response)

    stats = {
        "steps": 0,
        "control_steps": 0,
        "hits": 0,
        "serves": 0,
        "max_abs_action": 0.0,
        "max_abs_qvel": 0.0,
        "max_abs_tau": 0.0,
        "valid_frames": 0,
        "max_abs_raw_qdes_delta": 0.0,
        "max_abs_limited_qdes_delta": 0.0,
        "max_abs_motor_qdes_delta": 0.0,
        "max_abs_motor_qdes_speed": 0.0,
        "profile": ACTIVE_PROFILE.name,
        "xml": str(xml_path),
        "last_hit_step": -1,
        "last_hit_source": "",
        "last_hit_ball_x": float("nan"),
        "last_hit_paddle_x": float("nan"),
        "last_hit_contact_x": float("nan"),
    }
    trace_rows: list[dict[str, float | str]] = []
    trace_file = None
    trace_writer = None
    trace_write_count = 0

    def append_trace(row: dict[str, float | str]) -> None:
        nonlocal trace_file, trace_writer, trace_write_count
        trace_rows.append(row)
        if args.trace_csv is None:
            return
        if trace_writer is None:
            args.trace_csv.parent.mkdir(parents=True, exist_ok=True)
            trace_file = args.trace_csv.open("w", newline="")
            trace_writer = csv.DictWriter(trace_file, fieldnames=list(row.keys()))
            trace_writer.writeheader()
        trace_writer.writerow(row)
        trace_write_count += 1
        if args.trace_flush_every <= 1 or trace_write_count % args.trace_flush_every == 0:
            trace_file.flush()

    def close_trace() -> None:
        nonlocal trace_file
        if trace_file is not None:
            trace_file.flush()
            trace_file.close()
            trace_file = None

    def effort_limit() -> np.ndarray:
        return DAMIAO_EFFORT if args.actuator_mode == "real_deploy_preview" else EFFORT

    def velocity_limit() -> np.ndarray | None:
        return velocity_clip_limit_for_mode(args.actuator_mode, args.no_qvel_clip)

    def start_serve() -> None:
        nonlocal ball_active, hit_counted, next_serve_step
        p, v = serve.sample()
        set_ball_state(model, data, p, v)
        io.reset_ball_history()
        gate.reset()
        ball_active = True
        hit_counted = False
        next_serve_step = None
        stats["serves"] += 1

    def schedule_next_serve() -> None:
        nonlocal next_serve_step
        if trajectory is not None or static_ball is not None or args.no_serve:
            return
        next_serve_step = next_serve_step_after_inactive(
            stats["control_steps"],
            args.serve_pause_steps,
        )

    def park_ball() -> None:
        nonlocal ball_active, hit_counted
        set_ball_state(model, data, PARKED_BALL_POS, PARKED_BALL_VEL)
        io.reset_ball_history()
        gate.reset()
        ball_active = False
        hit_counted = False
        schedule_next_serve()

    def set_static_ball() -> None:
        nonlocal ball_active, hit_counted
        set_ball_state(model, data, static_ball, PARKED_BALL_VEL)
        ball_active = True
        hit_counted = False

    def sync_trajectory_ball() -> None:
        nonlocal ball_active, hit_counted, trajectory_cycle, trajectory_source_ball, trajectory_phase_s, trajectory_replay_active
        if trajectory is None:
            return
        sim_time = stats["steps"] * PHYSICS_DT
        phase = sim_time - float(args.trajectory_start_delay_s)
        if phase < 0.0:
            trajectory_phase_s = phase
            trajectory_replay_active = False
            if ball_active:
                park_ball()
            else:
                set_ball_state(model, data, PARKED_BALL_POS, PARKED_BALL_VEL)
            return
        if args.trajectory_loop:
            cycle = int(phase // max(trajectory_interval, trajectory.duration + 1e-6))
            replay_t = phase - cycle * trajectory_interval
        else:
            cycle = 0
            replay_t = phase
        if cycle != trajectory_cycle:
            trajectory_cycle = cycle
            trajectory_source_ball = -1
            io.reset_ball_history()
            gate.reset()
            hit_counted = False
        trajectory_phase_s = replay_t
        if 0.0 <= replay_t <= trajectory.duration:
            pos, vel, source_ball = trajectory.sample(replay_t)
            if source_ball != trajectory_source_ball:
                trajectory_source_ball = source_ball
                io.reset_ball_history()
                gate.reset()
                hit_counted = False
            if source_ball <= 0:
                set_ball_state(model, data, PARKED_BALL_POS, PARKED_BALL_VEL)
                ball_active = False
                trajectory_replay_active = False
                return
            set_ball_state(model, data, pos, vel)
            ball_active = True
            trajectory_replay_active = True
        else:
            trajectory_replay_active = False
            if ball_active:
                park_ball()
            else:
                set_ball_state(model, data, PARKED_BALL_POS, PARKED_BALL_VEL)

    if trajectory is not None:
        park_ball()
        sync_trajectory_ball()
    elif static_ball is not None:
        set_static_ball()
    elif args.no_serve:
        park_ball()
    else:
        start_serve()
    if args.initial_state_csv is not None:
        io.reset_policy_history(
            data.xpos[ball_bid].copy(),
            data.qvel[ball_vadr : ball_vadr + 3].copy(),
            valid_ball=False,
        )

    def control_tick() -> BallGateOutput:
        stats["control_steps"] += 1
        scheduled_serve_due = next_serve_step is not None and stats["control_steps"] >= next_serve_step
        periodic_serve_due = next_serve_step is None and serve.due(stats["control_steps"])
        if (
            trajectory is None
            and not args.no_serve
            and static_ball is None
            and not ball_active
            and (scheduled_serve_due or periodic_serve_due)
        ):
            start_serve()
        ball_pos = data.xpos[ball_bid].copy()
        ball_vel = data.qvel[ball_vadr : ball_vadr + 3].copy()
        gate_out = gate.update(ball_pos, ball_vel)
        if gate_out.engaged:
            stats["valid_frames"] += 1
        obs = io.observe(ball_pos, ball_vel, valid_ball=gate_out.engaged)
        action = action_for_gate(
            policy,
            obs,
            gate_out,
            invalid_ball_action=args.invalid_ball_action,
        )
        prev_q_des = io.q_des.copy()
        apply_qdes_limit = args.bridge_qdes_limit or args.real_response_model
        max_delta = max_delta_per_tick if apply_qdes_limit else None
        io.update_action(
            action,
            max_delta_per_tick=max_delta,
            update_motor_target=response_model is None,
        )
        raw_qdes_delta = io.raw_q_des - prev_q_des
        limited_qdes_delta = io.q_des - prev_q_des
        stats["max_abs_action"] = max(stats["max_abs_action"], float(np.max(np.abs(action))))
        stats["max_abs_raw_qdes_delta"] = max(
            stats["max_abs_raw_qdes_delta"],
            float(np.max(np.abs(raw_qdes_delta))),
        )
        stats["max_abs_limited_qdes_delta"] = max(
            stats["max_abs_limited_qdes_delta"],
            float(np.max(np.abs(limited_qdes_delta))),
        )
        if args.diag_every > 0 and stats["control_steps"] % args.diag_every == 0:
            paddle = io.paddle_touch_point()
            print(
                "[a1_sim2sim_diag] "
                f"step={stats['control_steps']} "
                f"action={np.round(action, 3).tolist()} "
                f"q={np.round(io.right_q(), 3).tolist()} "
                f"q_des={np.round(io.q_des, 3).tolist()} "
                f"motor_q_des={np.round(io.motor_q_des, 3).tolist()} "
                f"dq={np.round(io.right_dq(), 3).tolist()} "
                f"ball={np.round(ball_pos, 3).tolist()} "
                f"hit={np.round(io.last_ball_pred, 3).tolist()} "
                f"paddle={np.round(paddle, 3).tolist()} "
                f"dx_paddle_hit={float(paddle[0] - io.last_ball_pred[0]):.3f} "
                f"last_hit={stats['last_hit_source']}@{stats['last_hit_step']} "
                f"ball_x={float(stats['last_hit_ball_x']):.3f} "
                f"paddle_x={float(stats['last_hit_paddle_x']):.3f} "
                f"contact_x={float(stats['last_hit_contact_x']):.3f} "
                f"gate={int(gate_out.engaged)}:{gate_out.reason}"
            )
        if args.trace_csv is not None:
            computed_tau = io.estimate_mit_tau_raw()
            tau = np.clip(computed_tau, -effort_limit(), effort_limit())
            row: dict[str, float | str] = {
                "step": float(stats["control_steps"]),
                "source": f"mujoco_{args.actuator_mode}",
                "obs_max_abs": float(np.max(np.abs(obs))),
                "valid_ball": float(gate_out.engaged),
                "gate_live": float(gate_out.live),
                "gate_reason": gate_out.reason,
                "gate_first_bounce_x": float(gate_out.first_bounce_x),
                "gate_own_bounces": float(gate_out.own_bounces),
                "gate_hit_seen": float(gate_out.hit_seen),
                "hits": float(stats["hits"]),
                "serves": float(stats["serves"]),
                "bridge_qdes_limit": float(args.bridge_qdes_limit),
                "real_response_model": float(args.real_response_model),
                "sim_time_s": float(stats["steps"] * PHYSICS_DT),
                "trajectory_active": float(trajectory_replay_active),
                "trajectory_cycle": float(trajectory_cycle),
                "trajectory_source_ball": float(trajectory_source_ball),
                "trajectory_phase_s": float(trajectory_phase_s),
                "trajectory_file": "" if trajectory is None else str(trajectory.path),
                "initial_state_time_s": float(initial_state_time_s),
            }
            for i, value in enumerate(action, 1):
                row[f"action_{i}"] = float(value)
            for i, value in enumerate(io.right_q(), 1):
                row[f"q_{i}"] = float(value)
                row[f"q_rel_{i}"] = float(value - DEFAULT_RIGHT_Q[i - 1])
            for i, value in enumerate(io.raw_q_des, 1):
                row[f"raw_q_des_{i}"] = float(value)
            for i, value in enumerate(io.q_des, 1):
                row[f"q_des_{i}"] = float(value)
            for i, value in enumerate(io.motor_q_des, 1):
                row[f"motor_q_des_{i}"] = float(value)
            for i, value in enumerate(raw_qdes_delta, 1):
                row[f"raw_q_des_delta_{i}"] = float(value)
            for i, value in enumerate(limited_qdes_delta, 1):
                row[f"q_des_delta_{i}"] = float(value)
            for i, value in enumerate(max_delta_per_tick, 1):
                row[f"max_delta_per_tick_{i}"] = float(value)
            for i, value in enumerate(io.right_dq(), 1):
                row[f"dq_{i}"] = float(value)
            for i, value in enumerate(tau, 1):
                row[f"applied_tau_{i}"] = float(value)
                row[f"computed_tau_{i}"] = float(computed_tau[i - 1])
                row[f"effort_limit_{i}"] = float(effort_limit()[i - 1])
                row[f"vel_limit_{i}"] = float(np.nan if velocity_limit() is None else velocity_limit()[i - 1])
            for i, value in enumerate(ball_pos, 1):
                row[f"ball_{i}"] = float(value)
            for i, value in enumerate(ball_vel, 1):
                row[f"ball_vel_{i}"] = float(value)
            for i, value in enumerate(io.last_ball_pred, 1):
                row[f"ball_pred_{i}"] = float(value)
            paddle = io.paddle_touch_point()
            for i, value in enumerate(paddle, 1):
                row[f"paddle_{i}"] = float(value)
            append_trace(row)
        return gate_out

    def physics_step() -> None:
        nonlocal hit_counted
        data.qfrc_applied[:] = 0.0
        if trajectory is not None:
            sync_trajectory_ball()
        if static_ball is not None:
            set_static_ball()
        if not ball_active:
            set_ball_state(model, data, PARKED_BALL_POS, PARKED_BALL_VEL)
        io.enforce_hold_joints()
        io.enforce_motor_limits(velocity_limit())
        prev_motor_q_des = io.motor_q_des.copy()
        if response_model is not None:
            io.set_motor_q_des(response_model.step(io.q_des, PHYSICS_DT))
        else:
            io.set_motor_q_des(io.q_des)
        motor_delta = io.motor_q_des - prev_motor_q_des
        stats["max_abs_motor_qdes_delta"] = max(
            stats["max_abs_motor_qdes_delta"],
            float(np.max(np.abs(motor_delta))),
        )
        stats["max_abs_motor_qdes_speed"] = max(
            stats["max_abs_motor_qdes_speed"],
            float(np.max(np.abs(motor_delta)) / PHYSICS_DT),
        )
        if args.actuator_mode == "damiao_mit":
            desired_vel = response_model.response_velocity if response_model is not None else None
            tau = io.apply_damiao_mit(PHYSICS_DT, desired_vel=desired_vel)
        elif args.actuator_mode in ("torque_chain", "real_deploy_preview"):
            tau = io.apply_mit_pd(effort_limit())
        elif args.actuator_mode == "direct_response":
            tau = io.estimate_mit_tau(effort_limit())
            io.apply_direct_response(PHYSICS_DT)
            mujoco.mj_forward(model, data)
        else:
            tau = io.estimate_mit_tau(effort_limit())
            servo_vel = VEL_LIMIT if args.fast_servo else ISAAC_PLAY_SERVO_VEL * args.servo_vel_scale
            io.apply_position_servo(PHYSICS_DT, servo_vel, args.servo_tau)
            mujoco.mj_forward(model, data)
        stats["max_abs_tau"] = max(stats["max_abs_tau"], float(np.max(np.abs(tau))))
        stats["max_abs_qvel"] = max(stats["max_abs_qvel"], float(np.max(np.abs(io.right_dq()))))
        if ball_active and trajectory is None:
            _apply_ball_drag(model, data, ball_vadr)
            manual_hit = _maybe_manual_hit(io, data.qpos[ball_qadr : ball_qadr + 3], data.qvel[ball_vadr : ball_vadr + 3])
            if manual_hit:
                gate.register_paddle_hit()
                if not hit_counted:
                    stats["hits"] += 1
                    hit_counted = True
                stats["last_hit_step"] = stats["control_steps"]
                stats["last_hit_source"] = "manual"
                stats["last_hit_ball_x"] = float(data.qpos[ball_qadr])
                stats["last_hit_paddle_x"] = float(io.paddle_touch_point()[0])
                stats["last_hit_contact_x"] = float("nan")
        mujoco.mj_step(model, data)
        if trajectory is not None:
            sync_trajectory_ball()
            mujoco.mj_forward(model, data)
        if static_ball is not None:
            set_static_ball()
            mujoco.mj_forward(model, data)
        contact_pos = _ball_paddle_contact_pos(model, data) if ball_active else None
        if contact_pos is not None:
            gate.register_paddle_hit()
            if not hit_counted:
                stats["hits"] += 1
                hit_counted = True
            stats["last_hit_step"] = stats["control_steps"]
            stats["last_hit_source"] = "contact"
            stats["last_hit_ball_x"] = float(data.xpos[ball_bid][0])
            stats["last_hit_paddle_x"] = float(io.paddle_touch_point()[0])
            stats["last_hit_contact_x"] = float(contact_pos[0])
        if trajectory is None and ball_active and _dead_ball(data.xpos[ball_bid], data.qvel[ball_vadr : ball_vadr + 3]):
            if static_ball is not None:
                set_static_ball()
            else:
                park_ball()
        if args.actuator_mode == "isaac_approx":
            io.apply_position_servo(0.0)
            mujoco.mj_forward(model, data)
        elif args.actuator_mode == "direct_response":
            io.apply_direct_response(0.0)
            mujoco.mj_forward(model, data)
        io.enforce_motor_limits(velocity_limit())
        io.enforce_hold_joints()
        stats["steps"] += 1

    if args.headless_steps > 0:
        for k in range(args.headless_steps * DECIMATION):
            physics_step()
            if (k + 1) % DECIMATION == 0:
                control_tick()
        print(
            "[a1_sim2sim] headless "
            f"profile={ACTIVE_PROFILE.name} "
            f"control_steps={stats['control_steps']} serves={stats['serves']} hits={stats['hits']} "
            f"valid_frames={stats['valid_frames']} "
            f"max_abs_action={stats['max_abs_action']:.2f} "
            f"max_raw_qdes_delta={stats['max_abs_raw_qdes_delta']:.3f} "
            f"max_limited_qdes_delta={stats['max_abs_limited_qdes_delta']:.3f} "
            f"max_motor_qdes_delta={stats['max_abs_motor_qdes_delta']:.4f} "
            f"max_motor_qdes_speed={stats['max_abs_motor_qdes_speed']:.2f} "
            f"max_abs_qvel={stats['max_abs_qvel']:.2f} "
            f"max_abs_tau={stats['max_abs_tau']:.2f}/{float(np.max(EFFORT)):.1f} "
            f"ball={np.round(data.xpos[ball_bid], 3).tolist()}"
        )
        if args.trace_csv is not None:
            close_trace()
            print(f"[a1_sim2sim] wrote trace {args.trace_csv}")
        return stats

    import mujoco.viewer as mj_viewer

    print("[a1_sim2sim] viewer running. Ctrl-C or close the window to stop.")
    print(f"[a1_sim2sim] actuator_mode={args.actuator_mode}")
    print(f"[a1_sim2sim] real_response_model={args.real_response_model} max_delta_per_tick={np.round(max_delta_per_tick, 3).tolist()}")
    if trajectory is not None:
        print(
            "[a1_sim2sim] replay trajectory "
            f"path={trajectory.path} duration={trajectory.duration:.3f}s "
            f"loop={args.trajectory_loop} interval={trajectory_interval:.3f}s"
        )
    if args.trace_csv is not None:
        print(f"[a1_sim2sim] live trace: {args.trace_csv}")
    print(f"[a1_sim2sim] profile={ACTIVE_PROFILE.name}")
    print(f"[a1_sim2sim] scene: {xml_path}")
    try:
        with mj_viewer.launch_passive(model, data) as viewer:
            next_time = time.perf_counter()
            k = 0
            while viewer.is_running():
                step_start = time.perf_counter()
                physics_step()
                k += 1
                if k % DECIMATION == 0:
                    gate_out = control_tick()
                    if not args.no_hit_viz:
                        with viewer.lock():
                            _update_hit_point_visuals(
                                viewer.user_scn,
                                data.xpos[ball_bid].copy(),
                                io.last_ball_pred.copy(),
                                gate_out,
                            )
                    viewer.sync()
                next_time += PHYSICS_DT
                sleep_s = next_time - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif time.perf_counter() - step_start < PHYSICS_DT * 0.2:
                    time.sleep(PHYSICS_DT * 0.2)
    finally:
        if args.trace_csv is not None:
            close_trace()
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--headless-steps", type=int, default=0, help="Run N 50Hz control steps without viewer.")
    ap.add_argument("--serve-interval", type=int, default=250, help="Serve period in 50Hz control steps.")
    ap.add_argument(
        "--serve-pause-steps",
        type=int,
        default=1,
        help="Control-step pause after a dead/parked ball before the next serve. Default 1 matches training-style immediate ball reset.",
    )
    ap.add_argument("--no-serve", action="store_true", help="Keep the ball parked and never start serves.")
    ap.add_argument(
        "--static-ball",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Hold a zero-velocity ball at this position for gate/no-ball diagnostics.",
    )
    ap.add_argument("--diag-every", type=int, default=0, help="Print policy/control diagnostics every N control steps.")
    ap.add_argument("--trace-csv", type=Path, default=None, help="Write per-control-step diagnostics to CSV.")
    ap.add_argument("--trace-flush-every", type=int, default=10, help="Flush trace CSV every N written control rows.")
    ap.add_argument("--ball-trajectory-csv", type=Path, default=None, help="Replay a recorded ball trajectory CSV instead of sampling serves.")
    ap.add_argument("--trajectory-loop", action=argparse.BooleanOptionalAction, default=True, help="Loop --ball-trajectory-csv in viewer/headless runs.")
    ap.add_argument("--trajectory-loop-interval", type=float, default=0.0, help="Loop period in seconds. Default uses trajectory duration plus pause.")
    ap.add_argument("--trajectory-pause-s", type=float, default=1.0, help="Pause between trajectory loops when --trajectory-loop-interval is omitted.")
    ap.add_argument("--trajectory-start-delay-s", type=float, default=0.5, help="Delay before the first trajectory replay starts.")
    ap.add_argument("--seed", type=int, default=None, help="Seed for the MuJoCo serve sampler.")
    ap.add_argument("--initial-state-csv", type=Path, default=None, help="CSV containing q1..q7 and optional dq1..dq7 for initial robot state.")
    ap.add_argument("--initial-state-time-s", type=float, default=None, help="Interpolate --initial-state-csv at this time_s. Default uses the first row.")
    ap.add_argument("--servo-vel-scale", type=float, default=1.0, help="Scale the Isaac-play-calibrated visual servo velocity cap.")
    ap.add_argument("--servo-tau", type=float, default=0.25, help="First-order visual servo time constant in seconds.")
    ap.add_argument("--fast-servo", action="store_true", help="Use the training hard velocity limits for the visual servo.")
    ap.add_argument("--bridge-qdes-limit", action="store_true", help="Apply the sim2real bridge max_delta_per_tick limit to policy q_des.")
    ap.add_argument(
        "--invalid-ball-action",
        choices=["zero", "policy"],
        default="zero",
        help=(
            "Action source when the ball gate is invalid. 'zero' sends the ready-pose action "
            "instead of letting no-ball/sentinel observations trigger extra swings; 'policy' "
            "keeps the legacy raw-policy behavior."
        ),
    )
    ap.add_argument(
        "--real-response-model",
        action="store_true",
        help="Apply the A1 real-training q_des slew limiter and fitted second-order joint response.",
    )
    ap.add_argument(
        "--max-delta-per-tick",
        type=float,
        nargs=7,
        default=None,
        metavar=("R1", "R2", "R3", "R4", "R5", "R6", "R7"),
        help=(
            "Per-50Hz-tick q_des delta limit. If omitted, --real-response-model "
            "uses the 2.5/5 rad/s training envelope; --bridge-qdes-limit uses the legacy bridge envelope."
        ),
    )
    ap.add_argument(
        "--actuator-mode",
        choices=["isaac_approx", "direct_response", "torque_chain", "real_deploy_preview", "damiao_mit"],
        default="isaac_approx",
        help="Actuator model: visual Isaac approximation, direct second-order response, training torque chain, or DAMIAO real-deploy preview.",
    )
    ap.add_argument("--dynamic-pd", action="store_true", help="Use explicit MIT-PD torques instead of the stable position-servo sim2sim actuator.")
    ap.add_argument("--real-deploy-preview", dest="actuator_mode", action="store_const", const="real_deploy_preview", help="Alias for --actuator-mode real_deploy_preview.")
    ap.add_argument("--no-qvel-clip", action="store_true", help="Disable MuJoCo qvel clipping in torque modes. Diagnostic only.")
    ap.add_argument("--no-policy", action="store_true", help="Use zero raw action; useful for scene smoke tests.")
    ap.add_argument("--gate-confirm-frames", type=int, default=1, help="Consecutive live 50Hz frames required before ball is valid.")
    ap.add_argument("--gate-coast-frames", type=int, default=1, help="Consecutive invalid 50Hz frames before valid gate drops.")
    ap.add_argument("--no-hit-viz", action="store_true", help="Disable viewer debug marker for the predicted hit point.")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

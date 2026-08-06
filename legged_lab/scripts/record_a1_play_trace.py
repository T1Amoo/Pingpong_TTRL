"""Record finite-step A1 Isaac play traces for MuJoCo sim2sim comparison."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record A1 TT play trace.")
parser.add_argument("--task", type=str, default="a1_tt_eval")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--load_run", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--out", type=Path, default=Path("/tmp/a1_play_trace.csv"))
parser.add_argument("--predictor", action="store_true")
parser.add_argument("--no_noise", action="store_true")
parser.add_argument("--ball-trajectory-csv", type=Path, default=None)
parser.add_argument("--trajectory-start-delay-s", type=float, default=0.0)
parser.add_argument("--trajectory-loop", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--initial-state-csv", type=Path, default=None)
parser.add_argument("--initial-state-time-s", type=float, default=None)
parser.add_argument("--policy-obs-csv", type=Path, default=None, help="Use obs_N columns from this CSV as the actor input for policy inference.")
parser.add_argument("--raw-action-csv", type=Path, default=None, help="Use raw_action_N columns from this CSV instead of running policy inference.")
parser.add_argument("--trace-obs", action="store_true", help="Include the 195-D policy input as obs_N columns in the output CSV.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab_tasks.utils import get_checkpoint_path
try:
    from rsl_rl.rsl_rl.runners import OnPolicyRunner, OnPolicyPredictorRegressionRunner
except ImportError:
    from rsl_rl.runners import OnPolicyRunner, OnPolicyPredictorRegressionRunner

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils import task_registry  # noqa: E402


JOINTS = [f"r{i}" for i in range(1, 8)]


def _cpu_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _row(prefix: str, values) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {f"{prefix}{i + 1}": float(v) for i, v in enumerate(arr)}


def _env0(value, device, *, dtype=None):
    if value is None:
        value = torch.zeros((), device=device, dtype=dtype or torch.float32)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        return value[0]
    return value


def _load_initial_joint_state(path: Path, time_s: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    path = Path(path)
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


def _set_robot_initial_state(env, q: np.ndarray, dq: np.ndarray) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    action_ids = list(env.action_joint_ids)
    q_t = torch.as_tensor(q, device=env.device, dtype=env.robot.data.joint_pos.dtype).reshape(1, 7)
    dq_t = torch.as_tensor(dq, device=env.device, dtype=env.robot.data.joint_vel.dtype).reshape(1, 7)
    q_all = env.robot.data.joint_pos.clone()
    dq_all = env.robot.data.joint_vel.clone()
    q_all[:, action_ids] = q_t
    dq_all[:, action_ids] = dq_t
    env.robot.write_joint_state_to_sim(q_all, dq_all)
    env.robot.set_joint_position_target(q_all)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)
    if hasattr(env, "_reset_action_target_limiter"):
        env._reset_action_target_limiter(env_ids)
    if hasattr(env, "_reset_action_response_model"):
        env._reset_action_response_model(env_ids)
    q_action = env.robot.data.joint_pos[:, action_ids].clone()
    env.processed_actions = q_action
    if hasattr(env, "action_response_targets"):
        env.action_response_targets = q_action.clone()
    try:
        env.action_buffer.reset(env_ids)
    except Exception:
        pass


class BallTrajectoryReplay:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        rows: list[tuple[float, list[float], list[float]]] = []
        with self.path.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"segment_time_s", "x", "y", "z", "vx", "vy", "vz"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.path} missing columns: {sorted(missing)}")
            for row in reader:
                try:
                    t = float(row["segment_time_s"])
                    pos = [float(row["x"]), float(row["y"]), float(row["z"])]
                    vel = [float(row["vx"]), float(row["vy"]), float(row["vz"])]
                except (TypeError, ValueError):
                    continue
                values = np.asarray([t, *pos, *vel], dtype=np.float64)
                if not np.isfinite(values).all():
                    continue
                if rows and t <= rows[-1][0]:
                    t = rows[-1][0] + 1.0e-6
                rows.append((t, pos, vel))
        if len(rows) < 2:
            raise ValueError(f"{self.path} must contain at least two finite trajectory rows")
        self.time = np.asarray([r[0] for r in rows], dtype=np.float64)
        self.pos = np.asarray([r[1] for r in rows], dtype=np.float64)
        self.vel = np.asarray([r[2] for r in rows], dtype=np.float64)
        self.duration = float(self.time[-1])

    def sample(self, phase_s: float) -> tuple[np.ndarray, np.ndarray, bool]:
        active = 0.0 <= phase_s <= self.duration
        t = float(np.clip(phase_s, self.time[0], self.time[-1]))
        pos = np.asarray([np.interp(t, self.time, self.pos[:, i]) for i in range(3)], dtype=np.float64)
        vel = np.asarray([np.interp(t, self.time, self.vel[:, i]) for i in range(3)], dtype=np.float64)
        return pos, vel, active


class StepVectorReplay:
    """Per-control-tick vector replay from a CSV with prefix_0 or prefix_1 columns."""

    def __init__(self, path: Path, prefix: str, width: int | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with self.path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"{self.path} is empty")
        names = rows[0].keys()
        zero_cols = [f"{prefix}_{i}" for i in range(width or 0)]
        one_cols = [f"{prefix}_{i}" for i in range(1, (width or 0) + 1)]
        if width is None:
            indexed: list[tuple[int, str]] = []
            for name in names:
                if not name.startswith(prefix + "_"):
                    continue
                suffix = name[len(prefix) + 1 :]
                if suffix.isdigit():
                    indexed.append((int(suffix), name))
            if not indexed:
                raise ValueError(f"{self.path} has no {prefix}_N columns")
            indexed.sort()
            self.columns = [name for _, name in indexed]
        elif all(col in names for col in zero_cols):
            self.columns = zero_cols
        elif all(col in names for col in one_cols):
            self.columns = one_cols
        else:
            raise ValueError(f"{self.path} missing {prefix} columns width={width}")
        values = []
        for row in rows:
            values.append([float(row[col]) for col in self.columns])
        self.values = np.asarray(values, dtype=np.float32)

    def get(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, len(self.values) - 1))
        return self.values[idx].copy()


def _trajectory_phase(step: int, dt: float, replay: BallTrajectoryReplay) -> tuple[float, int, bool]:
    phase = float(step) * float(dt) - float(args.trajectory_start_delay_s)
    if args.trajectory_loop:
        period = max(replay.duration, 1.0e-6)
        cycle = int(np.floor(max(phase, 0.0) / period)) if phase >= 0.0 else -1
        phase = phase - cycle * period if phase >= 0.0 else phase
    else:
        cycle = 0 if phase >= 0.0 else -1
    active = 0.0 <= phase <= replay.duration
    return phase, cycle, active


def _set_ball_state(env, pos: np.ndarray, vel: np.ndarray) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    pos_t = torch.as_tensor(pos, device=env.device, dtype=env.ball.data.root_pos_w.dtype).reshape(1, 3)
    vel_t = torch.as_tensor(vel, device=env.device, dtype=env.ball.data.root_lin_vel_w.dtype).reshape(1, 3)
    pose = torch.zeros((env.num_envs, 7), device=env.device, dtype=env.ball.data.root_pos_w.dtype)
    pose[:, :3] = env.scene.env_origins + pos_t
    pose[:, 3] = 1.0
    root_vel = torch.zeros((env.num_envs, 6), device=env.device, dtype=env.ball.data.root_lin_vel_w.dtype)
    root_vel[:, :3] = vel_t
    env.ball.write_root_pose_to_sim(pose, env_ids)
    env.ball.write_root_velocity_to_sim(root_vel, env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)


def _sync_forced_ball_observation(env, *, fill_history: bool = False) -> torch.Tensor:
    env.compute_perception()
    env.compute_paddle_touch()
    env.compute_intermediate_values()
    env.delayed_perception = env.current_perception.clone()
    current_actor_obs, current_critic_obs = env.compute_current_observations_perception()
    env.current_actor_obs = current_actor_obs
    if env.add_noise:
        current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * env.noise_scale_vec
    if fill_history:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        env.actor_obs_buffer.reset(env_ids)
        env.critic_obs_buffer.reset(env_ids)
        for _ in range(env.actor_obs_buffer.max_length):
            env.actor_obs_buffer.append(current_actor_obs)
        for _ in range(env.critic_obs_buffer.max_length):
            env.critic_obs_buffer.append(current_critic_obs)
    else:
        if env.actor_obs_buffer._buffer is None:
            env.actor_obs_buffer.append(current_actor_obs)
        else:
            env.actor_obs_buffer._buffer[env.actor_obs_buffer._pointer] = current_actor_obs
        if env.critic_obs_buffer._buffer is None:
            env.critic_obs_buffer.append(current_critic_obs)
        else:
            env.critic_obs_buffer._buffer[env.critic_obs_buffer._pointer] = current_critic_obs
    actor_obs = env.actor_obs_buffer.buffer.reshape(env.num_envs, -1)
    critic_obs = env.critic_obs_buffer.buffer.reshape(env.num_envs, -1)
    actor_obs = torch.clip(actor_obs, -env.clip_obs, env.clip_obs)
    critic_obs = torch.clip(critic_obs, -env.clip_obs, env.clip_obs)
    env.extras["observations"] = {"critic": critic_obs}
    return actor_obs


def _apply_replay_to_env(
    env,
    replay: BallTrajectoryReplay,
    step: int,
    *,
    fill_history: bool = False,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    phase_s, cycle, active = _trajectory_phase(step, env.step_dt, replay)
    if active:
        pos, vel, _ = replay.sample(phase_s)
    else:
        pos = np.array([1.75, 1.35, 0.20], dtype=np.float64)
        vel = np.zeros(3, dtype=np.float64)
    _set_ball_state(env, pos, vel)
    if not active:
        env.mask_invalid[:] = True
        try:
            env._reset_prediction_buffers()
        except Exception:
            pass
    obs = _sync_forced_ball_observation(env, fill_history=fill_history)
    meta: dict[str, float | str] = {
        "trajectory_active": float(active),
        "trajectory_cycle": float(cycle),
        "trajectory_phase_s": float(phase_s),
        "trajectory_file": str(replay.path),
    }
    for i, value in enumerate(pos, 1):
        meta[f"replay_ball_{i}"] = float(value)
    for i, value in enumerate(vel, 1):
        meta[f"replay_ball_vel_{i}"] = float(value)
    return obs, meta


def main() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    agent_cfg.load_run = args.load_run
    agent_cfg.load_checkpoint = args.checkpoint

    env_cfg.scene.num_envs = 1
    env_cfg.noise.add_noise = not args.no_noise
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.domain_rand.perception_delay.enable = False

    env = task_registry.get_task_class(args.task)(env_cfg, headless=True)
    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume = get_checkpoint_path(log_root, args.load_run, args.checkpoint)
    runner_cls = OnPolicyPredictorRegressionRunner if args.predictor else OnPolicyRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=os.path.dirname(resume), device=agent_cfg.device)
    runner.load(resume, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.get_observations()
    replay = BallTrajectoryReplay(args.ball_trajectory_csv) if args.ball_trajectory_csv is not None else None
    policy_obs_replay = StepVectorReplay(args.policy_obs_csv, "obs") if args.policy_obs_csv is not None else None
    raw_action_replay = (
        StepVectorReplay(args.raw_action_csv, "raw_action", width=7)
        if args.raw_action_csv is not None
        else None
    )
    replay_meta: dict[str, float | str] = {}
    run_meta: dict[str, float | str] = {}
    if args.initial_state_csv is not None:
        initial_q, initial_dq, initial_time_s = _load_initial_joint_state(
            args.initial_state_csv,
            args.initial_state_time_s,
        )
        _set_robot_initial_state(env, initial_q, initial_dq)
        run_meta.update(
            {
                "initial_state_file": str(args.initial_state_csv),
                "initial_state_time_s": float(initial_time_s),
            }
        )
        run_meta.update(_row("initial_q_", initial_q))
        run_meta.update(_row("initial_dq_", initial_dq))
        print(
            "[a1_play_trace] initial_state "
            f"path={args.initial_state_csv} time_s={initial_time_s:.6f} "
            f"q={np.round(initial_q, 4).tolist()} dq={np.round(initial_dq, 4).tolist()}",
            flush=True,
        )
    if replay is not None:
        obs, replay_meta = _apply_replay_to_env(env, replay, 0, fill_history=args.initial_state_csv is not None)
    elif args.initial_state_csv is not None:
        obs = _sync_forced_ball_observation(env, fill_history=True)
    rows: list[dict[str, float]] = []
    action_ids = list(env.action_joint_ids)
    default_q = env.robot.data.default_joint_pos[0, action_ids]

    with torch.inference_mode():
        for step in range(args.steps):
            if replay is not None:
                obs, replay_meta = _apply_replay_to_env(env, replay, step)
            external_obs_max_abs = float("nan")
            if raw_action_replay is not None:
                action_np = raw_action_replay.get(step)
                action = torch.as_tensor(action_np, device=env.device, dtype=obs.dtype).reshape(1, -1)
                obs_in = obs[0].detach().cpu().numpy().copy()
                policy_input_source = "external_raw_action"
            else:
                if policy_obs_replay is not None:
                    obs_np = policy_obs_replay.get(step)
                    external_obs_max_abs = float(np.max(np.abs(obs_np)))
                    obs_for_policy = torch.as_tensor(obs_np, device=env.device, dtype=obs.dtype).reshape(1, -1)
                    policy_input_source = "external_policy_obs"
                else:
                    obs_for_policy = obs
                    policy_input_source = "sim_policy_obs"
                action = policy(obs_for_policy)
                obs_in = obs_for_policy[0].detach().cpu().numpy().copy()
            obs, _, _, _ = env.step(action)
            if replay is not None:
                obs, replay_meta = _apply_replay_to_env(env, replay, step + 1)
            if args.predictor:
                try:
                    runner._record_ball_positions()
                    runner._maybe_predict_and_update_env()
                    if replay is not None:
                        obs = _sync_forced_ball_observation(env)
                except Exception:
                    pass

            robot = env.robot.data
            q = robot.joint_pos[0, action_ids]
            dq = robot.joint_vel[0, action_ids]
            q_des = getattr(env, "processed_actions", default_q.unsqueeze(0))[0]
            motor_q_des = getattr(env, "action_response_targets", q_des.unsqueeze(0))[0]
            applied_tau = robot.applied_torque[0, action_ids]
            computed_tau = robot.computed_torque[0, action_ids]
            effort_limit = robot.joint_effort_limits[0, action_ids]
            vel_limit = robot.joint_vel_limits[0, action_ids]
            ball_pos = getattr(env, "ball_pos", torch.zeros((1, 3), device=env.device))[0]
            ball_pred = getattr(env, "ball_prediction", torch.zeros((1, 3), device=env.device))[0]
            ball_future = getattr(env, "ball_future_pose", torch.zeros((1, 3), device=env.device))[0]
            paddle_pos = getattr(env, "paddle_pos", torch.zeros((1, 3), device=env.device))[0]
            paddle_touch = getattr(env, "paddle_touch_point", torch.zeros((1, 3), device=env.device))[0]
            paddle_touch_local = paddle_touch - env.scene.env_origins[0]
            paddle_vel = getattr(env, "paddle_touch_point_vel", torch.zeros((1, 3), device=env.device))[0]
            paddle_dist = _env0(getattr(env, "paddel_ball_distance", None), env.device)
            mask_invalid = _env0(getattr(env, "mask_invalid", None), env.device, dtype=torch.bool)
            mask_before = _env0(getattr(env, "mask_before", None), env.device, dtype=torch.bool)
            mask_after = _env0(getattr(env, "mask_after", None), env.device, dtype=torch.bool)
            has_touch = _env0(getattr(env, "has_touch_paddle", None), env.device, dtype=torch.bool)
            active_hit = _env0(getattr(env, "active_paddle_hit", None), env.device, dtype=torch.bool)
            first_bounce = _env0(getattr(env, "has_touch_own_table_prev", None), env.device, dtype=torch.bool)
            ball_contact = _env0(getattr(env, "ball_contact", None), env.device)
            ball_contact_rew = _env0(getattr(env, "ball_contact_rew", None), env.device)
            ball_contact_raw = _env0(getattr(env, "ball_contact_raw_rew", None), env.device)
            table_success_event = _env0(
                env.get_opponent_table_success_event()
                if hasattr(env, "get_opponent_table_success_event")
                else None,
                env.device,
                dtype=torch.bool,
            )
            geometric_table_band = _env0(
                getattr(env, "has_touch_opponent_table_just_now", None),
                env.device,
                dtype=torch.bool,
            )
            reset_ids = getattr(env, "ball_reset_ids", torch.empty(0, dtype=torch.long, device=env.device))

            row = {
                "step": float(step),
                "source": "isaac_play_replay" if replay is not None else "isaac_play",
                "obs_max_abs": float(np.max(np.abs(obs_in))),
                "policy_input_source": policy_input_source,
                "policy_input_idx": float(step),
                "external_obs_max_abs": external_obs_max_abs,
                "mask_invalid": float(bool(mask_invalid)),
                "mask_before": float(bool(mask_before)),
                "mask_after": float(bool(mask_after)),
                "has_first_bounce_prev": float(bool(first_bounce)),
                "has_touch_paddle": float(bool(has_touch)),
                "active_paddle_hit": float(bool(active_hit)),
                "ball_reset": float(reset_ids.numel() > 0),
                "paddle_ball_dist": float(_cpu_np(paddle_dist)),
                "ball_contact": float(_cpu_np(ball_contact)),
                "ball_contact_rew": float(_cpu_np(ball_contact_rew)),
                "ball_contact_raw_rew": float(_cpu_np(ball_contact_raw)),
                "table_success_event": float(bool(table_success_event)),
                "geometric_table_band": float(bool(geometric_table_band)),
            }
            row.update(run_meta)
            row.update(replay_meta)
            row.update(_row("action_", _cpu_np(action[0])))
            row.update(_row("q_", _cpu_np(q)))
            row.update(_row("q_rel_", _cpu_np(q - default_q)))
            row.update(_row("q_des_", _cpu_np(q_des)))
            row.update(_row("motor_q_des_", _cpu_np(motor_q_des)))
            row.update(_row("dq_", _cpu_np(dq)))
            row.update(_row("applied_tau_", _cpu_np(applied_tau)))
            row.update(_row("computed_tau_", _cpu_np(computed_tau)))
            row.update(_row("effort_limit_", _cpu_np(effort_limit)))
            row.update(_row("vel_limit_", _cpu_np(vel_limit)))
            row.update(_row("ball_", _cpu_np(ball_pos)))
            row.update(_row("ball_pred_", _cpu_np(ball_pred)))
            row.update(_row("ball_future_", _cpu_np(ball_future)))
            if args.trace_obs:
                for i, value in enumerate(np.asarray(obs_in, dtype=np.float64).reshape(-1)):
                    row[f"obs_{i}"] = float(value)
            row.update(_row("paddle_", _cpu_np(paddle_pos)))
            row.update(_row("paddle_touch_", _cpu_np(paddle_touch_local)))
            row.update(_row("paddle_vel_", _cpu_np(paddle_vel)))
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    action_abs = np.asarray([[r[f"action_{i}"] for i in range(1, 8)] for r in rows])
    dq_abs = np.asarray([[r[f"dq_{i}"] for i in range(1, 8)] for r in rows])
    tau_abs = np.asarray([[r[f"applied_tau_{i}"] for i in range(1, 8)] for r in rows])
    print(
        "[a1_play_trace] "
        f"wrote={args.out} steps={len(rows)} "
        f"action_max={np.max(np.abs(action_abs)):.3f} action_p95={np.quantile(np.abs(action_abs), 0.95):.3f} "
        f"dq_max={np.max(np.abs(dq_abs)):.3f} dq_p95={np.quantile(np.abs(dq_abs), 0.95):.3f} "
        f"tau_max={np.max(np.abs(tau_abs)):.3f} tau_p95={np.quantile(np.abs(tau_abs), 0.95):.3f}",
        flush=True,
    )

    env.close()
    app.close()


if __name__ == "__main__":
    main()

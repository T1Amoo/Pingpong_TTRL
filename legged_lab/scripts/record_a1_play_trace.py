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


def main() -> None:
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    agent_cfg.load_run = args.load_run
    agent_cfg.load_checkpoint = args.checkpoint

    env_cfg.scene.num_envs = 1
    env_cfg.noise.add_noise = not args.no_noise
    env_cfg.domain_rand.events.push_robot = None

    env = task_registry.get_task_class(args.task)(env_cfg, headless=True)
    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume = get_checkpoint_path(log_root, args.load_run, args.checkpoint)
    runner_cls = OnPolicyPredictorRegressionRunner if args.predictor else OnPolicyRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=os.path.dirname(resume), device=agent_cfg.device)
    runner.load(resume, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.get_observations()
    rows: list[dict[str, float]] = []
    action_ids = list(env.action_joint_ids)
    default_q = env.robot.data.default_joint_pos[0, action_ids]

    with torch.inference_mode():
        for step in range(args.steps):
            action = policy(obs)
            obs_in = obs[0].detach().cpu().numpy().copy()
            obs, _, _, _ = env.step(action)
            if args.predictor:
                try:
                    runner._record_ball_positions()
                    runner._maybe_predict_and_update_env()
                except Exception:
                    pass

            robot = env.robot.data
            q = robot.joint_pos[0, action_ids]
            dq = robot.joint_vel[0, action_ids]
            q_des = getattr(env, "processed_actions", default_q.unsqueeze(0))[0]
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
            reset_ids = getattr(env, "ball_reset_ids", torch.empty(0, dtype=torch.long, device=env.device))

            row = {
                "step": float(step),
                "source": "isaac_play",
                "obs_max_abs": float(np.max(np.abs(obs_in))),
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
            }
            row.update(_row("action_", _cpu_np(action[0])))
            row.update(_row("q_", _cpu_np(q)))
            row.update(_row("q_rel_", _cpu_np(q - default_q)))
            row.update(_row("q_des_", _cpu_np(q_des)))
            row.update(_row("dq_", _cpu_np(dq)))
            row.update(_row("applied_tau_", _cpu_np(applied_tau)))
            row.update(_row("computed_tau_", _cpu_np(computed_tau)))
            row.update(_row("effort_limit_", _cpu_np(effort_limit)))
            row.update(_row("vel_limit_", _cpu_np(vel_limit)))
            row.update(_row("ball_", _cpu_np(ball_pos)))
            row.update(_row("ball_pred_", _cpu_np(ball_pred)))
            row.update(_row("ball_future_", _cpu_np(ball_future)))
            row.update(_row("paddle_", _cpu_np(paddle_pos)))
            row.update(_row("paddle_touch_", _cpu_np(paddle_touch_local)))
            row.update(_row("paddle_vel_", _cpu_np(paddle_vel)))
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
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

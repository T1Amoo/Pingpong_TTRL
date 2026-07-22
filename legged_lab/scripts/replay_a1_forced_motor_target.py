"""Replay an A1 right-arm motor target sequence through IsaacLab only.

This bypasses the policy and TTEnv action-response model.  Each 50 Hz row from
``--target-csv`` is written directly as the joint position target for every
physics substep, so the output isolates the implicit actuator tracking layer.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay forced A1 motor_q_des targets in IsaacLab.")
parser.add_argument("--task", type=str, default="a1_tt_real")
parser.add_argument("--target-csv", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--initial-state-csv", type=Path, default=None)
parser.add_argument("--initial-state-time-s", type=float, default=None)
parser.add_argument("--target-prefix", type=str, default="motor_q_des_")
parser.add_argument("--no-noise", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils import task_registry  # noqa: E402


def _row(prefix: str, values) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {f"{prefix}{i + 1}": float(v) for i, v in enumerate(arr)}


def _cpu_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _read_target_rows(path: Path, prefix: str, steps: int | None) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    required = [f"{prefix}{i}" for i in range(1, 8)]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} missing target columns: {missing}")
    if steps is not None:
        rows = rows[: int(steps)]
    return rows


def _target_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}{i}"]) for i in range(1, 8)], dtype=np.float64)


def _load_initial_joint_state(path: Path, time_s: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    q_cols = [f"q{i}" for i in range(1, 8)]
    dq_cols = [f"dq{i}" for i in range(1, 8)]
    required = {"time_s", *q_cols}
    missing = required - set(rows[0])
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
        env._reset_action_target_limiter(torch.arange(env.num_envs, device=env.device, dtype=torch.long))
    if hasattr(env, "_reset_action_response_model"):
        env._reset_action_response_model(torch.arange(env.num_envs, device=env.device, dtype=torch.long))


def main() -> None:
    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.scene.num_envs = 1
    env_cfg.noise.add_noise = False if args.no_noise else env_cfg.noise.add_noise
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.domain_rand.perception_delay.enable = False

    env = task_registry.get_task_class(args.task)(env_cfg, headless=True)
    action_ids = list(env.action_joint_ids)
    rows_in = _read_target_rows(args.target_csv, args.target_prefix, args.steps)

    initial_meta: dict[str, float | str] = {}
    if args.initial_state_csv is not None:
        q0, dq0, t0 = _load_initial_joint_state(args.initial_state_csv, args.initial_state_time_s)
        _set_robot_initial_state(env, q0, dq0)
        initial_meta.update(
            {
                "initial_state_file": str(args.initial_state_csv),
                "initial_state_time_s": float(t0),
            }
        )
        initial_meta.update(_row("initial_q_", q0))
        initial_meta.update(_row("initial_dq_", dq0))

    out_rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for local_step, src_row in enumerate(rows_in):
            target = _target_from_row(src_row, args.target_prefix)
            target_t = torch.as_tensor(target, device=env.device, dtype=env.robot.data.joint_pos.dtype).reshape(1, 7)
            for _ in range(env.cfg.sim.decimation):
                env.sim_step_counter += 1
                env.robot.set_joint_position_target(target_t, action_ids)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)

            robot = env.robot.data
            q = robot.joint_pos[0, action_ids]
            dq = robot.joint_vel[0, action_ids]
            applied_tau = robot.applied_torque[0, action_ids]
            computed_tau = robot.computed_torque[0, action_ids]
            effort_limit = robot.joint_effort_limits[0, action_ids]
            vel_limit = robot.joint_vel_limits[0, action_ids]
            row: dict[str, float | str] = {
                "step": float(src_row.get("step", local_step + 1)),
                "source": "isaac_forced_motor_target",
                "sim_time_s": float(src_row.get("sim_time_s", (local_step + 1) * env.step_dt)),
                "target_file": str(args.target_csv),
                "target_prefix": args.target_prefix,
            }
            if "trajectory_phase_s" in src_row:
                row["trajectory_phase_s"] = float(src_row["trajectory_phase_s"])
            row.update(initial_meta)
            row.update(_row("target_motor_q_des_", target))
            row.update(_row("q_", _cpu_np(q)))
            row.update(_row("dq_", _cpu_np(dq)))
            row.update(_row("applied_tau_", _cpu_np(applied_tau)))
            row.update(_row("computed_tau_", _cpu_np(computed_tau)))
            row.update(_row("effort_limit_", _cpu_np(effort_limit)))
            row.update(_row("vel_limit_", _cpu_np(vel_limit)))
            out_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    q_arr = np.asarray([[r[f"q_{i}"] for i in range(1, 8)] for r in out_rows], dtype=np.float64)
    target_arr = np.asarray([[r[f"target_motor_q_des_{i}"] for i in range(1, 8)] for r in out_rows], dtype=np.float64)
    err = q_arr - target_arr
    print(
        "[forced_motor_target] "
        f"wrote={os.path.abspath(args.out)} rows={len(out_rows)} "
        f"rmse_mean={np.sqrt(np.mean(err * err, axis=0)).mean():.6f} "
        f"p95={np.quantile(np.abs(err), 0.95):.6f} max={np.max(np.abs(err)):.6f}",
        flush=True,
    )
    env.close()
    app.close()


if __name__ == "__main__":
    main()

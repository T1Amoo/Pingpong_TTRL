"""Replay an A1 right-arm target sequence through IsaacLab only.

By default this bypasses the policy and TTEnv action-response model: each 50 Hz
row from ``--target-csv`` is written directly as the joint position target for
every physics substep.  ``--through-action-response`` instead treats each row
as the post-command-filter input to TTEnv's identified response model before
the target reaches the actuator.  This mode is used to validate a complete
response-plus-effort-bound training chain without running a policy.
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
parser.add_argument("--start-row", type=int, default=0)
parser.add_argument("--substeps-per-row", type=int, default=None)
parser.add_argument("--initial-state-csv", type=Path, default=None)
parser.add_argument("--initial-state-time-s", type=float, default=None)
parser.add_argument("--target-prefix", type=str, default="motor_q_des_")
parser.add_argument("--target-index-base", type=int, choices=(0, 1), default=1)
parser.add_argument("--through-action-response", action="store_true")
parser.add_argument(
    "--disable-torque-projection",
    action="store_true",
    help="Diagnostic A/B: keep the task response but bypass its torque projection.",
)
parser.add_argument("--no-noise", action="store_true")
parser.add_argument("--initial-time-column", type=str, default="time_s")
parser.add_argument("--initial-q-prefix", type=str, default="q")
parser.add_argument("--initial-dq-prefix", type=str, default="dq")
parser.add_argument("--initial-index-base", type=int, choices=(0, 1), default=1)
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


def _read_target_rows(
    path: Path,
    prefix: str,
    index_base: int,
    start_row: int,
    steps: int | None,
) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    required = [f"{prefix}{index_base + i}" for i in range(7)]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} missing target columns: {missing}")
    start_row = int(start_row)
    if start_row < 0 or start_row >= len(rows):
        raise ValueError(f"start_row={start_row} outside [0,{len(rows)})")
    rows = rows[start_row:]
    if steps is not None:
        rows = rows[: int(steps)]
    return rows


def _target_from_row(row: dict[str, str], prefix: str, index_base: int) -> np.ndarray:
    return np.asarray(
        [float(row[f"{prefix}{index_base + i}"]) for i in range(7)],
        dtype=np.float64,
    )


def _load_initial_joint_state(
    path: Path,
    time_s: float | None,
    *,
    time_column: str,
    q_prefix: str,
    dq_prefix: str,
    index_base: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    q_cols = [f"{q_prefix}{index_base + i}" for i in range(7)]
    dq_cols = [f"{dq_prefix}{index_base + i}" for i in range(7)]
    required = {time_column, *q_cols}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    times = np.asarray([float(row[time_column]) for row in rows], dtype=np.float64)
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
    if args.through_action_response:
        if not env_cfg.robot.action_response_model_enable:
            raise ValueError(
                f"task {args.task!r} does not enable TTEnv's action-response model"
            )
        # Frozen replay: remove per-environment response DR while retaining the
        # task's nominal response and physical actuator limits.
        env_cfg.robot.action_response_fn_scale_range = (1.0, 1.0)
        env_cfg.robot.action_response_zeta_scale_range = (1.0, 1.0)
        env_cfg.robot.action_response_gain_scale_range = (1.0, 1.0)
        env_cfg.robot.action_response_delay_jitter_s = (0.0,) * 7
        env_cfg.robot.action_response_bias_jitter_rad = (0.0,) * 7
        env_cfg.robot.action_response_accel_limit_scale_range = (1.0, 1.0)
        if args.disable_torque_projection:
            env_cfg.robot.action_response_torque_projection_enable = False

    env = task_registry.get_task_class(args.task)(env_cfg, headless=True)
    action_ids = list(env.action_joint_ids)
    rows_in = _read_target_rows(
        args.target_csv,
        args.target_prefix,
        args.target_index_base,
        args.start_row,
        args.steps,
    )
    substeps_per_row = (
        env.cfg.sim.decimation
        if args.substeps_per_row is None
        else int(args.substeps_per_row)
    )
    if substeps_per_row <= 0:
        raise ValueError("substeps_per_row must be positive")

    initial_meta: dict[str, float | str] = {}
    if args.initial_state_csv is not None:
        q0, dq0, t0 = _load_initial_joint_state(
            args.initial_state_csv,
            args.initial_state_time_s,
            time_column=args.initial_time_column,
            q_prefix=args.initial_q_prefix,
            dq_prefix=args.initial_dq_prefix,
            index_base=args.initial_index_base,
        )
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
            target = _target_from_row(
                src_row, args.target_prefix, args.target_index_base
            )
            target_t = torch.as_tensor(target, device=env.device, dtype=env.robot.data.joint_pos.dtype).reshape(1, 7)
            motor_target_t = target_t
            for _ in range(substeps_per_row):
                env.sim_step_counter += 1
                if args.through_action_response:
                    motor_target_t = env._apply_action_response_model(target_t, env.physics_dt)
                env.robot.set_joint_position_target(motor_target_t, action_ids)
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
                "sim_time_s": float(
                    src_row.get(
                        "sim_time_s",
                        (local_step + 1) * substeps_per_row * env.physics_dt,
                    )
                ),
                "target_file": str(args.target_csv),
                "target_prefix": args.target_prefix,
                "target_index_base": args.target_index_base,
                "substeps_per_row": substeps_per_row,
                "through_action_response": int(args.through_action_response),
            }
            if "trajectory_phase_s" in src_row:
                row["trajectory_phase_s"] = float(src_row["trajectory_phase_s"])
            for source_key in ("t", "stamp", "phase", "freq_hz", "sample_index"):
                if source_key in src_row:
                    row[f"source_{source_key}"] = src_row[source_key]
            row.update(initial_meta)
            row.update(_row("target_motor_q_des_", target))
            row.update(_row("response_target_", _cpu_np(motor_target_t[0])))
            row.update(
                _row(
                    "response_torque_demand_",
                    _cpu_np(env.action_response_torque_demand[0]),
                )
            )
            row.update(
                _row(
                    "response_torque_projected_",
                    _cpu_np(env.action_response_torque_projected[0]),
                )
            )
            row.update(
                _row(
                    "response_torque_clipped_",
                    _cpu_np(env.action_response_torque_clipped[0]).astype(np.int64),
                )
            )
            q_source_cols = [
                f"{args.initial_q_prefix}{args.initial_index_base + i}"
                for i in range(7)
            ]
            tau_source_cols = [
                f"actual_tau_{args.initial_index_base + i}" for i in range(7)
            ]
            if all(col in src_row for col in q_source_cols):
                row.update(
                    _row(
                        "source_actual_q_",
                        [float(src_row[col]) for col in q_source_cols],
                    )
                )
            if all(col in src_row for col in tau_source_cols):
                row.update(
                    _row(
                        "source_actual_tau_",
                        [float(src_row[col]) for col in tau_source_cols],
                    )
                )
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

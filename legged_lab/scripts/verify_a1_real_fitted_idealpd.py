"""Verify that A1_TT_REAL_FITTED_CFG can track the identified response target.

This does not refit the motor model. It feeds the already fitted second-order
q trajectory to the high-bandwidth IdealPD actuator and checks whether IsaacLab
joint position follows that target tightly enough for training.

Example:
    OMNI_KIT_ACCEPT_EULA=YES /home/woan/.conda/envs/pingpong/bin/python -u \
      legged_lab/scripts/verify_a1_real_fitted_idealpd.py --headless
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=0.002, help="Physics dt used for the tracking check.")
parser.add_argument("--warmup-steps", type=int, default=100, help="Steps to hold the first target before recording.")
parser.add_argument("--ignore-s", type=float, default=0.2, help="Initial recorded seconds ignored in metrics.")
parser.add_argument("--max-rmse-rad", type=float, default=0.002, help="Per-joint RMSE pass threshold.")
parser.add_argument("--max-abs-rad", type=float, default=0.01, help="Per-joint max abs error pass threshold.")
parser.add_argument("--output-dir", type=str, default=None, help="Directory for CSV/JSON/PNG outputs.")
parser.add_argument("--joints", type=str, default="1,2,3,4,5,6,7", help="Comma-separated joint indices to test.")
parser.add_argument("--max-duration-s", type=float, default=None, help="Optional duration crop for quick gain scans.")
parser.add_argument("--generated-chirp", action="store_true", help="Generate a synthetic q target instead of reading fitted CSVs.")
parser.add_argument("--chirp-start-hz", type=float, default=2.0, help="Synthetic chirp start frequency.")
parser.add_argument("--chirp-end-hz", type=float, default=5.0, help="Synthetic chirp end frequency.")
parser.add_argument("--chirp-duration-s", type=float, default=20.0, help="Synthetic chirp duration.")
parser.add_argument("--chirp-amplitude-rad", type=float, default=0.08, help="Synthetic chirp amplitude.")
parser.add_argument("--chirp-ramp-s", type=float, default=1.0, help="Smoothstep ramp-in duration for synthetic chirp.")
parser.add_argument(
    "--actuator-type",
    choices=("idealpd", "implicit"),
    default="idealpd",
    help="Right-arm actuator used for this verification.",
)
parser.add_argument("--ideal-kp", type=float, default=None, help="Override IdealPD stiffness for all right-arm joints.")
parser.add_argument("--ideal-kd", type=float, default=None, help="Override IdealPD damping for all right-arm joints.")
parser.add_argument("--ideal-effort", type=float, default=None, help="Override IdealPD effort limits for all right-arm joints.")
parser.add_argument("--ideal-velocity", type=float, default=None, help="Override IdealPD velocity limits for all right-arm joints.")
parser.add_argument("--ideal-kp-list", type=str, default=None, help="Seven comma-separated stiffness values.")
parser.add_argument("--ideal-kd-list", type=str, default=None, help="Seven comma-separated damping values.")
parser.add_argument("--ideal-effort-list", type=str, default=None, help="Seven comma-separated effort limits.")
parser.add_argument("--ideal-velocity-list", type=str, default=None, help="Seven comma-separated velocity limits.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402

import sys  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from legged_lab.assets.a1.a1 import A1_RIGHT_ARM_JOINTS, A1_TT_REAL_FITTED_CFG  # noqa: E402


FIT_TARGETS = {
    1: (
        "系统辨识/joint1/20260713/kp300_kd3.5/sim/"
        "j1_chirp0.1-2hz_amp0.08_real_180643_isaaclab_fitted_second_order.csv",
        "isaac_fitted_q",
    ),
    2: (
        "系统辨识/joint2/20260714/kp300_kd3.5/merged/"
        "j2_default_chirp0.1-2hz_amp0.08_kp300_kd3.5_with_bias_real_isaaclab_compare.csv",
        "with_bias_second_order",
    ),
    3: (
        "系统辨识/joint3/20260714/kp300_kd3.5/sim/"
        "j3_default_chirp0.1-2hz_amp0.08_with_bias_second_order_isaaclab_fitted.csv",
        "isaac_fitted_q",
    ),
    4: (
        "系统辨识/joint4/20260714/kp120_kd1.0/sim/"
        "j4_default_chirp0.1-2hz_amp0.08_kp120_kd1.0_isaaclab_fitted_second_order.csv",
        "isaac_fitted_q",
    ),
    5: (
        "系统辨识/joint5/20260714/kp120_kd1.0/sim/"
        "j5_default_chirp0.1-2hz_amp0.08_kp120_kd1.0_isaaclab_fitted_second_order.csv",
        "isaac_fitted_q",
    ),
    6: (
        "系统辨识/joint6/20260714/kp120_kd1.0/sim/"
        "j6_default_chirp0.1-2hz_amp0.08_kp120_kd1.0_isaaclab_fitted_second_order.csv",
        "isaac_fitted_q",
    ),
    7: (
        "系统辨识/joint7/20260714/kp120_kd1.0/sim/"
        "j7_default_chirp0.1-2hz_amp0.08_kp120_kd1.0_isaaclab_fitted_second_order.csv",
        "isaac_fitted_q",
    ),
}


def load_target(joint_index: int) -> tuple[np.ndarray, np.ndarray, Path]:
    rel_path, q_col = FIT_TARGETS[joint_index]
    path = WORKSPACE_ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    times = []
    q_values = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "relative_s" not in reader.fieldnames:
            raise ValueError(f"{path} has no relative_s column")
        if q_col not in reader.fieldnames:
            raise ValueError(f"{path} has no {q_col} column")
        for row in reader:
            t = float(row["relative_s"])
            q = float(row[q_col])
            if np.isfinite(t) and np.isfinite(q):
                times.append(t)
                q_values.append(q)
    if len(times) < 2:
        raise ValueError(f"{path} has too few valid samples")
    t_arr = np.asarray(times, dtype=np.float64)
    q_arr = np.asarray(q_values, dtype=np.float64)
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    q_arr = q_arr[order]
    t_arr = t_arr - t_arr[0]
    if args.max_duration_s is not None:
        keep = t_arr <= float(args.max_duration_s)
        t_arr = t_arr[keep]
        q_arr = q_arr[keep]
        if len(t_arr) < 2:
            raise ValueError(f"{path} has too few samples after --max-duration-s crop")
    return t_arr, q_arr, path


def generated_chirp_target(
    robot: Articulation,
    joint_ids: list[int],
    joint_index: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    duration = float(args.chirp_duration_s if args.max_duration_s is None else args.max_duration_s)
    times = np.arange(0.0, duration + 0.5 * args.dt, args.dt, dtype=np.float64)
    f0 = float(args.chirp_start_hz)
    f1 = float(args.chirp_end_hz)
    k = (f1 - f0) / max(duration, 1.0e-9)
    phase = 2.0 * np.pi * (f0 * times + 0.5 * k * times * times)
    ramp_x = np.clip(times / max(float(args.chirp_ramp_s), 1.0e-9), 0.0, 1.0)
    envelope = ramp_x * ramp_x * (3.0 - 2.0 * ramp_x)
    center = float(robot.data.default_joint_pos[0, joint_ids[joint_index - 1]].item())
    target = center + float(args.chirp_amplitude_rad) * envelope * np.sin(phase)
    source = f"generated_chirp_{f0:g}-{f1:g}hz_amp{args.chirp_amplitude_rad:g}_duration{duration:g}s"
    return times, target, source


def reset_robot_to_pose(robot: Articulation, sim: sim_utils.SimulationContext, q_all: torch.Tensor):
    robot.write_joint_state_to_sim(q_all, torch.zeros_like(q_all))
    robot.set_joint_position_target(q_all)
    robot.reset()
    robot.write_data_to_sim()
    sim.forward()


def run_joint(
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    joint_ids: list[int],
    joint_index: int,
    times: np.ndarray,
    targets: np.ndarray,
) -> tuple[list[dict], dict]:
    joint_id = joint_ids[joint_index - 1]
    base_q = robot.data.default_joint_pos.clone()
    q0 = base_q.clone()
    q0[:, joint_id] = float(targets[0])
    reset_robot_to_pose(robot, sim, q0)

    for _ in range(args.warmup_steps):
        robot.set_joint_position_target(q0)
        robot.write_data_to_sim()
        sim.step()
        robot.update(args.dt)

    duration = float(times[-1])
    steps = int(np.ceil(duration / args.dt)) + 1
    rows = []
    for step in range(steps):
        t = step * args.dt
        q_target = float(np.interp(t, times, targets))
        q_cmd = base_q.clone()
        q_cmd[:, joint_id] = q_target
        robot.set_joint_position_target(q_cmd)
        robot.write_data_to_sim()
        sim.step()
        robot.update(args.dt)
        q_actual = float(robot.data.joint_pos[0, joint_id].item())
        rows.append(
            {
                "joint": joint_index,
                "joint_name": A1_RIGHT_ARM_JOINTS[joint_index - 1],
                "relative_s": t,
                "target_q": q_target,
                "idealpd_actual_q": q_actual,
                "error_rad": q_actual - q_target,
            }
        )

    metric_rows = [row for row in rows if row["relative_s"] >= args.ignore_s]
    err = np.asarray([row["error_rad"] for row in metric_rows], dtype=np.float64)
    metric = {
        "joint": joint_index,
        "joint_name": A1_RIGHT_ARM_JOINTS[joint_index - 1],
        "rmse_rad": float(np.sqrt(np.mean(np.square(err)))),
        "max_abs_rad": float(np.max(np.abs(err))),
        "mean_abs_rad": float(np.mean(np.abs(err))),
        "duration_s": duration,
        "samples": len(rows),
    }
    metric["pass"] = metric["rmse_rad"] <= args.max_rmse_rad and metric["max_abs_rad"] <= args.max_abs_rad
    return rows, metric


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path: Path, all_rows: list[dict]):
    fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
    for joint_index, ax in enumerate(axes, start=1):
        rows = [row for row in all_rows if row["joint"] == joint_index]
        t = np.asarray([row["relative_s"] for row in rows])
        target = np.asarray([row["target_q"] for row in rows])
        actual = np.asarray([row["idealpd_actual_q"] for row in rows])
        ax.plot(t, target, label="second_order_q_target", linewidth=1.4)
        ax.plot(t, actual, label="idealpd_actual_q", linewidth=1.0, linestyle="--")
        ax.set_ylabel(f"j{joint_index} rad")
        ax.grid(True, alpha=0.25)
        if joint_index == 1:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("relative_s")
    fig.suptitle("A1 real-fitted second-order q target tracked by high-bandwidth IdealPD")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_joint_values(value_list: str | None, scalar: float | None, default: float) -> dict[str, float]:
    if value_list is not None:
        values = [float(item.strip()) for item in value_list.split(",") if item.strip()]
        if len(values) != len(A1_RIGHT_ARM_JOINTS):
            raise ValueError(f"expected 7 values, got {len(values)}: {value_list}")
        return dict(zip(A1_RIGHT_ARM_JOINTS, values))
    value = default if scalar is None else float(scalar)
    return {joint: value for joint in A1_RIGHT_ARM_JOINTS}


def main():
    output_dir = Path(args.output_dir) if args.output_dir else (
        WORKSPACE_ROOT / "系统辨识/reports/a1_tt_real_v1_idealpd_verify"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(A1_TT_REAL_FITTED_CFG).replace(prim_path="/World/Robot")
    if args.actuator_type == "implicit":
        kp = parse_joint_values(args.ideal_kp_list, args.ideal_kp, 5000.0)
        kd = parse_joint_values(args.ideal_kd_list, args.ideal_kd, 100.0)
        effort = parse_joint_values(args.ideal_effort_list, args.ideal_effort, 1.0e5)
        velocity = parse_joint_values(args.ideal_velocity_list, args.ideal_velocity, 1.0e3)
        cfg.actuators["right_arm"] = ImplicitActuatorCfg(
            joint_names_expr=A1_RIGHT_ARM_JOINTS,
            effort_limit_sim=effort,
            velocity_limit_sim=velocity,
            stiffness=kp,
            damping=kd,
        )
    if any(value is not None for value in (args.ideal_kp, args.ideal_kd, args.ideal_effort, args.ideal_velocity)):
        actuator = cfg.actuators["right_arm"]
        if args.ideal_kp is not None:
            actuator.stiffness = {joint: float(args.ideal_kp) for joint in A1_RIGHT_ARM_JOINTS}
        if args.ideal_kd is not None:
            actuator.damping = {joint: float(args.ideal_kd) for joint in A1_RIGHT_ARM_JOINTS}
        if args.ideal_effort is not None:
            effort = {joint: float(args.ideal_effort) for joint in A1_RIGHT_ARM_JOINTS}
            if args.actuator_type == "idealpd":
                actuator.effort_limit = effort
            actuator.effort_limit_sim = effort
        if args.ideal_velocity is not None:
            velocity = {joint: float(args.ideal_velocity) for joint in A1_RIGHT_ARM_JOINTS}
            if args.actuator_type == "idealpd":
                actuator.velocity_limit = velocity
            actuator.velocity_limit_sim = velocity
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    robot = Articulation(cfg)
    sim.reset()

    joint_ids, joint_names = robot.find_joints(A1_RIGHT_ARM_JOINTS, preserve_order=True)
    if list(joint_names) != list(A1_RIGHT_ARM_JOINTS):
        raise RuntimeError(f"Unexpected joint order: {joint_names}")

    all_rows = []
    metrics = []
    source_files = {}
    joint_indices = [int(item.strip()) for item in args.joints.split(",") if item.strip()]
    for joint_index in joint_indices:
        if args.generated_chirp:
            times, targets, source_path = generated_chirp_target(robot, joint_ids, joint_index)
        else:
            times, targets, source_path = load_target(joint_index)
        source_files[f"j{joint_index}"] = str(source_path)
        rows, metric = run_joint(robot, sim, joint_ids, joint_index, times, targets)
        all_rows.extend(rows)
        metrics.append(metric)
        print(
            f"j{joint_index}: rmse={metric['rmse_rad']:.6f} rad, "
            f"max_abs={metric['max_abs_rad']:.6f} rad, pass={metric['pass']}"
        )

    write_csv(output_dir / "idealpd_tracking_rows.csv", all_rows)
    write_csv(output_dir / "idealpd_tracking_metrics.csv", metrics)
    plot_results(output_dir / "idealpd_tracking_overview.png", all_rows)

    summary = {
        "pass": all(metric["pass"] for metric in metrics),
        "dt": args.dt,
        "warmup_steps": args.warmup_steps,
        "ignore_s": args.ignore_s,
        "max_rmse_rad": args.max_rmse_rad,
        "max_abs_rad": args.max_abs_rad,
        "source_files": source_files,
        "metrics": metrics,
    }
    with (output_dir / "idealpd_tracking_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"output_dir={output_dir}")
    print(f"RESULT={'PASS' if summary['pass'] else 'FAIL'}")


try:
    main()
finally:
    app.close()

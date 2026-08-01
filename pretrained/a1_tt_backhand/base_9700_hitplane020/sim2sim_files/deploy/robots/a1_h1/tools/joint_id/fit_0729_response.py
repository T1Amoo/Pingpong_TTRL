#!/usr/bin/python3
"""Fit 2026-07-29 A1 right-arm real actuator response data.

The 0729 folder contains raw real-robot single-joint chirp/step recordings.
This wrapper filters valid ``real/`` trials, maps them to the existing
``fit_second_order_actuator`` column format, fits one response model per joint
from chirp data, and validates the fitted model on step data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from fit_second_order_actuator import fit_model, simulate_base_response


RIGHT_ARM_JOINTS = [f"r{i}" for i in range(1, 8)]


@dataclass(frozen=True)
class Trial:
    csv_path: Path
    manifest_path: Path
    pose: str
    joint: int
    signal_type: str
    real_kp: tuple[float, ...]
    real_kd: tuple[float, ...]
    chirp_start_hz: float
    chirp_end_hz: float
    freq_hz: float
    row_count: int


@dataclass(frozen=True)
class TrialSeries:
    trial: Trial
    relative_s: np.ndarray
    command_q: np.ndarray
    actual_q: np.ndarray
    actual_dq: np.ndarray


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing manifest for 0729 trial: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _float_list(values: Any, *, size: int, default: float) -> tuple[float, ...]:
    if not isinstance(values, list):
        return tuple([default] * size)
    out = []
    for value in values[:size]:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(default)
    while len(out) < size:
        out.append(default)
    return tuple(out)


def discover_trials(root: Path, pose: str) -> list[Trial]:
    pose_dir = root / pose
    if not pose_dir.exists():
        raise FileNotFoundError(f"missing 0729 pose directory: {pose_dir}")
    trials: list[Trial] = []
    for csv_path in sorted(pose_dir.glob("j[1-7]/real/*.csv")):
        rows = _row_count(csv_path)
        if rows <= 0:
            continue
        manifest_path = csv_path.with_suffix(".manifest.json")
        manifest = _load_manifest(manifest_path)
        joint = int(manifest.get("joint_index") or csv_path.parent.parent.name.removeprefix("j"))
        signal_type = str(manifest.get("signal_type") or "").strip()
        if signal_type not in {"chirp", "step"}:
            continue
        trials.append(
            Trial(
                csv_path=csv_path,
                manifest_path=manifest_path,
                pose=pose,
                joint=joint,
                signal_type=signal_type,
                real_kp=_float_list(manifest.get("real_kp"), size=7, default=math.nan),
                real_kd=_float_list(manifest.get("real_kd"), size=7, default=math.nan),
                chirp_start_hz=float(manifest.get("chirp_start_hz", 0.1)),
                chirp_end_hz=float(manifest.get("chirp_end_hz", 0.1)),
                freq_hz=float(manifest.get("freq_hz", math.nan)),
                row_count=rows,
            )
        )
    return trials


def _get_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except KeyError as exc:
        raise KeyError(f"missing column {key!r}") from exc


def load_trial_series(trial: Trial) -> TrialSeries:
    rows = _read_rows(trial.csv_path)
    if not rows:
        raise ValueError(f"empty 0729 trial: {trial.csv_path}")
    target_key = f"target_q{trial.joint}"
    actual_key = f"actual_q{trial.joint}"
    actual_dq_key = f"actual_dq{trial.joint}"
    t = np.array([_get_float(row, "t") for row in rows], dtype=np.float64)
    command = np.array([_get_float(row, target_key) for row in rows], dtype=np.float64)
    actual = np.array([_get_float(row, actual_key) for row in rows], dtype=np.float64)
    actual_dq = np.array([_get_float(row, actual_dq_key) for row in rows], dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(command) & np.isfinite(actual) & np.isfinite(actual_dq)
    if int(valid.sum()) < 2:
        raise ValueError(f"not enough finite samples in {trial.csv_path}")
    t = t[valid]
    command = command[valid]
    actual = actual[valid]
    actual_dq = actual_dq[valid]
    order = np.argsort(t)
    t = t[order]
    command = command[order]
    actual = actual[order]
    actual_dq = actual_dq[order]
    return TrialSeries(trial=trial, relative_s=t - t[0], command_q=command, actual_q=actual, actual_dq=actual_dq)


def estimate_velocity_limit(
    max_real_dq: dict[str, float],
    *,
    margin: float = 1.35,
    min_limit: float = 4.0,
    max_limit: float = 30.0,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for joint in RIGHT_ARM_JOINTS:
        measured = abs(float(max_real_dq.get(joint, 0.0)))
        limit = math.ceil(measured * margin)
        out[joint] = float(min(max_limit, max(min_limit, limit)))
    return out


def write_merged_csv(path: Path, series: TrialSeries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_s", "target_raw_q", "cmd_q", "real_actual_q"])
        writer.writeheader()
        for t, command, actual in zip(series.relative_s, series.command_q, series.actual_q):
            writer.writerow(
                {
                    "relative_s": _format_csv_float(t),
                    "target_raw_q": _format_csv_float(command),
                    "cmd_q": _format_csv_float(command),
                    "real_actual_q": _format_csv_float(actual),
                }
            )


def _format_csv_float(value: float) -> str:
    return f"{float(value):.12g}"


def _predict_response(series: TrialSeries, fit: dict[str, Any], max_step_s: float) -> np.ndarray:
    u_mean = float(fit["u_mean"])
    x, v = simulate_base_response(
        series.relative_s,
        series.command_q - u_mean,
        float(fit["fn_hz"]),
        float(fit["zeta"]),
        float(fit["delay_s"]),
        max_step_s,
    )
    feature = x + float(fit.get("tau_zero_s", 0.0)) * v
    return float(fit["intercept"]) + float(fit["linear_gain"]) * feature


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    residual = pred - target
    return float(np.sqrt(np.mean(residual * residual)))


def _max_abs(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(pred - target)))


def _sanitize_fit(fit: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "model",
        "fn_hz",
        "zeta",
        "delay_s",
        "tau_zero_s",
        "intercept",
        "linear_gain",
        "bias_at_u_mean",
        "u_mean",
        "output_bias",
        "rmse_all",
        "rmse_fit_raw",
        "maxabs_fit",
        "success",
        "objective",
    }
    out: dict[str, Any] = {}
    for key in keep:
        value = fit.get(key)
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def _choose_latest(trials: list[Trial]) -> Trial:
    if not trials:
        raise ValueError("cannot choose from empty trial list")
    return sorted(trials, key=lambda trial: trial.csv_path.name)[-1]


def fit_pose(root: Path, pose: str, args: argparse.Namespace) -> dict[str, Any]:
    trials = discover_trials(root, pose)
    by_joint: dict[int, dict[str, list[Trial]]] = {
        joint: {"chirp": [], "step": []}
        for joint in range(1, 8)
    }
    for trial in trials:
        by_joint[trial.joint][trial.signal_type].append(trial)

    joint_summaries: list[dict[str, Any]] = []
    max_real_dq: dict[str, float] = {}
    for joint in range(1, 8):
        chirp = _choose_latest(by_joint[joint]["chirp"])
        chirp_series = load_trial_series(chirp)
        joint_name = RIGHT_ARM_JOINTS[joint - 1]
        max_real_dq[joint_name] = float(np.max(np.abs(chirp_series.actual_dq)))
        merged_path = args.output_dir / "merged" / f"j{joint}_{pose}_chirp.csv"
        write_merged_csv(merged_path, chirp_series)
        fit_mask = (
            (chirp_series.relative_s >= chirp_series.relative_s[0] + args.ignore_start_s)
            & (chirp_series.relative_s <= chirp_series.relative_s[-1] - args.ignore_end_s)
        )
        if int(fit_mask.sum()) < 50:
            raise ValueError(f"fit window too small for {chirp.csv_path}")
        fit = fit_model(
            args.model,
            chirp_series.relative_s,
            chirp_series.command_q,
            chirp_series.actual_q,
            fit_mask,
            args.max_step_s,
            args.seed + joint,
            args.output_bias,
            args.fn_min_hz,
            args.fn_max_hz,
            args.delay_max_s,
        )
        pred = _predict_response(chirp_series, fit, args.max_step_s)
        step_reports = []
        for step_trial in sorted(by_joint[joint]["step"], key=lambda trial: trial.csv_path.name):
            step_series = load_trial_series(step_trial)
            max_real_dq[joint_name] = max(max_real_dq[joint_name], float(np.max(np.abs(step_series.actual_dq))))
            step_merged = args.output_dir / "merged" / f"j{joint}_{pose}_step_{step_trial.csv_path.stem}.csv"
            write_merged_csv(step_merged, step_series)
            step_pred = _predict_response(step_series, fit, args.max_step_s)
            step_reports.append(
                {
                    "csv": str(step_trial.csv_path),
                    "rows": step_trial.row_count,
                    "rmse": _rmse(step_pred, step_series.actual_q),
                    "maxabs": _max_abs(step_pred, step_series.actual_q),
                    "merged_csv": str(step_merged),
                }
            )

        joint_summaries.append(
            {
                "joint": joint,
                "joint_name": joint_name,
                "fit_csv": str(chirp.csv_path),
                "fit_rows": chirp.row_count,
                "fit_merged_csv": str(merged_path),
                "chirp_start_hz": chirp.chirp_start_hz,
                "chirp_end_hz": chirp.chirp_end_hz,
                "real_kp": chirp.real_kp[joint - 1],
                "real_kd": chirp.real_kd[joint - 1],
                "fit": _sanitize_fit(fit),
                "fit_recomputed_rmse": _rmse(pred, chirp_series.actual_q),
                "fit_recomputed_maxabs": _max_abs(pred, chirp_series.actual_q),
                "step_validation": step_reports,
            }
        )

    velocity_limit = estimate_velocity_limit(max_real_dq)
    for item in joint_summaries:
        item["max_real_dq"] = max_real_dq[item["joint_name"]]
        item["real_velocity_limit"] = velocity_limit[item["joint_name"]]

    return {
        "source_root": str(root),
        "pose": pose,
        "model": args.model,
        "fn_min_hz": args.fn_min_hz,
        "fn_max_hz": args.fn_max_hz,
        "delay_max_s": args.delay_max_s,
        "ignore_start_s": args.ignore_start_s,
        "ignore_end_s": args.ignore_end_s,
        "joints": joint_summaries,
        "max_real_dq": max_real_dq,
        "real_velocity_limit": velocity_limit,
    }


def _joint_sort_key(name: str) -> int:
    if name.startswith("r") and name[1:].isdigit():
        return int(name[1:])
    return 999


def format_python_dict(values: dict[str, float], *, indent: int = 4) -> str:
    pad = " " * indent
    lines = ["{"]
    for key in sorted(values, key=_joint_sort_key):
        lines.append(f'{pad}"{key}": {repr(float(values[key]))},')
    lines.append("}")
    return "\n".join(lines)


def _dict_from_summary(summary: dict[str, Any], key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in summary["joints"]:
        name = str(item["joint_name"])
        if key == "real_kp":
            value = item["real_kp"]
        elif key == "real_kd":
            value = item["real_kd"]
        elif key == "real_velocity_limit":
            value = item["real_velocity_limit"]
        else:
            value = item["fit"][key]
        out[name] = float(value)
    return out


def _array_literal(values: dict[str, float]) -> str:
    ordered = [values[joint] for joint in RIGHT_ARM_JOINTS]
    return "np.array([" + ", ".join(repr(float(v)) for v in ordered) + "], dtype=np.float64)"


def write_constant_snippets(summary: dict[str, Any], output_dir: Path) -> None:
    maps = {
        "node_kp": _dict_from_summary(summary, "real_kp"),
        "node_kd": _dict_from_summary(summary, "real_kd"),
        "vel": _dict_from_summary(summary, "real_velocity_limit"),
        "u_mean": _dict_from_summary(summary, "u_mean"),
        "fn_hz": _dict_from_summary(summary, "fn_hz"),
        "zeta": _dict_from_summary(summary, "zeta"),
        "delay_s": _dict_from_summary(summary, "delay_s"),
        "gain": _dict_from_summary(summary, "linear_gain"),
        "intercept": _dict_from_summary(summary, "intercept"),
        "tau_zero_s": _dict_from_summary(summary, "tau_zero_s"),
    }

    a1_text = "\n".join(
        [
            "# Generated by tools/joint_id/fit_0729_response.py from mentor 0729 real data.",
            "_REAL_FITTED_NODE_KP = " + format_python_dict(maps["node_kp"]),
            "_REAL_FITTED_NODE_KD = " + format_python_dict(maps["node_kd"]),
            "_REAL_FITTED_VEL = " + format_python_dict(maps["vel"]),
            "_REAL_FITTED_RESPONSE_U_MEAN = " + format_python_dict(maps["u_mean"]),
            "_REAL_FITTED_RESPONSE_FN_HZ = " + format_python_dict(maps["fn_hz"]),
            "_REAL_FITTED_RESPONSE_ZETA = " + format_python_dict(maps["zeta"]),
            "_REAL_FITTED_RESPONSE_DELAY_S = " + format_python_dict(maps["delay_s"]),
            "_REAL_FITTED_RESPONSE_GAIN = " + format_python_dict(maps["gain"]),
            "_REAL_FITTED_RESPONSE_INTERCEPT = " + format_python_dict(maps["intercept"]),
            "_REAL_FITTED_RESPONSE_TAU_ZERO_S = " + format_python_dict(maps["tau_zero_s"]),
            "",
        ]
    )
    policy_text = "\n".join(
        [
            "# Generated by tools/joint_id/fit_0729_response.py from mentor 0729 real data.",
            "KP = " + _array_literal(maps["node_kp"]),
            "KD = " + _array_literal(maps["node_kd"]),
            "DAMIAO_MIT_KP = " + _array_literal(maps["node_kp"]),
            "DAMIAO_MIT_KD = " + _array_literal(maps["node_kd"]),
            "DAMIAO_MIT_VEL = " + _array_literal(maps["vel"]),
            "REAL_RESPONSE_U_MEAN = " + _array_literal(maps["u_mean"]),
            "REAL_RESPONSE_FN_HZ = " + _array_literal(maps["fn_hz"]),
            "REAL_RESPONSE_ZETA = " + _array_literal(maps["zeta"]),
            "REAL_RESPONSE_DELAY_S = " + _array_literal(maps["delay_s"]),
            "REAL_RESPONSE_GAIN = " + _array_literal(maps["gain"]),
            "REAL_RESPONSE_BIAS_RAD = " + _array_literal(
                {
                    joint: maps["intercept"][joint] - maps["u_mean"][joint]
                    for joint in RIGHT_ARM_JOINTS
                }
            ),
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "a1_constants_snippet.py").write_text(a1_text, encoding="utf-8")
    (output_dir / "policy_io_constants_snippet.py").write_text(policy_text, encoding="utf-8")


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "joint",
        "real_kp",
        "real_kd",
        "fn_hz",
        "zeta",
        "delay_s",
        "gain",
        "intercept",
        "u_mean",
        "rmse_fit_raw",
        "maxabs_fit",
        "step_rmse_mean",
        "max_real_dq",
        "real_velocity_limit",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in summary["joints"]:
            steps = item["step_validation"]
            writer.writerow(
                {
                    "joint": item["joint_name"],
                    "real_kp": item["real_kp"],
                    "real_kd": item["real_kd"],
                    "fn_hz": item["fit"]["fn_hz"],
                    "zeta": item["fit"]["zeta"],
                    "delay_s": item["fit"]["delay_s"],
                    "gain": item["fit"]["linear_gain"],
                    "intercept": item["fit"]["intercept"],
                    "u_mean": item["fit"]["u_mean"],
                    "rmse_fit_raw": item["fit"]["rmse_fit_raw"],
                    "maxabs_fit": item["fit"]["maxabs_fit"],
                    "step_rmse_mean": float(np.mean([step["rmse"] for step in steps])) if steps else math.nan,
                    "max_real_dq": item["max_real_dq"],
                    "real_velocity_limit": item["real_velocity_limit"],
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/woan/桌面/0729"))
    parser.add_argument("--pose", default="backhand", choices=["backhand", "forehand"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="second_order", choices=["second_order", "lead_lag"])
    parser.add_argument("--ignore-start-s", type=float, default=2.0)
    parser.add_argument("--ignore-end-s", type=float, default=0.5)
    parser.add_argument("--max-step-s", type=float, default=0.0025)
    parser.add_argument("--output-bias", choices=["free", "none"], default="free")
    parser.add_argument("--fn-min-hz", type=float, default=0.4)
    parser.add_argument("--fn-max-hz", type=float, default=20.0)
    parser.add_argument("--delay-max-s", type=float, default=0.14)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = fit_pose(args.root, args.pose, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / f"{args.pose}_0729_{args.model}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_csv(summary, args.output_dir / f"{args.pose}_0729_{args.model}_summary.csv")
    write_constant_snippets(summary, args.output_dir)
    print(f"wrote {summary_json}")
    print(f"wrote {args.output_dir / f'{args.pose}_0729_{args.model}_summary.csv'}")
    print(f"wrote {args.output_dir / 'a1_constants_snippet.py'}")
    print(f"wrote {args.output_dir / 'policy_io_constants_snippet.py'}")
    for item in summary["joints"]:
        fit = item["fit"]
        steps = item["step_validation"]
        step_mean = float(np.mean([step["rmse"] for step in steps])) if steps else math.nan
        print(
            item["joint_name"],
            f"kp={item['real_kp']:.3g}",
            f"kd={item['real_kd']:.3g}",
            f"fn={fit['fn_hz']:.3f}Hz",
            f"zeta={fit['zeta']:.3f}",
            f"delay={1000.0 * fit['delay_s']:.1f}ms",
            f"fit_rmse={fit['rmse_fit_raw']:.5f}",
            f"step_rmse_mean={step_mean:.5f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

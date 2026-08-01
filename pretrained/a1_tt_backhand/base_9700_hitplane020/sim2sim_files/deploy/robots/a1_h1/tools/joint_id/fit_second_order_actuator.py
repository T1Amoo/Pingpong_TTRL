#!/usr/bin/python3
"""Fit simple closed-loop actuator dynamics from q_des command to measured q."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def fit_local_sine(t: np.ndarray, y: np.ndarray, freq_hz: float) -> tuple[float, float, float, float]:
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if t.size < 12:
        return math.nan, math.nan, math.nan, math.nan
    omega = 2.0 * math.pi * freq_hz
    x = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
    offset, sin_c, cos_c = [float(v) for v in coeff]
    pred = x @ coeff
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else math.nan
    return offset, math.hypot(sin_c, cos_c), math.atan2(cos_c, sin_c), r2


def load_merged(path: Path, input_column: str, output_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_rows(path)
    if not rows:
        raise SystemExit(f"empty merged CSV: {path}")
    t = np.array([get_float(r, "relative_s") for r in rows], dtype=float)
    u = np.array([get_float(r, input_column) for r in rows], dtype=float)
    y = np.array([get_float(r, output_column) for r in rows], dtype=float)
    valid = np.isfinite(t) & np.isfinite(u) & np.isfinite(y)
    return t[valid], u[valid], y[valid]


def delayed_input(t: np.ndarray, u_rel: np.ndarray, delay_s: float) -> np.ndarray:
    return np.interp(t - max(0.0, delay_s), t, u_rel, left=u_rel[0], right=u_rel[-1])


def simulate_base_response(
    t: np.ndarray,
    u_rel: np.ndarray,
    fn_hz: float,
    zeta: float,
    delay_s: float,
    max_step_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate x'' + 2*zeta*wn*x' + wn^2*x = wn^2*u_rel(t-delay)."""
    wn = 2.0 * math.pi * max(fn_hz, 1e-6)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    u_delay = delayed_input(t, u_rel, delay_s)
    x[0] = u_delay[0]
    v[0] = 0.0

    def acc(tt: float, xx: float, vv: float) -> float:
        uu = float(np.interp(tt - delay_s, t, u_rel, left=u_rel[0], right=u_rel[-1]))
        return wn * wn * (uu - xx) - 2.0 * zeta * wn * vv

    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        if dt <= 0.0:
            x[i] = x[i - 1]
            v[i] = v[i - 1]
            continue
        n = max(1, int(math.ceil(dt / max_step_s)))
        h = dt / n
        xx = float(x[i - 1])
        vv = float(v[i - 1])
        tt = float(t[i - 1])
        for _ in range(n):
            # Semi-implicit integration is stable enough for this low-order fit.
            aa = acc(tt, xx, vv)
            vv += aa * h
            xx += vv * h
            tt += h
        x[i] = xx
        v[i] = vv
    return x, v


def solve_linear_output(
    feature: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    output_bias: str,
    u0: float,
) -> tuple[float, float, np.ndarray]:
    if output_bias == "none":
        denom = float(np.dot(feature[fit_mask], feature[fit_mask]))
        if denom <= 1e-12:
            gain = 0.0
        else:
            gain = float(np.dot(feature[fit_mask], y[fit_mask] - u0) / denom)
        intercept = u0
        return intercept, gain, intercept + gain * feature

    a = np.column_stack([np.ones(int(fit_mask.sum())), feature[fit_mask]])
    coeff, *_ = np.linalg.lstsq(a, y[fit_mask], rcond=None)
    intercept, gain = [float(v) for v in coeff]
    return intercept, gain, intercept + gain * feature


def model_bounds(
    model: str,
    *,
    fn_min_hz: float = 0.4,
    fn_max_hz: float = 8.0,
    delay_max_s: float = 0.14,
) -> list[tuple[float, float]]:
    if fn_min_hz <= 0.0:
        raise ValueError(f"fn_min_hz must be positive, got {fn_min_hz}")
    if fn_max_hz < fn_min_hz:
        raise ValueError(f"fn_max_hz must be >= fn_min_hz, got {fn_max_hz} < {fn_min_hz}")
    if delay_max_s < 0.0:
        raise ValueError(f"delay_max_s must be non-negative, got {delay_max_s}")
    base = [(float(fn_min_hz), float(fn_max_hz)), (0.03, 2.5), (0.0, float(delay_max_s))]
    if model == "second_order":
        return base
    if model == "lead_lag":
        return [*base, (0.0, 0.16)]
    raise ValueError(model)


def refine_bounded(objective, x0: np.ndarray, bounds: list[tuple[float, float]]):
    from scipy.optimize import minimize

    result = minimize(
        objective,
        np.asarray(x0, dtype=float),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 600, "xtol": 1e-8, "ftol": 1e-10},
    )
    result.x = np.clip(np.asarray(result.x, dtype=float), [lo for lo, _ in bounds], [hi for _, hi in bounds])
    return result


def fit_model(
    model: str,
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    max_step_s: float,
    seed: int,
    output_bias: str,
    fn_min_hz: float = 0.4,
    fn_max_hz: float = 8.0,
    delay_max_s: float = 0.14,
) -> dict:
    u0 = float(np.mean(u[fit_mask]))
    u_rel = u - u0
    cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}

    def evaluate(params: np.ndarray) -> tuple[float, dict]:
        fn_hz = float(params[0])
        zeta = float(params[1])
        delay_s = float(params[2])
        tau_zero_s = float(params[3]) if model == "lead_lag" else 0.0
        key = (round(fn_hz, 8), round(zeta, 8), round(delay_s, 8))
        if key in cache:
            x, v = cache[key]
        else:
            x, v = simulate_base_response(t, u_rel, fn_hz, zeta, delay_s, max_step_s)
            if len(cache) < 4096:
                cache[key] = (x, v)
        feature = x + tau_zero_s * v
        intercept, gain, y_hat = solve_linear_output(feature, y, fit_mask, output_bias, u0)
        residual = y_hat[fit_mask] - y[fit_mask]
        rmse = float(np.sqrt(np.mean(residual * residual)))
        # Mild regularization prevents the zero from absorbing noise-only derivative gain.
        if model == "lead_lag":
            rmse += 0.0005 * abs(tau_zero_s) / 0.05
        return rmse, {
            "fn_hz": fn_hz,
            "zeta": zeta,
            "delay_s": delay_s,
            "tau_zero_s": tau_zero_s,
            "intercept": intercept,
            "linear_gain": gain,
            "bias_at_u_mean": intercept - u0,
            "u_mean": u0,
            "output_bias": output_bias,
            "y_hat": y_hat,
            "feature": feature,
            "rmse_fit": rmse,
        }

    bounds = model_bounds(model, fn_min_hz=fn_min_hz, fn_max_hz=fn_max_hz, delay_max_s=delay_max_s)

    from scipy.optimize import differential_evolution

    result_de = differential_evolution(
        lambda p: evaluate(np.asarray(p))[0],
        bounds=bounds,
        seed=seed,
        polish=False,
        maxiter=70,
        popsize=10,
        tol=1e-8,
        updating="immediate",
        workers=1,
    )
    result = refine_bounded(lambda p: evaluate(np.asarray(p))[0], result_de.x, bounds)
    _, info = evaluate(np.asarray(result.x))
    y_hat = info["y_hat"]
    all_residual = y_hat - y
    fit_residual = y_hat[fit_mask] - y[fit_mask]
    info.update(
        {
            "model": model,
            "rmse_all": float(np.sqrt(np.mean(all_residual * all_residual))),
            "rmse_fit_raw": float(np.sqrt(np.mean(fit_residual * fit_residual))),
            "maxabs_fit": float(np.max(np.abs(fit_residual))),
            "success": bool(result.success),
            "objective": float(result.fun),
        }
    )
    return info


def local_metrics(
    t: np.ndarray,
    u: np.ndarray,
    series: dict[str, np.ndarray],
    freq_start: float,
    freq_end: float,
    window_cycles: float,
    min_window_s: float,
    max_window_s: float,
    step_s: float,
) -> list[dict[str, float | str]]:
    duration = float(t[-1] - t[0])
    out: list[dict[str, float | str]] = []
    centers = np.arange(t[0] + max_window_s * 0.5, t[-1] - max_window_s * 0.5, step_s)
    for center in centers:
        alpha = (center - t[0]) / duration
        freq = freq_start + (freq_end - freq_start) * alpha
        window_s = min(max(min_window_s, window_cycles / max(freq, 1e-6)), max_window_s)
        mask = (t >= center - 0.5 * window_s) & (t <= center + 0.5 * window_s)
        if int(mask.sum()) < 20:
            continue
        u_offset, u_amp, u_phase, u_r2 = fit_local_sine(t[mask], u[mask], freq)
        if not np.isfinite(u_amp) or u_amp < 1e-6:
            continue
        for label, y in series.items():
            y_offset, y_amp, y_phase, y_r2 = fit_local_sine(t[mask], y[mask], freq)
            phase_lag = wrap_to_pi(u_phase - y_phase)
            out.append(
                {
                    "series": label,
                    "center_s": center,
                    "freq_hz": freq,
                    "bias_rad": y_offset - u_offset,
                    "gain": y_amp / u_amp,
                    "phase_lag_s": phase_lag / (2.0 * math.pi * freq),
                    "input_r2": u_r2,
                    "response_r2": y_r2,
                }
            )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--input-column", default="cmd_q")
    ap.add_argument("--output-column", default="real_actual_q")
    ap.add_argument("--models", nargs="+", choices=["second_order", "lead_lag"], default=["second_order", "lead_lag"])
    ap.add_argument("--ignore-start-s", type=float, default=2.0)
    ap.add_argument("--ignore-end-s", type=float, default=0.5)
    ap.add_argument("--max-step-s", type=float, default=0.0025)
    ap.add_argument("--fn-min-hz", type=float, default=0.4)
    ap.add_argument("--fn-max-hz", type=float, default=8.0)
    ap.add_argument("--delay-max-s", type=float, default=0.14)
    ap.add_argument(
        "--output-bias",
        choices=["free", "none"],
        default="free",
        help="free fits an output intercept; none fixes the output center to the input mean.",
    )
    ap.add_argument("--freq-start", type=float, default=0.1)
    ap.add_argument("--freq-end", type=float, default=2.0)
    ap.add_argument("--window-cycles", type=float, default=2.0)
    ap.add_argument("--min-window-s", type=float, default=2.0)
    ap.add_argument("--max-window-s", type=float, default=4.0)
    ap.add_argument("--step-s", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", type=Path, required=True, help="Prediction CSV.")
    ap.add_argument("--params-output", type=Path, required=True)
    ap.add_argument("--metrics-output", type=Path, required=True)
    ap.add_argument("--plot", type=Path, required=True)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    t, u, y = load_merged(args.merged, args.input_column, args.output_column)
    fit_mask = (t >= t[0] + args.ignore_start_s) & (t <= t[-1] - args.ignore_end_s)
    if int(fit_mask.sum()) < 50:
        raise SystemExit("fit window too small")

    fits = [
        fit_model(
            model,
            t,
            u,
            y,
            fit_mask,
            args.max_step_s,
            args.seed,
            args.output_bias,
            args.fn_min_hz,
            args.fn_max_hz,
            args.delay_max_s,
        )
        for model in args.models
    ]
    predictions = {fit["model"]: fit["y_hat"] for fit in fits}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred_fields = ["relative_s", "input_q", "real_q"] + [f"pred_{fit['model']}" for fit in fits]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        for i in range(len(t)):
            row = {"relative_s": t[i], "input_q": u[i], "real_q": y[i]}
            for fit in fits:
                row[f"pred_{fit['model']}"] = fit["y_hat"][i]
            writer.writerow(row)

    params = []
    for fit in fits:
        params.append({k: v for k, v in fit.items() if k not in {"y_hat", "feature"}})
    args.params_output.parent.mkdir(parents=True, exist_ok=True)
    args.params_output.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metric_series = {"real": y}
    metric_series.update({f"pred_{name}": pred for name, pred in predictions.items()})
    metrics = local_metrics(
        t,
        u,
        metric_series,
        args.freq_start,
        args.freq_end,
        args.window_cycles,
        args.min_window_s,
        args.max_window_s,
        args.step_s,
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = list(metrics[0].keys())
    with args.metrics_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=False)
    axes[0].plot(t, u, color="black", linewidth=1.1, label=args.input_column)
    axes[0].plot(t, y, color="tab:green", linewidth=1.2, label=args.output_column)
    for fit in fits:
        axes[0].plot(t, fit["y_hat"], linewidth=1.0, label=f"pred_{fit['model']}")
    axes[0].set_title("time-domain fit")
    axes[0].set_ylabel("q (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3)

    high = t >= max(t[0], t[-1] - 10.0)
    axes[1].plot(t[high], u[high], color="black", linewidth=1.1, label=args.input_column)
    axes[1].plot(t[high], y[high], color="tab:green", linewidth=1.2, label=args.output_column)
    for fit in fits:
        axes[1].plot(t[high], fit["y_hat"][high], linewidth=1.0, label=f"pred_{fit['model']}")
    axes[1].set_title("last 10 seconds")
    axes[1].set_ylabel("q (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=3)

    for fit in fits:
        axes[2].plot(t, fit["y_hat"] - y, linewidth=1.0, label=f"{fit['model']} residual")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("prediction residual")
    axes[2].set_ylabel("pred - real (rad)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    for label in metric_series:
        data = [m for m in metrics if m["series"] == label]
        freq = np.array([float(m["freq_hz"]) for m in data])
        gain = np.array([float(m["gain"]) for m in data])
        axes[3].plot(freq, gain, linewidth=1.0, label=label)
    axes[3].axhline(1.0, color="black", linewidth=0.8)
    axes[3].set_title("local gain relative to command")
    axes[3].set_xlabel("frequency (Hz)")
    axes[3].set_ylabel("gain")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=8, ncol=3)

    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160)

    print(f"wrote {args.output}")
    print(f"wrote {args.params_output}")
    print(f"wrote {args.metrics_output}")
    print(f"wrote {args.plot}")
    for fit in fits:
        print(
            fit["model"],
            f"rmse_fit={fit['rmse_fit_raw']:.6f}",
            f"rmse_all={fit['rmse_all']:.6f}",
            f"fn_hz={fit['fn_hz']:.4f}",
            f"zeta={fit['zeta']:.4f}",
            f"delay_s={fit['delay_s']:.4f}",
            f"tau_zero_s={fit['tau_zero_s']:.4f}",
            f"gain={fit['linear_gain']:.4f}",
            f"bias={fit['bias_at_u_mean']:.6f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnose whether A1 serves can farm contact with a static or barely moving paddle.

Example:
    OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/diagnose_a1_contact_shortcut.py \
        --task a1_tt --headless --num_envs 64 --steps 180 --noise-stds 0,0.02,0.05
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="A1 table-tennis static-contact shortcut diagnostic.")
parser.add_argument("--task", type=str, default="a1_tt", help="Task name to instantiate.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs.")
parser.add_argument("--steps", type=int, default=180, help="Control steps to run per action case.")
parser.add_argument("--seed", type=int, default=42, help="Seed reset before each action case.")
parser.add_argument(
    "--noise-stds",
    type=str,
    default="0,0.02,0.05",
    help="Comma-separated raw action Gaussian stds to test. 0 means exactly zero action.",
)
parser.add_argument("--serve-y-center", type=float, default=None, help="Temporarily override cfg.ball.serve_y_center.")
parser.add_argument("--serve-y-start", type=float, default=None, help="Temporarily override cfg.ball.serve_y_start.")
parser.add_argument("--serve-y-wide", type=float, default=None, help="Temporarily override cfg.ball.serve_y_wide.")
parser.add_argument("--hit-target-y-min", type=float, default=None, help="Temporarily override hit_target_y_range min.")
parser.add_argument("--hit-target-y-max", type=float, default=None, help="Temporarily override hit_target_y_range max.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils import task_registry  # noqa: E402


def _fmt_rate(count: int, total: int) -> str:
    return f"{count}/{total} ({100.0 * count / max(total, 1):.1f}%)"


def _quantiles(values: torch.Tensor) -> str:
    if values.numel() == 0:
        return "n/a"
    qs = torch.quantile(values.float(), torch.tensor([0.05, 0.50, 0.95], device=values.device))
    return f"p05={qs[0].item():.4f}, p50={qs[1].item():.4f}, p95={qs[2].item():.4f}"


@torch.no_grad()
def run_case(env, noise_std: float, steps: int, seed: int):
    device = env.device
    env.seed(seed)
    env_ids = torch.arange(env.num_envs, device=device)
    env.reset(env_ids)

    raw_touched = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
    gated_touched = torch.zeros_like(raw_touched)
    raw_peak = torch.zeros(env.num_envs, device=device)
    gated_peak = torch.zeros_like(raw_peak)
    min_dist = torch.full((env.num_envs,), float("inf"), device=device)
    max_paddle_speed = torch.zeros(env.num_envs, device=device)
    first_raw_step = torch.full((env.num_envs,), -1, device=device, dtype=torch.long)
    first_raw_ball = torch.zeros(env.num_envs, 3, device=device)
    first_raw_paddle = torch.zeros_like(first_raw_ball)
    first_raw_paddle_vel = torch.zeros_like(first_raw_ball)
    first_raw_ball_vel = torch.zeros_like(first_raw_ball)
    first_raw_own_bounce = torch.zeros(env.num_envs, device=device, dtype=torch.bool)

    for step_idx in range(steps):
        if noise_std == 0.0:
            actions = torch.zeros(env.num_envs, env.num_actions, device=device)
        else:
            actions = torch.randn(env.num_envs, env.num_actions, device=device) * noise_std

        env.step(actions)

        raw_peak = torch.maximum(raw_peak, env.ball_contact_raw_rew)
        gated_peak = torch.maximum(gated_peak, env.ball_contact_rew)
        raw_now = env.ball_contact > 0.0
        gated_now = env.has_touch_paddle
        raw_touched |= raw_now | (env.ball_contact_raw_rew > 0.0)
        gated_touched |= gated_now | (env.ball_contact_rew > 0.0)
        min_dist = torch.minimum(min_dist, env.paddel_ball_distance)
        max_paddle_speed = torch.maximum(max_paddle_speed, torch.linalg.norm(env.paddle_touch_point_vel, dim=1))

        first_now = raw_now & (first_raw_step < 0)
        if first_now.any():
            ball_local = env.ball.data.root_pos_w - env.scene.env_origins
            paddle_local = env.paddle_touch_point - env.scene.env_origins
            first_raw_step[first_now] = step_idx
            first_raw_ball[first_now] = ball_local[first_now]
            first_raw_paddle[first_now] = paddle_local[first_now]
            first_raw_paddle_vel[first_now] = env.paddle_touch_point_vel[first_now]
            first_raw_ball_vel[first_now] = env.ball.data.root_lin_vel_w[first_now]
            first_raw_own_bounce[first_now] = env.has_touch_own_table_prev[first_now]

    touched_idx = torch.nonzero(raw_touched, as_tuple=False).squeeze(-1)
    gated_idx = torch.nonzero(gated_touched, as_tuple=False).squeeze(-1)
    first_idx = torch.nonzero(first_raw_step >= 0, as_tuple=False).squeeze(-1)

    print("=" * 80)
    print(f"case noise_std={noise_std:g}  steps={steps}  num_envs={env.num_envs}")
    print(f"raw contact envs:    {_fmt_rate(int(raw_touched.sum().item()), env.num_envs)}")
    print(f"gated contact envs:  {_fmt_rate(int(gated_touched.sum().item()), env.num_envs)}")
    print(f"raw peak score:      {_quantiles(raw_peak[touched_idx])}")
    print(f"gated peak score:    {_quantiles(gated_peak[gated_idx])}")
    print(f"min paddle distance: {_quantiles(min_dist[torch.isfinite(min_dist)])} m")
    print(f"max paddle speed:    {_quantiles(max_paddle_speed)} m/s")
    if first_idx.numel() > 0:
        print(f"first raw step:      {_quantiles(first_raw_step[first_idx].float())}")
        print(
            "first raw ball xyz:  "
            f"mean={first_raw_ball[first_idx].mean(dim=0).detach().cpu().tolist()} "
            f"std={first_raw_ball[first_idx].std(dim=0, unbiased=False).detach().cpu().tolist()}"
        )
        print(
            "first raw paddle xyz:"
            f" mean={first_raw_paddle[first_idx].mean(dim=0).detach().cpu().tolist()} "
            f"std={first_raw_paddle[first_idx].std(dim=0, unbiased=False).detach().cpu().tolist()}"
        )
        print(
            "first raw paddle vel:"
            f" mean={first_raw_paddle_vel[first_idx].mean(dim=0).detach().cpu().tolist()} "
            f"std={first_raw_paddle_vel[first_idx].std(dim=0, unbiased=False).detach().cpu().tolist()}"
        )
        print(
            "first raw ball vel:  "
            f"mean={first_raw_ball_vel[first_idx].mean(dim=0).detach().cpu().tolist()} "
            f"std={first_raw_ball_vel[first_idx].std(dim=0, unbiased=False).detach().cpu().tolist()}"
        )
        print(
            "own bounce before first raw contact: "
            f"{_fmt_rate(int(first_raw_own_bounce[first_idx].sum().item()), int(first_idx.numel()))}"
        )
    else:
        print("first raw contact details: n/a")

    return {
        "noise_std": noise_std,
        "raw_rate": float(raw_touched.float().mean().item()),
        "gated_rate": float(gated_touched.float().mean().item()),
        "raw_peak_mean": float(raw_peak.mean().item()),
        "gated_peak_mean": float(gated_peak.mean().item()),
    }


def main():
    noise_stds = [float(x.strip()) for x in args_cli.noise_stds.split(",") if x.strip()]
    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    env_class = task_registry.get_task_class(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = args_cli.seed
    if args_cli.serve_y_center is not None:
        env_cfg.ball.serve_y_center = args_cli.serve_y_center
    if args_cli.serve_y_start is not None:
        env_cfg.ball.serve_y_start = args_cli.serve_y_start
    if args_cli.serve_y_wide is not None:
        env_cfg.ball.serve_y_wide = args_cli.serve_y_wide
    if args_cli.hit_target_y_min is not None or args_cli.hit_target_y_max is not None:
        y_min, y_max = env_cfg.robot.hit_target_y_range
        if args_cli.hit_target_y_min is not None:
            y_min = args_cli.hit_target_y_min
        if args_cli.hit_target_y_max is not None:
            y_max = args_cli.hit_target_y_max
        env_cfg.robot.hit_target_y_range = (y_min, y_max)

    env = env_class(env_cfg, args_cli.headless)
    try:
        print(
            f"[diagnose] task={args_cli.task} num_envs={env.num_envs} "
            f"steps={args_cli.steps} action_scale={env.action_scale} "
            f"require_active_contact={getattr(env.cfg.ball, 'require_active_contact', False)} "
            f"serve_y_center={env.cfg.ball.serve_y_center} "
            f"serve_y_start={env.cfg.ball.serve_y_start} "
            f"serve_y_wide={env.cfg.ball.serve_y_wide} "
            f"hit_target_y_range={env.cfg.robot.hit_target_y_range}"
        )
        results = []
        for idx, noise_std in enumerate(noise_stds):
            results.append(run_case(env, noise_std, args_cli.steps, args_cli.seed + idx))

        print("=" * 80)
        print("summary")
        for item in results:
            print(
                f"noise={item['noise_std']:g}: "
                f"raw_rate={100.0 * item['raw_rate']:.1f}% "
                f"gated_rate={100.0 * item['gated_rate']:.1f}% "
                f"raw_peak_mean={item['raw_peak_mean']:.4f} "
                f"gated_peak_mean={item['gated_peak_mean']:.4f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

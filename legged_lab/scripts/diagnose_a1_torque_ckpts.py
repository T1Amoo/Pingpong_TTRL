"""Compare A1 TT checkpoints for action, torque, and velocity saturation."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
from isaaclab.app import AppLauncher
from rsl_rl.rsl_rl.runners import OnPolicyRunner

from legged_lab.utils import task_registry
import legged_lab.utils.cli_args as cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Diagnose A1 TT checkpoint torque usage.")
parser.add_argument("--task", type=str, default="a1_tt_eval")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--predictor", action="store_true")
parser.add_argument("--checkpoints", type=str, required=True, help="Comma-separated model_*.pt list.")
parser.add_argument("--steps", type=int, default=300, help="Control steps per checkpoint.")
parser.add_argument("--near_dist", type=float, default=0.30)
parser.add_argument("--out", type=str, default="/tmp/a1_torque_ckpt_diag")
parser.add_argument(
    "--curriculum_offset_from_ckpt",
    action="store_true",
    help="Set sim_step_counter to ckpt_iter * decimation * num_steps_per_env before each rollout.",
)
parser.add_argument(
    "--real_effort",
    action="store_true",
    help="Disable proximal effort curriculum and use base action effort limits.",
)
parser.add_argument(
    "--sample_actions",
    action="store_true",
    help="Sample from the policy action distribution instead of using the deterministic inference mean.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa: F401,F403,E402
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402


def _ckpt_iter(checkpoint: str) -> int:
    match = re.search(r"model_(\d+)\.pt", checkpoint)
    return int(match.group(1)) if match else 0


def _as_float(value) -> float:
    return float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)


def _percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return _as_float(torch.quantile(values.float(), q))


def _bucket_stats(values: list[torch.Tensor], joint_names: list[str], limit_values: torch.Tensor | None = None) -> dict:
    if not values:
        zero = [0.0 for _ in joint_names]
        return {
            "count": 0,
            "global_max": 0.0,
            "global_p95": 0.0,
            "per_joint_max": zero,
            "per_joint_p95": zero,
            "per_joint_sat95_frac": zero,
            "worst_joint": "",
            "worst_joint_max": 0.0,
        }
    data = torch.cat(values, dim=0).float().abs().cpu()
    per_joint_max = data.max(dim=0).values
    per_joint_p95 = torch.quantile(data, 0.95, dim=0)
    per_joint_sat = (data >= 0.95).float().mean(dim=0)
    worst_idx = int(torch.argmax(per_joint_max).item())
    return {
        "count": int(data.shape[0]),
        "global_max": _as_float(data.max()),
        "global_p95": _percentile(data.flatten(), 0.95),
        "per_joint_max": [_as_float(x) for x in per_joint_max],
        "per_joint_p95": [_as_float(x) for x in per_joint_p95],
        "per_joint_sat95_frac": [_as_float(x) for x in per_joint_sat],
        "worst_joint": joint_names[worst_idx],
        "worst_joint_max": _as_float(per_joint_max[worst_idx]),
    }


def _reset_predictor_history(runner):
    for name, value in (
        ("_traj_len", 0),
        ("_traj_write_idx", 0),
        ("_pred_call_count", 0),
    ):
        if hasattr(runner, name):
            setattr(runner, name, value)
    for name in ("_warn_no_pred", "_warn_short_hist"):
        if hasattr(runner, name):
            delattr(runner, name)


def _collect_rollout(env, runner, policy, checkpoint: str, steps: int, near_dist: float, sample_actions: bool) -> dict:
    device = env.device
    all_ids = torch.arange(env.num_envs, device=device)
    env.reset(all_ids)
    obs, _ = env.get_observations()

    action_ids = list(env.action_joint_ids)
    joint_names = list(env.action_joint_names)
    base_limit = env._base_action_effort_limit.detach().clone()

    if args_cli.real_effort:
        env.cfg.robot.effort_curriculum_steps = 0
        env.robot.write_joint_effort_limit_to_sim(base_limit, action_ids)
    elif args_cli.curriculum_offset_from_ckpt:
        iter_num = _ckpt_iter(checkpoint)
        nspe = int(getattr(runner, "num_steps_per_env", 24) or 24)
        env.sim_step_counter = iter_num * int(env.cfg.sim.decimation) * nspe

    action_abs: list[torch.Tensor] = []
    applied_ratio_sim = {"all": [], "near": [], "hit": []}
    applied_ratio_real = {"all": [], "near": [], "hit": []}
    computed_ratio_real = {"all": [], "near": [], "hit": []}
    vel_ratio = {"all": [], "near": [], "hit": []}

    serve_success_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    serve_hit_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    succ_total = 0
    hit_total = 0
    serve_total = 0
    effort_scale_max = 1.0

    for _ in range(steps):
        with torch.no_grad():
            if sample_actions:
                actor_obs = runner.obs_normalizer(obs.to(runner.device))
                actions = runner.alg.policy.act(actor_obs).detach()
            else:
                actions = policy(obs)
        action_abs.append(actions.detach().abs().cpu())
        obs, _, _, _ = env.step(actions)
        if args_cli.predictor:
            with torch.no_grad():
                runner._record_ball_positions()
                runner._maybe_predict_and_update_env()

        ids = env.action_joint_ids
        applied = env.robot.data.applied_torque[:, ids].detach().abs()
        computed = env.robot.data.computed_torque[:, ids].detach().abs()
        sim_limit = torch.clamp(env.robot.data.joint_effort_limits[:, ids].detach().abs(), min=1e-6)
        real_limit = torch.clamp(base_limit, min=1e-6)
        jvel = env.robot.data.joint_vel[:, ids].detach().abs()
        vlim = torch.clamp(env.robot.data.joint_vel_limits[:, ids].detach().abs(), min=1e-6)

        effort_scale_max = max(effort_scale_max, _as_float((sim_limit[:, :3] / real_limit[:, :3]).max()))
        all_mask = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        near_mask = (env.paddel_ball_distance < near_dist) & (~env.mask_invalid)
        hit_mask = env.ball_contact_rew > 0.0

        for name, mask in (("all", all_mask), ("near", near_mask), ("hit", hit_mask)):
            if mask.any():
                applied_ratio_sim[name].append((applied[mask] / sim_limit[mask]).cpu())
                applied_ratio_real[name].append((applied[mask] / real_limit[mask]).cpu())
                computed_ratio_real[name].append((computed[mask] / real_limit[mask]).cpu())
                vel_ratio[name].append((jvel[mask] / vlim[mask]).cpu())

        event_mask = env.get_opponent_table_success_event()
        serve_success_flag |= event_mask
        serve_hit_flag |= hit_mask
        if hasattr(env, "ball_reset_ids") and env.ball_reset_ids is not None:
            reset_ids = env.ball_reset_ids
            if reset_ids.numel() > 0:
                serve_total += int(reset_ids.numel())
                succ_total += int(serve_success_flag[reset_ids].sum().item())
                hit_total += int(serve_hit_flag[reset_ids].sum().item())
                serve_success_flag[reset_ids] = False
                serve_hit_flag[reset_ids] = False

    action_data = torch.cat(action_abs, dim=0) if action_abs else torch.empty(0, len(joint_names))
    action_clip_frac = (action_data >= (0.98 * float(env.clip_actions))).float().mean(dim=0) if action_data.numel() else torch.zeros(len(joint_names))

    result = {
        "checkpoint": checkpoint,
        "steps": steps,
        "num_envs": env.num_envs,
        "serve_total": serve_total,
        "hit_total": hit_total,
        "success_total": succ_total,
        "hit_rate": hit_total / max(serve_total, 1),
        "success_rate": succ_total / max(serve_total, 1),
        "joint_names": joint_names,
        "base_effort_limit": [_as_float(x) for x in base_limit[0]],
        "effort_scale_max": effort_scale_max,
        "action_abs_global_max": _as_float(action_data.max()) if action_data.numel() else 0.0,
        "action_abs_global_p95": _percentile(action_data.flatten(), 0.95) if action_data.numel() else 0.0,
        "sample_actions": sample_actions,
        "action_clip_frac": [_as_float(x) for x in action_clip_frac],
        "applied_over_sim": {k: _bucket_stats(v, joint_names) for k, v in applied_ratio_sim.items()},
        "applied_over_real": {k: _bucket_stats(v, joint_names) for k, v in applied_ratio_real.items()},
        "computed_over_real": {k: _bucket_stats(v, joint_names) for k, v in computed_ratio_real.items()},
        "vel_over_limit": {k: _bucket_stats(v, joint_names) for k, v in vel_ratio.items()},
    }
    return result


def main():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 5
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    env = task_registry.get_task_class(args_cli.task)(env_cfg, args_cli.headless)

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    first_checkpoint = args_cli.checkpoints.split(",")[0].strip()
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, first_checkpoint)
    log_dir = os.path.dirname(resume_path)
    if args_cli.predictor:
        from rsl_rl.rsl_rl.runners import OnPolicyPredictorRegressionRunner

        runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for checkpoint in [x.strip() for x in args_cli.checkpoints.split(",") if x.strip()]:
        resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, checkpoint)
        print(f"[TORQUE] loading {checkpoint}: {resume_path}", flush=True)
        runner.load(resume_path, load_optimizer=False)
        _reset_predictor_history(runner)
        policy = runner.get_inference_policy(device=env.device)
        result = _collect_rollout(
            env,
            runner,
            policy,
            checkpoint,
            args_cli.steps,
            args_cli.near_dist,
            args_cli.sample_actions,
        )
        results.append(result)
        hit = result["hit_rate"]
        succ = result["success_rate"]
        all_real = result["applied_over_real"]["all"]
        hit_real = result["applied_over_real"]["hit"]
        vel_hit = result["vel_over_limit"]["hit"]
        print(
            "[TORQUE_SUMMARY] "
            f"{checkpoint} hit={hit:.3f} succ={succ:.3f} "
            f"eff_scale={result['effort_scale_max']:.2f} "
            f"all_real_max={all_real['global_max']:.2f} all_real_p95={all_real['global_p95']:.2f} "
            f"hit_real_max={hit_real['global_max']:.2f} hit_real_p95={hit_real['global_p95']:.2f} "
            f"hit_worst={hit_real['worst_joint']}:{hit_real['worst_joint_max']:.2f} "
            f"hit_vel_max={vel_hit['global_max']:.2f}",
            flush=True,
        )

    json_path = out_dir / "torque_ckpt_diag.json"
    csv_path = out_dir / "torque_ckpt_diag_summary.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "serve_total",
                "hit_rate",
                "success_rate",
                "effort_scale_max",
                "action_abs_p95",
                "all_real_max",
                "all_real_p95",
                "near_real_max",
                "near_real_p95",
                "hit_real_max",
                "hit_real_p95",
                "hit_worst_joint",
                "hit_worst_joint_max",
                "hit_vel_max",
                "hit_vel_p95",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "checkpoint": r["checkpoint"],
                    "serve_total": r["serve_total"],
                    "hit_rate": r["hit_rate"],
                    "success_rate": r["success_rate"],
                    "effort_scale_max": r["effort_scale_max"],
                    "action_abs_p95": r["action_abs_global_p95"],
                    "all_real_max": r["applied_over_real"]["all"]["global_max"],
                    "all_real_p95": r["applied_over_real"]["all"]["global_p95"],
                    "near_real_max": r["applied_over_real"]["near"]["global_max"],
                    "near_real_p95": r["applied_over_real"]["near"]["global_p95"],
                    "hit_real_max": r["applied_over_real"]["hit"]["global_max"],
                    "hit_real_p95": r["applied_over_real"]["hit"]["global_p95"],
                    "hit_worst_joint": r["applied_over_real"]["hit"]["worst_joint"],
                    "hit_worst_joint_max": r["applied_over_real"]["hit"]["worst_joint_max"],
                    "hit_vel_max": r["vel_over_limit"]["hit"]["global_max"],
                    "hit_vel_p95": r["vel_over_limit"]["hit"]["global_p95"],
                }
            )
    print(f"[TORQUE] wrote {json_path}", flush=True)
    print(f"[TORQUE] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()

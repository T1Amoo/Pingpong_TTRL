"""Record the trained g1_tt policy step-by-step for C++ deploy equivalence testing.

Deterministic (perception delay fixed at lag=0, noise off post-init), 1 env.
Outputs ref.npz + replay.csv + robot.csv.

jp_* columns in robot.csv are joint_pos_rel (joint_pos - default_joint_pos), policy order.
C++ ReplayArticulation should treat jp_* as already-relative for joint_pos_rel obs,
OR set default_joint_pos=0 and joint_pos=jp_*.
"""
import argparse
import csv
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="g1_tt_eval")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--load_run", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", default="/tmp/g1_tt_deploy_ref.npz")
AppLauncher.add_app_launcher_args(ap)
args, _ = ap.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import sys
import numpy as np
import torch
from isaaclab_rl.rsl_rl import export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
try:
    from rsl_rl.runners import OnPolicyPredictorRegressionRunner
except ImportError:
    from rsl_rl.rsl_rl.runners import OnPolicyPredictorRegressionRunner

from legged_lab.envs import *  # noqa: F401,F403
from legged_lab.utils.task_registry import task_registry

env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
env_cfg.scene.num_envs = 1

# Keep add_noise True during construction so init_obs_buffer can build delayed_perception;
# we disable noise on the env object after init (see below).

# Fix perception delay at lag=0 for determinism (keep enable=True so buffer code path runs).
env_cfg.domain_rand.perception_delay.params["min_delay"] = 0
env_cfg.domain_rand.perception_delay.params["max_delay"] = 0

env = task_registry.get_task_class(args.task)(env_cfg, headless=True)

# Disable noise injection after construction (env.add_noise is read in compute_observations)
env.add_noise = False

log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
resume = get_checkpoint_path(log_root, args.load_run, args.checkpoint)
runner = OnPolicyPredictorRegressionRunner(
    env,
    agent_cfg.to_dict(),
    log_dir=os.path.dirname(resume),
    device=agent_cfg.device,
)
runner.load(resume, load_optimizer=False)
policy = runner.get_inference_policy(device=env.device)

# Export policy.onnx alongside the checkpoint (exported/ dir next to the checkpoint)
_export_dir = os.path.join(os.path.dirname(resume), "exported")
os.makedirs(_export_dir, exist_ok=True)
export_policy_as_onnx(
    runner.alg.policy, normalizer=runner.obs_normalizer,
    path=_export_dir, filename="policy.onnx"
)
print(f"[record] exported policy.onnx to {_export_dir}/policy.onnx", flush=True)

oj = env.obs_joint_ids
keys = [
    "actor_obs", "action",
    "ang_vel", "proj_grav", "joint_pos_rel", "joint_vel", "last_action",
    "delayed_perception", "ball_pos", "robot_pos", "ball_pred", "heading", "root_quat",
]
rec = {k: [] for k in keys}

obs, _ = env.get_observations()
# Ensure ball_pos / robot_pos / delayed_perception are initialised before the first step
env.compute_perception()
with torch.inference_mode():
    for i in range(args.steps):
        action = policy(obs)
        r = env.robot
        rec["actor_obs"].append(obs[0].cpu().numpy().copy())
        rec["action"].append(action[0].cpu().numpy().copy())
        rec["ang_vel"].append(r.data.root_ang_vel_b[0].cpu().numpy().copy())
        rec["proj_grav"].append(r.data.projected_gravity_b[0].cpu().numpy().copy())
        rec["joint_pos_rel"].append(
            (r.data.joint_pos[0, oj] - r.data.default_joint_pos[0, oj]).cpu().numpy().copy()
        )
        rec["joint_vel"].append(r.data.joint_vel[0, oj].cpu().numpy().copy())
        rec["last_action"].append(
            env.action_buffer._circular_buffer.buffer[0, -1, :].cpu().numpy().copy()
        )
        rec["delayed_perception"].append(env.delayed_perception[0].cpu().numpy().copy())
        rec["ball_pos"].append(env.ball_pos[0].cpu().numpy().copy())
        rec["robot_pos"].append(env.robot_pos[0].cpu().numpy().copy())
        rec["ball_pred"].append(env.ball_prediction[0].cpu().numpy().copy())
        rec["heading"].append(np.array([float(r.data.heading_w[0])], dtype=np.float32))
        rec["root_quat"].append(r.data.root_quat_w[0].cpu().numpy().copy())  # w,x,y,z
        obs, _, _, _ = env.step(action)
        try:
            runner._record_ball_positions()
            runner._maybe_predict_and_update_env()
        except Exception:
            pass

meta = dict(
    history_length=int(env.cfg.robot.actor_obs_history_length),
    num_actions=int(env.num_actions),
    default_joint_pos=r.data.default_joint_pos[0, oj].cpu().numpy().tolist(),
    note="actor_obs frame-major [frame0(all terms)...]; joints in obs_joint_ids order",
)
np.savez(
    args.out,
    meta=np.array([str(meta)]),
    **{k: np.array(v, dtype=np.float32) for k, v in rec.items()},
)
print(
    f"[record] wrote {args.out}: {args.steps} steps, "
    f"actor_obs={rec['actor_obs'][0].shape}, action={rec['action'][0].shape}",
    flush=True,
)

# replay.csv: ball+robot perception per step (delayed_perception split: [ball3, robot3]) + heading
csvp = args.out.replace(".npz", "_replay.csv")
with open(csvp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["bx", "by", "bz", "rx", "ry", "rz", "heading"])
    for i in range(len(rec["delayed_perception"])):
        dp = rec["delayed_perception"][i]
        hd = float(rec["heading"][i][0])
        w.writerow([dp[0], dp[1], dp[2], dp[3], dp[4], dp[5], hd])
print(f"[record] wrote {csvp}", flush=True)

# robot.csv: per-step robot proprio in policy order for the C++ ReplayArticulation.
# jp_* are joint_pos_rel (joint_pos - default_joint_pos), policy order.
robp = args.out.replace(".npz", "_robot.csv")
with open(robp, "w", newline="") as f:
    w = csv.writer(f)
    hdr = (
        ["jp_%d" % k for k in range(23)]
        + ["jv_%d" % k for k in range(23)]
        + ["av0", "av1", "av2", "pg0", "pg1", "pg2", "qw", "qx", "qy", "qz"]
        + ["la_%d" % k for k in range(23)]
    )
    w.writerow(hdr)
    for i in range(len(rec["actor_obs"])):
        jp = rec["joint_pos_rel"][i]
        jv = rec["joint_vel"][i]
        av = rec["ang_vel"][i]
        pg = rec["proj_grav"][i]
        q = rec["root_quat"][i]
        la = rec["last_action"][i]
        w.writerow([*jp.tolist(), *jv.tolist(), *av.tolist(), *pg.tolist(), *q.tolist(), *la.tolist()])
print(f"[record] wrote {robp}", flush=True)

env.close()
app.close()

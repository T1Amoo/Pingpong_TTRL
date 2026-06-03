"""Instrumented rollout: load trained g1_tt policy, log paddle orientation & ball geometry at contact.

Confirms whether the robot hits with the blade face pointing DOWN (normal_z<0) and where the
ball goes afterwards (outgoing vx). Headless, no GUI. Writes summary to /tmp/g1_contact_probe.txt.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="g1_tt_eval")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=2500)
parser.add_argument("--load_run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, default="/tmp/g1_contact_probe.txt")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import os, torch
import isaaclab.utils.math as mu
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyPredictorRegressionRunner
from legged_lab.envs import *  # noqa
from legged_lab.utils.task_registry import task_registry

env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
env_cfg.scene.num_envs = args.num_envs
env = task_registry.get_task_class(args.task)(env_cfg, headless=True)

log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
resume = get_checkpoint_path(log_root, args.load_run, args.checkpoint)
runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(resume), device=agent_cfg.device)
runner.load(resume, load_optimizer=False)
policy = runner.get_inference_policy(device=env.device)

N = env.num_envs
dev = env.device
zaxis = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(N, 3)
xaxis = torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(N, 3)
yaxis = torch.tensor([[0.0, 1.0, 0.0]], device=dev).expand(N, 3)
prev_touch = torch.zeros(N, dtype=torch.bool, device=dev)
pend = torch.full((N,), -1, dtype=torch.long, device=dev)
nz, nx, bz, outvx = [], [], [], []
fwd_w, lat_w, nrm_w = [], [], []   # world dir of local +X,+Y,+Z at contact (avg)

obs, _ = env.get_observations()
with torch.inference_mode():
    for step in range(args.steps):
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        try:
            runner._record_ball_positions(); runner._maybe_predict_and_update_env()
        except Exception:
            pass
        pid = env._paddle_body_id
        q = env.robot.data.body_quat_w[:, pid, :]
        nrm = mu.quat_apply(q, zaxis)          # world blade-face normal
        fwd = mu.quat_apply(q, xaxis)          # world forearm/extension (+X)
        lat = mu.quat_apply(q, yaxis)          # world lateral (+Y)
        touch = env.has_touch_paddle
        new_hit = touch & ~prev_touch
        if new_hit.any():
            for i in new_hit.nonzero(as_tuple=False).squeeze(-1).tolist():
                nz.append(float(nrm[i, 2])); nx.append(float(nrm[i, 0]))
                bz.append(float(env.ball.data.root_pos_w[i, 2]))
                fwd_w.append([float(fwd[i, 0]), float(fwd[i, 1]), float(fwd[i, 2])])
                lat_w.append([float(lat[i, 0]), float(lat[i, 1]), float(lat[i, 2])])
                nrm_w.append([float(nrm[i, 0]), float(nrm[i, 1]), float(nrm[i, 2])])
            pend[new_hit] = step + 6
        due = (pend == step)
        if due.any():
            for i in due.nonzero(as_tuple=False).squeeze(-1).tolist():
                outvx.append(float(env.ball.data.root_lin_vel_w[i, 0]))
            pend[due] = -1
        prev_touch = touch.clone()

def stats(name, xs):
    if not xs: return f"{name}: n=0"
    xs = sorted(xs); n = len(xs); mean = sum(xs)/n; med = xs[n//2]
    return f"{name}: n={n} mean={mean:+.3f} median={med:+.3f} min={xs[0]:+.3f} max={xs[-1]:+.3f}"

lines = [
    f"task={args.task} ckpt={os.path.basename(resume)} hits_recorded={len(nz)}",
    stats("blade_normal_z @contact (DOWN if <0)", nz),
    stats("blade_normal_x @contact (toward opponent if >0)", nx),
    stats("ball_world_z @contact (table=0.76)", bz),
    stats("ball vx +6 steps after hit (toward opponent if >0)", outvx),
]
if nz:
    frac_down = sum(1 for v in nz if v < -0.3) / len(nz)
    frac_fwd_vx = sum(1 for v in outvx if v > 0.3) / max(1, len(outvx))
    lines.append(f"fraction face-down (nz<-0.3): {frac_down:.2f} | fraction ball-forward (vx>0.3): {frac_fwd_vx:.2f}")
    def avg3(L):
        n=len(L); return [round(sum(r[k] for r in L)/n, 3) for k in range(3)]
    lines.append(f"avg world dir of local +X (forearm) @contact: {avg3(fwd_w)}")
    lines.append(f"avg world dir of local +Y (lateral) @contact: {avg3(lat_w)}")
    lines.append(f"avg world dir of local +Z (face normal) @contact: {avg3(nrm_w)}")
    lines.append("(opponent is world +x; pick the local axis whose world-x is most positive to be the new face normal)")
out = "\n".join(lines)
print(out)
with open(args.out, "w") as f:
    f.write(out + "\n")
env.close(); app.close()

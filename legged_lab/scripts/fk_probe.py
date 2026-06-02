"""G1 TT probe: load env, report joints/paddle body, hold zero action, print paddle vs base geometry."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="g1_tt")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--out", type=str, default="/tmp/g1_probe_result.txt")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import torch
from legged_lab.envs import *  # noqa
from legged_lab.utils.task_registry import task_registry

_lines = []
def report(msg):
    print(msg, flush=True)
    _lines.append(str(msg))

env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
env_cfg.scene.num_envs = 1
env_class = task_registry.get_task_class(args.task)
env = env_class(env_cfg, headless=True)

robot = env.robot
report("[probe] num joints: " + str(robot.num_joints))
report("[probe] joint names: " + str(robot.joint_names))
pid = env._paddle_body_id
report("[probe] paddle body: " + str(robot.body_names[pid]) + " idx " + str(pid))

# zero-action hold
act = torch.zeros(env.num_envs, env.num_actions, device=env.device)
zmin = 1e9
z0 = float(env.robot.data.root_link_pos_w[0, 2].item())
ztraj = []
for i in range(args.steps):
    env.step(act)
    z = float(env.robot.data.root_link_pos_w[0, 2].item())
    zmin = min(zmin, z)
    if i % 25 == 0 or i == args.steps - 1:
        ztraj.append((i, round(z, 3)))
zfinal = float(env.robot.data.root_link_pos_w[0, 2].item())
report("[probe] pelvis z init: " + str(round(z0, 3)))
report("[probe] pelvis z trajectory: " + str(ztraj))
report("[probe] pelvis z final: " + str(round(zfinal, 3)))
report("[probe] min pelvis z over hold: " + str(round(zmin, 3)))

# paddle vs base geometry (ready stance)
import isaaclab.utils.math as mu
paddle_pos = robot.data.body_pos_w[:, pid, :]
paddle_quat = robot.data.body_quat_w[:, pid, :]
off = torch.tensor(env_cfg.robot.paddle_offset, device=env.device).unsqueeze(0).expand_as(paddle_pos)
face = paddle_pos + mu.quat_apply(paddle_quat, off)
base = robot.data.root_link_pos_w
rel = (face - base)[0]
report("[probe] paddle_face - base (x,y,z): " + str([round(float(v),3) for v in rel]))

with open(args.out, "w") as f:
    f.write("\n".join(_lines) + "\n")
print("[probe] wrote results to " + args.out, flush=True)
env.close()
app.close()

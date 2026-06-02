"""G1 TT probe: load env, report joints/paddle body, hold zero action, print paddle vs base geometry."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="g1_tt")
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import torch
from legged_lab.envs import *  # noqa
from legged_lab.utils.task_registry import task_registry

env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
env_cfg.scene.num_envs = 1
env_class = task_registry.get_task_class(args.task)
env = env_class(env_cfg, headless=True)

robot = env.robot
print("[probe] num joints:", robot.num_joints)
print("[probe] joint names:", robot.joint_names)
pid = env._paddle_body_id
print("[probe] paddle body:", robot.body_names[pid], "idx", pid)

# zero-action hold
act = torch.zeros(env.num_envs, env.num_actions, device=env.device)
zmin = 1e9
for i in range(args.steps):
    env.step(act)
    z = float(env.robot.data.root_link_pos_w[0, 2].item())
    zmin = min(zmin, z)
print("[probe] min pelvis z over hold:", round(zmin, 3))

# paddle vs base geometry (ready stance)
import isaaclab.utils.math as mu
paddle_pos = robot.data.body_pos_w[:, pid, :]
paddle_quat = robot.data.body_quat_w[:, pid, :]
off = torch.tensor(env_cfg.robot.paddle_offset, device=env.device).unsqueeze(0).expand_as(paddle_pos)
face = paddle_pos + mu.quat_apply(paddle_quat, off)
base = robot.data.root_link_pos_w
rel = (face - base)[0]
print("[probe] paddle_face - base (x,y,z):", [round(float(v),3) for v in rel])
env.close()
app.close()

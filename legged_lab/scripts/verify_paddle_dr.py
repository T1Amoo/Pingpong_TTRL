"""Verify the v9 paddle-restitution domain randomization actually applies PER-ENV.

Guards against the no-op trap (cf. the zero_23dof no-op bug): if the paddle body's per-env
material restitution does NOT vary across envs after the startup events, the DR is inert and
we MUST NOT spend a training run on it.

Reads back robot.root_physx_view material properties for ONLY the right_tt_paddle_link shapes
and checks they span the configured restitution_range across envs.

Run (cloud):
  OMNI_KIT_ACCEPT_EULA=YES /root/miniconda3/envs/pingpong/bin/python \
    -m legged_lab.scripts.verify_paddle_dr --num_envs 64 --headless
Exit 0 = PASS (restitution varies per-env), 1 = FAIL (inert / no-op).
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="g1_tt_dr")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import torch  # noqa: F401

from legged_lab.envs import *  # noqa: F401,F403  (registers tasks)
from legged_lab.utils import task_registry


def main():
    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_class = task_registry.get_task_class(args_cli.task)
    # __init__ applies all "startup" events (incl. paddle_restitution) after sim.reset().
    env = env_class(env_cfg, True)

    robot = env.robot
    # material buffer: [num_envs, total_shapes, 3] with col 2 = restitution
    mat = robot.root_physx_view.get_material_properties()
    restit = mat[..., 2]

    # locate the paddle body's shape index range (mirrors randomize_rigid_body_material).
    # The paddle is fixed-joint-merged into the wrist body, so target that body.
    body_ids, body_names = robot.find_bodies("right_wrist_roll_rubber_hand")
    bid = body_ids[0]
    num_shapes_per_body = []
    for link_path in robot.root_physx_view.link_paths[0]:
        lv = robot._physics_sim_view.create_rigid_body_view(link_path)
        num_shapes_per_body.append(lv.max_shapes)
    start = sum(num_shapes_per_body[:bid])
    end = start + num_shapes_per_body[bid]
    paddle_restit = restit[:, start:end]

    pmin = float(paddle_restit.min())
    pmax = float(paddle_restit.max())
    pstd = float(paddle_restit.std())
    # restitution of the OTHER (non-paddle) robot shapes — should stay ~0 (the inherited .* event)
    other = torch.cat([restit[:, :start], restit[:, end:]], dim=1)
    print(f"[VERIFY] task={args_cli.task} num_envs={args_cli.num_envs} "
          f"paddle_body='{body_names[0]}' shape_idx=[{start},{end})", flush=True)
    print(f"[VERIFY] paddle restitution per-env: min={pmin:.3f} max={pmax:.3f} std={pstd:.3f}", flush=True)
    print(f"[VERIFY] other-body restitution: min={float(other.min()):.4f} max={float(other.max()):.4f}", flush=True)
    print(f"[VERIFY] sample paddle restitution (first 12 envs): "
          f"{[round(float(v), 3) for v in paddle_restit[:12, 0]]}", flush=True)

    ok = (pstd > 0.05) and (pmax > 0.10) and (pmin >= 0.0)
    if ok:
        print("[VERIFY] PASS: paddle restitution varies per-env — DR is effective.", flush=True)
    else:
        print("[VERIFY] FAIL: paddle restitution does NOT vary per-env — DR is a NO-OP. "
              "Do not train; revisit the mechanism.", flush=True)
    os._exit(0 if ok else 1)


main()

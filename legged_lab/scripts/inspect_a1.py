"""Inspect the canonical A1 table-tennis articulation.

Examples:
    OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/inspect_a1.py --headless
    OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/inspect_a1.py --preview
    OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/inspect_a1.py --preview --pose 0.569,-0.692,0.717,1.13,-1.24,0.0314,0.772
"""

import argparse
import copy
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--settle", action="store_true", help="Use a free root and drop from z=0.3 for height checks.")
parser.add_argument("--lift", type=float, default=None, help="Override sj init value.")
parser.add_argument("--preview", action="store_true", help="Run headed and hold the pose for visual inspection.")
parser.add_argument("--pose", type=str, default=None, help="Override r1..r7: 7 comma-separated radians.")
parser.add_argument("--steps", type=int, default=400, help="Number of sim steps for non-preview inspection.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
if args.preview:
    args.headless = False
app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from assets.a1.a1 import A1_TT_CFG  # noqa: E402


PADDLE_BODY = "Link_r_paddle"
RIGHT_WHEELS = ("Link_lun_r", "Link_lun_l")


def build_cfg():
    cfg = copy.deepcopy(A1_TT_CFG).replace(prim_path="/World/Robot")
    cfg.init_state.pos = (0.0, 0.0, 0.3 if args.settle else A1_TT_CFG.init_state.pos[2])
    cfg.spawn.articulation_props.fix_root_link = not args.settle

    joint_pos = dict(cfg.init_state.joint_pos)
    if args.lift is not None:
        joint_pos["sj"] = args.lift
    if args.pose is not None:
        vals = [float(v.strip()) for v in args.pose.split(",") if v.strip()]
        assert len(vals) == 7, "--pose needs 7 comma-separated values for r1..r7"
        for i, value in enumerate(vals, start=1):
            joint_pos[f"r{i}"] = value
    cfg.init_state.joint_pos = joint_pos
    return cfg


def print_robot_state(robot, title):
    body_pos = robot.data.body_pos_w[0]
    body_quat = robot.data.body_quat_w[0]
    base_idx = robot.body_names.index("base_link")
    paddle_idx = robot.body_names.index(PADDLE_BODY)

    print("=" * 72)
    print(title)
    print("JOINTS:", robot.joint_names)
    print("BODIES:", robot.body_names)
    print(f"num_joints={robot.num_joints}  num_bodies={robot.num_bodies}")

    base = body_pos[base_idx].tolist()
    paddle = body_pos[paddle_idx].tolist()
    paddle_quat = body_quat[paddle_idx].tolist()
    print(f"base_link     xyz=({base[0]:+.4f}, {base[1]:+.4f}, {base[2]:+.4f})")
    print(f"{PADDLE_BODY:13s} xyz=({paddle[0]:+.4f}, {paddle[1]:+.4f}, {paddle[2]:+.4f})")
    print(
        f"{PADDLE_BODY:13s} quat(wxyz)=({paddle_quat[0]:+.4f}, {paddle_quat[1]:+.4f}, "
        f"{paddle_quat[2]:+.4f}, {paddle_quat[3]:+.4f})"
    )

    for name in RIGHT_WHEELS:
        idx = robot.body_names.index(name)
        pos = body_pos[idx].tolist()
        print(f"{name:13s} xyz=({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")

    print("\nPER-BODY WORLD POSITIONS:")
    for i, (name, pos) in enumerate(zip(robot.body_names, body_pos.tolist())):
        print(f"  [{i:02d}] {name:40s}  x={pos[0]:8.4f}  y={pos[1]:8.4f}  z={pos[2]:8.4f}")

    all_pos = robot.data.body_pos_w[0].cpu()
    print(f"NaN check: {'FAIL' if torch.any(torch.isnan(all_pos)) else 'OK'}")


cfg = build_cfg()
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
gp = sim_utils.GroundPlaneCfg()
gp.func("/World/ground", gp)
robot = Articulation(cfg)
sim.reset()

print(f"init joint r1..r7 = {[round(cfg.init_state.joint_pos[f'r{i}'], 4) for i in range(1, 8)]}")
print(f"init sj = {cfg.init_state.joint_pos.get('sj')}")
print(f"fix_root_link = {cfg.spawn.articulation_props.fix_root_link}")

if args.preview:
    print("PREVIEW: holding canonical A1_TT_CFG pose. Close the window or Ctrl+C to stop.")
    i = 0
    while app.is_running():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.current_time)
        if i % 400 == 0:
            paddle_idx = robot.body_names.index(PADDLE_BODY)
            paddle = robot.data.body_pos_w[0, paddle_idx].tolist()
            print(f"[{i:6d}] {PADDLE_BODY} world=({paddle[0]:+.3f}, {paddle[1]:+.3f}, {paddle[2]:+.3f})")
        i += 1
else:
    for i in range(args.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.current_time)
        if i % 100 == 0:
            base_z = robot.data.body_pos_w[0, 0, 2].item()
            print(f"  step {i:4d}  base_link z = {base_z:.4f}")
    print_robot_state(robot, "A1 INSPECTION")

app.close()

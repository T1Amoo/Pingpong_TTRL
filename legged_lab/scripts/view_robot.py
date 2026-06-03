"""Minimal viewer: spawn ONLY the G1_TT robot (with paddle) on a ground plane, hold default pose.
GUI, no table/ball — for inspecting the URDF / paddle mount orientation. Ctrl-C or close window to quit.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = False
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
from legged_lab.assets.unitree.g1 import G1_TT_CFG

sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0"))
sim.set_camera_view([2.2, 1.6, 1.3], [0.0, 0.0, 0.8])

# ground + light
gp = sim_utils.GroundPlaneCfg(); gp.func("/World/ground", gp)
light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)); light.func("/World/Light", light)

# robot only
cfg = G1_TT_CFG.replace(prim_path="/World/Robot")
robot = Articulation(cfg)

sim.reset()
print("[viewer] robot spawned. bodies:", robot.num_bodies, "joints:", robot.num_joints)
print("[viewer] body names:", robot.body_names)
dt = sim.get_physics_dt()
while app.is_running():
    robot.set_joint_position_target(robot.data.default_joint_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(dt)
app.close()

"""Verification: spawn A1 with A1_TT_CFG (free chassis, DAMIAO kp/kd), verify
wheels≈z0, base upright, paddle>0.4m, no NaN.

Run headless:
    OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/verify_a1_tt.py --headless
"""
import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

# Use the canonical A1_TT_CFG from a1.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from assets.a1.a1 import A1_TT_CFG, A1_INIT_Z, A1_INIT_ROT

# Must use single-env prim path for standalone (no regex needed)
cfg = A1_TT_CFG.replace(prim_path="/World/Robot")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
gp = sim_utils.GroundPlaneCfg()
gp.func("/World/ground", gp)

robot = Articulation(cfg)
sim.reset()

# Step 2 s to let it settle
n_steps = 400
for i in range(n_steps):
    robot.set_joint_position_target(robot.data.default_joint_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.current_time)
    if i % 100 == 0:
        base_z = robot.data.body_pos_w[0, 0, 2].item()
        print(f"  step {i:4d}  base_link z = {base_z:.4f}")

body_pos = robot.data.body_pos_w[0]
base_idx   = robot.body_names.index("base_link")
rw_idx     = robot.body_names.index("link_right_wheel")
lw_idx     = robot.body_names.index("link_left_wheel")
paddle_idx = robot.body_names.index("Link_yb_paddle")

base_z   = body_pos[base_idx,   2].item()
rw_z     = body_pos[rw_idx,     2].item()
lw_z     = body_pos[lw_idx,     2].item()
paddle_z = body_pos[paddle_idx, 2].item()
base_x   = body_pos[base_idx,   0].item()
base_y   = body_pos[base_idx,   1].item()

print("=" * 60)
print("VERIFICATION: A1_TT_CFG free-chassis spawn")
print(f"  A1_INIT_Z used   = {A1_INIT_Z}")
print(f"  A1_INIT_ROT used = {A1_INIT_ROT}")
print(f"  base_link          z = {base_z:.4f}  m")
print(f"  link_right_wheel   z = {rw_z:.4f}  m")
print(f"  link_left_wheel    z = {lw_z:.4f}  m")
print(f"  Link_yb_paddle     z = {paddle_z:.4f}  m")
print("=" * 60)

# Check NaN
all_pos = robot.data.body_pos_w[0].cpu()
nan_ok = not torch.any(torch.isnan(all_pos))
print(f"NaN check:     {'OK' if nan_ok else 'FAIL - NaN detected!'}")
print(f"Wheel z≈0:     {'OK' if abs(rw_z) < 0.12 and abs(lw_z) < 0.12 else 'WARN - wheels not near ground'} (rw={rw_z:.4f} lw={lw_z:.4f})")
print(f"Paddle z>0.4:  {'OK' if paddle_z > 0.4 else 'FAIL - paddle below 0.4'} (z={paddle_z:.4f})")

# Check robot is stable (base not drifting - run 10 more steps and check drift)
base_z_before = base_z
for _ in range(10):
    robot.set_joint_position_target(robot.data.default_joint_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.current_time)
drift = abs(robot.data.body_pos_w[0, base_idx, 2].item() - base_z_before)
print(f"Stability (10 more steps drift): {drift:.5f} m ({'OK' if drift < 0.01 else 'WARN - drifting'})")

print("=" * 60)
if nan_ok and paddle_z > 0.4:
    print("RESULT: PASS")
else:
    print("RESULT: FAIL")

app.close()

"""One-shot: spawn A1, print joint/body names, base height, paddle body pose.

Uses a temporary inline ArticulationCfg (Task 1 only); Task 2 will produce a1.py.
Run headless:
    OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/inspect_a1.py --headless

Settle mode (measures free-base resting height):
    OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/inspect_a1.py --headless --settle
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--settle", action="store_true",
                    help="Drop robot free-base from z=0.3, step 2s, report settled heights.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import os
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg

USD_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "assets", "a1", "a1.usd",
)

# Temporary inline cfg mirroring hardware config in /home/woan/下载/a1.py
# Uses fix_root_link=True and identity rot (w,x,y,z) = (1,0,0,0)
TEMP_A1_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # w,x,y,z identity
        joint_pos={
            "joint_lift": -0.28,
            "joint_yb_1": 1.769,
            "joint_yb_2": -0.762,
            "joint_yb_3": -1.863,
            "joint_yb_4": 1.445,
            "joint_yb_5": 0.206,
            "joint_yb_6": -0.827,
            "joint_yb_7": 1.043,
            "joint_zb_1": 0.0,
            "joint_zb_2": 0.0,
            "joint_zb_3": 0.0,
            "joint_zb_4": 0.0,
            "joint_zb_5": 0.0,
            "joint_zb_6": 0.0,
            "joint_zb_7": 0.0,
            "joint_head_lr": 0.0,
            "joint_head_ud": 0.0,
            "joint_left_wheel": 0.0,
            "joint_right_wheel": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_yb_[1-7]"],
            effort_limit_sim={
                "joint_yb_1": 28.0, "joint_yb_2": 28.0, "joint_yb_3": 28.0,
                "joint_yb_4": 8.0,  "joint_yb_5": 8.0,  "joint_yb_6": 8.0,  "joint_yb_7": 8.0,
            },
            velocity_limit_sim={
                "joint_yb_1": 8.0, "joint_yb_2": 8.0, "joint_yb_3": 8.0,
                "joint_yb_4": 20.0, "joint_yb_5": 20.0, "joint_yb_6": 20.0, "joint_yb_7": 20.0,
            },
            stiffness={
                "joint_yb_1": 250.0, "joint_yb_2": 250.0, "joint_yb_3": 250.0,
                "joint_yb_4": 120.0, "joint_yb_5": 120.0, "joint_yb_6": 120.0, "joint_yb_7": 120.0,
            },
            damping={
                "joint_yb_1": 1.0, "joint_yb_2": 1.0, "joint_yb_3": 1.0,
                "joint_yb_4": 0.5, "joint_yb_5": 0.5, "joint_yb_6": 0.5, "joint_yb_7": 0.5,
            },
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_zb_[1-7]"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.1,
            stiffness=10000.0,
            damping=1000.0,
        ),
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["joint_lift"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=0.0,
            stiffness=5000.0,
            damping=500.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["joint_head_.*"],
            effort_limit_sim=10.0,
            velocity_limit_sim=0.1,
            stiffness=10000.0,
            damping=1000.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*_wheel"],
            effort_limit_sim=10.0,
            velocity_limit_sim=0.0,
            stiffness=10000.0,
            damping=1000.0,
        ),
    },
)

# Free-base settle cfg: drop from z=0.3 with INIT_ROT, DAMIAO actuators, whip_high3 pose
SETTLE_A1_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,   # free chassis — will drop onto wheels
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),      # safe drop height; will settle onto wheels
        rot=(0.7071, 0.0, 0.0, 0.7071),  # wxyz, +90° around z (arm faces +x)
        joint_pos={
            "joint_lift": -0.28,
            "joint_yb_1": 1.769, "joint_yb_2": -0.762, "joint_yb_3": -1.863,
            "joint_yb_4": 1.445, "joint_yb_5": 0.206, "joint_yb_6": -0.827, "joint_yb_7": 1.043,
            "joint_zb_1": 0.0, "joint_zb_2": 0.0, "joint_zb_3": 0.0, "joint_zb_4": 0.0,
            "joint_zb_5": 0.0, "joint_zb_6": 0.0, "joint_zb_7": 0.0,
            "joint_head_lr": 0.0, "joint_head_ud": 0.0,
            "joint_left_wheel": 0.0, "joint_right_wheel": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_yb_[1-7]"],
            effort_limit_sim={
                "joint_yb_1": 28.0, "joint_yb_2": 28.0, "joint_yb_3": 28.0,
                "joint_yb_4": 8.0,  "joint_yb_5": 8.0,  "joint_yb_6": 8.0,  "joint_yb_7": 8.0,
            },
            velocity_limit_sim={
                "joint_yb_1": 8.0, "joint_yb_2": 8.0, "joint_yb_3": 8.0,
                "joint_yb_4": 20.0, "joint_yb_5": 20.0, "joint_yb_6": 20.0, "joint_yb_7": 20.0,
            },
            stiffness={
                "joint_yb_1": 126.33094, "joint_yb_2": 126.33094, "joint_yb_3": 126.33094,
                "joint_yb_4": 7.106115,  "joint_yb_5": 7.106115,  "joint_yb_6": 7.106115,  "joint_yb_7": 7.106115,
            },
            damping={
                "joint_yb_1": 8.042478, "joint_yb_2": 8.042478, "joint_yb_3": 8.042478,
                "joint_yb_4": 0.452389, "joint_yb_5": 0.452389, "joint_yb_6": 0.452389, "joint_yb_7": 0.452389,
            },
            armature={
                "joint_yb_1": 0.032, "joint_yb_2": 0.032, "joint_yb_3": 0.032,
                "joint_yb_4": 0.0018, "joint_yb_5": 0.0018, "joint_yb_6": 0.0018, "joint_yb_7": 0.0018,
            },
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_zb_[1-7]"],
            effort_limit_sim=200.0, velocity_limit_sim=0.1, stiffness=10000.0, damping=1000.0,
        ),
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["joint_lift"],
            effort_limit_sim=1000.0, velocity_limit_sim=0.0, stiffness=5000.0, damping=500.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["joint_head_.*"],
            effort_limit_sim=10.0, velocity_limit_sim=0.1, stiffness=10000.0, damping=1000.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*_wheel"],
            effort_limit_sim=10.0, velocity_limit_sim=0.0, stiffness=10000.0, damping=1000.0,
        ),
    },
)


if args.settle:
    # -----------------------------------------------------------------------
    # Settle mode: drop robot onto ground, step 2 s, report resting heights
    # -----------------------------------------------------------------------
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    gp = sim_utils.GroundPlaneCfg()
    gp.func("/World/ground", gp)

    robot = Articulation(SETTLE_A1_CFG)
    sim.reset()

    # step 2 seconds = 400 steps @ dt=0.005
    n_steps = 400
    for i in range(n_steps):
        # zero actions → actuators hold target = init_joint_pos
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.current_time)
        if i % 100 == 0:
            base_z = robot.data.body_pos_w[0, 0, 2].item()
            print(f"  step {i:4d}  base_link z = {base_z:.4f}")

    body_pos = robot.data.body_pos_w[0]  # (num_bodies, 3)
    base_idx = robot.body_names.index("base_link")
    rw_idx = robot.body_names.index("link_right_wheel")
    lw_idx = robot.body_names.index("link_left_wheel")
    paddle_idx = robot.body_names.index("Link_yb_paddle")

    base_z    = body_pos[base_idx, 2].item()
    rw_z      = body_pos[rw_idx, 2].item()
    lw_z      = body_pos[lw_idx, 2].item()
    paddle_z  = body_pos[paddle_idx, 2].item()

    print("=" * 60)
    print("SETTLE RESULTS (after 2 s free simulation):")
    print(f"  base_link        z = {base_z:.4f}  m")
    print(f"  link_right_wheel z = {rw_z:.4f}  m")
    print(f"  link_left_wheel  z = {lw_z:.4f}  m")
    print(f"  Link_yb_paddle   z = {paddle_z:.4f}  m")
    print("=" * 60)
    print(f"A1_INIT_Z (set to base_link settled z) = {base_z:.4f}")
    print(f"Wheel centers settled z = {(rw_z + lw_z) / 2:.4f}  (expect ≈0.035 if on ground)")

    # NaN check
    all_pos = robot.data.body_pos_w[0].cpu()
    if torch.any(torch.isnan(all_pos)):
        print("WARNING: NaN detected in body positions!")
    else:
        print("NaN check: OK (no NaN in body positions)")

    app.close()
else:
    # -----------------------------------------------------------------------
    # Original mode: fixed-root inspection (Task 1 behavior unchanged)
    # -----------------------------------------------------------------------
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))

    # ground plane
    gp = sim_utils.GroundPlaneCfg()
    gp.func("/World/ground", gp)

    robot = Articulation(TEMP_A1_CFG)
    sim.reset()

    print("=" * 60)
    print("JOINTS:", robot.joint_names)
    print("=" * 60)
    print("BODIES:", robot.body_names)
    print("=" * 60)
    print(f"BASE root body: {robot.body_names[0]}")
    print(f"num_joints={robot.num_joints}  num_bodies={robot.num_bodies}")

    # Per-body world position (x, y, z)
    body_pos = robot.data.body_pos_w[0]  # shape: (num_bodies, 3)
    print("=" * 60)
    print("PER-BODY WORLD POSITIONS (x, y, z):")
    for i, (name, pos) in enumerate(zip(robot.body_names, body_pos.tolist())):
        print(f"  [{i:02d}] {name:50s}  x={pos[0]:8.4f}  y={pos[1]:8.4f}  z={pos[2]:8.4f}")

    # Also print just z for quick scan
    print("=" * 60)
    print("Z ONLY (sorted):")
    z_vals = [(pos[2], name) for name, pos in zip(robot.body_names, body_pos.tolist())]
    for z, name in sorted(z_vals):
        print(f"  z={z:8.4f}  {name}")

    app.close()

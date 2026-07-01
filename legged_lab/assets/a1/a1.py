"""A1 AGV arm robot config for table tennis (free chassis, right-arm-only training)."""
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

A1_USD_PATH = os.path.join(os.path.dirname(__file__), "a1.usd")

# DAMIAO motors (ω_n=2π·10=62.83): kp=armature·ω_n², kd≈2·2·armature·ω_n
_KP = {"joint_yb_1": 126.33094, "joint_yb_2": 126.33094, "joint_yb_3": 126.33094,
       "joint_yb_4": 7.106115, "joint_yb_5": 7.106115, "joint_yb_6": 7.106115, "joint_yb_7": 7.106115}
_KD = {"joint_yb_1": 8.042478, "joint_yb_2": 8.042478, "joint_yb_3": 8.042478,
       "joint_yb_4": 0.452389, "joint_yb_5": 0.452389, "joint_yb_6": 0.452389, "joint_yb_7": 0.452389}
_ARM = {"joint_yb_1": 0.032, "joint_yb_2": 0.032, "joint_yb_3": 0.032,
        "joint_yb_4": 0.0018, "joint_yb_5": 0.0018, "joint_yb_6": 0.0018, "joint_yb_7": 0.0018}
_EFFORT = {"joint_yb_1": 28.0, "joint_yb_2": 28.0, "joint_yb_3": 28.0,
           "joint_yb_4": 8.0, "joint_yb_5": 8.0, "joint_yb_6": 8.0, "joint_yb_7": 8.0}
_VEL = {"joint_yb_1": 8.0, "joint_yb_2": 8.0, "joint_yb_3": 8.0,
        "joint_yb_4": 20.0, "joint_yb_5": 20.0, "joint_yb_6": 20.0, "joint_yb_7": 20.0}

# Robot FACING (whole-body orientation), from a1_facts.md measured link positions under identity rot:
# the chassis is built spread along Y — both wheels on ±y (±0.146), both arms on ±y (±0.24), torso/head
# clustered at x≈+0.04 — so the body's FRONT is +x. The robot spawns at x=-1.8 with the table at x=0,
# so identity already makes the whole robot FACE THE TABLE (+x). The right arm (yb) sits on the -y side =
# the robot's right when facing the table (human-like); its 7 DoF swing the paddle forward to hit.
# Do NOT use +90° — that points the arm's *rest* pose at the table but turns the *body* sideways (+y),
# which reads as "robot not facing the table". Body-facing wins; the arm reaches forward via its joints.
A1_INIT_ROT = (1.0, 0.0, 0.0, 0.0)  # wxyz identity — whole robot faces +x (the table)

# Settled free-base height: measured by inspect_a1.py --settle (drop from z=0.3, step 2 s).
# base_link settled z=0.0282 m; wheel centers z=0.0629 m (≈0.035 above base_link).
# See docs/superpowers/plans/a1_facts.md ## BASE_Z_SETTLED.
A1_INIT_Z = 0.0282

A1_TT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=A1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=10.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,   # free chassis; arm reaction can wobble it
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.8, 0.0, A1_INIT_Z),
        rot=A1_INIT_ROT,
        joint_pos={
            # Task 4: lift kept at -0.28 — changing lift has negligible effect on paddle height
            # (arm droop on soft distal joints kp=7.1 dominates; lift=-0.28 gives blade z≈0.935 m,
            # lift=-0.45 only gains +0.006 m). Blade center at 0.935 m is within ~0.065 m of 1.0 target.
            "joint_lift": -0.28,   # range [-0.8,-0.05]; kept at design default
            # FOREHAND ready pose (was backhand). From /tmp probe single-joint scan on the old pose:
            #   yb_2 -1.57 (-0.762->-2.33) swings the paddle to the RIGHT (-y) side, front, ~0.89 m high;
            #   yb_7 -pi (1.043->-2.10) rolls the blade in place so the FOREHAND face points at the table (+x).
            # Old backhand pose was: yb=[1.769,-0.762,-1.863,1.445,0.206,-0.827,1.043] (paddle front-centered).
            "joint_yb_1": 1.769, "joint_yb_2": -2.33, "joint_yb_3": -1.863,
            "joint_yb_4": 1.445, "joint_yb_5": 0.206, "joint_yb_6": -0.827, "joint_yb_7": -2.10,
            "joint_zb_1": 0.0, "joint_zb_2": 0.0, "joint_zb_3": 0.0, "joint_zb_4": 0.0,
            "joint_zb_5": 0.0, "joint_zb_6": 0.0, "joint_zb_7": 0.0,
            "joint_head_lr": 0.0, "joint_head_ud": 0.0,
            "joint_left_wheel": 0.0, "joint_right_wheel": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,   # 5% margin inside hard limits: joint_pos_target_limits engages before hard stop
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_yb_[1-7]"],
            effort_limit_sim=_EFFORT, velocity_limit_sim=_VEL,
            stiffness=_KP, damping=_KD, armature=_ARM,
        ),
        "left_arm": ImplicitActuatorCfg(joint_names_expr=["joint_zb_[1-7]"],
            effort_limit_sim=200.0, velocity_limit_sim=0.1, stiffness=10000.0, damping=1000.0),
        "lift": ImplicitActuatorCfg(joint_names_expr=["joint_lift"],
            effort_limit_sim=1000.0, velocity_limit_sim=0.0, stiffness=5000.0, damping=500.0),
        "head": ImplicitActuatorCfg(joint_names_expr=["joint_head_.*"],
            effort_limit_sim=10.0, velocity_limit_sim=0.1, stiffness=10000.0, damping=1000.0),
        "wheels": ImplicitActuatorCfg(joint_names_expr=["joint_.*_wheel"],
            effort_limit_sim=10.0, velocity_limit_sim=0.0, stiffness=10000.0, damping=1000.0),
    },
)

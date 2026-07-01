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

# From a1_facts.md (Task 1): quaternion (w,x,y,z) that stands the robot upright + arm toward +x
A1_INIT_ROT = (0.7071, 0.0, 0.0, 0.7071)  # wxyz, +90° around z

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
            "joint_lift": -0.28,   # tuned in Task 4 for G1-prior paddle-ready height
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

"""A1 AGV arm robot config for table tennis (parked chassis, right-arm-only training)."""
import copy
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

A1_USD_PATH = os.path.join(os.path.dirname(__file__), "X1_URDF_V1_1", "X1_URDF_V1_1.usd")

A1_RIGHT_ARM_JOINTS = [f"r{i}" for i in range(1, 8)]
A1_LEFT_ARM_JOINTS = [f"l{i}" for i in range(1, 8)]
A1_WHEEL_JOINTS = [
    "lun_r", "lun_l",
    "wxl_1_2", "wxl_1_1", "wxl_2_2", "wxl_2_1",
    "wxl_3_2", "wxl_3_1", "wxl_4_2", "wxl_4_1",
]

# DAMIAO motors. Gains hand-tuned 2026-07-06 (was kp=armature·ω_n² formula): proximal r1-3 kp=200/kd=3.5,
# distal r4-7 kp=90/kd=0.5.
_KP = {joint: (200.0 if idx < 3 else 90.0) for idx, joint in enumerate(A1_RIGHT_ARM_JOINTS)}
_KD = {joint: (3.5 if idx < 3 else 0.5) for idx, joint in enumerate(A1_RIGHT_ARM_JOINTS)}
_ARM = {joint: (0.032 if idx < 3 else 0.0018) for idx, joint in enumerate(A1_RIGHT_ARM_JOINTS)}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_EFFORT_SCALE = _env_float("A1_TT_EFFORT_SCALE", 1.0)
_PROX_EFFORT_SCALE = _env_float("A1_TT_PROX_EFFORT_SCALE", _EFFORT_SCALE)
_DIST_EFFORT_SCALE = _env_float("A1_TT_DIST_EFFORT_SCALE", _EFFORT_SCALE)
_EFFORT = {
    joint: (28.0 * _PROX_EFFORT_SCALE if idx < 3 else 8.0 * _DIST_EFFORT_SCALE)
    for idx, joint in enumerate(A1_RIGHT_ARM_JOINTS)
}   # peak (rated is 9/3); override with A1_TT_EFFORT_SCALE for diagnostics.
_VEL = {joint: (8.0 if idx < 3 else 20.0) for idx, joint in enumerate(A1_RIGHT_ARM_JOINTS)}

# OpenArm-like weekend actuator probe. The A1 arm is mapped onto the OpenArm joint
# 3-4 / 5-7 motor envelope requested for the scratch reproduction run.
_OPENARM_LIKE_KP = {joint: 80.0 for joint in A1_RIGHT_ARM_JOINTS}
_OPENARM_LIKE_KD = {joint: 4.0 for joint in A1_RIGHT_ARM_JOINTS}
_OPENARM_LIKE_EFFORT = {
    joint: (27.0 if idx < 4 else 7.0)
    for idx, joint in enumerate(A1_RIGHT_ARM_JOINTS)
}
_OPENARM_LIKE_VEL = {
    joint: (2.175 if idx < 4 else 2.61)
    for idx, joint in enumerate(A1_RIGHT_ARM_JOINTS)
}

# Robot FACING (whole-body orientation), from a1_facts.md measured link positions under identity rot:
# the chassis is built spread along Y — original drive wheels on ±y (±0.146), arms on ±y (±0.24),
# torso/head clustered at x≈+0.04 — so the body's FRONT is +x. We add passive front/rear
# support wheels in the USD at ±x to avoid rocking on the original two-wheel line. The robot
# spawns at x=-1.8 with the table at x=0,
# so identity already makes the whole robot FACE THE TABLE (+x). The right arm (r*) sits on the -y side =
# the robot's right when facing the table (human-like); its 7 DoF swing the paddle forward to hit.
# Do NOT use +90° — that points the arm's *rest* pose at the table but turns the *body* sideways (+y),
# which reads as "robot not facing the table". Body-facing wins; the arm reaches forward via its joints.
A1_INIT_ROT = (1.0, 0.0, 0.0, 0.0)  # wxyz identity — whole robot faces +x (the table)
A1_INIT_Y = 0.76

# Settled free-base height: measured by inspect_a1.py --settle (drop from z=0.3, step 2 s).
# base_link settled z=0.0282 m; wheel centers z=0.0629 m (≈0.035 above base_link).
# See docs/superpowers/plans/a1_facts.md ## BASE_Z_SETTLED.
A1_INIT_Z = 0.0282

# Real robot calibration, 2026-07-08: the maximum usable r1 joint centerline height is 1.15 m.
# The URDF chain gives r1_z ~= base_z + sj_origin_z + r0_origin_z + sj, where
# sj_origin_z=1.2107 and r0_origin_z=0.025. Older sim settings implicitly allowed sj=0,
# i.e. r1_z ~= 1.264 m, which is outside the current hardware envelope.
A1_R1_CENTER_HEIGHT_M = 1.15
A1_LIFT_SJ_FOR_R1_CENTER = A1_R1_CENTER_HEIGHT_M - (A1_INIT_Z + 1.2107 + 0.025)

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
            # Parked-AGV bootstrap: the policy has no chassis actions, so the base must not
            # remain a free body that can be exploited through arm reaction torque.
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.8, A1_INIT_Y, A1_INIT_Z),
        rot=A1_INIT_ROT,
        joint_pos={
            # Match the real maximum r1 centerline height (1.15 m), not the URDF's sj=0
            # maximum (about 1.264 m).
            "sj": A1_LIFT_SJ_FOR_R1_CENTER,   # range [-0.85, 0]; about -0.114
            # Forehand ready pose supplied from visual inspection. The paddle is double-sided,
            # so the task reward accepts either local +Y or -Y as the hitting face.
            "r1": 0.569, "r2": -0.692, "r3": 0.717,
            "r4": 1.13, "r5": -1.24, "r6": 0.0314, "r7": 0.772,
            "l1": 0.0, "l2": 0.0, "l3": 0.0, "l4": 0.0,
            "l5": 0.0, "l6": 0.0, "l7": 0.0,
            "t01": 0.0, "t02": 0.0,
            "lun_r": 0.0, "lun_l": 0.0,
            "wxl_1_2": 0.0, "wxl_1_1": 0.0,
            "wxl_2_2": 0.0, "wxl_2_1": 0.0,
            "wxl_3_2": 0.0, "wxl_3_1": 0.0,
            "wxl_4_2": 0.0, "wxl_4_1": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,   # 5% margin inside hard limits: joint_pos_target_limits engages before hard stop
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=A1_RIGHT_ARM_JOINTS,
            effort_limit_sim=_EFFORT, velocity_limit_sim=_VEL,
            stiffness=_KP, damping=_KD, armature=_ARM,
        ),
        "left_arm": ImplicitActuatorCfg(joint_names_expr=A1_LEFT_ARM_JOINTS,
            effort_limit_sim=200.0, velocity_limit_sim=0.1, stiffness=10000.0, damping=1000.0),
        "lift": ImplicitActuatorCfg(joint_names_expr=["sj"],
            effort_limit_sim=1000.0, velocity_limit_sim=0.0, stiffness=5000.0, damping=500.0),
        "head": ImplicitActuatorCfg(joint_names_expr=["t01", "t02"],
            effort_limit_sim=10.0, velocity_limit_sim=0.1, stiffness=10000.0, damping=1000.0),
        "wheels": ImplicitActuatorCfg(joint_names_expr=A1_WHEEL_JOINTS,
            effort_limit_sim=10.0, velocity_limit_sim=0.0, stiffness=10000.0, damping=1000.0),
    },
)

A1_TT_OPENARM_CFG = copy.deepcopy(A1_TT_CFG)
A1_TT_OPENARM_CFG.actuators["right_arm"] = ImplicitActuatorCfg(
    joint_names_expr=A1_RIGHT_ARM_JOINTS,
    effort_limit_sim=_OPENARM_LIKE_EFFORT,
    velocity_limit_sim=_OPENARM_LIKE_VEL,
    stiffness=_OPENARM_LIKE_KP,
    damping=_OPENARM_LIKE_KD,
)

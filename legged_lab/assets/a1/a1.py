"""A1 AGV arm robot config for table tennis (parked chassis, right-arm-only training)."""
import copy
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg

try:
    from legged_lab.actuators import DamiaoMITActuatorCfg
except ModuleNotFoundError:
    from actuators import DamiaoMITActuatorCfg

A1_USD_PATH_ORIGINAL = os.path.join(os.path.dirname(__file__), "X1_URDF_V1_1", "X1_URDF_V1_1.usd")
A1_USD_PATH_SJ_FIXED = os.path.join(os.path.dirname(__file__), "X1_URDF_V1_1", "X1_URDF_V1_1_sj_fixed.usd")
A1_USD_PATH = A1_USD_PATH_SJ_FIXED
# v9: V1_3 CAD (arm ~6% heavier than V1_1, matches the latest SolidWorks export), sj fixed,
# 0.15 kg paddle kept as a separate Link_r_paddle body (converted WITHOUT --merge-joints).
A1_USD_PATH_V1_3 = os.path.join(os.path.dirname(__file__), "X1_URDF_V1_3", "X1_URDF_V1_3_paddle.usd")

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

# DM-Dog-style weekend actuator probe. This is not the old OpenArm implicit run:
# keep the public Damiao/OpenArm motor envelope as the initial numeric prior, but
# put the policy behind delayed explicit PD so it cannot lean on an ideal implicit
# joint servo. These values should be replaced by system-ID curves next week.
_DAMIAO_DELAYED_KP = {joint: 80.0 for joint in A1_RIGHT_ARM_JOINTS}
_DAMIAO_DELAYED_KD = {joint: 4.0 for joint in A1_RIGHT_ARM_JOINTS}
_DAMIAO_DELAYED_EFFORT = dict(_OPENARM_LIKE_EFFORT)
_DAMIAO_DELAYED_VEL = dict(_OPENARM_LIKE_VEL)

# System-ID deployment gains and second-order response at the FixStand pose.
# r1-r3 were refit after the 2026-07-25 motor replacement; r4-r7 keep the
# 2026-07-14 baseline.
_REAL_FITTED_NODE_KP = {joint: (300.0 if idx < 3 else 120.0) for idx, joint in enumerate(A1_RIGHT_ARM_JOINTS)}
_REAL_FITTED_NODE_KD = {joint: (3.5 if idx < 3 else 1.0) for idx, joint in enumerate(A1_RIGHT_ARM_JOINTS)}
_REAL_FITTED_RESPONSE_U_MEAN = {
    "r1": 0.5681831170861339,
    "r2": -0.6928691673392671,
    "r3": 0.7161501174932786,
    "r4": 1.1293556605846493,
    "r5": -1.2407980020381792,
    "r6": 0.030473524919208673,
    "r7": 0.7714033875755423,
}
_REAL_FITTED_RESPONSE_FN_HZ = {
    "r1": 6.661038037881058,
    "r2": 5.008208552071931,
    "r3": 6.984778406823649,
    "r4": 4.341336283530257,
    "r5": 15.30971682198707,
    "r6": 8.223759975617558,
    "r7": 18.61140865177779,
}
_REAL_FITTED_RESPONSE_ZETA = {
    "r1": 0.16764263679500072,
    "r2": 0.1876649639165574,
    "r3": 0.28653637194779724,
    "r4": 0.22981041868709606,
    "r5": 0.7941795710126007,
    "r6": 0.565832498155223,
    "r7": 1.3208910165925782,
}
_REAL_FITTED_RESPONSE_DELAY_S = {
    "r1": 0.03502917289780583,
    "r2": 0.032912611967056964,
    "r3": 0.02843821965716936,
    "r4": 0.018006420135349824,
    "r5": 0.017563104629677986,
    "r6": 0.013993930820317215,
    "r7": 0.014997124673895237,
}
_REAL_FITTED_RESPONSE_GAIN = {
    "r1": 0.9832701113210972,
    "r2": 0.9741513252336587,
    "r3": 0.9991903500772663,
    "r4": 1.0027218616565117,
    "r5": 0.9998451719960618,
    "r6": 1.0020846023179375,
    "r7": 1.0005302866606762,
}
_REAL_FITTED_RESPONSE_INTERCEPT = {
    "r1": 0.5535098365417602,
    "r2": -0.6800868502823569,
    "r3": 0.714945234033157,
    "r4": 1.097829467273751,
    "r5": -1.2411862500648883,
    "r6": 0.0315228173080221,
    "r7": 0.771172217142933,
}

# The fitted motor response itself is applied in TTEnv. This solver-side implicit
# actuator is intentionally much harder than the identified motor loop, so it
# tracks the response target without adding another explicit torque-PD dynamic.
_REAL_FITTED_TRACKING_KP = {
    joint: (30000.0 if joint == "r2" else 20000.0)
    for joint in A1_RIGHT_ARM_JOINTS
}
_REAL_FITTED_TRACKING_KD = {joint: 100.0 for joint in A1_RIGHT_ARM_JOINTS}
_REAL_FITTED_TRACKING_EFFORT = {joint: 1.0e9 for joint in A1_RIGHT_ARM_JOINTS}
_REAL_FITTED_TRACKING_VEL = {joint: 1.0e9 for joint in A1_RIGHT_ARM_JOINTS}

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

# Real robot calibration: r1 joint centerline height is fixed at 1.15 m.
# V1_3 uses a fixed sj joint, so its URDF origin is baked to 1.09680245585163 m:
# base_z(0.0282) + sj_origin_z + r0_origin_z(0.025) = 1.15 m.
# A movable sj offset is therefore zero for this current asset.
A1_R1_CENTER_HEIGHT_M = 1.15
A1_LIFT_SJ_FOR_R1_CENTER = 0.0

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
            # The fixed-sj USD bakes the real r1 centerline height (1.15 m) into
            # the Link_sj origin, so sj is no longer an articulated DOF here.
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

A1_TT_DAMIAO_DELAYED_CFG = copy.deepcopy(A1_TT_CFG)
A1_TT_DAMIAO_DELAYED_CFG.actuators["right_arm"] = DelayedPDActuatorCfg(
    joint_names_expr=A1_RIGHT_ARM_JOINTS,
    effort_limit=_DAMIAO_DELAYED_EFFORT,
    velocity_limit=_DAMIAO_DELAYED_VEL,
    stiffness=_DAMIAO_DELAYED_KP,
    damping=_DAMIAO_DELAYED_KD,
    armature=0.01,
    min_delay=1,
    max_delay=3,
)

A1_TT_REAL_FITTED_CFG = copy.deepcopy(A1_TT_CFG)
A1_TT_REAL_FITTED_CFG.actuators["right_arm"] = ImplicitActuatorCfg(
    joint_names_expr=A1_RIGHT_ARM_JOINTS,
    effort_limit_sim=_REAL_FITTED_TRACKING_EFFORT,
    velocity_limit_sim=_REAL_FITTED_TRACKING_VEL,
    stiffness=_REAL_FITTED_TRACKING_KP,
    damping=_REAL_FITTED_TRACKING_KD,
)

A1_TT_REAL_TORQUE_ONLY_CFG = copy.deepcopy(A1_TT_CFG)
A1_TT_REAL_TORQUE_ONLY_CFG.actuators["right_arm"] = DamiaoMITActuatorCfg(
    joint_names_expr=A1_RIGHT_ARM_JOINTS,
    effort_limit=_EFFORT,
    velocity_limit=_VEL,
    effort_limit_sim=_REAL_FITTED_TRACKING_EFFORT,
    velocity_limit_sim=_REAL_FITTED_TRACKING_VEL,
    stiffness=_REAL_FITTED_NODE_KP,
    damping=_REAL_FITTED_NODE_KD,
    armature=_ARM,
    control_dt=0.002,
    command_delay_s=0.0,
    command_velocity_limit=_VEL,
    use_command_velocity=False,
    response_model_enable=True,
    response_fn_hz=_REAL_FITTED_RESPONSE_FN_HZ,
    response_zeta=_REAL_FITTED_RESPONSE_ZETA,
    response_delay_s=_REAL_FITTED_RESPONSE_DELAY_S,
    response_linear_gain=_REAL_FITTED_RESPONSE_GAIN,
    response_intercept=_REAL_FITTED_RESPONSE_INTERCEPT,
    response_u_mean=_REAL_FITTED_RESPONSE_U_MEAN,
    response_tau_zero_s=0.0,
    torque_time_constant=0.0,
    viscous_friction=0.0,
    coulomb_friction=0.0,
    torque_speed_limit_enable=True,
    brake_effort_limit=_EFFORT,
)

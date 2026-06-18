import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ

G1_TT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAAC_ASSET_DIR}/unitree/g1_description/g1_23dof_tt_paddle.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False,
            linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8, solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-2.0, 0.0, 0.76),   # stance >=60cm from table (was -1.6=23cm); matches hit_plane_x
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.6, "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.6, "right_wrist_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim={".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0,
                              ".*_hip_pitch_joint": 88.0, ".*_knee_joint": 139.0},
            velocity_limit_sim={".*_hip_yaw_joint": 32.0, ".*_hip_roll_joint": 20.0,
                                ".*_hip_pitch_joint": 32.0, ".*_knee_joint": 20.0},
            stiffness={".*_hip_pitch_joint": STIFFNESS_7520_14, ".*_hip_roll_joint": STIFFNESS_7520_22,
                       ".*_hip_yaw_joint": STIFFNESS_7520_14, ".*_knee_joint": STIFFNESS_7520_22},
            damping={".*_hip_pitch_joint": DAMPING_7520_14, ".*_hip_roll_joint": DAMPING_7520_22,
                     ".*_hip_yaw_joint": DAMPING_7520_14, ".*_knee_joint": DAMPING_7520_22},
            armature={".*_hip_pitch_joint": ARMATURE_7520_14, ".*_hip_roll_joint": ARMATURE_7520_22,
                      ".*_hip_yaw_joint": ARMATURE_7520_14, ".*_knee_joint": ARMATURE_7520_22},
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0, velocity_limit_sim=37.0,
            stiffness=2.0 * STIFFNESS_5020, damping=2.0 * DAMPING_5020, armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=139, velocity_limit_sim=20.0,
            stiffness=STIFFNESS_7520_22, damping=DAMPING_7520_22, armature=ARMATURE_7520_22,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                              ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=25.0, velocity_limit_sim=37.0,
            stiffness=STIFFNESS_5020, damping=DAMPING_5020, armature=ARMATURE_5020,
        ),
    },
)

# G1 23-DoF locomotion (velocity) task — stable stand + basic walk, joint order == g1_tt
# (leg-first: legs 0-11 / waist 12 / Larm 13-17 / Rarm 18-22) so the exported onnx deploys
# under the same joint_ids_map as TableTennis. Built on LeggedEnv (no ball/table/paddle task).
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree.g1 import G1_TT_CFG
from legged_lab.envs.base.legged_env_config import LeggedEnvCfg, LeggedAgentCfg
from legged_lab.envs.base.legged_config import RewardCfg

# leg-first 23-DoF order (identical to g1_tt G1_JOINT_NAMES) -> consistent deploy joint_ids_map
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
]


@configclass
class G1LocomotionRewardCfg(RewardCfg):
    # velocity tracking (the task)
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"std": 0.5})
    # stability / smoothness (params reused from the proven g1_tt cfg; all BaseEnv-safe)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*_ankle_roll_link).*"), "threshold": 1.0},
    )
    # keep arms/waist/hip-yaw-roll near default so a velocity policy walks without flailing
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint"])})
    joint_deviation_waist = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])})
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])})


@configclass
class G1LocomotionEnvCfg(LeggedEnvCfg):
    reward = G1LocomotionRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.002
        self.sim.decimation = 10  # 50 Hz (matches g1_tt / deploy step_dt 0.02)
        self.scene.robot = G1_TT_CFG          # same G1+paddle asset as TT (real robot has paddle)
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.robot.num_actions = 23
        self.robot.num_joints = 23
        self.robot.terminate_contacts_body_names = ["torso_link"]
        self.robot.feet_body_names = [".*_ankle_roll_link"]
        # base RobotCfg leaves these as [] (LeggedEnv needs scalars: get_phase() fmod, height term)
        self.robot.phase_dt = 0.8          # gait-phase clock period (s)
        self.robot.min_base_height = 0.3   # robot_height(pelvis-above-feet)=0.486 in the squat default
                                           # (knee 0.669); 0.5 terminated every step. 0.3 catches real falls.
        self.robot.max_base_height = 1.0   # terminate if above (jump/launch)
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.observations.joint_names = G1_JOINT_NAMES
        self.actions.joint_names = G1_JOINT_NAMES
        # modest command ranges: stable stand + basic walk (not aggressive)
        self.commands.ranges.lin_vel_x = (-0.5, 0.8)
        self.commands.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class G1LocomotionAgentCfg(LeggedAgentCfg):
    experiment_name = "g1_locomotion"
    logger = "tensorboard"
    save_interval = 200
    max_iterations = 15000

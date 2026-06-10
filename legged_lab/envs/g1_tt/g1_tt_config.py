# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import  RslRlPpoAlgorithmCfg
import legged_lab.mdp as mdp
from legged_lab.assets.unitree.g1 import G1_TT_CFG
from legged_lab.assets.table_tennis.table import TABLE_CFG
from legged_lab.assets.table_tennis.ball import BALL_CFG
from legged_lab.envs.base.tt_env_config import (  # noqa:F401
    TTAgentCfg,
    TTEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    CurriculumCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class G1TableTennisRewardCfg(RewardCfg):
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.02)
    energy = RewTerm(func=mdp.energy, weight=-1.5e-3)
    energy_ankle = RewTerm(func=mdp.energy, weight=-2e-3,params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.025)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-80.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*_ankle_roll_link).*"), "threshold": 1.0},
    )
    penalty_robot_table_proximity_x = RewTerm(
        func=mdp.penalty_robot_table_proximity_x,
        weight=-20.0,
        params={ "min_distance": 0.15, "std":0.07},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-2.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1000.0)

    hit_unstable_support = RewTerm(
        func=mdp.hit_unstable_support,
        weight=-10,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link")},
    )

    feet_orientation_L = RewTerm(
        func=mdp.body_orientation_l2,
        weight = -4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_link")},
    )
    feet_orientation_R = RewTerm(
        func=mdp.body_orientation_l2,
        weight = -4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_link")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    paddel_head_too_near = RewTerm(
        func=mdp.paddel_too_near_humanoid,
        weight=-100,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]), "threshold": 0.3},  # head_link merges into torso under fixed-joint import

    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), "threshold": 0.2},
    )
    feet_really_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-10,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), "threshold": 0.15},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"])},
    )

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])})
    joint_deviation_left_arm = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_.*", "left_elbow_joint", "left_wrist_roll_joint"])})
    joint_deviation_right_arm = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_shoulder_.*", "right_elbow_joint", "right_wrist_roll_joint"])})
    joint_deviation_torso = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])})

    reward_contact = RewTerm(
        func=mdp.reward_contact,
        weight=150.0,
    )


    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target,
        weight=2.0,
        params={
            "std_ee": 0.5,
            "threshold": 0.15
        },
    )

    reward_future_dis_ro = RewTerm(
        func=mdp.reward_future_body_target,
        weight=5.0,
        params={
            "std_ro": 0.5,
            "threshold": 0.05
        },
    )

    reward_future_vel_base = RewTerm(
        func=mdp.reward_future_vel_target,
        weight=5.0,
        params={
            "vel_std": 1.2,
            "threshold" : 0.1,
        },
    )

    reward_future_landing_dis= RewTerm(
        func=mdp.reward_future_landing_dis,
        weight=60.0,
        params={
            "threshold": 3.0
        }
    )


    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net,
        params={"std_h": 0.4,'z_target':0.76+0.35}, #table height +  height above table
        weight=100.0
    )

    reward_table_success = RewTerm(
        func=mdp.reward_table_success,
        weight=100.0,
    )





@configclass
class G1TableTennisEnvCfg(TTEnvCfg):

    reward = G1TableTennisRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        #For fast validation, we can use a faster simulation time step
        # self.sim.dt = 0.005
        # self.sim.decimation = 4 # 50 Hz
        #NOTE: The following parameters are set to match the original T1 configuration.
        # For sim2real, recommand to use the following settings:
        #######################################################
        self.sim.dt = 0.002
        self.sim.decimation = 10 # 50 Hz
        #######################################################
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_TT_CFG
        self.scene.table = TABLE_CFG
        self.scene.ball = BALL_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.robot.terminate_contacts_body_names = ["torso_link"]
        self.robot.feet_body_names = [".*_ankle_roll_link"]
        self.robot.num_actions = 23
        self.robot.num_joints = 23
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        # reset-joint groups (base tt_env_config uses Booster names) -> G1: locomotion = legs+waist+left arm, manipulation = right (hitting) arm
        self.domain_rand.events.reset_locomotion_joints.params["asset_cfg"].joint_names = [
            "waist_yaw_joint", ".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint",
            "left_shoulder_.*", "left_elbow_joint", "left_wrist_roll_joint",
        ]
        self.domain_rand.events.reset_manipulation_joints.params["asset_cfg"].joint_names = [
            "right_shoulder_.*", "right_elbow_joint", "right_wrist_roll_joint",
        ]
        # paddle / hitting geometry (real-mesh PACE adapter: contact/blade center at wrist +X 0.337m)
        self.robot.paddle_body_name = "right_wrist_roll_rubber_hand"
        self.robot.paddle_offset = (0.337, 0.0, 0.0)
        self.robot.hit_body_height = 0.685   # FK: steady pelvis height in ready stance
        self.robot.paddle_y_offset = -0.55   # hitting-extension lateral offset (was -0.227 ready-stance; caused ~0.37m paddle-ball gap -> hit~0). ~T1's -0.60.
        G1_JOINT_NAMES = [
            "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
            "left_ankle_pitch_joint","left_ankle_roll_joint",
            "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint",
            "right_ankle_pitch_joint","right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
            "left_elbow_joint","left_wrist_roll_joint",
            "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
            "right_elbow_joint","right_wrist_roll_joint",
        ]
        self.observations.joint_names = G1_JOINT_NAMES
        self.actions.joint_names = G1_JOINT_NAMES

        # ---- FROM-SCRATCH idle+rally training (g1_tt_idle3) ----
        # No warm-start: learn hitting AND no-ball idle together behind the unified HARD
        # validity gate (tt_env.compute_current_observations_perception). Both idle1/idle2
        # warm-start fine-tunes DIVERGED (action_rate -1e5..-1e6 EVERY iter): clip_actions
        # was the wrong lever (it clips only the APPLIED action, not the obs / action_rate
        # penalty, which both use the raw network output), and swapping the home sentinel
        # under a warm policy was OOD. From scratch avoids that mismatch.
        # 1) Bounce serves (in-court, no volley) with a difficulty curriculum easy->hard.
        self.ball.serve_bounce_enable = True
        self.ball.serve_bounce_x_range = (-0.95, -0.75)        # easy: shallow, centered, reachable
        self.ball.serve_bounce_vz_range = (1.7, 1.9)           # easy: gentle moderate arc
        self.ball.serve_y_start = 0.08                         # easy: nearly centered
        self.ball.serve_bounce_x_range_hard = (-1.35, -0.60)   # hard: full depth
        self.ball.serve_bounce_vz_range_hard = (1.3, 2.1)      # hard: flat-fast + high-slow
        self.ball.serve_y_wide = 0.72                          # hard: corners (table half 0.7625)
        self.ball.serve_curriculum_steps = 300000              # ~12.5k iter to full difficulty (gentle from scratch)
        # 2) No-ball idle injection — in real play NO-BALL is the MAJORITY of time, so the
        #    final ratio is idle-heavy (7 s no-ball / 3 s ball per 10 s = 70% idle), with
        #    LONG contiguous 7 s windows so the policy learns to hold home indefinitely.
        #    BUT a random init left mostly idle learns "just stand, never risk hitting"
        #    (idle is safe, hitting risks a fall) -> local optimum. So ramp the no-ball gap
        #    from 0 (c=0: all-ball, bootstrap hitting) up to 7 s over no_ball_curriculum_steps
        #    (~12.5k iter, parallel to the serve-difficulty curriculum). Late training is the
        #    realistic idle-heavy distribution; early training learns to hit first.
        self.ball.no_ball_period_s = 10.0
        self.ball.ball_active_s = 3.0                          # final: 3 s ball, 7 s no-ball
        self.ball.no_ball_curriculum_steps = 300000            # ramp 0 -> 7 s gap over ~12.5k iter
        # clip_actions stays at the base 100 (NOT 20 — clipping the applied action decouples
        # the network output from the dynamics and does nothing for the obs/action_rate path).
        # 3) Randomization (kept): wide perception/action delay for real/deploy latency.
        self.domain_rand.perception_delay.params["max_delay"] = 15   # 2..15 steps = 4-30 ms
        self.domain_rand.action_delay.enable = True
        self.domain_rand.action_delay.params["min_delay"] = 1
        self.domain_rand.action_delay.params["max_delay"] = 3        # 2-6 ms command delay



@configclass
class G1TT_EvalEnvCfg(G1TableTennisEnvCfg):
    """Eval variant: identical to G1TableTennisEnvCfg but with extended episode length.
    """
    def __post_init__(self):
        super().__post_init__()
        self.scene.max_episode_length_s = 99999999999 # prevent frequent reset
        self.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (-0.41, -0.4),
            "y": (0.3, 0.4),#(-0.4, 0.4),
            "yaw": (-0.1, 0.1),
            }
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
            "yaw": (-0.02, 0.02),
            }
        # serving range — eval uses a FIXED medium bounce distribution (bounce path is
        # active via inherited serve_bounce_enable; the ball_speed_* below are dead code).
        self.ball.serve_bounce_x_range = (-1.15, -0.80)   # fixed medium depth
        self.ball.serve_bounce_vz_range = (1.5, 1.9)
        self.ball.serve_y_start = 0.45                    # fixed medium lateral spread
        self.ball.serve_y_wide = 0.45
        self.ball.serve_curriculum_steps = 0   # eval: fixed serve distribution (no curriculum widening)
        self.ball.no_ball_period_s = 0.0       # eval: NO no-ball injection -> clean hitting success rate
        #   (test no-ball idle separately via the TT_SERVE_PERIOD / TT_NO_SERVE env hooks)

@configclass
class G1TableTennisAgentCfg(TTAgentCfg):
    experiment_name: str = "g1_tt_idle3"
    logger = "tensorboard"
    save_interval = 100
    max_iterations = 100000

    # Auxiliary predictor configuration used by OnPolicyPredictorRegressionRunner
    # Ignored by the standard OnPolicyRunner.
    predictor = {
        "history_len": 5,
        "traj_max_len": 128,
        "hidden_sizes": [64, 64],
        "lr": 0.5e-3,
        "epochs_per_update": 1,
        "batch_size": 1024,
        "train_until_iters":20,
    }

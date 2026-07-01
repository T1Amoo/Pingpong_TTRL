# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Licensed under BSD-3-Clause.

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import legged_lab.mdp as mdp
from legged_lab.assets.a1.a1 import A1_TT_CFG
from legged_lab.assets.table_tennis.table import TABLE_CFG
from legged_lab.assets.table_tennis.ball import BALL_CFG
from legged_lab.envs.base.tt_env_config import (  # noqa:F401
    TTAgentCfg,
    TTEnvCfg,
    RewardCfg,
)

A1_ARM_JOINTS = [
    "joint_yb_1", "joint_yb_2", "joint_yb_3", "joint_yb_4",
    "joint_yb_5", "joint_yb_6", "joint_yb_7",
]


@configclass
class A1TableTennisRewardCfg(RewardCfg):
    # --- base-shake penalty (arm reaction wobbles the free chassis) ---
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.02)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # --- arm smoothness / limits ---
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.025)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_pos_target_limits = RewTerm(func=mdp.joint_pos_target_limits, weight=-0.1)   # was -1.0; quadratic+unbounded -> value-fn bomb (critic diverged ~iter1000). Now bounded by clip_actions=10 + soft 0.95; keep as a mild nudge only.
    joint_deviation_right_arm = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_ARM_JOINTS)},
    )
    # --- don't crash into table ---
    penalty_robot_table_proximity_x = RewTerm(
        func=mdp.penalty_robot_table_proximity_x,
        weight=-20.0,
        params={"min_distance": 0.15, "std": 0.07},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    # --- ready-pose regularization when no playable ball ---
    # reward_idle_pose is G1-specific (hardcoded 23-joint ready vector); drop for A1.
    reward_idle_stand = RewTerm(func=mdp.reward_idle_stand, weight=0.5)
    # --- ball / hitting core ---
    reward_contact = RewTerm(func=mdp.reward_contact, weight=150.0)
    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target,
        weight=2.0,
        params={"std_ee": 0.5, "threshold": 0.15},
    )
    reward_future_dis_ro = RewTerm(
        func=mdp.reward_future_body_target,
        weight=5.0,
        params={"std_ro": 0.5, "threshold": 0.05},
    )
    reward_future_vel_base = RewTerm(
        func=mdp.reward_future_vel_target,
        weight=5.0,
        params={"vel_std": 1.2, "threshold": 0.1},
    )
    reward_future_landing_dis = RewTerm(
        func=mdp.reward_future_landing_dis,
        weight=60.0,
        params={"threshold": 3.0},
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net,
        weight=100.0,
        params={"std_h": 0.4, "z_target": 0.76 + 0.35},
    )
    reward_table_success = RewTerm(func=mdp.reward_table_success, weight=100.0)


@configclass
class A1TableTennisEnvCfg(TTEnvCfg):
    reward = A1TableTennisRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.002
        self.sim.decimation = 10  # 50 Hz
        # Bound applied action target: processed = clip(action,±10)*scale(0.25)+default -> target within default±2.5rad.
        # Base default is 100 (~no clip) -> targets ran unbounded past joint limits, joint_pos_target_limits (quadratic)
        # exploded the critic value fn ~iter1000. 10 covers the hitting workspace while capping the runaway (raw hit ~32).
        self.normalization.clip_actions = 10.0
        self.scene.height_scanner.enable_height_scan = False
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.robot = A1_TT_CFG
        self.scene.table = TABLE_CFG
        self.scene.ball = BALL_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        # bodies that resolve against the contact sensor / collision
        self.robot.terminate_contacts_body_names = ["base_link"]
        self.robot.feet_body_names = ["link_(right|left)_wheel"]
        self.robot.num_actions = 7
        self.robot.num_joints = 7
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        # DR reset joint groups: only the right arm is active (no locomotion joints)
        self.domain_rand.events.reset_locomotion_joints.params["asset_cfg"].joint_names = A1_ARM_JOINTS[:1]
        self.domain_rand.events.reset_manipulation_joints.params["asset_cfg"].joint_names = A1_ARM_JOINTS
        # base is free but should not be reset-scattered like a walking robot
        self.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        }
        # paddle geometry (Task 4): body origin is at joint attachment (bottom of handle);
        # blade rubber face center is ~8.5 cm above body origin in local +z
        # (mesh z range: handle −0.08..0 m, blade 0..0.17 m, blade center z≈0.085 m).
        self.robot.paddle_body_name = "Link_yb_paddle"
        self.robot.paddle_offset = (0.0, 0.0, 0.085)
        # hit_body_height: settled base_link z ≈ 0.028 m (from a1_facts.md BASE_Z_SETTLED)
        self.robot.hit_body_height = 0.028
        self.robot.paddle_y_offset = -0.55
        self.robot.hit_plane_x = -1.8
        self.observations.joint_names = A1_ARM_JOINTS
        self.actions.joint_names = A1_ARM_JOINTS
        # serve: G1 v11 easy distribution
        self.ball.serve_bounce_enable = True
        self.ball.serve_bounce_x_range = (-0.90, -0.76)
        self.ball.serve_bounce_vz_range = (1.2, 1.6)
        self.ball.serve_y_start = 0.5
        self.ball.serve_curriculum_steps = 0   # easy-only for first run
        self.ball.no_ball_period_s = 0.0


@configclass
class A1TT_EvalEnvCfg(A1TableTennisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.max_episode_length_s = 99999999999
        self.ball.serve_curriculum_steps = 0


@configclass
class A1TableTennisAgentCfg(TTAgentCfg):
    experiment_name: str = "a1_tt"
    logger = "tensorboard"
    save_interval = 100
    max_iterations = 30000
    predictor = {
        "history_len": 5,
        "traj_max_len": 128,
        "hidden_sizes": [64, 64],
        "lr": 0.5e-3,
        "epochs_per_update": 1,
        "batch_size": 1024,
        "train_until_iters": 20,
    }

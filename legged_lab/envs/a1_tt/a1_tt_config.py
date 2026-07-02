# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Licensed under BSD-3-Clause.

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import legged_lab.mdp as mdp
from legged_lab.assets.a1.a1 import A1_RIGHT_ARM_JOINTS, A1_TT_CFG
from legged_lab.assets.table_tennis.table import TABLE_CFG
from legged_lab.assets.table_tennis.ball import BALL_CFG
from legged_lab.envs.base.tt_env_config import (  # noqa:F401
    TTAgentCfg,
    TTEnvCfg,
    RewardCfg,
)

A1_ARM_JOINTS = list(A1_RIGHT_ARM_JOINTS)
A1_GROUND_CONTACT_BODIES = ["Link_lun_(r|l)", "Link_wxl_.*"]


@configclass
class A1TableTennisRewardCfg(RewardCfg):
    # --- base-shake penalty (arm reaction wobbles the free chassis) ---
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.2)
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
    paddle_face_x = RewTerm(
        func=mdp.paddle_face_x_alignment,
        weight=3.0,
        params={"local_axis": "y"},
    )
    reward_contact = RewTerm(func=mdp.reward_contact, weight=70.0)
    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target,
        weight=20.0,
        params={"std_ee": 0.5, "threshold": 0.15},
    )
    reward_future_dis_ro = RewTerm(
        func=mdp.reward_future_body_target,
        weight=0.0,
        params={"std_ro": 0.5, "threshold": 0.05},
    )
    reward_future_vel_base = RewTerm(
        func=mdp.reward_future_vel_target,
        weight=0.0,
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
        self.scene.terrain_static_friction = 4.0
        self.scene.terrain_dynamic_friction = 3.0
        self.scene.terrain_friction_combine_mode = "max"
        # bodies that resolve against the contact sensor / collision
        self.robot.terminate_contacts_body_names = ["base_link"]
        self.robot.feet_body_names = A1_GROUND_CONTACT_BODIES
        self.robot.num_actions = 7
        self.robot.num_joints = 7
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]

        # ---- Parked-AGV bootstrap: no chassis/domain randomization ----
        # The policy controls only the right arm. For v2 we keep the base reset deterministic and
        # remove startup DR/delays so arm learning is not hidden behind chassis wobble. The one
        # remaining startup material event is deterministic (single bucket, equal ranges) and
        # targets only the four wheel bodies to override low/default USD wheel friction.
        self.domain_rand.events.add_base_mass = None
        self.domain_rand.events.physics_material.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=A1_GROUND_CONTACT_BODIES
        )
        self.domain_rand.events.physics_material.params["static_friction_range"] = (4.0, 4.0)
        self.domain_rand.events.physics_material.params["dynamic_friction_range"] = (3.0, 3.0)
        self.domain_rand.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.domain_rand.events.physics_material.params["num_buckets"] = 1
        self.domain_rand.events.physics_material.params["make_consistent"] = True

        # reset_base: no randomization. Keep the term but zero every range so each episode is a clean
        # deterministic reset to the spawn pose. (Dropping the term would leave the base at its terminal
        # pose across resets — scene.reset() does not rewrite the root state; only this event does.)
        self.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

        # push_robot: disabled — a free unactuated base cannot recover from an external push.
        self.domain_rand.events.push_robot = None

        # reset_locomotion_joints: disabled. It is a biped hip/knee "scale" reset and does not
        # apply to this parked AGV arm task.
        self.domain_rand.events.reset_locomotion_joints = None

        # reset_manipulation_joints: small ready-pose jitter prevents overfitting to a single
        # static paddle intercept without turning this into broad domain randomization.
        self.domain_rand.events.reset_manipulation_joints.params["asset_cfg"].joint_names = A1_ARM_JOINTS
        self.domain_rand.events.reset_manipulation_joints.params["position_range"] = (-0.03, 0.03)

        self.domain_rand.action_delay.enable = False
        self.domain_rand.perception_delay.enable = False
        self.noise.add_noise = True
        # paddle geometry (Task 4): body origin is at joint attachment (bottom of handle);
        # blade rubber face center is ~8.5 cm above body origin in local +z
        # (mesh z range: handle −0.08..0 m, blade 0..0.17 m, blade center z≈0.085 m).
        self.robot.paddle_body_name = "Link_r_paddle"
        self.robot.paddle_offset = (0.0, 0.0, 0.085)
        # hit_body_height: settled base_link z ≈ 0.028 m (from a1_facts.md BASE_Z_SETTLED)
        self.robot.hit_body_height = 0.028
        self.robot.home_y = 0.76
        # Latest X1_URDF_V1_1 ready pose probe: paddle_touch_point y≈0.10 with home_y=0.76.
        self.robot.paddle_y_offset = -0.66
        # A1 base stays near x=-1.8, but the forehand blade is ~0.38 m in front of the base.
        # The hitting guidance plane must therefore live near the reachable blade x, not on
        # the base x plane.
        self.robot.hit_plane_x = -1.42
        # Fixed hit/intercept plane, matching the G1 task semantics: the learned predictor may
        # output 3 values, but x is the plane anchor and only y/z should meaningfully vary.
        self.robot.hit_target_x_range = (self.robot.hit_plane_x, self.robot.hit_plane_x)
        self.robot.hit_target_y_range = (0.08, 0.30)
        # Latest X1_URDF_V1_1 ready pose probe: zero-action touch z≈1.30 initially,
        # settling toward ≈1.25 over the first ~120 control steps.
        self.robot.hit_target_z_range = (1.08, 1.32)
        self.observations.joint_names = A1_ARM_JOINTS
        self.actions.joint_names = A1_ARM_JOINTS
        # serve: narrow, easy forehand-only distribution around the corrected blade region.
        # Lateral target is intentionally offset from the static ready-paddle center; verified by
        # diagnose_a1_contact_shortcut.py that zero/tiny actions no longer get raw paddle contact.
        # First bounce is deep and launch-vz is moderate so the post-bounce path is flatter
        # and reaches the A1 hit plane instead of dying short near the net.
        self.ball.serve_bounce_enable = True
        self.ball.serve_bounce_x_range = (-1.24, -0.96)
        self.ball.serve_bounce_x_range_hard = (-1.24, -0.96)
        self.ball.serve_bounce_vz_range = (1.85, 2.25)
        self.ball.serve_bounce_vz_range_hard = (1.85, 2.25)
        self.ball.serve_y_center = 0.20
        self.ball.serve_y_start = 0.06
        self.ball.serve_y_wide = 0.08
        self.ball.require_active_contact = True
        self.ball.active_contact_min_paddle_speed = 0.12
        self.ball.active_contact_min_forward_speed = -0.05
        self.ball.active_contact_require_own_bounce = False
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
    experiment_name: str = "a1_tt_v2"
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
        "train_until_iters": 200,
    }

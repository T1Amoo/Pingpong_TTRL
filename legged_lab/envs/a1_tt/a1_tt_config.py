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
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.003)   # v5: -0.01->-0.003; biggest motion penalty, was suppressing the swing
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0005)            # v5: -0.001->-0.0005
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_pos_target_limits = RewTerm(func=mdp.joint_pos_target_limits, weight=-0.1)   # was -1.0; quadratic+unbounded -> value-fn bomb (critic diverged ~iter1000). Now bounded by clip_actions=10 + soft 0.95; keep as a mild nudge only.
    joint_deviation_right_arm = RewTerm(
        func=mdp.joint_deviation_l1_idle,   # v5: idle-only (no-ball); was joint_deviation_l1 every step -> fought the swing
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
    # v4: kill the "hover in the ball's path and farm positioning/orientation" exploit seen in play
    # (v3 hit ~10%, returned 0%). paddle_face + future_dis_ee are dense and DON'T need contact, so a
    # parked pose farmed ~18 reward without striking. Cut them hard; let the un-fakeable outcome ladder
    # (contact -> pass_net -> landing -> table_success) + a body-block penalty drive real forehand swings.
    paddle_face_x = RewTerm(
        func=mdp.paddle_face_x_alignment,
        weight=0.5,   # v9: 3.0->0.5. v8's 3.0 (+dis_ee 6) let the policy FARM dense shaping (hover near intercept,
                      # face +x) to ~2 reward WITHOUT ever contacting -> 0 hits. Cut it so contact must drive reward.
        params={"local_axis": "y"},
    )
    # v3(#3): penalize the ball approaching non-paddle mid-arm links (Link_r3..r6) -> stop body-blocking,
    # force hitting with the paddle blade.
    penalty_ball_body_block = RewTerm(
        func=mdp.penalty_ball_body_block,
        weight=-20.0,
        params={"body_regex": "Link_r[3-6]", "threshold": 0.09},
    )
    reward_contact = RewTerm(func=mdp.reward_contact, weight=150.0)   # v7: 40->150 to match G1 (A1 was severely under-rewarding contact)
    reward_sweet_contact = RewTerm(
        func=mdp.reward_paddle_sweet_contact,
        weight=35.0,   # v11: first-contact sweet-spot bonus; downstream return rewards are also quality-scaled.
    )
    # v6: v5 killed ball-tracking (future_dis_ee 0.1) -> paddle camped + wrist-jittered, never moved to the
    # ball (play: paddle static <4cm, 28% hit). RESTORE tracking so the paddle goes to the intercept, AND add
    # a forward-swing reward so it drives THROUGH the ball toward the table instead of passively camping.
    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target,
        weight=2.0,   # v9: 6.0->2.0 (back to G1 level). v8's 6.0 made hovering-near-intercept farmable without
                      # contact -> 0 hits. Keep tracking guidance but let the un-fakeable contact/pass_net/table dominate.
        params={"std_ee": 0.5, "threshold": 0.15},
    )
    reward_swing_through = RewTerm(
        func=mdp.reward_swing_through,
        weight=3.0,   # v6 NEW: forward paddle speed (+x, toward net) while near the ball -> swing through, not camp.
        params={"near_dist": 0.30},
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
    reward_table_success = RewTerm(func=mdp.reward_table_success, weight=150.0)   # v4: was 100; the true un-fakeable success terminal, make it the top prize

    # v10: fine-tune at real motor limits. The v9 policies frequently demand computed torque far
    # beyond the motor limit and rely on clipping. Penalize that demand without dominating contact.
    joint_computed_torque_limit = RewTerm(
        func=mdp.joint_computed_torque_limit_l2,
        weight=-0.01,
        params={
            "threshold": 0.9,
            "max_ratio": 3.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=A1_ARM_JOINTS),
        },
    )


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
        self.robot.action_scale = 0.25
        # v10: train directly at real motor effort (r1-r3 28Nm, r4-r7 8Nm). Warm-start from v9
        # model_800 instead of replaying the old 4x proximal exploration curriculum.
        self.robot.effort_curriculum_start_scale = 1.0
        self.robot.effort_curriculum_steps = 0
        self.robot.effort_curriculum_num_joints = 0
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
        # A1 base stays near x=-1.8. v11 moves the intercept plane back from -1.42 to -1.55
        # so the paddle does not have to fold into the table/body-side serve path.
        self.robot.hit_plane_x = -1.55
        # Fixed hit/intercept plane, matching the G1 task semantics: the learned predictor may
        # output 3 values, but x is the plane anchor and only y/z should meaningfully vary.
        self.robot.hit_target_x_range = (self.robot.hit_plane_x, self.robot.hit_plane_x)
        # v3: spread the intercept target across the measured forehand-reachable envelope. v11 keeps
        # that broad y/z envelope but anchors it on the new -1.55 hit plane.
        self.robot.hit_target_y_range = (0.0, 0.55)
        self.robot.hit_target_z_range = (0.90, 1.25)
        self.observations.joint_names = A1_ARM_JOINTS
        self.actions.joint_names = A1_ARM_JOINTS
        # v10 serve: keep the bounce forehand-reachable but move it away from the robot body side.
        # v9 used y ~= 0.21..0.33, which made the arm fold back toward the chassis. Start with a
        # narrower y ~= 0.08..0.16 real-effort adaptation window and widen only after it stabilizes.
        self.ball.serve_bounce_enable = True
        self.ball.serve_bounce_x_range = (-1.24, -0.96)
        self.ball.serve_bounce_x_range_hard = (-1.24, -0.96)
        self.ball.serve_bounce_vz_range = (1.60, 2.10)
        self.ball.serve_bounce_vz_range_hard = (1.60, 2.10)
        self.ball.serve_y_center = 0.12
        self.ball.serve_y_start = 0.04
        self.ball.serve_y_wide = 0.12
        self.ball.require_active_contact = True
        self.ball.active_contact_min_paddle_speed = 0.12
        self.ball.active_contact_min_forward_speed = -0.05
        self.ball.active_contact_require_own_bounce = False
        # v12: a hit only counts near the fixed intercept plane. v11 still let the
        # paddle touch early in front of the -1.55 target; make plane quality part
        # of the first-contact reward and downstream return rewards.
        self.ball.active_contact_hit_plane_margin = 0.10
        self.ball.hit_plane_contact_radius = 0.10
        self.ball.hit_plane_contact_core_radius = 0.03
        self.ball.hit_plane_contact_gate_outcomes = True
        self.ball.hit_plane_contact_outcome_floor = 0.5
        # v11: latch first valid hit quality in the paddle face plane. Keep a 50% floor on
        # pass-net / landing / table-success rewards so a successful return still teaches,
        # but full credit requires a centered hit instead of edge scraping.
        self.ball.sweet_contact_radius = 0.08
        self.ball.sweet_contact_core_radius = 0.03
        self.ball.sweet_contact_face_axis = "y"
        self.ball.sweet_contact_gate_outcomes = True
        self.ball.sweet_contact_outcome_floor = 0.5
        self.ball.serve_curriculum_steps = 0   # easy-only for first run
        self.ball.no_ball_period_s = 0.0


@configclass
class A1TT_EvalEnvCfg(A1TableTennisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.max_episode_length_s = 99999999999
        self.ball.serve_curriculum_steps = 0
        # Runtime serve probes for GUI/debug without another config edit.
        import os as _os
        _y_center = _os.environ.get("TT_SERVE_Y_CENTER")
        _y_half = _os.environ.get("TT_SERVE_Y_HALF")
        _vz_lo = _os.environ.get("TT_SERVE_VZ_LO")
        _vz_hi = _os.environ.get("TT_SERVE_VZ_HI")
        if _y_center is not None:
            self.ball.serve_y_center = float(_y_center)
        if _y_half is not None:
            self.ball.serve_y_start = float(_y_half)
            self.ball.serve_y_wide = float(_y_half)
        if _vz_lo and _vz_hi:
            self.ball.serve_bounce_vz_range = (float(_vz_lo), float(_vz_hi))
            self.ball.serve_bounce_vz_range_hard = (float(_vz_lo), float(_vz_hi))


@configclass
class A1TableTennisAgentCfg(TTAgentCfg):
    experiment_name: str = "a1_tt_v12"
    empirical_normalization = True   # v3: normalize observations for critic stability (v2 diverged, value_loss->1e9)
    logger = "tensorboard"
    save_interval = 100      # 2026-07-06: ckpt every 100 iters (finer, for post-hoc ckpt selection)
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

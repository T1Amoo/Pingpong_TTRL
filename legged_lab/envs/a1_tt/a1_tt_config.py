# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Licensed under BSD-3-Clause.

import copy
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import legged_lab.mdp as mdp
from legged_lab.assets.a1.a1 import (
    A1_RIGHT_ARM_JOINTS,
    A1_INIT_Z,
    A1_USD_PATH_V1_3,
    A1_TT_CFG,
    A1_TT_DAMIAO_DELAYED_CFG,
    A1_TT_OPENARM_CFG,
    A1_TT_REAL_FITTED_CFG,
    A1_TT_REAL_TORQUE_ONLY_CFG,
)
from legged_lab.assets.table_tennis.table import TABLE_CFG
from legged_lab.assets.table_tennis.ball import BALL_CFG
from legged_lab.envs.base.tt_env_config import (  # noqa:F401
    TTAgentCfg,
    TTEnvCfg,
    RewardCfg,
)

A1_ARM_JOINTS = list(A1_RIGHT_ARM_JOINTS)
A1_GROUND_CONTACT_BODIES = ["Link_lun_(r|l)", "Link_wxl_.*"]
A1_TT_RAW_STEPS_PER_ITER = 240  # num_steps_per_env(24) * sim.decimation(10)
A1_REAL_V5_EASY_ITERS = 5000
A1_REAL_V5_RAMP_ITERS = 5000
A1_REAL_V5_SERVE_CURRICULUM_START = A1_REAL_V5_EASY_ITERS * A1_TT_RAW_STEPS_PER_ITER
A1_REAL_V5_SERVE_CURRICULUM_STEPS = A1_REAL_V5_RAMP_ITERS * A1_TT_RAW_STEPS_PER_ITER
A1_REAL_V7_EASY_ITERS = 30000
A1_REAL_V7_RAMP_ITERS = 30000
A1_REAL_V7_SERVE_CURRICULUM_START = A1_REAL_V7_EASY_ITERS * A1_TT_RAW_STEPS_PER_ITER
A1_REAL_V7_SERVE_CURRICULUM_STEPS = A1_REAL_V7_RAMP_ITERS * A1_TT_RAW_STEPS_PER_ITER
A1_REAL_TORQUE_LIMIT_NM = (
    28.0,  # r1
    28.0,  # r2
    28.0,  # r3
    8.0,   # r4
    8.0,   # r5
    8.0,   # r6
    8.0,   # r7
)
A1_DEPLOY_QDES_MAX_DELTA_PER_TICK = (
    0.05,  # r1
    0.05,  # r2
    0.05,  # r3
    0.10,  # r4
    0.10,  # r5
    0.10,  # r6
    0.10,  # r7
)
A1_DAMIAO_PEAK_TORQUE_NM = (
    27.0,  # r1
    27.0,  # r2
    27.0,  # r3
    27.0,  # r4
    7.0,   # r5
    7.0,   # r6
    7.0,   # r7
)
A1_DAMIAO_NO_LOAD_SPEED_RAD_S = (
    2.175,  # r1
    2.175,  # r2
    2.175,  # r3
    2.175,  # r4
    2.61,   # r5
    2.61,   # r6
    2.61,   # r7
)
A1_REAL_FITTED_NODE_KP = (300.0, 300.0, 300.0, 120.0, 120.0, 120.0, 120.0)
A1_REAL_FITTED_NODE_KD = (3.5, 3.5, 3.5, 1.0, 1.0, 1.0, 1.0)
A1_REAL_FITTED_U_MEAN = (
    0.5681831170861339,
    -0.6928691673392671,
    0.7161501174932786,
    1.1293556605846493,
    -1.2407980020381792,
    0.030473524919208673,
    0.7714033875755423,
)
A1_REAL_FITTED_FN_HZ = (
    6.661038037881058,
    5.008208552071931,
    6.984778406823649,
    4.341336283530257,
    15.30971682198707,
    8.223759975617558,
    18.61140865177779,
)
A1_REAL_FITTED_ZETA = (
    0.16764263679500072,
    0.1876649639165574,
    0.28653637194779724,
    0.22981041868709606,
    0.7941795710126007,
    0.565832498155223,
    1.3208910165925782,
)
A1_REAL_FITTED_DELAY_S = (
    0.03502917289780583,
    0.032912611967056964,
    0.02843821965716936,
    0.018006420135349824,
    0.017563104629677986,
    0.013993930820317215,
    0.014997124673895237,
)
A1_REAL_FITTED_GAIN = (
    0.9832701113210972,
    0.9741513252336587,
    0.9991903500772663,
    1.0027218616565117,
    0.9998451719960618,
    1.0020846023179375,
    1.0005302866606762,
)
A1_REAL_FITTED_BIAS_RAD = (
    -0.014673280544373668,
    0.01278231705691013,
    -0.0012048834601215974,
    -0.03152619331089834,
    -0.0003882480267090038,
    0.0010492923888134296,
    -0.00023117043260922898,
)
A1_REAL_DEPLOY_MAX_DELTA_PER_TRAIN_TICK = (
    # Conservative exploration envelope at TTEnv's 50 Hz policy step:
    # r1-r3 <= 2.5 rad/s, r4-r7 <= 5.0 rad/s. This is intentionally below the
    # unloaded 100 Hz arm-node max_delta_per_cycle envelope to leave load margin.
    0.05,  # r1
    0.05,  # r2
    0.05,  # r3
    0.10,  # r4
    0.10,  # r5
    0.10,  # r6
    0.10,  # r7
)

# v8 (2026-07-27): first-order low-pass replacing the bang-bang rate_limit. Per-joint tau, NOT
# uniform: tau >= 1/(4*pi*fn*zeta) means only the underdamped proximal joints (r1/r2/r4,
# fn~4-7 Hz, zeta~0.17-0.23) need large tau (~0.08-0.10) to suppress their resonance; the
# well-damped wrist/forearm (r5/r6 zeta 0.57-0.79, r7 zeta 1.32) barely ring, so a uniform 0.10
# needlessly throttled them (offline replay: r7 hit-velocity dropped ~4x vs tau=0.05). Distal
# joints therefore run at 0.05 to keep the paddle snap. r7 is held at 0.10 (conservative) rather
# than its tiny model-implied tau because it is the empirically jittery joint whose buzz lives
# ABOVE the fitted 0.1-2 Hz identification band -> refine from the planned motor-current data.
# vel_limit matches the deploy servo_velocity_limit.
A1_REAL_DEPLOY_LOWPASS_TAU_S = (0.10, 0.10, 0.08, 0.10, 0.05, 0.05, 0.10)
A1_REAL_DEPLOY_LOWPASS_VEL_LIMIT = (1.0, 1.2, 1.8, 1.6, 4.0, 3.2, 8.0)


def _env_float(name: str):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


def _env_int(name: str):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _env_str(name: str):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def _override_attr(obj, attr: str, env_name: str, cast):
    value = cast(env_name)
    if value is not None:
        setattr(obj, attr, value)
        print(f"[A1TTCfg] {env_name}: {attr}={value}")


def _override_reward_weight(reward_cfg, term_name: str, env_name: str):
    value = _env_float(env_name)
    if value is not None:
        getattr(reward_cfg, term_name).weight = value
        print(f"[A1TTCfg] {env_name}: {term_name}.weight={value}")


def _apply_deploy_reward_overrides(reward_cfg):
    # Environment-variable knobs for short 4096-env ablations. Values are dumped
    # into env.yaml because they mutate the actual config object before training.
    for env_name, term_name in (
        ("TT_REWARD_PADDLE_FACE_X", "paddle_face_x"),
        ("TT_REWARD_FUTURE_DIS_EE", "reward_future_dis_ee"),
        ("TT_REWARD_SWING_THROUGH", "reward_swing_through"),
        ("TT_REWARD_CONTACT", "reward_contact"),
        ("TT_REWARD_SWEET_CONTACT", "reward_sweet_contact"),
        ("TT_REWARD_PASS_NET", "reward_future_pass_net"),
        ("TT_REWARD_LANDING_DIS", "reward_future_landing_dis"),
        ("TT_REWARD_TABLE_SUCCESS", "reward_table_success"),
        ("TT_REWARD_ACTION_RATE_L2", "action_rate_l2"),
        ("TT_REWARD_ACTION_L2", "action_l2"),
        ("TT_REWARD_JOINT_POS_TARGET_LIMITS", "joint_pos_target_limits"),
        ("TT_REWARD_ACTION_TARGET_SLEW_LIMIT", "action_target_slew_limit"),
        ("TT_REWARD_TORQUE_LIMIT", "joint_computed_torque_limit"),
    ):
        if hasattr(reward_cfg, term_name):
            _override_reward_weight(reward_cfg, term_name, env_name)


def _apply_agent_overrides(agent_cfg):
    _override_attr(agent_cfg.policy, "init_noise_std", "TT_PPO_INIT_NOISE_STD", _env_float)
    _override_attr(agent_cfg.algorithm, "learning_rate", "TT_PPO_LEARNING_RATE", _env_float)
    _override_attr(agent_cfg.algorithm, "entropy_coef", "TT_PPO_ENTROPY_COEF", _env_float)
    _override_attr(agent_cfg.algorithm, "desired_kl", "TT_PPO_DESIRED_KL", _env_float)
    _override_attr(agent_cfg.algorithm, "num_learning_epochs", "TT_PPO_NUM_EPOCHS", _env_int)
    _override_attr(agent_cfg.algorithm, "num_mini_batches", "TT_PPO_NUM_MINI_BATCHES", _env_int)
    _override_attr(agent_cfg, "num_steps_per_env", "TT_PPO_NUM_STEPS_PER_ENV", _env_int)

    schedule = _env_str("TT_PPO_SCHEDULE")
    if schedule is not None:
        agent_cfg.algorithm.schedule = schedule
        print(f"[A1TTCfg] TT_PPO_SCHEDULE: schedule={schedule}")

    run_name = _env_str("TT_RUN_NAME")
    if run_name is not None:
        agent_cfg.run_name = run_name
        print(f"[A1TTCfg] TT_RUN_NAME: run_name={run_name}")


@configclass
class A1TableTennisRewardCfg(RewardCfg):
    # --- base-shake penalty (arm reaction wobbles the free chassis) ---
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.2)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # --- arm smoothness / limits ---
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)    # v12: raise from -0.003 to reduce bang-bang action targets.
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)             # v12: raise from -0.0005; still below G1's -0.002.
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_pos_target_limits = RewTerm(func=mdp.joint_pos_target_limits, weight=-0.5)   # v12: raise from -0.1; still below G1's -1.0 after earlier critic instability.
    joint_deviation_right_arm = RewTerm(
        func=mdp.joint_deviation_l1_idle,   # v5: idle-only (no-ball); was joint_deviation_l1 every step -> fought the swing
        weight=-0.1,                        # v11: -0.05 -> -0.1 (still mask_invalid-gated, pairs with reward_arm_ready_idle)
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_ARM_JOINTS)},
    )
    # --- don't crash into table ---
    penalty_robot_table_proximity_x = RewTerm(
        func=mdp.penalty_robot_table_proximity_x,
        weight=-20.0,
        params={"min_distance": 0.15, "std": 0.07},
    )
    penalty_arm_table_collision = RewTerm(
        func=mdp.arm_table_collision,
        weight=-40.0,
    )
    penalty_arm_table_stuck_contact = RewTerm(
        func=mdp.arm_table_stuck_contact,
        weight=-20.0,
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    # --- ready-pose regularization when no playable ball ---
    # reward_idle_pose is G1-specific (hardcoded 23-joint ready vector); drop for A1.
    reward_idle_stand = RewTerm(func=mdp.reward_idle_stand, weight=0.5)
    # v11: mask_invalid-gated ready-pose shaping (NOT idle injection; no_ball_period_s stays 0).
    # Strengthens the too-weak joint_deviation_l1_idle (-0.05) that let v10 flail when no ball
    # (idle |action|~4.7, r1/r3/r7 driven 1.2-1.6 rad off home). exp ready bonus + arm-velocity
    # penalty, both ZERO while a ball is playable -> never competes with the swing.
    reward_arm_ready_idle = RewTerm(
        func=mdp.reward_arm_ready_idle,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_ARM_JOINTS), "k": 4.0},
    )
    penalty_arm_vel_idle = RewTerm(
        func=mdp.penalty_arm_vel_idle,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_ARM_JOINTS)},
    )
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
        params={"std_ee": 0.5, "threshold": 0.08, "z_weight": 2.5},
    )
    penalty_paddle_above_target = RewTerm(
        func=mdp.penalty_paddle_above_future_target,
        weight=-2.0,
        params={"margin": 0.12, "max_error": 0.50},
    )
    reward_swing_through = RewTerm(
        func=mdp.reward_swing_through,
        weight=1.0,   # only reward forward swing once the blade is near the target y/z; avoids high-overhead farming.
        params={"near_dist": 0.30, "target_yz_gate": 0.18},
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
class A1TableTennisDeployRewardCfg(A1TableTennisRewardCfg):
    action_target_slew_limit = RewTerm(
        func=mdp.action_target_slew_limit_l2,
        weight=-0.02,
    )


@configclass
class A1TableTennisDamiaoRewardCfg(A1TableTennisRewardCfg):
    motor_speed_margin = RewTerm(
        func=mdp.motor_speed_margin_l2,
        weight=-0.25,
        params={
            "soft_ratio": 0.85,
            "no_load_speed": A1_DAMIAO_NO_LOAD_SPEED_RAD_S,
            "asset_cfg": SceneEntityCfg("robot", joint_names=A1_ARM_JOINTS),
        },
    )
    speed_torque_limit_violation = RewTerm(
        func=mdp.speed_torque_limit_violation_l2,
        weight=-4.0,
        params={
            "peak_torque": A1_DAMIAO_PEAK_TORQUE_NM,
            "no_load_speed": A1_DAMIAO_NO_LOAD_SPEED_RAD_S,
            "min_torque_fraction": 0.0,
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
        # A1 base stays near x=-1.8. v13 moves the intercept plane back from -1.55 to -1.60:
        # play/sim2real showed the arm could graze the table when asked to hit too far forward.
        # Prefer fixing the target geometry before adding a sparse table-collision penalty.
        self.robot.hit_plane_x = -1.60
        # Fixed hit/intercept plane, matching the G1 task semantics: the learned predictor may
        # output 3 values, but x is the plane anchor and only y/z should meaningfully vary.
        self.robot.hit_target_x_range = (self.robot.hit_plane_x, self.robot.hit_plane_x)
        # v3: spread the intercept target across the measured forehand-reachable envelope. v13 keeps
        # that broad y/z envelope but anchors it on the new -1.60 hit plane.
        self.robot.hit_target_y_range = (0.0, 0.55)
        self.robot.hit_target_z_range = (0.90, 1.25)
        self.observations.joint_names = A1_ARM_JOINTS
        self.actions.joint_names = A1_ARM_JOINTS
        # v5 serve curriculum, following the successful G1 fixed-iteration pattern:
        #   0..5k:    current easy serve, y ~= 0.08..0.16, z at x=-1.60 ~= 0.96..1.14.
        #   5k..10k:  ramp to hard, expanding mostly z/speed and only a little y.
        #   10k..30k: consolidate at hard.
        # Keep the hard distribution forehand-reachable: y ~= 0.00..0.24 and estimated
        # z at the fixed hit plane stays inside the target band, ~= 0.92..1.20.
        self.ball.serve_bounce_enable = True
        self.ball.serve_bounce_x_range = (-1.24, -0.96)
        self.ball.serve_bounce_x_range_hard = (-1.30, -0.92)
        self.ball.serve_bounce_vz_range = (1.60, 2.10)
        self.ball.serve_bounce_vz_range_hard = (1.45, 2.35)
        self.ball.serve_y_center = 0.12
        self.ball.serve_y_start = 0.04
        self.ball.serve_y_wide = 0.12
        self.ball.require_active_contact = True
        self.ball.active_contact_min_paddle_speed = 0.12
        self.ball.active_contact_min_forward_speed = -0.05
        self.ball.active_contact_require_own_bounce = False
        # v12: a hit only counts near the fixed intercept plane. v11 still let the
        # paddle touch early in front of the fixed target; make plane quality part
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
        self.ball.serve_curriculum_perf_gated = False
        self.ball.serve_curriculum_phase_start = A1_REAL_V5_SERVE_CURRICULUM_START
        self.ball.serve_curriculum_steps = A1_REAL_V5_SERVE_CURRICULUM_STEPS
        self.ball.no_ball_period_s = 0.0


@configclass
class A1TableTennisDeployEnvCfg(A1TableTennisEnvCfg):
    reward = A1TableTennisDeployRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # a1_tt_real_v4: train through the measured real-arm closed-loop response,
        # but first apply the same per-cycle q_des slew clamp used by deployment.
        # The second-order model should see only commands the real arm node would
        # allow through its raw_q -> cmd_q limiter.
        self.scene.robot = A1_TT_REAL_FITTED_CFG
        self.robot.action_target_rate_limit_enable = True
        self.robot.action_target_max_delta_per_tick = A1_REAL_DEPLOY_MAX_DELTA_PER_TRAIN_TICK
        self.robot.action_response_model_enable = True
        self.robot.action_response_u_mean = A1_REAL_FITTED_U_MEAN
        self.robot.action_response_fn_hz = A1_REAL_FITTED_FN_HZ
        self.robot.action_response_zeta = A1_REAL_FITTED_ZETA
        self.robot.action_response_delay_s = A1_REAL_FITTED_DELAY_S
        self.robot.action_response_gain = A1_REAL_FITTED_GAIN
        self.robot.action_response_bias_rad = A1_REAL_FITTED_BIAS_RAD
        # Keep training hit-first. Prior idle/no-ball curricula hurt receiving stability;
        # invalid/no-ball still uses the sentinel observation path, but we do not sample a
        # dedicated no-ball phase in tonight's scratch run.
        self.ball.no_ball_period_s = 0.0
        self.ball.ball_active_s = 0.0
        self.ball.no_ball_curriculum_steps = 0
        self.ball.idle_reward_ramp_steps = 0
        self.ball.curriculum_phase1_steps = 0
        # Stay at the real effort envelope from the first iteration. Do not revive the
        # older proximal over-torque curriculum for this deploy-alignment run.
        self.robot.effort_curriculum_start_scale = 1.0
        self.robot.effort_curriculum_steps = 0
        self.robot.effort_curriculum_num_joints = 0
        # v4 4096 stable run: keep non-contact shaping as guidance only and make
        # the un-fakeable outcome ladder dominate the gradient.
        self.reward.paddle_face_x.weight = 0.25
        self.reward.reward_future_dis_ee.weight = 1.0
        self.reward.reward_swing_through.weight = 0.25
        self.reward.reward_contact.weight = 300.0
        self.reward.reward_sweet_contact.weight = 70.0
        self.reward.reward_future_pass_net.weight = 150.0
        self.reward.reward_future_landing_dis.weight = 90.0
        self.reward.reward_table_success.weight = 300.0
        self.reward.penalty_paddle_above_target.weight = -5.0
        _apply_deploy_reward_overrides(self.reward)


@configclass
class A1TableTennisTorqueOnlyEnvCfg(A1TableTennisDeployEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = A1_TT_REAL_TORQUE_ONLY_CFG
        # a1_tt_real_v7: raw q_des after deploy slew limiting goes directly to
        # explicit MIT torque with 28/8 Nm limits. The fitted second-order
        # response lives inside the Damiao actuator here, so keep TTEnv's
        # action-response filter disabled to avoid double filtering.
        self.robot.action_response_model_enable = False
        self.robot.action_response_u_mean = ()
        self.robot.action_response_fn_hz = ()
        self.robot.action_response_zeta = ()
        self.robot.action_response_delay_s = ()
        self.robot.action_response_gain = ()
        self.robot.action_response_bias_rad = ()
        # The explicit actuator uses a large PhysX effort_limit_sim to avoid
        # double-clipping; score the true Damiao envelope explicitly here.
        self.reward.joint_computed_torque_limit.weight = -0.03
        self.reward.joint_computed_torque_limit.params["limit"] = A1_REAL_TORQUE_LIMIT_NM
        # Default 100k schedule: 0..30k easy, 30k..60k expand, 60k..100k consolidate.
        # Override the curriculum phase with A1_REAL_V7_CURRICULUM_START_ITER and
        # A1_REAL_V7_CURRICULUM_RAMP_ITERS for mid-run range-expansion resumes.
        # Hard range is wider than v6, but stays on the reachable forehand side.
        self.ball.serve_bounce_x_range = (-1.24, -0.96)
        self.ball.serve_bounce_x_range_hard = (-1.32, -0.88)
        self.ball.serve_bounce_vz_range = (1.60, 2.10)
        self.ball.serve_bounce_vz_range_hard = (1.45, 2.40)
        self.ball.serve_y_center = 0.17
        self.ball.serve_y_start = 0.04
        self.ball.serve_y_wide = 0.17
        self.ball.serve_curriculum_perf_gated = False
        curriculum_start_iter = _env_int("A1_REAL_V7_CURRICULUM_START_ITER")
        curriculum_ramp_iters = _env_int("A1_REAL_V7_CURRICULUM_RAMP_ITERS")
        self.ball.serve_curriculum_phase_start = (
            A1_REAL_V7_SERVE_CURRICULUM_START
            if curriculum_start_iter is None
            else curriculum_start_iter * A1_TT_RAW_STEPS_PER_ITER
        )
        self.ball.serve_curriculum_steps = (
            A1_REAL_V7_SERVE_CURRICULUM_STEPS
            if curriculum_ramp_iters is None
            else curriculum_ramp_iters * A1_TT_RAW_STEPS_PER_ITER
        )
        if curriculum_start_iter is not None:
            print(
                "[A1TTCfg] A1_REAL_V7_CURRICULUM_START_ITER: "
                f"serve_curriculum_phase_start={self.ball.serve_curriculum_phase_start}"
            )
        if curriculum_ramp_iters is not None:
            print(
                "[A1TTCfg] A1_REAL_V7_CURRICULUM_RAMP_ITERS: "
                f"serve_curriculum_steps={self.ball.serve_curriculum_steps}"
            )


@configclass
class A1TableTennisTorqueLowpassEnvCfg(A1TableTennisTorqueOnlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # a1_tt_real_v8: same DamiaoMIT actuator + serve curriculum as v7, but the command
        # shaping switches from the bang-bang rate_limit (which excited the identified
        # underdamped 4-7 Hz proximal resonance -> hardware jitter) to the first-order low-pass
        # that deployment already uses. Training and deploy now share the command shaping, so the
        # policy learns against the non-ringing command instead of relying on sim's harmless ring.
        self.robot.action_target_rate_limit_enable = False
        self.robot.action_target_max_delta_per_tick = ()
        self.robot.action_target_lowpass_enable = True
        self.robot.action_target_lowpass_tau_s = A1_REAL_DEPLOY_LOWPASS_TAU_S
        self.robot.action_target_lowpass_vel_limit = A1_REAL_DEPLOY_LOWPASS_VEL_LIMIT
        # v8 also re-centers the robot to table-relative y=0 (the real robot sits on the table
        # centerline). v7 trained at y=0.76 and deployment shifted every ball by +0.76 to
        # compensate; moving the robot to y=0 removes that deploy-side offset (deploy ball bridge
        # origin_in_training_world y: 0.76 -> 0.0). Relative geometry is preserved -- base, home_y
        # and the serve bounce center all shift by -0.76, so the forehand paddle
        # (home_y + paddle_y_offset = 0 - 0.66 = -0.66) and the ball still meet the same way.
        # deepcopy so v7 (which shares A1_TT_REAL_TORQUE_ONLY_CFG) keeps its y=0.76 spawn.
        robot_cfg = copy.deepcopy(A1_TT_REAL_TORQUE_ONLY_CFG)
        robot_cfg.init_state.pos = (-1.8, 0.0, A1_INIT_Z)
        self.scene.robot = robot_cfg
        self.robot.home_y = 0.0
        self.ball.serve_y_center = -0.59   # v7 0.17 shifted by -0.76 (table half-width 0.7625: hard spread -0.59+-0.17 stays on-table)
        # hit_target_y_range clamps ball_future_pose.y (the hit target the actor/critic see and the
        # plausibility gate validates). v7 used (0.0, 0.55) for the robot at y=0.76; the y=0 recenter
        # must shift it by -0.76 too, else the target y is clamped to the old +y band (~0) while the
        # paddle/gate sit at home_y+paddle_offset=-0.66 -> pred_usable never true, policy chases a
        # target ~0.6 m off the ball. Shift (0.0,0.55) -> (-0.76,-0.21).
        self.robot.hit_target_y_range = (-0.76, -0.21)


@configclass
class A1TableTennisV13TestEnvCfg(A1TableTennisTorqueLowpassEnvCfg):
    # TEMP (2026-07-28): run the v8 policy on the V1_3 (heavier arm + 0.15 paddle) USD for a
    # visual test only. Not for training, not committed. Swaps just the robot USD; everything
    # else (actuator, lowpass, y=0 geometry, ball/table) stays as v8.
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = "/home/woan/下载/X1_URDF_V1_3_test/X1_URDF_V1_3_paddle.usd"


@configclass
class A1TableTennisV9EnvCfg(A1TableTennisTorqueLowpassEnvCfg):
    # v9 (2026-07-28): train on the V1_3 CAD (arm ~6% heavier, matches latest export) + reward
    # changes to stop the v8 "twist the paddle to graze the ball instead of moving the arm" habit.
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = A1_USD_PATH_V1_3
        # (1) reward contact QUALITY not raw contact (raw contact is farmable by wrist-angling the
        #     blade into a passing ball); tighten the sweet radius so edge grazes score low.
        self.reward.reward_contact.weight = 60.0
        self.reward.reward_sweet_contact.weight = 100.0
        self.ball.sweet_contact_radius = 0.05
        # (2) sharper paddle-POSITION tracking. r7 roll cannot move the paddle position (COM on
        #     axis), so a strong/sharp position reward can only be satisfied by moving the arm.
        self.reward.reward_future_dis_ee.weight = 3.5
        self.reward.reward_future_dis_ee.params["std_ee"] = 0.35
        # (3)+(4) per-joint action penalty: cheap torque-limited proximal (needs to move), expensive
        #     low-inertia wrist r7 (was farming twist). weight (rate/l2) is modulated per joint.
        _AW = (0.5, 0.5, 0.5, 0.7, 1.0, 1.0, 2.0)
        self.reward.action_l2.func = mdp.action_l2_weighted
        self.reward.action_l2.params = {"weights": _AW}
        self.reward.action_rate_l2.func = mdp.action_rate_l2_weighted
        self.reward.action_rate_l2.params = {"weights": _AW}


@configclass
class A1TableTennisV10EnvCfg(A1TableTennisV9EnvCfg):
    # v10 (2026-07-28): (a) new default/ready pose, (b) delaymotor(DamiaoMIT) response params
    #   swapped to the 2026-07-28 re-fit values (<=2Hz chirp; non-unique, may revert if v10 worse),
    #   (c) hit-plane/target/serve geometry rebuilt for the new paddle ready position.
    #   FK: paddle world (-1.383,-0.725,1.278) -> (-1.578,-0.849,1.296), Δ=(-0.194,-0.124,+0.018).
    def __post_init__(self):
        super().__post_init__()  # V1_3 + v9 anti-wrist-twist rewards + lowpass + y=0 recenter
        # (a) new default/ready pose
        NEW_POSE = {"r1": -0.505, "r2": -1.13, "r3": 1.13, "r4": 1.02,
                    "r5": -0.7, "r6": 0.0, "r7": -1.3}
        self.scene.robot.init_state.joint_pos.update(NEW_POSE)
        # (b) KEEP official DamiaoMIT response dynamics (validated ~7.5mrad). This-session re-fit
        #     is non-unique/degenerate (fn/zeta/delay trade off under <=2Hz data, no gain), so we
        #     do NOT swap fn/zeta/delay/gain. Only re-anchor the operating point to the new pose
        #     (u_mean/intercept were the OLD ready pose) so the response model holds the new pose.
        act = self.scene.robot.actuators["right_arm"]
        act.response_u_mean = dict(NEW_POSE)
        act.response_intercept = dict(NEW_POSE)
        # (c) rebuilt hit geometry: paddle x ~unchanged (new -1.578 ~ old hit_plane -1.60),
        #     y shifts -0.12, z +0.02. Serve x/vz unchanged (hit-plane x barely moved).
        self.robot.hit_plane_x = -1.58
        self.robot.hit_target_x_range = (-1.58, -1.58)
        self.robot.hit_target_y_range = (-0.88, -0.33)
        self.robot.hit_target_z_range = (0.92, 1.27)
        self.robot.paddle_y_offset = -0.72
        self.ball.serve_y_center = -0.71


@configclass
class A1TableTennisV11EnvCfg(A1TableTennisV10EnvCfg):
    # v11 (2026-07-30): inherits V10 (forehand ready pose + rebuilt hit geometry + y=0), and
    #   (a) swaps DamiaoMIT response dynamics to the 2026-07-29 re-fit (0.1-6Hz chirp + 0.4Hz
    #       step, measured AT the forehand ready pose). Unlike the <=2Hz 0728 fit, this has
    #       chirp+step so near-end fn/zeta/delay/gain are well-determined; the (fn,zeta,delay,
    #       gain) tuple reproduces the measured response (chirp_rmse<0.017), so the fn/delay
    #       degeneracy does NOT hurt sim fidelity. Wrist fn was unidentifiable (>~20Hz, capped
    #       at the 30Hz search ceiling by the 100Hz-log / 6Hz-chirp bandwidth) -> use 30Hz;
    #       exact value is irrelevant for such a fast joint, and the measured wrist delay/zeta
    #       are kept (they are what matter). u_mean/intercept stay = forehand pose (from V10).
    #   (b) reward: mask_invalid-gated ready-pose shaping (reward_arm_ready_idle +
    #       penalty_arm_vel_idle, both zero while a ball is playable; no idle injection,
    #       no_ball_period_s stays 0) to fix v10's no-ball whole-arm flailing (idle |action|
    #       ~4.7, r1/r3/r7 driven 1.2-1.6 rad off home) WITHOUT softening the swing.
    def __post_init__(self):
        super().__post_init__()  # V10: forehand pose + geometry + u_mean/intercept=forehand
        act = self.scene.robot.actuators["right_arm"]
        # 2026-07-29 forehand fit (系统辨识/0729/fit/A1_0729_actuator_params.csv).
        act.response_fn_hz = {
            "r1": 15.835, "r2": 2.806, "r3": 18.0, "r4": 3.648,
            "r5": 30.0, "r6": 30.0, "r7": 30.0,          # wrist fn unidentifiable -> "high"
        }
        act.response_zeta = {
            "r1": 0.271, "r2": 0.555, "r3": 0.752, "r4": 0.721,
            "r5": 0.331, "r6": 1.058, "r7": 0.782,
        }
        act.response_delay_s = {
            "r1": 0.0362, "r2": 0.0, "r3": 0.0441, "r4": 0.0,
            "r5": 0.0443, "r6": 0.0381, "r7": 0.0376,
        }
        act.response_linear_gain = {
            "r1": 0.9832, "r2": 0.9796, "r3": 0.9275, "r4": 1.0885,
            "r5": 0.9649, "r6": 1.0135, "r7": 1.0027,
        }
        # (c) robot back to the v7-style offset y=+0.76 (undo the v8 y=0 recenter). The forehand
        #     pose reaches -y, so at +0.76 the paddle meets the ball near table center
        #     (home_y+paddle_offset = 0.76-0.72 = +0.04, on-table). Shift every world-y term by
        #     +0.76 (paddle_y_offset is base-relative -> unchanged); relative task == v10.
        self.scene.robot.init_state.pos = (-1.8, 0.76, A1_INIT_Z)
        self.robot.home_y = 0.76
        self.ball.serve_y_center = 0.05                 # v10 -0.71 + 0.76
        self.robot.hit_target_y_range = (-0.12, 0.43)   # v10 (-0.88, -0.33) + 0.76


@configclass
class A1TableTennisV12EnvCfg(A1TableTennisV11EnvCfg):
    # v12 (2026-07-30): "forward-reach" hit geometry. In v10/v11 the ready forehand paddle sat
    #   essentially ON the hit plane (FK: ready touch_point x=-1.578 vs hit_plane_x=-1.58, Δ=1.5mm)
    #   -> the paddle just waits in place, no active forward swing is trained. To force an active
    #   forward-reach hit, move the hit plane ~0.16m IN FRONT of the ready blade (toward the net,
    #   +x). The blade is already ~above the near table edge, so instead of pushing the hit plane
    #   further forward (table-collision risk) we move the BASE BACK 0.20m and push the hit plane
    #   forward 0.04m; net push = ready->hit ≈ 0.16m, with 0.25m clearance to the near edge.
    #
    #   Geometry (arm pose fixed -> ready blade tracks base: ready_x ≈ base_x + 0.222):
    #     base x  -1.8 -> -2.0     (world/env-local; joint pose UNCHANGED, so u_mean/intercept stay)
    #     hit_plane_x  -1.58 -> -1.62
    #     ready blade x  -1.578 -> ≈ -1.778   (FK)
    #     push (hit - ready)  ~0 -> +0.158 ≈ 0.16m ✓
    #     hit_plane to near edge (x=-1.37)  0.21m -> 0.25m ✓
    #   base=-2.0 matches the validated G1 stance (g1.py:40 "stance >=60cm from table").
    #   y/z windows unchanged (x-shift does not move them). Serve stays v11 initially; the ball
    #   now travels 4cm further (-x) while descending -> verified in-window by the TT_SERVE_PROBE
    #   physical crossing histogram (bump serve_bounce_vz only if z drops below the window).
    #   No new push/wait rewards: forward-reach is driven by geometry + reward_future_dis_ee
    #   (pulls blade to the clamped hit target) + reward_swing_through (0.25); adding reward
    #   shaping risks critic divergence (g1_tt_critic_divergence_termination_penalty_2026-06-30).
    def __post_init__(self):
        super().__post_init__()  # V11: forehand pose + 0729 fit + lowpass + ready-idle + y=0.76
        self.scene.robot.init_state.pos = (-2.0, 0.76, A1_INIT_Z)
        self.robot.hit_plane_x = -1.62
        self.robot.hit_target_x_range = (-1.62, -1.62)


@configclass
class A1TableTennisOpenArmEnvCfg(A1TableTennisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = A1_TT_OPENARM_CFG
        self.robot.action_target_rate_limit_enable = False
        self.robot.action_target_max_delta_per_tick = ()
        # Match the hit-first real/deploy flow, but without q_des pre-limit/filter.
        self.ball.no_ball_period_s = 0.0
        self.ball.ball_active_s = 0.0
        self.ball.no_ball_curriculum_steps = 0
        self.ball.idle_reward_ramp_steps = 0
        self.ball.curriculum_phase1_steps = 0
        self.robot.effort_curriculum_start_scale = 1.0
        self.robot.effort_curriculum_steps = 0
        self.robot.effort_curriculum_num_joints = 0


@configclass
class A1TableTennisDamiaoEnvCfg(A1TableTennisEnvCfg):
    reward = A1TableTennisDamiaoRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = A1_TT_DAMIAO_DELAYED_CFG
        # Route-A weekend run: model actuator dynamics in sim instead of forcing
        # deployment-side q_des slew clipping into the training loop.
        self.robot.action_target_rate_limit_enable = False
        self.robot.action_target_max_delta_per_tick = ()
        self.ball.no_ball_period_s = 0.0
        self.ball.ball_active_s = 0.0
        self.ball.no_ball_curriculum_steps = 0
        self.ball.idle_reward_ramp_steps = 0
        self.ball.curriculum_phase1_steps = 0
        self.robot.effort_curriculum_start_scale = 1.0
        self.robot.effort_curriculum_steps = 0
        self.robot.effort_curriculum_num_joints = 0


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
    experiment_name: str = "a1_tt_v13"
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


@configclass
class A1TableTennisDeployAgentCfg(A1TableTennisAgentCfg):
    experiment_name: str = "a1_tt_real_v6"
    run_name = "scratch_sjfixed_v4ppo_paddleabove_5k_easy_5k_ramp_20k_hold"
    resume = False
    max_iterations = 30000

    def __post_init__(self):
        super().__post_init__()
        # v6: keep the fixed-sj / serve-curriculum changes, but return PPO to the
        # v4 stable settings after v5 showed std/action/value runaway on 4096 envs.
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.entropy_coef = 0.006
        self.algorithm.desired_kl = 0.01
        self.algorithm.num_mini_batches = 64
        self.algorithm.schedule = "adaptive"
        _apply_agent_overrides(self)


@configclass
class A1TableTennisTorqueOnlyAgentCfg(A1TableTennisDeployAgentCfg):
    experiment_name: str = "a1_tt_real_v7"
    run_name = "scratch_damiao_mit_30k_easy_30k_expand_40k_hold"
    max_iterations = 100000


@configclass
class A1TableTennisTorqueLowpassAgentCfg(A1TableTennisTorqueOnlyAgentCfg):
    experiment_name: str = "a1_tt_real_v8"
    run_name = "scratch_lowpass_tau0p10"
    max_iterations = 100000


@configclass
class A1TableTennisV13TestAgentCfg(A1TableTennisTorqueLowpassAgentCfg):
    experiment_name: str = "a1_tt_real_v8"  # find logs/a1_tt_real_v8/pulled_14000/model_14000.pt


@configclass
class A1TableTennisV9AgentCfg(A1TableTennisTorqueLowpassAgentCfg):
    experiment_name: str = "a1_tt_real_v9"
    run_name = "scratch_v13_rewardfix"
    max_iterations = 100000


@configclass
class A1TableTennisV10AgentCfg(A1TableTennisTorqueLowpassAgentCfg):
    experiment_name: str = "a1_tt_real_v10"
    run_name = "scratch_newpose_officialdamiao_5k_10k_5k"
    max_iterations = 20000


@configclass
class A1TableTennisV11AgentCfg(A1TableTennisTorqueLowpassAgentCfg):
    experiment_name: str = "a1_tt_real_v11"
    run_name = "scratch_fit0729_readyidle_y076_10k_10k_10k"
    max_iterations = 30000


@configclass
class A1TableTennisV12AgentCfg(A1TableTennisTorqueLowpassAgentCfg):
    experiment_name: str = "a1_tt_real_v12"
    run_name = "scratch_forwardhit_base-2.0_hitplane-1.62_predictor"
    max_iterations = 30000


@configclass
class A1TableTennisOpenArmAgentCfg(A1TableTennisAgentCfg):
    experiment_name: str = "a1_tt_openarm_v1"
    run_name = "scratch_openarm_implicit"
    resume = False
    max_iterations = 100000


@configclass
class A1TableTennisDamiaoAgentCfg(A1TableTennisAgentCfg):
    experiment_name: str = "a1_tt_damiao_v1"
    run_name = "scratch_delayed_pd_torque_speed"
    resume = False
    max_iterations = 100000

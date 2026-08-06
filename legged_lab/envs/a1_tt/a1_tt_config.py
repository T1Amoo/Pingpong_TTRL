# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Licensed under BSD-3-Clause.

import copy
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import legged_lab.mdp as mdp
from legged_lab.physics import a1_backhand_v2_contract as backhand_v2
from legged_lab.physics import a1_backhand_v3_contract as backhand_v3
from legged_lab.physics import a1_backhand_v4_contract as backhand_v4
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

# Backhand base imported from mentor model_9700, plus the 2026-08-01 natural-swing
# q_des->q refit.  The fit uses the *post-tau* command seen by the SDK and is a
# closed-loop black-box response; pair it with A1_TT_REAL_FITTED_CFG (high-bandwidth
# tracking) rather than the explicit Damiao actuator to avoid applying two motor
# dynamics in series.  Validation on the new hardware-latency trace is 3.1--8.2 mrad.
A1_BACKHAND_READY_Q = (1.450, -0.762, -2.050, 1.445, 0.206, -0.827, 1.043)
A1_BACKHAND_RESPONSE_FN_HZ = (3.166, 9.959, 9.899, 5.892, 20.000, 8.572, 19.455)
A1_BACKHAND_RESPONSE_ZETA = (0.414, 0.559, 0.728, 0.325, 0.889, 0.615, 0.565)
A1_BACKHAND_RESPONSE_DELAY_S = (0.010, 0.040, 0.040, 0.030, 0.040, 0.030, 0.040)
A1_BACKHAND_RESPONSE_DELAY_JITTER_S = (0.005, 0.010, 0.010, 0.008, 0.008, 0.008, 0.008)
A1_BACKHAND_RESPONSE_GAIN = (0.9965, 1.0006, 1.0060, 0.9771, 1.0060, 1.0045, 0.9948)
A1_BACKHAND_RESPONSE_BIAS_RAD = (-0.0208, 0.0004, -0.0095, -0.0084, -0.0050, 0.0041, 0.0003)
A1_BACKHAND_LOWPASS_TAU_RANGE_S = (
    (0.080, 0.130),  # r1
    (0.080, 0.130),  # r2
    (0.065, 0.105),  # r3
    (0.080, 0.130),  # r4
    (0.040, 0.065),  # r5
    (0.040, 0.065),  # r6
    (0.075, 0.130),  # r7
)


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
    # v13: first-contact forward-speed bonus (un-farmable; fires once per rally at the hit).
    # OFF by default (weight 0.0) so v9-v12 are unchanged; v13 turns it on to force an active
    # swing-into-the-ball instead of a static block.
    reward_approach_velocity = RewTerm(
        func=mdp.reward_approach_velocity,
        weight=0.0,
    )
    # v14: FIRST-CONTACT paddle-normal-vs-ball-velocity alignment (un-farmable, fires once per rally).
    # OFF by default (0.0) so v9-v13 are unchanged; v14 turns it on to stop glancing/"往身侧打" hits
    # by paying for meeting the ball square-on (normal parallel to the incoming ball line at contact).
    reward_hit_direction = RewTerm(
        func=mdp.reward_hit_direction,
        weight=0.0,
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
class A1TableTennisBackhandRewardCfg(A1TableTennisRewardCfg):
    """Success-first reward subset available in this repository.

    The delivered mentor snapshot names additional private backhand shaping
    terms whose source was not included.  This task keeps its verified outcome
    ladder and uses the current un-farmable approach/direction terms instead of
    pretending those missing functions can be resumed bit-for-bit.
    """

    def __post_init__(self):
        self.action_rate_l2.weight = -0.015
        self.action_l2.weight = -0.002
        self.joint_deviation_right_arm.weight = -0.05
        self.paddle_face_x.weight = 0.0
        self.reward_contact.weight = 260.0
        self.reward_sweet_contact.weight = 50.0
        self.reward_future_dis_ee.weight = 1.0
        self.penalty_paddle_above_target.weight = -4.0
        self.reward_swing_through.weight = 0.75
        self.reward_approach_velocity.weight = 3.0
        self.reward_hit_direction.weight = 8.0
        self.reward_future_landing_dis.weight = 90.0
        self.reward_future_pass_net.weight = 180.0
        self.reward_table_success.weight = 300.0
        self.reward_arm_ready_idle.weight = 3.0
        self.penalty_arm_vel_idle.weight = -0.10
        # The closed-loop black-box actuator intentionally does not expose the
        # real SDK torque, so a simulated computed-torque penalty is misleading.
        self.joint_computed_torque_limit.weight = 0.0


@configclass
class A1TableTennisBackhandV4RewardCfg(A1TableTennisBackhandRewardCfg):
    """A1-only v4 reward terms; historical backhand configs stay bit-for-bit unchanged."""

    penalty_early_paddle_forward = RewTerm(
        func=mdp.penalty_early_paddle_forward,
        weight=0.0,
        params={
            "release_s": backhand_v4.EARLY_HOLD_RELEASE_S,
            "ramp_s": backhand_v4.EARLY_HOLD_RAMP_S,
            "min_retraction_m": backhand_v4.EARLY_MIN_RETRACTION_M,
            "max_excess_m": backhand_v4.EARLY_MAX_EXCESS_M,
        },
    )
    penalty_late_paddle_backtrack = RewTerm(
        func=mdp.penalty_late_paddle_backtrack,
        weight=0.0,
        params={
            "window_s": backhand_v4.LATE_BACKTRACK_WINDOW_S,
            "speed_scale_mps": backhand_v4.LATE_BACKTRACK_SPEED_SCALE_MPS,
        },
    )
    penalty_contact_lateral_paddle_speed = RewTerm(
        func=mdp.penalty_contact_lateral_paddle_speed,
        weight=0.0,
        params={
            "deadband_mps": backhand_v4.CONTACT_LATERAL_SPEED_DEADBAND_MPS,
            "ramp_mps": backhand_v4.CONTACT_LATERAL_SPEED_RAMP_MPS,
        },
    )
    penalty_predicted_landing_outside_table = RewTerm(
        func=mdp.penalty_a1_predicted_landing_outside_table,
        weight=0.0,
        params={
            "half_penalty_distance_m": (
                backhand_v4.LANDING_OUTSIDE_HALF_PENALTY_DISTANCE_M
            ),
        },
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
        # Preserve v1-v3's historical landing projection for reproducibility.
        # V4 overrides this old 2.7 g/Cd=.47 value with the actual 3.4 g ball
        # and Cd=.4378 force-field coefficient.
        self.ball.landing_drag_accel_k = 0.13398310891143134
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
        # A1 table-tennis material contract: the ball is 0.95, the table is
        # 0.95, and the paddle is fixed at 0.75.  PhysX uses combine=min, so
        # ball-paddle=0.75 while ball-table=0.95.  Keep this event on the base
        # A1 task so all future A1 training versions inherit the same contact.
        self.domain_rand.events.paddle_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["Link_r_paddle"]),
                "static_friction_range": (0.5, 0.5),
                "dynamic_friction_range": (0.5, 0.5),
                "restitution_range": (0.75, 0.75),
                "num_buckets": 1,
                "make_consistent": True,
            },
        )

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
class A1TableTennisV13EnvCfg(A1TableTennisV12EnvCfg):
    # v13 (2026-07-31): v12 (forward-reach geometry) only learned STATIC pre-positioning -- the
    #   serve distribution was narrow enough (hard lateral half-width 0.17m, centered 0.05 low in
    #   the reachable window) that one fixed paddle spot covered every serve, so camping at the
    #   intercept dominated an active swing. v13 breaks this with THREE coordinated changes;
    #   geometry (base -2.0 / hit_plane -1.62) stays from v12.
    #
    #   (1) WIDEN + RE-CENTER the serve to FILL the reachable window so no single pre-position
    #       covers it -> the arm MUST track laterally and re-reach each ball. The reachable intercept
    #       window is hit_target_y_range=(-0.12,0.43) (center 0.155); the old serve centered at 0.05
    #       wasted the +y reach and spilled -y out of reach. Re-center + widen (hard end; easy start
    #       stays gentle for the stage-1 bootstrap):
    #         serve_y_center          0.05 -> 0.15  (match reachable-window center 0.155)
    #         serve_y_wide            0.17 -> 0.25  (fills [-0.10,0.40] ⊂ reachable (-0.12,0.43))
    #         serve_bounce_x_range_hard  (-1.32,-0.88) -> (-1.34,-1.00)  (depth; SHALLOW end raised
    #                                             -0.86 -> -1.00: probe showed the shallowest near-net
    #                                             serves arrive lowest -- deep bounce = higher z at the
    #                                             plane, shallow bounce = lower z -- so trimming the
    #                                             shallowest lifts the low-z tail into the reach window)
    #         serve_bounce_vz_range_hard (1.45, 2.40) -> (1.50, 2.55)    (arc height / speed spread)
    #       Verified in-window by TT_SERVE_PROBE (physical crossing histogram) at c=1: with shallow end
    #       at -0.86 the z p5 sat ~0.70-0.75 (below the 0.92 floor -- vz was NOT the lever, bounce depth
    #       was); raising the shallow end to -1.00 pulls the low-z tail up. y is well-centered
    #       (p50~0.18 on window center 0.155).
    #   (2) reward_approach_velocity (0 -> 3.0): first-contact forward blade speed -> pay for MEETING
    #       the ball with a moving paddle (un-farmable, fires once per rally). Forces swing-through.
    #   (3) STRENGTHEN the no-ball return-to-default shaping (already mask_invalid-gated, and
    #       mask_invalid includes has_touch_paddle -> the post-hit "no ball" window):
    #         reward_arm_ready_idle  2.0 -> 5.0   (strong pull back to the retracted default after a
    #                                             hit -> can't camp at the extended hit pose)
    #         penalty_arm_vel_idle  -0.05 -> -0.15
    #   Guardrails: termination_penalty stays -100 (critic-safe); from scratch (warm-start + this
    #   distribution shift diverges per a1_tt_v12_from_scratch_warmstart_diverges); curriculum
    #   5k-10k-5k (easy 0-5k -> ramp 5k-15k -> consolidate 15k-20k, via watchdog START=5000/
    #   RAMP=10000, TARGET 20000) to avoid the c~=0.3 critic blow-up; monitor value_loss<100 past 15k.
    def __post_init__(self):
        super().__post_init__()  # V12: forward-reach geometry (base -2.0, hit_plane -1.62)
        # (1) widen + re-center serve to fill the reachable window (hard end; easy start unchanged)
        self.ball.serve_y_center = 0.15
        self.ball.serve_y_wide = 0.25
        self.ball.serve_bounce_x_range_hard = (-1.34, -1.00)
        self.ball.serve_bounce_vz_range_hard = (1.50, 2.55)
        # (2) first-contact approach-velocity reward
        self.reward.reward_approach_velocity.weight = 3.0
        # (3) strengthen no-ball return-to-default shaping
        self.reward.reward_arm_ready_idle.weight = 5.0
        self.reward.penalty_arm_vel_idle.weight = -0.15


@configclass
class A1TableTennisV14EnvCfg(A1TableTennisV13EnvCfg):
    # v14 (2026-07-31): fix the two structural gaps seen in v13 sim2sim play (still iter ~4k, but
    #   these are design gaps that MORE training only mitigates, not fixes):
    #   (1) "往身侧打" -- the paddle meets the ball glancingly and sends it sideways. The only
    #       orientation shaping was paddle_face_x (weight 0.5, aligns the normal to the FIXED world
    #       ±x every step, unrelated to the incoming ball and farmable by hovering). v14 adds
    #       reward_hit_direction: FIRST-CONTACT bonus for the blade normal being parallel to the
    #       ball's incoming velocity line (square-on -> returns it back over the net). Un-farmable
    #       (fires once per rally at the hit, like reward_approach_velocity), so a camped/glancing
    #       paddle scores 0. Weight 8.0 (bounded [0,1] per rally; well under contact=150).
    #   (2) "先去位置等着" -- reward_future_dis_ee was dense every step, so the optimum was to run to
    #       the (static) intercept early and camp. v14 scales it by TIME-TO-HIT (ball_future_t):
    #       early = down-weighted to the floor (0.4), near contact = full. Kills the camp-early
    #       optimum while keeping enough early guidance for the torque-limited arm to travel in time.
    #   approach_velocity (v13, forward blade speed at first contact) already forces the hit to happen
    #   WITH motion (a static paddle scores 0 there) -> combined with (2) the blade arrives at the
    #   intercept moving forward exactly as the ball crosses the plane; no explicit timing term needed
    #   (an explicit |t_hit - t_arrival| penalty risks the c~=0.3 critic blow-up, avoided).
    #   Guardrails unchanged from v13: termination_penalty -100, from scratch (warm-start diverges),
    #   5k-10k-5k curriculum, monitor value_loss<100 past 15k.
    def __post_init__(self):
        super().__post_init__()  # V13: widen serve + approach_velocity + strong idle + fwd-reach geom
        # (1) first-contact paddle-normal-vs-ball-velocity alignment (fix "往身侧打")
        self.reward.reward_hit_direction.weight = 8.0
        # (2) time-gate the dense intercept-tracking reward (fix "先去位置等着")
        self.reward.reward_future_dis_ee.params["time_gate_ref"] = 0.6
        self.reward.reward_future_dis_ee.params["time_gate_floor"] = 0.4


@configclass
class A1TableTennisBackhandEnvCfg(A1TableTennisEnvCfg):
    """2026-08-01 backhand scratch task aligned to the current camera/SDK path."""

    reward = A1TableTennisBackhandRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        # Frozen mentor geometry and ready pose.  Use the V1_3 paddle asset, but
        # execute the newly fitted closed-loop q_des->q trajectory through a
        # high-bandwidth tracking articulation (no duplicate MIT dynamics).
        robot_cfg = copy.deepcopy(A1_TT_REAL_FITTED_CFG)
        robot_cfg.spawn.usd_path = A1_USD_PATH_V1_3
        robot_cfg.init_state.pos = (-1.8, 0.0, A1_INIT_Z)
        robot_cfg.init_state.joint_pos.update(
            {joint: q for joint, q in zip(A1_ARM_JOINTS, A1_BACKHAND_READY_Q)}
        )
        self.scene.robot = robot_cfg
        self.robot.home_y = 0.0
        self.robot.paddle_offset = (0.0, 0.0, 0.0)
        self.robot.paddle_y_offset = -0.03
        self.robot.hit_body_height = 0.028
        self.robot.hit_plane_x = -1.243
        self.robot.hit_target_x_range = (-1.243, -1.243)
        # The old mentor box covered only 3/10 of today's measured crossings.
        # This conservative envelope contains the measured p5..p95 tails while
        # retaining the original center as the easy curriculum start.
        self.robot.hit_target_y_range = (-0.06, 0.20)
        self.robot.hit_target_z_range = (0.84, 1.14)

        self.robot.action_target_rate_limit_enable = False
        self.robot.action_target_max_delta_per_tick = ()
        self.robot.action_target_lowpass_enable = True
        self.robot.action_target_lowpass_tau_s = A1_REAL_DEPLOY_LOWPASS_TAU_S
        self.robot.action_target_lowpass_vel_limit = A1_REAL_DEPLOY_LOWPASS_VEL_LIMIT
        self.robot.action_target_lowpass_tau_range_s = A1_BACKHAND_LOWPASS_TAU_RANGE_S
        self.robot.action_target_lowpass_vel_limit_scale_range = (0.85, 1.15)

        self.robot.action_response_model_enable = True
        self.robot.action_response_u_mean = A1_BACKHAND_READY_Q
        self.robot.action_response_fn_hz = A1_BACKHAND_RESPONSE_FN_HZ
        self.robot.action_response_zeta = A1_BACKHAND_RESPONSE_ZETA
        self.robot.action_response_delay_s = A1_BACKHAND_RESPONSE_DELAY_S
        self.robot.action_response_gain = A1_BACKHAND_RESPONSE_GAIN
        self.robot.action_response_bias_rad = A1_BACKHAND_RESPONSE_BIAS_RAD
        self.robot.action_response_fn_scale_range = (0.85, 1.15)
        self.robot.action_response_zeta_scale_range = (0.80, 1.25)
        self.robot.action_response_gain_scale_range = (0.98, 1.02)
        self.robot.action_response_delay_jitter_s = A1_BACKHAND_RESPONSE_DELAY_JITTER_S
        self.robot.action_response_bias_jitter_rad = (0.006, 0.004, 0.004, 0.008, 0.006, 0.006, 0.004)

        # One 50 Hz action tick covers scheduler/ROS phase uncertainty beyond
        # the per-joint delay already present in the fitted motor response.
        self.domain_rand.action_delay.enable = True
        self.domain_rand.action_delay.params = {"min_delay": 0, "max_delay": 1}

        # Timestamped 60 Hz camera path measured today.  Latency is sampled as
        # a persistent per-rally mode; the model then applies the same two-frame
        # acquisition, alpha-beta tracking, one-bounce extrapolation and 50 Hz
        # sample consumption as deployment.
        self.domain_rand.perception_delay.enable = False
        camera = self.domain_rand.camera_observation
        camera.enable = True
        camera.fps = 60.0
        camera.acquire_frames = 2
        camera.reset_gap_s = 0.25
        camera.coast_max_s = 0.12
        camera.dropout_prob = 0.01
        # 2026-08-04 real traces after host clocks settled put essentially every
        # rally in the 45--80 ms source-to-receive band (median 56--60 ms). Make
        # that the normal mode while retaining fast-path and rare-tail coverage.
        camera.latency_mode_weights = (0.15, 0.75, 0.10)
        # Fresh-lock ZED exposure age remains 23.5--27.4 ms; TensorRT, publish,
        # DDS transport and policy sampling account for the remaining age.
        camera.latency_ranges_s = ((0.010, 0.045), (0.045, 0.080), (0.080, 0.120))
        camera.position_noise_std = (0.004, 0.004, 0.006)
        camera.filter_alpha = 0.65
        camera.filter_beta = 0.10
        camera.max_extrapolation_s = 0.16
        camera.x_range = (-1.50, 1.20)
        camera.y_range = (-0.30, 0.30)
        camera.z_range = (0.76, 1.70)
        camera.extrapolate_to_now = True
        camera.gravity_mps2 = -9.81
        camera.table_bounce_enable = True
        camera.table_ball_center_z = 0.78
        camera.table_restitution = 0.95
        # Camera noise is owned by the model above; do not add the legacy
        # elementwise perception noise a second time.
        self.noise.noise_scales.perception = 0.0

        self.ball.ball_max_eposide_length = 1.8
        self.ball.ball_reset_repeat = 1
        self.ball.max_serve_per_episode = 5
        self.ball.serve_bounce_enable = True
        # Start exactly on the mentor distribution, then expand toward today's
        # hand-fed y/z/vx envelope instead of clamping real balls to a tiny box.
        self.ball.serve_bounce_x_range = (-1.053, -0.773)
        self.ball.serve_bounce_vz_range = (0.0, 0.45)
        self.ball.serve_y_center = 0.041
        # y is sampled at the nominal bounce point, then keeps spreading before
        # x=-1.243. Back-project today's hit-plane envelope instead of applying
        # its full half-width directly at the bounce (which produced
        # y_hit p5/p95 ~= -0.082/0.263 and wasted ~31% of hard-stage serves).
        self.ball.serve_y_center_hard = 0.055
        self.ball.serve_y_start = 0.020
        self.ball.serve_bounce_x_range_hard = (-1.10, -0.35)
        self.ball.serve_bounce_vz_range_hard = (0.50, 2.00)
        self.ball.serve_y_wide = 0.105
        self.ball.serve_curriculum_perf_gated = False
        # 30k v1: 0--10k mentor/easy, 10k--20k linear widening, 20k--30k
        # full-range consolidation. Curriculum clock is raw physics substeps.
        self.ball.serve_curriculum_phase_start = 10000 * A1_TT_RAW_STEPS_PER_ITER
        self.ball.serve_curriculum_steps = 10000 * A1_TT_RAW_STEPS_PER_ITER

        self.ball.require_active_contact = True
        self.ball.active_contact_min_paddle_speed = 0.18
        self.ball.active_contact_min_forward_speed = 0.32
        self.ball.active_contact_require_own_bounce = False
        self.ball.active_contact_hit_plane_margin = 0.11
        self.ball.sweet_contact_radius = 0.08
        self.ball.sweet_contact_core_radius = 0.03
        self.ball.sweet_contact_gate_outcomes = True
        self.ball.sweet_contact_outcome_floor = 0.5
        self.ball.hit_plane_contact_radius = 0.11
        self.ball.hit_plane_contact_core_radius = 0.04
        self.ball.hit_plane_contact_gate_outcomes = True
        self.ball.hit_plane_contact_outcome_floor = 0.5
        self.ball.no_ball_period_s = 0.0
        self.ball.ball_active_s = 0.0
        self.ball.no_ball_curriculum_steps = 0
        self.ball.idle_reward_ramp_steps = 0
        self.ball.curriculum_phase1_steps = 0


@configclass
class A1TableTennisBackhandV2EnvCfg(A1TableTennisBackhandEnvCfg):
    """Backhand v2: real r1 height plus net-clearing high/slow serves."""

    def __post_init__(self):
        super().__post_init__()

        # The current V1_3 USD bakes the measured r1 centerline at 1.15 m.  Do
        # not compensate the world-space hit target downward: the fresh policy
        # must learn the corrected kinematics.
        self.robot.hit_target_y_range = backhand_v2.HIT_TARGET_Y_RANGE
        self.robot.hit_target_z_range = backhand_v2.HIT_TARGET_Z_RANGE

        # Stage 1 is a moderate, physically net-clearing bootstrap rather than
        # v1's nominal mentor box (which produced many real-physics net faults).
        # Stage 2 widens depth/vz to include correlated high and slow arrivals;
        # reset-time rejection enforces the final physical contract below.
        self.ball.serve_bounce_x_range = backhand_v2.EASY_BOUNCE_X_RANGE
        self.ball.serve_bounce_vz_range = backhand_v2.EASY_BOUNCE_VZ_RANGE
        self.ball.serve_y_center = backhand_v2.EASY_Y_CENTER
        self.ball.serve_y_start = backhand_v2.EASY_Y_HALF
        self.ball.serve_bounce_x_range_hard = backhand_v2.HARD_BOUNCE_X_RANGE
        self.ball.serve_bounce_vz_range_hard = backhand_v2.HARD_BOUNCE_VZ_RANGE
        self.ball.serve_y_center_hard = backhand_v2.HARD_Y_CENTER
        self.ball.serve_y_wide = backhand_v2.HARD_Y_HALF

        self.ball.serve_flight_rejection_enable = True
        self.ball.serve_net_x = 0.0
        self.ball.serve_net_center_z_min = backhand_v2.NET_CENTER_Z_MIN
        self.ball.serve_net_prediction_margin = backhand_v2.NET_PREDICTION_MARGIN
        self.ball.serve_physical_bounce_x_range = backhand_v2.PHYSICAL_BOUNCE_X_RANGE
        self.ball.serve_arrival_y_range = backhand_v2.HIT_TARGET_Y_RANGE
        self.ball.serve_arrival_z_range = backhand_v2.PREFLIGHT_HIT_Z_RANGE
        self.ball.serve_arrival_abs_vx_range = backhand_v2.HIT_ARRIVAL_ABS_VX_RANGE
        self.ball.serve_drag_accel_k = backhand_v2.DRAG_ACCEL_K
        self.ball.serve_table_restitution = backhand_v2.TABLE_RESTITUTION
        self.ball.serve_table_dynamic_friction = backhand_v2.TABLE_DYNAMIC_FRICTION
        self.ball.serve_rejection_max_attempts = 16
        # The actor/critic/reward target is the fixed x=-1.243 intersection,
        # not the historical post-bounce apex projected onto that plane.
        self.ball.hit_plane_target_from_serve_probe = True

        # 20k deployment schedule: 5k fixed easy + 10k linear expansion + 5k hold.
        self.ball.serve_curriculum_phase_start = (
            backhand_v2.CURRICULUM_EASY_ITERS * A1_TT_RAW_STEPS_PER_ITER
        )
        self.ball.serve_curriculum_steps = (
            backhand_v2.CURRICULUM_RAMP_ITERS * A1_TT_RAW_STEPS_PER_ITER
        )
        # Slow accepted balls need enough time to complete their returned arc;
        # v1's 1.8 s timeout truncated this outcome signal.
        self.ball.ball_max_eposide_length = 2.4


@configclass
class A1TableTennisBackhandV3EnvCfg(A1TableTennisBackhandV2EnvCfg):
    """Backhand v3: real-serve envelope plus independent actuator DR."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the calibrated base/ready/hit plane and contact material from
        # v2, but move the actual arrival contract onto the 2026-08-04 hand-fed
        # ball distribution with modest measured-tail margins.
        self.robot.hit_target_y_range = backhand_v3.HIT_TARGET_Y_RANGE
        self.robot.hit_target_z_range = backhand_v3.HIT_TARGET_Z_RANGE
        self.ball.serve_bounce_x_range = backhand_v3.EASY_BOUNCE_X_RANGE
        self.ball.serve_bounce_vz_range = backhand_v3.EASY_BOUNCE_VZ_RANGE
        self.ball.serve_y_center = backhand_v3.EASY_Y_CENTER
        self.ball.serve_y_start = backhand_v3.EASY_Y_HALF
        self.ball.serve_bounce_x_range_hard = backhand_v3.HARD_BOUNCE_X_RANGE
        self.ball.serve_bounce_vz_range_hard = backhand_v3.HARD_BOUNCE_VZ_RANGE
        self.ball.serve_y_center_hard = backhand_v3.HARD_Y_CENTER
        self.ball.serve_y_wide = backhand_v3.HARD_Y_HALF
        self.ball.serve_arrival_y_range = backhand_v3.HIT_TARGET_Y_RANGE
        self.ball.serve_arrival_z_range = backhand_v3.PREFLIGHT_HIT_Z_RANGE
        self.ball.serve_arrival_abs_vx_range = backhand_v3.HIT_ARRIVAL_ABS_VX_RANGE
        self.ball.serve_curriculum_phase_start = (
            backhand_v3.CURRICULUM_EASY_ITERS * A1_TT_RAW_STEPS_PER_ITER
        )
        self.ball.serve_curriculum_steps = (
            backhand_v3.CURRICULUM_RAMP_ITERS * A1_TT_RAW_STEPS_PER_ITER
        )

        # v2 sampled one fn/zeta/gain scale for the whole arm.  The real
        # traces require widening r4 (and r1 high-frequency gain) without
        # degrading the already well-aligned wrist joints.
        self.robot.action_response_fn_scale_range = backhand_v3.RESPONSE_FN_SCALE_RANGES
        self.robot.action_response_zeta_scale_range = backhand_v3.RESPONSE_ZETA_SCALE_RANGES
        self.robot.action_response_gain_scale_range = backhand_v3.RESPONSE_GAIN_SCALE_RANGES
        self.robot.action_response_delay_s = backhand_v3.RESPONSE_DELAY_S
        self.robot.action_response_delay_jitter_s = backhand_v3.RESPONSE_DELAY_JITTER_S
        self.robot.action_response_accel_limit_rad_s2 = (
            backhand_v3.RESPONSE_ACCEL_LIMIT_RAD_S2
        )
        self.robot.action_response_accel_limit_scale_range = (
            backhand_v3.RESPONSE_ACCEL_LIMIT_SCALE_RANGES
        )


@configclass
class A1TableTennisBackhandV3EvalEnvCfg(A1TableTennisBackhandV3EnvCfg):
    """Backhand-v3 final-range eval with the same physical rejection."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.max_episode_length_s = 99999999999
        self.ball.serve_bounce_x_range = self.ball.serve_bounce_x_range_hard
        self.ball.serve_bounce_vz_range = self.ball.serve_bounce_vz_range_hard
        self.ball.serve_y_center = self.ball.serve_y_center_hard
        self.ball.serve_y_start = self.ball.serve_y_wide
        self.ball.serve_curriculum_steps = 0
        self.ball.serve_curriculum_phase_start = 0


@configclass
class A1TableTennisBackhandV4EnvCfg(A1TableTennisBackhandV3EnvCfg):
    """Backhand v4: preserve v3 physics and enforce a one-way timed swing."""

    reward = A1TableTennisBackhandV4RewardCfg()

    def __post_init__(self):
        super().__post_init__()

        # Empirical real-serve mixture.  The easy phase sits near the measured
        # median; the hard proposal retains a small slow tail without making
        # high/slow balls the dominant visual mode.
        self.ball.serve_bounce_x_range = backhand_v4.EASY_BOUNCE_X_RANGE
        self.ball.serve_bounce_vz_range = backhand_v4.EASY_BOUNCE_VZ_RANGE
        self.ball.serve_y_center = backhand_v4.EASY_Y_CENTER
        self.ball.serve_y_start = backhand_v4.EASY_Y_HALF
        self.ball.serve_bounce_x_range_hard = backhand_v4.HARD_BOUNCE_X_RANGE
        self.ball.serve_bounce_vz_range_hard = backhand_v4.HARD_BOUNCE_VZ_RANGE
        self.ball.serve_y_center_hard = backhand_v4.HARD_Y_CENTER
        self.ball.serve_y_wide = backhand_v4.HARD_Y_HALF
        self.ball.serve_tail_candidate_weight = 0.0
        self.ball.serve_tail_candidate_weight_hard = (
            backhand_v4.TAIL_CANDIDATE_WEIGHT_HARD
        )
        self.ball.serve_tail_bounce_x_range = backhand_v4.EASY_BOUNCE_X_RANGE
        self.ball.serve_tail_bounce_vz_range = backhand_v4.EASY_BOUNCE_VZ_RANGE
        self.ball.serve_tail_bounce_x_range_hard = (
            backhand_v4.TAIL_BOUNCE_X_RANGE_HARD
        )
        self.ball.serve_tail_bounce_vz_range_hard = (
            backhand_v4.TAIL_BOUNCE_VZ_RANGE_HARD
        )

        # v2/v3 applied the complete 3-D intercept-distance reward from the
        # first valid camera observation.  With the stable true-plane target,
        # v3 learned to reach forward immediately, backtrack, then swing again.
        # Keep early y/z tracking, but open x tracking only in the final window.
        self.reward.reward_future_dis_ee.func = mdp.reward_a1_timed_future_ee_target
        self.reward.reward_future_dis_ee.params.update(
            {
                "x_time_gate_ref": backhand_v4.X_TRACKING_WINDOW_S,
                "x_time_gate_floor": backhand_v4.X_TRACKING_FLOOR,
            }
        )
        self.reward.penalty_early_paddle_forward.weight = (
            backhand_v4.EARLY_FORWARD_PENALTY_WEIGHT
        )
        self.reward.penalty_late_paddle_backtrack.weight = (
            backhand_v4.LATE_BACKTRACK_PENALTY_WEIGHT
        )
        self.reward.penalty_contact_lateral_paddle_speed.weight = (
            backhand_v4.CONTACT_LATERAL_SPEED_PENALTY_WEIGHT
        )
        self.reward.penalty_contact_lateral_paddle_speed.func = (
            mdp.penalty_a1_latched_contact_lateral_paddle_speed
        )
        self.reward.reward_approach_velocity.func = mdp.reward_a1_latched_approach_velocity
        self.reward.reward_hit_direction.func = (
            mdp.reward_a1_latched_horizontal_hit_direction
        )
        self.reward.reward_hit_direction.weight = backhand_v4.HIT_DIRECTION_REWARD_WEIGHT

        # First contact rewards use latched pre-impact paddle/ball state;
        # landing/pass-net wait for the physical outgoing velocity instead of
        # consuming the geometric 7 cm proximity event.
        self.ball.post_impact_outcome_enable = True
        self.ball.post_impact_min_outgoing_vx_mps = (
            backhand_v4.POST_IMPACT_MIN_OUTGOING_VX_MPS
        )
        self.ball.post_impact_timeout_s = backhand_v4.POST_IMPACT_TIMEOUT_S
        self.ball.landing_drag_accel_k = backhand_v4.DRAG_ACCEL_K
        self.ball.table_success_bounce_event_enable = True

        # Spin is sampled from a compact correlated prior fitted to filtered
        # real serves, then injected only after the own-table bounce. Magnus
        # remains zero until serve preflight and target propagation are made
        # spin-aware in a separately validated version.
        self.ball.post_bounce_spin_enable = True
        self.ball.post_bounce_spin_easy_scale = backhand_v4.POST_BOUNCE_SPIN_EASY_SCALE
        self.ball.post_bounce_spin_hard_scale = backhand_v4.POST_BOUNCE_SPIN_HARD_SCALE
        self.ball.post_bounce_spin_magnitude_jitter = (
            backhand_v4.POST_BOUNCE_SPIN_MAGNITUDE_JITTER
        )

        # v2 real returns had median outgoing vy=-1.96 m/s and were mostly
        # sideways/long, yet the old 3 m landing threshold still paid strongly.
        self.reward.reward_future_landing_dis.func = mdp.reward_a1_landing_target_quality
        self.reward.reward_future_landing_dis.params = {
            "target_x": backhand_v4.LANDING_TARGET_X,
            "target_y": backhand_v4.LANDING_TARGET_Y,
            "half_reward_radius_m": (
                backhand_v4.LANDING_TARGET_HALF_REWARD_RADIUS_M
            ),
        }
        self.reward.reward_future_landing_dis.weight = backhand_v4.LANDING_REWARD_WEIGHT
        self.reward.penalty_predicted_landing_outside_table.weight = (
            backhand_v4.LANDING_OUTSIDE_PENALTY_WEIGHT
        )
        self.reward.reward_future_pass_net.func = mdp.reward_a1_future_pass_net
        self.reward.reward_future_pass_net.params = {
            "std_h": backhand_v4.PASS_NET_HEIGHT_STD_M,
            "z_target": 0.76 + 0.35,
            "net_x": 0.0,
            "min_center_z": backhand_v4.PASS_NET_MIN_CENTER_Z,
            "clearance_ramp_m": backhand_v4.PASS_NET_CLEARANCE_RAMP_M,
            "horizontal_drag_accel_k": backhand_v4.DRAG_ACCEL_K,
        }
        self.reward.reward_table_success.func = mdp.reward_a1_table_success_event
        self.reward.reward_table_success.weight = (
            backhand_v4.TABLE_SUCCESS_REWARD_WEIGHT
        )


@configclass
class A1TableTennisBackhandV4EvalEnvCfg(A1TableTennisBackhandV4EnvCfg):
    """Backhand-v4 final-range eval with the training swing timing."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.max_episode_length_s = 99999999999
        self.ball.serve_bounce_x_range = self.ball.serve_bounce_x_range_hard
        self.ball.serve_bounce_vz_range = self.ball.serve_bounce_vz_range_hard
        self.ball.serve_y_center = self.ball.serve_y_center_hard
        self.ball.serve_y_start = self.ball.serve_y_wide
        self.ball.serve_tail_candidate_weight = self.ball.serve_tail_candidate_weight_hard
        self.ball.serve_tail_bounce_x_range = self.ball.serve_tail_bounce_x_range_hard
        self.ball.serve_tail_bounce_vz_range = self.ball.serve_tail_bounce_vz_range_hard
        self.ball.post_bounce_spin_easy_scale = self.ball.post_bounce_spin_hard_scale
        self.ball.serve_curriculum_steps = 0
        self.ball.serve_curriculum_phase_start = 0


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
class A1TableTennisBackhandEvalEnvCfg(A1TableTennisBackhandEnvCfg):
    """Default A1 eval/play task: final-range backhand-v1, never legacy forehand."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.max_episode_length_s = 99999999999
        # TTEnv treats a zero-length curriculum as c=0, so explicitly promote
        # the hard endpoint to the base sampler for deterministic final-range eval.
        self.ball.serve_bounce_x_range = self.ball.serve_bounce_x_range_hard
        self.ball.serve_bounce_vz_range = self.ball.serve_bounce_vz_range_hard
        self.ball.serve_y_center = self.ball.serve_y_center_hard
        self.ball.serve_y_start = self.ball.serve_y_wide
        self.ball.serve_curriculum_steps = 0
        self.ball.serve_curriculum_phase_start = 0


@configclass
class A1TableTennisBackhandV2EvalEnvCfg(A1TableTennisBackhandV2EnvCfg):
    """Backhand-v2 hard-range eval with the same physical serve rejection."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.max_episode_length_s = 99999999999
        self.ball.serve_bounce_x_range = self.ball.serve_bounce_x_range_hard
        self.ball.serve_bounce_vz_range = self.ball.serve_bounce_vz_range_hard
        self.ball.serve_y_center = self.ball.serve_y_center_hard
        self.ball.serve_y_start = self.ball.serve_y_wide
        self.ball.serve_curriculum_steps = 0
        self.ball.serve_curriculum_phase_start = 0


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
class A1TableTennisV13AgentCfg(A1TableTennisTorqueLowpassAgentCfg):
    experiment_name: str = "a1_tt_real_v13"
    run_name = "scratch_widenserve_approachvel_strongidle_5k10k5k"
    max_iterations = 20000


@configclass
class A1TableTennisV14AgentCfg(A1TableTennisTorqueLowpassAgentCfg):
    experiment_name: str = "a1_tt_real_v14"
    run_name = "scratch_hitdir_timegate_dis_ee_5k10k5k"
    max_iterations = 20000


@configclass
class A1TableTennisBackhandAgentCfg(A1TableTennisDeployAgentCfg):
    experiment_name: str = "a1_tt_backhand_real_v1_y055h105"
    run_name = "scratch_backhand_camera_age35_tau_delay_dr_servey055h105_10k10k10k"
    resume = False
    max_iterations = 30000
    predictor = {
        "history_len": 5,
        "traj_max_len": 128,
        "hidden_sizes": [64, 64],
        "lr": 0.5e-3,
        "epochs_per_update": 1,
        "batch_size": 1024,
        "train_until_iters": 1000,
    }


@configclass
class A1TableTennisBackhandV2AgentCfg(A1TableTennisBackhandAgentCfg):
    experiment_name: str = "a1_tt_backhand_real_v2_r115_netclear_highslow_paddle075"
    run_name = "scratch_r115_netclear_highslow_paddle075_camera_tau_delay_5k10k5k"
    resume = False
    max_iterations = backhand_v2.MAX_ITERATIONS
    # The serve distribution reaches its high/slow endpoint at iter 15k.
    # Keep fitting through the ramp and early hold so the learned marker
    # does not saturate near the old v1 z ceiling on high balls.
    predictor = {
        "history_len": 5,
        "traj_max_len": 128,
        "hidden_sizes": [64, 64],
        "lr": 0.2e-3,
        "epochs_per_update": 1,
        "batch_size": 1024,
        "train_until_iters": 17000,
        # Backfill each serve's histories only after Isaac's physical ball
        # actually crosses the fixed hit plane.  This makes the supervised y/z
        # label a measured intersection rather than an analytic apex proxy.
        "target_mode": "actual_hit_plane",
        # Causal validation uses the previous-tick prediction and the physical
        # crossing revealed on the current tick.  Report every 50 iterations;
        # warn if y/z or high/slow-ball errors stay above the acceptance bands.
        "validation_window_samples": 4096,
        "validation_interval_iters": 50,
        "validation_min_samples": 256,
        "validation_warn_start_iter": 100,
        "validation_warn_patience": 3,
        "validation_warn_min_rel_improvement": 0.03,
        "validation_warn_y_mae_m": 0.08,
        "validation_warn_z_mae_m": 0.08,
        "validation_warn_hard_z_mae_m": 0.10,
        "validation_slow_abs_vx_mps": 2.2,
        "validation_high_z_m": 1.30,
    }


@configclass
class A1TableTennisBackhandV3AgentCfg(A1TableTennisBackhandV2AgentCfg):
    experiment_name: str = "a1_tt_backhand_real_v3_realserve_predictor_r4dr"
    run_name = "scratch_realserve142_truehit_camera157510_perjointdr_r4sat_5k10k5k"
    resume = False
    max_iterations = backhand_v3.MAX_ITERATIONS


@configclass
class A1TableTennisBackhandV4AgentCfg(A1TableTennisBackhandV3AgentCfg):
    experiment_name: str = "a1_tt_backhand_real_v4_timing_return"
    run_name = "scratch_realserve_truehit_camera157510_oneway_return_5k10k5k"
    resume = False
    max_iterations = backhand_v4.MAX_ITERATIONS


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

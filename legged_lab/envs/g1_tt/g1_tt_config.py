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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import  RslRlPpoAlgorithmCfg
import os
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
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)   # v13: reverted to v7's original value (the v8 3x -3.75e-7 over-stiffened the policy -> near-frozen body in sim2sim; v7 could play at -1.25e-7)
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

    # --- ankle-overrun fix (2026-06-18): real robot pins L/R ankle_roll at the ±0.262 clip
    # limit ~100% of the time (no-ball AND during hits) because training NEVER penalized the
    # COMMANDED target exceeding limits (dof_pos_limits罚 actual angle, physics-clamped -> never
    # fires; energy_ankle ~0 when vel~0). These three close that loop. Weights are smoke/early-
    # iter tunable (too strong -> suppresses legit hitting swing; too weak -> ankle still saturates).
    joint_pos_target_limits = RewTerm(func=mdp.joint_pos_target_limits, weight=-1.0)  # penalize越限 INTENT (key)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)                            # weak raw-action magnitude reg
    joint_deviation_ankle = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])})

    reward_contact = RewTerm(
        func=mdp.reward_contact,
        weight=150.0,
    )

    # Dense positive bonus for actively standing when there is NO playable ball
    # (mask_invalid). Counters the freeze-collapse (action_rate->0 -> fall -> ep_len
    # 297<->5 limit cycle) that both idle3 (fast ramp) and idle4 (slow ramp) hit: the
    # sparse -1000 termination penalty did not pull the policy out of the freeze basin, so
    # pay a dense upright+calm bonus while idle. Zero when a ball is present -> never
    # competes with hitting.
    reward_idle_stand = RewTerm(
        func=mdp.reward_idle_stand,
        weight=0.5,
    )

    # HITTER-style reference-stand-pose tracking when no playable ball: reward joints
    # matching the default stand pose (an explicit STABLE configuration to hold). This is
    # the real freeze fix the earlier idle rewards lacked (they never said WHICH pose).
    reward_idle_pose = RewTerm(
        func=mdp.reward_idle_pose,
        weight=1.0,
        params={"k": 1.0},
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
        # paddle / hitting geometry (real-mesh PACE adapter: contact/blade center at wrist +X 0.30m;
        # extension shortened 0.16->0.123 in build_tt_urdf.py to match the REAL 30cm wrist->paddle-center)
        self.robot.paddle_body_name = "right_wrist_roll_rubber_hand"
        self.robot.paddle_offset = (0.30, 0.0, 0.0)
        self.robot.hit_body_height = 0.685   # FK: steady pelvis height in ready stance
        self.robot.paddle_y_offset = -0.55   # hitting-extension lateral offset (was -0.227 ready-stance; caused ~0.37m paddle-ball gap -> hit~0). ~T1's -0.60.
        # Robot >=40cm from table. Table own edge x=-1.37; stance/hit-plane at -1.8 (~43cm; v11
        # moved -2.0 -> -1.8 because -2.0 felt too far from the table). HOME/intercept-clamp/
        # give-up/terminal/idle anchors all derive from hit_plane_x in tt_env.py. Serve re-tuned
        # (vz dropped ~0.2 vs the -2.0 serve) so the ball reaches -1.8 at ready z~1.0 — see below.
        self.robot.hit_plane_x = -1.8
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
        # ===== v14 SPEED + HEIGHT curriculum (2026-06-29): warm-start from v13 model_14900 =====
        # v13 ramped height up to z~1.17 (deep bounce -1.10) -> sim2sim showed 1.17 too HIGH (extreme
        # arm-raise) + chronic action_rate spikes -20..-42 during the ramp. Fix: LOWER + CAP the height
        # band to z~0.92-1.10 (cap the ceiling, drop the floor toward the ready paddle height 0.885 so
        # the robot reaches up LESS, and adds some lower balls). bounce_x=HEIGHT knob, vz=SPEED knob.
        # Verified all-hittable (reach -1.8, bounce in court, clear net):
        #   easy c=0: xb(-0.90,-0.76) vz(1.2,1.6) -> z 1.00-1.05, spd 3.6-4.3 (fast, = what model_14900 knows)
        #   hard c=1: xb(-0.85,-0.72) vz(1.5,3.0) -> z 0.92-1.10 (HEIGHT variety, capped) , spd 2.9-3.8, react 0.7-1.3s
        self.ball.serve_bounce_x_range = (-0.90, -0.76)        # easy: shallow bounce -> z~1.0-1.05 (matches 14900)
        self.ball.serve_bounce_vz_range = (1.2, 1.6)           # easy c=0: FAST (~4.0 m/s) = what model_14900 knows
        self.ball.serve_y_start = 0.5                          # lateral range (FIXED; NOT widened — that collapsed v12)
        self.ball.serve_bounce_x_range_hard = (-0.85, -0.72)   # hard: height band CAPPED (deep end -0.85 -> z<=1.10, no extreme raise)
        self.ball.serve_bounce_vz_range_hard = (1.5, 3.0)      # hard c=1: z 0.92-1.10 height variety (low vz->higher/faster, high vz->lower/slower)
        self.ball.serve_y_wide = 0.5                           # SAME as easy: NO lateral widening
        # ===== v13 SPEED curriculum schedule (sim_step-keyed, 240 raw steps/iter). WARM-START =====
        # from v12 model_10000 (TT_SIM_STEP_OFFSET=10000*240 continues the clock). model_10000 already
        # plays FAST center balls, so:
        #   c=0  (iter 10000 -> 15000): FAST-only serve (vz 1.2-1.6). 5k iters to settle on the new
        #                               shallow-bounce serve (mild OOD vs 10000's deep bounce) before
        #                               anything new is added.
        #   ramp (iter 15000 -> 25000): vz upper end grows 1.6 -> 2.7, ADDING progressively SLOWER
        #                               balls (down to ~3.0 m/s, 1.1 s reaction). Lateral range FIXED.
        #   c=1  (iter 25000 -> 30000): consolidate on the full fast+slow speed mix. TARGET=30000.
        # NO lateral widening (v12 collapse cause), NO idle / no-ball (no_ball_period_s=0).
        self.ball.serve_curriculum_perf_gated = False          # FIXED-iter schedule (not success-gated)
        self.ball.serve_curriculum_phase_start = 3600000       # iter 15000: fast-only until here (15000*240)
        self.ball.serve_curriculum_steps = 3600000             # v14b GENTLER ramp: add difficulty over 15000 iter (15000->30000) so the critic keeps up (#3; v14 critic diverged at c~0.32 / iter~18220 under the old 10000-iter ramp)
        self.ball.no_ball_period_s = 0.0       # v7: NO no-ball injection (v6 diverged ~iter39k when
        self.ball.ball_active_s = 3.0          #   no-ball ramped full; idle-region instability. Deploy clip handles no-ball.
        # idle/no-ball params below are INERT (no_ball_period_s=0) — kept for reference only.
        self.ball.curriculum_phase1_steps = 720000
        self.ball.idle_reward_ramp_steps = 96000
        self.ball.no_ball_curriculum_steps = 192000
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
        # diagnostic: override the per-ball flight timeout via env (default keeps training value).
        # Lets us test whether table_success is compressed by the 1.8s ball timeout cutting returns
        # off mid-flight (TT_BALL_MAX=5 vs 1.8 on the same ckpt isolates timeout-truncation).
        self.ball.ball_max_eposide_length = float(os.environ.get("TT_BALL_MAX", str(self.ball.ball_max_eposide_length)))
        # diagnostic: match eval paddle restitution to the trained physics (e.g. v10's 0.75),
        # else eval at the base ~0.005 underrates a policy trained on a bouncier paddle.
        if os.environ.get("TT_EVAL_PADDLE_REST"):
            _rest = float(os.environ["TT_EVAL_PADDLE_REST"])
            self.domain_rand.events.paddle_restitution = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_roll_rubber_hand"]),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (_rest, _rest),
                    "num_buckets": 64,
                },
            )
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
        # serving range — eval uses a FIXED bounce distribution (bounce path is
        # active via inherited serve_bounce_enable; the ball_speed_* below are dead code).
        self.ball.serve_bounce_x_range = (-0.80, -0.76)   # v13 eval easy: FAST-only serve (matches training c=0)
        self.ball.serve_bounce_vz_range = (1.2, 1.6)
        self.ball.serve_y_start = 0.5                     # eval: fixed lateral (NOT widened)
        self.ball.serve_y_wide = 0.5
        self.ball.serve_curriculum_steps = 0   # eval: fixed serve distribution (no curriculum widening)
        self.ball.no_ball_period_s = 0.0       # eval: NO no-ball injection -> clean hitting success rate
        #   (test no-ball idle separately via the TT_SERVE_PERIOD / TT_NO_SERVE env hooks)


@configclass
class G1TT_EvalHardEnvCfg(G1TT_EvalEnvCfg):
    """HARD eval (v13): the training c=1.0 SPEED distribution = fast+slow mix at the shallow bounce.
    g1_tt_eval is FAST-only and saturates high; this adds the SLOW balls to discriminate slow-ball
    robustness across ckpts. Fixed distribution (serve_curriculum_steps=0)."""
    def __post_init__(self):
        super().__post_init__()
        self.ball.serve_bounce_x_range = (-0.85, -0.72)   # v14 hard: capped+lowered HEIGHT band (z 0.92-1.10)
        self.ball.serve_bounce_vz_range = (1.5, 3.0)      # v14 hard: height variety, spd 2.9-3.8
        self.ball.serve_y_start = 0.5                     # NO lateral widening
        self.ball.serve_y_wide = 0.5
        self.ball.serve_curriculum_steps = 0


@configclass
class G1TableTennisAgentCfg(TTAgentCfg):
    experiment_name: str = "g1_tt_v8"
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


@configclass
class G1TableTennisDREnvCfg(G1TableTennisEnvCfg):
    """v9: domain-randomize the PADDLE (right_tt_paddle_link) contact restitution per-env so the
    learned swing lands the ball on the table across a RANGE of paddle bounciness — the sim2real
    fix for the 'kill' overshoots ("杀球出界"): the real rubber is bouncier than the sim's fixed
    paddle restitution (~0.005 from the inherited .* material event), so v7/v8 returns fly long on
    the real robot.

    Only the ball-PADDLE contact restitution varies. The ball-TABLE bounce (the ~0.8 = global
    default 0.8 combined w/ ball 0.9) is left untouched, so table physics stay realistic. The
    ball returns via the paddle SWING velocity (as in real TT), with restitution adding the
    elastic component — randomizing it teaches the policy to modulate the swing so the ball
    lands regardless of how bouncy the contact is.

    Mechanism: a startup event overriding ONLY the paddle blade body's per-env material. It is
    added AFTER the inherited physics_material event (.* @ restitution 0.0-0.005) so it WINS on
    the paddle shapes. VERIFY with scripts/verify_paddle_dr.py before training — the readback
    paddle restitution must vary per-env (guards against the no-op trap)."""

    def __post_init__(self):
        super().__post_init__()
        # NOTE: restitution_range is the key tunable. Span from near the current sim value up to
        # clearly bouncier-than-real so the policy is robust to the real rubber. Friction kept at
        # the inherited .* ranges (paddle friction is not the sim2real issue here).
        # The paddle's collision is fixed-joint-MERGED into the wrist body by Isaac's URDF importer,
        # so we randomize the wrist body (= paddle_body_name "right_wrist_roll_rubber_hand"), whose
        # shapes carry the paddle blade — there is no standalone right_tt_paddle_link body.
        self.domain_rand.events.paddle_restitution = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_roll_rubber_hand"]),
                "static_friction_range": (0.6, 1.0),
                "dynamic_friction_range": (0.4, 0.8),
                "restitution_range": (0.75, 0.75),
                "num_buckets": 64,
            },
        )


@configclass
class G1TableTennisDRAgentCfg(G1TableTennisAgentCfg):
    experiment_name: str = "g1_tt_v14"

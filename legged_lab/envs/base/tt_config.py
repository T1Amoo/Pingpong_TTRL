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

import math
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from legged_lab.utils.env_utils.scene import SceneCfg

import legged_lab.mdp as mdp


@configclass
class RewardCfg:
    pass

@configclass
class CurriculumCfg:
    pass

@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False
    prim_body_name: str = MISSING
    resolution: float = 0.1
    size: tuple = (1.6, 1.0)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)


@configclass
class BaseSceneCfg:
    max_episode_length_s: float = 10.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    seed: int = 42
    robot: ArticulationCfg = MISSING
    table: RigidObjectCfg = MISSING
    ball: RigidObjectCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    terrain_static_friction: float = 1.0
    terrain_dynamic_friction: float = 1.0
    terrain_friction_combine_mode: str = "multiply"
    height_scanner: HeightScannerCfg = HeightScannerCfg()


@configclass
class RobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []
    feet_body_names: list = []
    num_actions: int = 21
    num_joints: int = 21
    effort_limit_scale: float = 1.0
    # --- proximal effort curriculum (A1: high torque early to escape the exploration trap where
    # slow proximal joints can't be moved by random actions, then anneal to real torque). Disabled
    # (steps=0) by default so other robots are unaffected. Units of effort_curriculum_steps = CONTROL
    # steps (= training_iters * num_steps_per_env=24). Scales the first `num_joints` action joints from
    # start_scale (at step 0) linearly down to 1.0 (at effort_curriculum_steps), then holds 1.0.
    effort_curriculum_start_scale: float = 1.0
    effort_curriculum_steps: int = 0
    effort_curriculum_num_joints: int = 0
    # Optional deploy-style command limiter. This clamps the per-control-step change of
    # processed q_des after action_scale/default-pose conversion, before it is written to
    # the actuator. Units are rad/control-step in action joint order.
    action_target_rate_limit_enable: bool = False
    action_target_max_delta_per_tick: tuple = ()
    # --- Table-tennis paddle / hitting geometry (defaults match Booster T1) ---
    paddle_body_name: str = "right_hand_link"   # body the paddle is rigidly attached to
    paddle_offset: tuple = (0.0, -0.345, 0.0)   # paddle face center offset in that body's local frame
    hit_body_height: float = 0.69               # target body height for robot_future_pos
    home_y: float = 0.0                         # fixed robot base home y in table frame
    paddle_y_offset: float = -0.60              # lateral base->paddle offset in ready stance
    robot_vel_max: float = 7.0                  # clamp for robot_future_vel target
    hit_plane_x: float = -1.6                   # robot stance / hit-plane x (env-local). HOME, intercept
                                                # clamp, give-up/terminal lines & idle anchor all derive
                                                # from this. G1 overrides to -2.0 (robot >=60cm from table).
    # Optional reachable clamp for the analytic hit target. Defaults are inert and tasks can
    # tighten them when a fixed base/arm should not chase unreachable intercepts.
    hit_target_x_range: tuple = (-100.0, 100.0)
    hit_target_y_range: tuple = (-100.0, 100.0)
    hit_target_z_range: tuple = (-100.0, 100.0)

@configclass
class BallCfg:
    ball_speed_x_range: tuple = (-5.5, -4.5)
    ball_speed_y_range: tuple = (-0.8,0.8)
    ball_speed_z_range: tuple = (1.6, 1.7)
    ball_pos_y_range: tuple = (-0.2, 0.2)
    contact_threshold: float = 0.06
    # Sweet-spot shaping is opt-in per task. It latches the first valid paddle hit's
    # in-plane center quality and can scale downstream return rewards without making
    # "hover near paddle center" a standalone dense objective.
    sweet_contact_radius: float = 0.0
    sweet_contact_core_radius: float = 0.0
    sweet_contact_face_axis: str = "y"
    sweet_contact_gate_outcomes: bool = False
    sweet_contact_outcome_floor: float = 1.0
    # If enabled, distance alone is not a paddle hit. The paddle must actively swing into
    # the ball, which prevents a serve trajectory from farming reward on a static blade.
    require_active_contact: bool = False
    active_contact_min_paddle_speed: float = 0.0
    active_contact_min_forward_speed: float = -100.0
    active_contact_require_own_bounce: bool = False
    # Optional first-contact hit-plane quality. The hard margin gates whether a
    # paddle touch is a valid hit; the radius/core latch a [0,1] quality that can
    # scale contact and downstream return rewards.
    active_contact_hit_plane_margin: float = 0.28
    hit_plane_contact_radius: float = 0.0
    hit_plane_contact_core_radius: float = 0.0
    hit_plane_contact_gate_outcomes: bool = False
    hit_plane_contact_outcome_floor: float = 1.0
    ball_max_eposide_length: float = 1.5
    ball_reset_repeat: int = 5
    num_new_serves = 2
    max_serve_per_episode: int = 5
    # --- serve curriculum (0 = disabled -> use the ranges above directly). When >0,
    # reset_ball interpolates each range from the base (above) to the _wide target
    # over this many control steps, plus varies serve height by ball_pos_z_delta_wide. ---
    serve_curriculum_steps: int = 0
    ball_speed_x_range_wide: tuple = (-5.5, -4.5)
    ball_speed_y_range_wide: tuple = (-0.8, 0.8)
    ball_speed_z_range_wide: tuple = (1.6, 1.7)
    ball_pos_y_range_wide: tuple = (-0.2, 0.2)
    ball_pos_z_delta_wide: tuple = (0.0, 0.0)
    # --- RALLY serve (bounce-point parametrization, used when serve_bounce_enable=True).
    # Sample a TARGET BOUNCE POINT in the robot's own half and back-compute the launch
    # velocity so the ball always bounces in-court (no volleys). Lateral half-width
    # grows from serve_y_start -> serve_y_wide over serve_curriculum_steps. ---
    serve_bounce_enable: bool = False
    serve_bounce_x_range: tuple = (-1.25, -0.65)   # depth: mid + deep court (x in robot half [-1.37,0])
    serve_bounce_vz_range: tuple = (1.5, 1.9)      # launch vz -> arc height / bounce timing
    serve_y_center: float = 0.0                     # lateral center for bounce-target sampling
    serve_y_start: float = 0.05                    # initial lateral half-width (centered)
    serve_y_wide: float = 0.65                     # final lateral half-width (table half-width 0.7625)
    # --- no-ball idle training (0 = off). Every no_ball_period_s the ball is active for
    # ball_active_s, then teleported away (no-ball + mask_invalid) for the rest, so the
    # policy learns a stable idle at the home sentinel when there is no incoming ball. ---
    no_ball_period_s: float = 0.0
    ball_active_s: float = 0.0
    no_ball_curriculum_steps: int = 0   # ramp the no-ball gap from 0 -> (period-active) over this many control steps
    # --- idle10 (A): hit-first curriculum. Both the no-ball ramp AND the idle reward START at
    # curriculum_phase1_steps; before that the ball is always present and the idle reward is 0
    # (pure-hitting bootstrap = the proven from-scratch hitting recipe, no idle to neglect).
    # After phase1 the idle reward ramps 0->1 over idle_reward_ramp_steps while the no-ball gap
    # ramps over no_ball_curriculum_steps. Keep idle_reward_ramp_steps < no_ball_curriculum_steps
    # so the stabilizing idle_pose reference always LEADS the no-ball difficulty (the idle3
    # freeze-collapse happened when no-ball outpaced an inadequate reference). 0 = legacy (ramp
    # from control step 0, no phase-1 hold).
    curriculum_phase1_steps: int = 0
    idle_reward_ramp_steps: int = 0     # control steps to ramp the idle reward 0->1 after phase1
    # difficulty curriculum HARD targets (lerp from the easy ranges above over
    # serve_curriculum_steps). Default = same as easy -> no expansion.
    serve_bounce_x_range_hard: tuple = (-1.25, -0.65)
    serve_bounce_vz_range_hard: tuple = (1.5, 1.9)
    # idle10 (A): delay the START of the serve difficulty curriculum (RAW sim_step_counter
    # units) so stage-1 is a fixed-EASY hitting bootstrap; difficulty ramps only after this.
    # 0 = ramp from step 0 (legacy).
    serve_curriculum_phase_start: int = 0
    # idle12: PERFORMANCE-GATED serve difficulty. When enabled, the easy->hard factor c is no
    # longer a function of sim_step; instead it advances only while the running success-return
    # rate (succ_ema) is >= serve_succ_window, by a step sized so a sustained pass takes
    # serve_c_ramp_iters training iters to go c:0->1. c=1 then freezes (consolidation). The gate
    # self-limits: if a difficulty yields un-returnable serves, success drops below the window
    # and c stops advancing. Threshold is overridable at runtime via env TT_SUCC_WINDOW.
    serve_curriculum_perf_gated: bool = False
    serve_succ_window: float = 0.6
    serve_c_ramp_iters: int = 10000


@configclass
class TableCfg:
    table_opponent_contact_x: tuple = (0.0, 1.37)
    table_opponent_contact_y: tuple = (-0.7625, 0.7625)
    table_opponent_contact_z: tuple = (0.70, 0.85) # table height - ball radius + margin
    table_own_contact_x: tuple = (-1.37, 0.0)
    table_own_contact_y: tuple = (-0.7625, 0.7625)
    table_own_contact_z: tuple = (0.70, 0.85)

@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0
    robot_pos: float = 1.0
    ball_pos: float = 1.0
    ball_linvel: float = 1.0
    perception: float = 1.0 #assuming robot_pos, ball_pos, ball_linvel can be treated uniformly
    ball_state: float = 1.0

@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0) # TODO: set zero for student policy
    lin_vel_y: tuple = (-0.5, 0.5) # TODO: set zero for student policy
    ang_vel_z: tuple = (-1.0, 1.0) # TODO: set zero for student policy
    heading: tuple = (-math.pi, math.pi) # TODO: set smaller range for student policy if needed, else set zero


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (10.0, 10.0) # TODO: Check and change to match episode length if required
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 1.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class NoiseScalesCfg:
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1
    ball_pos: float = 0.0 #TODO: tune this based on perception noise
    ball_linvel: float = 0.0 #TODO: tune this based on perception noise
    robot_pos: float = 0.0  # TODO: tune this based on perception noise
    perception: float = 0.0 #TODO: assuming uniform noise across perception, tune this based on perception noise
    ball_state: float = 0.0


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_locomotion_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Waist",".*_Hip_.*", ".*_Knee_.*",".*_Ankle_.*","Left_Elbow_.*","Left_Shoulder_.*","AAHead_yaw","Head_pitch"])
        },
    )
    reset_manipulation_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Right_Elbow_.*","Right_Shoulder_.*"])
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class ActionDelayCfg:
    enable: bool = False
    params: dict = {"max_delay": 5, "min_delay": 0}

@configclass
class PerceptionDelayCfg:
    enable: bool = True
    params: dict = {"max_delay": 4, "min_delay": 3}

@configclass
class DomainRandCfg:
    events: EventCfg = EventCfg()
    action_delay: ActionDelayCfg = ActionDelayCfg()
    perception_delay: PerceptionDelayCfg = PerceptionDelayCfg()


@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15


@configclass
class SimCfg:
    dt: float = 0.005
    decimation: int = 4
    physx: PhysxCfg = PhysxCfg()

@configclass
class ActionsCfg:
    joint_names: list = []
    """List of joint names or regex expressions that the action will be mapped to."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""

@configclass
class ObservationsCfg:
    joint_names: list = []
    """List of joint names or regex expressions that the action will be mapped to."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""

# @configclass
# class TTSceneCfg(SceneCfg):
#     def __init__(self, config: BaseSceneCfg, physics_dt, step_dt):
#         super().__init__(config, physics_dt, step_dt)
#         self.table: RigidObjectCfg = config.table
#         self.table.prim_path = "{ENV_REGEX_NS}/Table"

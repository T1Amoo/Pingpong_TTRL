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

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import isaaclab.utils.math as math_utils
import numpy as np
import os
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager, CurriculumManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils import configclass
from rsl_rl.env import VecEnv

from legged_lab.envs.base.tt_env_config import TTEnvCfg
from legged_lab.envs.base.tt_config import BaseSceneCfg
from legged_lab.utils.env_utils.scene import SceneCfg

# ! Aerodynamics : BEGIN
from legged_lab.physics.aerodynamics import AeroForceField
# ! Aerodynamics : END

#TODO: move this functio to tt_config.py
@configclass
class TTSceneCfg(SceneCfg):
    def __init__(self, config: BaseSceneCfg, physics_dt, step_dt):
        super().__init__(config, physics_dt, step_dt)
        self.table: RigidObjectCfg = config.table
        self.table.prim_path = "{ENV_REGEX_NS}/Table"

        self.ball: RigidObjectCfg = config.ball
        self.ball.prim_path = "{ENV_REGEX_NS}/Ball"

        # visualization object for predicted ball pose
        self.ball_future: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BallFuture",
            spawn=sim_utils.SphereCfg(
                radius=0.035,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    metallic=0.0,
                    roughness=0.5,
                ),
            ),
        )
        # visualization object for learned model ball prediction (different color)
        self.ball_pred: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BallPred",
            spawn=sim_utils.SphereCfg(
                radius=0.04,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.7, 0.0),  # orange/yellow to distinguish
                    metallic=0.0,
                    roughness=0.5,
                ),
            ),
        )
        # self.robot_future: RigidObjectCfg = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/RobotFuturePos",
        #     spawn=sim_utils.SphereCfg(
        #         radius=0.03,  # slightly larger than BallFuture (0.02)
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=True,
        #         ),
        #         collision_props=sim_utils.CollisionPropertiesCfg(
        #             collision_enabled=False,
        #         ),
        #         visual_material=sim_utils.PreviewSurfaceCfg(
        #             diffuse_color=(0.1, 0.4, 1.0),  # different color, e.g., blue-ish
        #             metallic=0.0,
        #             roughness=0.5,
        #         ),
        #     ),
        # )

        # visualization object for predicted robot velocity
        # ArrowCfg = getattr(sim_utils, "ArrowCfg", None)
        # if ArrowCfg is not None:
        #     arrow_spawn = ArrowCfg(
        #         shaft_length=0.9,
        #         shaft_radius=0.01,
        #         head_length=0.1,
        #         head_radius=0.03,
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=True,
        #         ),
        #         collision_props=sim_utils.CollisionPropertiesCfg(
        #             collision_enabled=False,
        #         ),
        #         visual_material=sim_utils.PreviewSurfaceCfg(
        #             diffuse_color=(1.0, 0.0, 0.0),
        #             metallic=0.0,
        #             roughness=0.5,
        #         ),
        #     )
        # else:
        #     arrow_spawn = sim_utils.CuboidCfg(
        #         size=(1.0, 0.02, 0.02),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=True,
        #         ),
        #         collision_props=sim_utils.CollisionPropertiesCfg(
        #             collision_enabled=False,
        #         ),
        #         visual_material=sim_utils.PreviewSurfaceCfg(
        #             diffuse_color=(1.0, 0.0, 0.0),
        #             metallic=0.0,
        #             roughness=0.5,
        #         ),
        #     )

        # self.robot_future_vel: RigidObjectCfg = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/RobotFutureVel",
        #     spawn=arrow_spawn,
        # )

class TTEnv(VecEnv):
    def __init__(self, cfg: TTEnvCfg, headless):
        self.cfg: TTEnvCfg

        self.cfg = cfg
        self._is_closed = False
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="min",
                restitution_combine_mode="min",
                restitution=0.95,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = TTSceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.table: RigidObject = self.scene["table"]
        self.ball: RigidObject = self.scene["ball"]
        self.ball_future_visual: RigidObject = self.scene["ball_future"]
        self.ball_pred_visual: RigidObject = self.scene["ball_pred"]
        # self.robot_future_vel_visual: RigidObject = self.scene["robot_future_vel"]
        # self.robot_future_pos_visual: RigidObject = self.scene["robot_future"]

        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        
        # ! Aerodynamics Init : BEGIN
        # ! Initialize before the buffer and the environment reset
        self.aero = AeroForceField(
            device=str(self.device), 
            radius_m=0.020,
            air_density=1.225,
            # drag_coeff = 0.0 # ! Set to zero to disable aerodynamics
            drag_coeff = 0.4378 # ! Match the real world testing
        )
        # ! Aerodynamics Init : am

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)
    
    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    def init_buffers(self):
        self.extras = {}

        self.max_ball_episode_length_s = self.cfg.ball.ball_max_eposide_length
        self.max_ball_serve_per_episode = self.cfg.ball.max_serve_per_episode
        self.max_ball_episode_length = np.ceil(self.max_ball_episode_length_s / self.step_dt)
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        # self.max_episode_length_s = self.max_ball_episode_length_s * self.cfg.ball.ball_reset_repeat * self.cfg.ball.num_new_serves
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)

        # self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.num_actions = self.cfg.robot.num_actions   # actuated joints only
        self.num_joints = self.cfg.robot.num_joints
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        self.num_perception = 6 # ball_pos(3) + robot_pos(3) = 9

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        self.action_target_slew_excess_l2 = torch.zeros(self.num_envs, device=self.device)
        self.action_target_slew_clip_frac = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.perception_buffer = DelayBuffer(
            self.cfg.domain_rand.perception_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.perception_buffer.compute(
            torch.zeros(self.num_envs, self.num_perception, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.perception_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.perception_delay.params["min_delay"],
                high=self.cfg.domain_rand.perception_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.perception_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
        self._init_camera_observation_model()

        # resolve the joints over which the action term is applied
        self.action_joint_ids, self.action_joint_names = self.robot.find_joints(
            self.cfg.actions.joint_names, preserve_order=self.cfg.actions.preserve_order
        )
        self.obs_joint_ids, self.obs_joint_names = self.robot.find_joints(
            self.cfg.observations.joint_names, preserve_order=self.cfg.observations.preserve_order
        )
        self._action_target_rate_limit_enable = bool(
            getattr(self.cfg.robot, "action_target_rate_limit_enable", False)
        )
        self._action_target_max_delta = None
        if self._action_target_rate_limit_enable:
            target_delta = tuple(getattr(self.cfg.robot, "action_target_max_delta_per_tick", ()) or ())
            if len(target_delta) != self.num_actions:
                raise ValueError(
                    "robot.action_target_max_delta_per_tick must have one value per action "
                    f"joint ({self.num_actions}), got {len(target_delta)}"
                )
            self._action_target_max_delta = torch.tensor(
                target_delta,
                device=self.device,
                dtype=self.robot.data.default_joint_pos.dtype,
            ).unsqueeze(0)
        self._action_target_lowpass_enable = bool(
            getattr(self.cfg.robot, "action_target_lowpass_enable", False)
        )
        self._action_target_lowpass_tau = None
        self._action_target_lowpass_vel = None
        self._action_target_lowpass_tau_base = None
        self._action_target_lowpass_vel_base = None
        self._action_target_lowpass_tau_ranges = None
        self._action_target_lowpass_vel_scale_range = (1.0, 1.0)
        if self._action_target_lowpass_enable:
            if self._action_target_rate_limit_enable:
                raise ValueError(
                    "action_target_lowpass_enable and action_target_rate_limit_enable are "
                    "mutually exclusive; enable only one command-shaping path."
                )
            tau = tuple(getattr(self.cfg.robot, "action_target_lowpass_tau_s", ()) or ())
            if len(tau) != self.num_actions:
                raise ValueError(
                    "robot.action_target_lowpass_tau_s must have one value per action "
                    f"joint ({self.num_actions}), got {len(tau)}"
                )
            self._action_target_lowpass_tau_base = torch.tensor(
                tau, device=self.device, dtype=self.robot.data.default_joint_pos.dtype
            ).unsqueeze(0)
            self._action_target_lowpass_tau = self._action_target_lowpass_tau_base.expand(
                self.num_envs, -1
            ).clone()
            tau_ranges = tuple(
                tuple(float(v) for v in pair)
                for pair in (getattr(self.cfg.robot, "action_target_lowpass_tau_range_s", ()) or ())
            )
            if tau_ranges:
                if len(tau_ranges) != self.num_actions or any(
                    len(pair) != 2 or pair[0] <= 0.0 or pair[1] < pair[0]
                    for pair in tau_ranges
                ):
                    raise ValueError(
                        "robot.action_target_lowpass_tau_range_s must contain one positive "
                        f"(min,max) pair per action joint; got {tau_ranges}"
                    )
                self._action_target_lowpass_tau_ranges = torch.tensor(
                    tau_ranges,
                    device=self.device,
                    dtype=self.robot.data.default_joint_pos.dtype,
                )
            vel = tuple(getattr(self.cfg.robot, "action_target_lowpass_vel_limit", ()) or ())
            if vel:
                if len(vel) != self.num_actions:
                    raise ValueError(
                        "robot.action_target_lowpass_vel_limit must be empty or one value per "
                        f"action joint ({self.num_actions}), got {len(vel)}"
                    )
                self._action_target_lowpass_vel_base = torch.tensor(
                    vel, device=self.device, dtype=self.robot.data.default_joint_pos.dtype
                ).unsqueeze(0)
                self._action_target_lowpass_vel = self._action_target_lowpass_vel_base.expand(
                    self.num_envs, -1
                ).clone()
                scale_range = tuple(
                    float(v)
                    for v in getattr(
                        self.cfg.robot,
                        "action_target_lowpass_vel_limit_scale_range",
                        (1.0, 1.0),
                    )
                )
                if len(scale_range) != 2 or scale_range[0] <= 0.0 or scale_range[1] < scale_range[0]:
                    raise ValueError(
                        "robot.action_target_lowpass_vel_limit_scale_range must be a positive "
                        f"(min,max) pair, got {scale_range}"
                    )
                self._action_target_lowpass_vel_scale_range = scale_range
        self._last_processed_actions = self.robot.data.default_joint_pos[:, self.action_joint_ids].clone()
        self._init_action_response_model()

        _paddle_ids, _ = self.robot.find_bodies(self.cfg.robot.paddle_body_name)
        assert len(_paddle_ids) == 1, f"paddle_body_name resolved to {len(_paddle_ids)} bodies"
        self._paddle_body_id = _paddle_ids[0]

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)

        self.table_cfg = SceneEntityCfg(name="table")
        self.table_cfg.resolve(self.scene)

        self.ball_cfg = SceneEntityCfg(name="ball")
        self.ball_cfg.resolve(self.scene)

        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ball_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.ball_reset_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # will store env ids that had their ball reset in the most recent step
        self.ball_reset_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.reset_ball_state_buf = torch.zeros(self.num_envs, 13, device=self.device, dtype=torch.float)
        # Curriculum clock (physics-substep units; cs = //decimation = 50 Hz control steps).
        # Seed from TT_SIM_STEP_OFFSET so a watchdog RESUME continues the curriculum at the
        # resumed iteration instead of restarting at stage 1 (the watchdog sets it to
        # resumed_iter * decimation * num_steps_per_env). 0 (default) = fresh start.
        self.sim_step_counter = int(os.environ.get("TT_SIM_STEP_OFFSET", "0") or 0)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # ball-related buffers
        self.has_touch_paddle = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_touch_paddle_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ball_landing_dis_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ball_contact_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.ball_contact_raw_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.paddle_sweet_contact_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.paddle_sweet_contact_latch = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.paddle_hit_plane_latch = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.active_paddle_hit = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_first_bounce = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_first_bounce_prev = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_touch_own_table = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_touch_own_table_prev = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_touch_opo_table_prev = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.has_return_own_table2_prev = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # idle10 fix: second-bounce dead-ball detection + decoupled true-no-ball mask
        self.has_second_bounce = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.left_after_bounce = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.mask_no_ball = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # --- performance-gated serve difficulty curriculum (idle12 warm-start) ---
        # serve_c: easy->hard factor [0,1]; succ_ema: running success-return rate that gates it.
        self.serve_c = float(os.environ.get("TT_SERVE_C_INIT", "0.0"))
        self.succ_ema = 0.0
        self.hit_ema = 0.0
        self.opo_ema = 0.0
        self._curri_log_ctr = 0

        self.penalty_ball_to_floor = torch.zeros(self.num_envs, device=self.device)
        self.reward_table_success = torch.zeros(self.num_envs, device=self.device)
        self.reward_vel_prev = torch.zeros(self.num_envs, device=self.device)
        self.penalty_table_fail = torch.zeros(self.num_envs, device=self.device)

        self.touch_info = []

        # storage for predicted future ball pose
        self.ball_future_pose_vis = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_future_pose = torch.zeros(self.num_envs, 3, device=self.device)
        # learned prediction (from auxiliary model)
        self.ball_prediction_vis = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_prediction = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_future_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_future_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_future_t = torch.zeros(self.num_envs, 1, device=self.device)
        self.mask_invalid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.mask_terminal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.robot_future_vel = torch.zeros(self.num_envs, 3, device=self.device)

        self.paddle_touch_point = torch.zeros(self.num_envs, 3, device=self.device)
        self.paddle_touch_point_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_linvel_prev = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot.write_joint_effort_limit_to_sim(self.robot.data.joint_effort_limits[:, self.action_joint_ids] * self.cfg.robot.effort_limit_scale, self.action_joint_ids)
        # base (real) effort limits of the action joints, cached for the proximal effort curriculum
        self._base_action_effort_limit = (self.robot.data.joint_effort_limits[:, self.action_joint_ids] * self.cfg.robot.effort_limit_scale).clone()

        self._reset_prediction_buffers()
        self.init_obs_buffer()
        # --- Quadratic-drag model constant for ball dynamics (scalar k) ---
        # k = 0.5 * rho * Cd * A / m
        try:
            rho = 1.225  # kg/m^3
            cd = 0.47    # sphere drag coefficient
            radius = 0.02  # m
            mass = 0.0027  # kg
            area = float(np.pi) * (radius ** 2)
            self.ball_drag_k = float(0.5 * rho * cd * area / max(1e-6, mass))
        except Exception:
            self.ball_drag_k = 0.13

    def update_ball_future_visual(self):
        if self.headless:
            return
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = self.ball_future_pose_vis
        pose[:, 3] = 1.0
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.ball_future_visual.write_root_pose_to_sim(pose, env_ids)
        self.scene.write_data_to_sim()
    def update_ball_pred_visual(self):
        if self.headless:
            return
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = self.ball_prediction_vis
        pose[:, 3] = 1.0
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.ball_pred_visual.write_root_pose_to_sim(pose, env_ids)
        self.scene.write_data_to_sim()

    def _ball_prediction_plausible(self, pred: torch.Tensor) -> torch.Tensor:
        """Reject early/OOD predictor outputs before they become actor targets."""
        hx = self.cfg.robot.hit_plane_x
        y_center = self.cfg.robot.home_y + self.cfg.robot.paddle_y_offset
        finite = torch.isfinite(pred).all(dim=-1)
        return (
            finite
            & (pred[:, 0] > hx - 0.50)
            & (pred[:, 0] < hx + 0.30)
            & (torch.abs(pred[:, 1] - y_center) < 0.45)
            & (pred[:, 2] > 0.85)
            & (pred[:, 2] < 1.55)
        )

    def _project_prediction_to_target_geometry(self, pred: torch.Tensor) -> torch.Tensor:
        """Match learned predictions to the task's analytic hit-target geometry."""
        tx_min, tx_max = getattr(self.cfg.robot, "hit_target_x_range", (-100.0, 100.0))
        if tx_min <= -99.0 and tx_max >= 99.0:
            return pred

        projected = pred.clone()
        projected[:, 0] = torch.clamp(projected[:, 0], min=float(tx_min), max=float(tx_max))
        return projected

    def _prediction_sentinel(self, env_ids=None, dtype=None) -> torch.Tensor:
        """Fixed home hit target used when no learned/analytic live target is available."""
        if dtype is None:
            dtype = self.ball_prediction.dtype
        base = torch.tensor(
            [
                self.cfg.robot.hit_plane_x,
                self.cfg.robot.home_y + self.cfg.robot.paddle_y_offset,
                self.cfg.robot.hit_body_height + 0.2,
            ],
            device=self.device,
            dtype=dtype,
        ).unsqueeze(0)
        count = self.num_envs if env_ids is None else len(env_ids)
        return base.expand(count, -1).clone()

    def _reset_prediction_buffers(self, env_ids=None) -> None:
        """Prime future/predictor targets with the same sentinel used by sim2sim."""
        sentinel = self._prediction_sentinel(env_ids=env_ids, dtype=self.ball_prediction.dtype)
        if env_ids is None:
            self.ball_future_pose.copy_(sentinel)
            self.ball_prediction.copy_(sentinel)
            self.ball_future_pose_vis.copy_(sentinel + self.scene.env_origins)
            self.ball_prediction_vis.copy_(sentinel + self.scene.env_origins)
            self.ball_future_t.zero_()
            return

        self.ball_future_pose[env_ids] = sentinel
        self.ball_prediction[env_ids] = sentinel
        self.ball_future_pose_vis[env_ids] = sentinel + self.scene.env_origins[env_ids]
        self.ball_prediction_vis[env_ids] = sentinel + self.scene.env_origins[env_ids]
        self.ball_future_t[env_ids] = 0.0

    def update_robot_future_pos_visual(self):
        if self.headless:
            return
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        # robot_future_pos is in env-local coords; add env origins to place in world
        pose[:, :3] = self.robot_future_pos + self.scene.env_origins
        pose[:, 3] = 1.0  # identity quat (w, x, y, z) where w=1, xyz=0
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot_future_pos_visual.write_root_pose_to_sim(pose, env_ids)
        self.scene.write_data_to_sim()
    # def update_robot_future_vel_visual(self):
    #     if self.headless:
    #         return
    #     base_pos = self.robot.data.root_link_pos_w
    #     vel = self.robot_future_vel
    #     pose = torch.zeros((self.num_envs, 7), device=self.device)
    #     ref_dir = torch.zeros_like(vel)
    #     ref_dir[:, 0] = 1.0
    #     dirs = torch.nn.functional.normalize(vel, dim=-1, eps=1e-6)
    #     # place arrow above robot and start at robot origin
    #     pose[:, :3] = base_pos + torch.tensor([0.0, 0.0, 0.5], device=self.device) + dirs * 0.2
    #     cross = torch.cross(ref_dir, dirs, dim=-1)
    #     w = 1.0 + (ref_dir * dirs).sum(dim=-1, keepdim=True)
    #     quat = torch.cat([w, cross], dim=-1)
    #     quat = torch.nn.functional.normalize(quat, dim=-1, eps=1e-6)
    #     pose[:, 3:] = quat
    #     env_ids = torch.arange(self.num_envs, device=self.device)
    #     self.robot_future_vel_visual.write_root_pose_to_sim(pose, env_ids)
    #     self.scene.write_data_to_sim()
        

    def compute_current_observations(self):
        robot = self.robot
        table = self.table
        ball = self.ball
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        heading = robot.data.heading_w
        # command = self.command_generator.command
        joint_pos = robot.data.joint_pos[:, self.obs_joint_ids] - robot.data.default_joint_pos[:, self.obs_joint_ids]
        joint_vel = robot.data.joint_vel[:, self.obs_joint_ids]
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        ball_pos = ball.data.root_link_pos_w - table.data.root_link_pos_w
        # ball_pos = self.ball.data.root_pos_w - self.scene.env_origins  
        ball_linvel = self.ball.data.root_lin_vel_w
        robot_pos = robot.data.root_link_pos_w - table.data.root_link_pos_w
        pred_ok = self._ball_prediction_plausible(self.ball_prediction).unsqueeze(-1)
        ball_pred_safe = torch.where(pred_ok, self.ball_prediction, self.ball_future_pose)

        # Relative target offset (x,y): use ball prediction shifted by paddle offset as desired robot target
        ball_target_xy = torch.stack([
            ball_pred_safe[:, 0] - 0.1,
            ball_pred_safe[:, 1] - self.cfg.robot.paddle_y_offset,
        ], dim=1)
        rel_target_xy = (ball_target_xy - robot_pos[:, :2]) * self.obs_scales.robot_pos

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                ball_pos * self.obs_scales.ball_pos,
                robot_pos * self.obs_scales.robot_pos,
                # self.paddle_touch_point* self.obs_scales.ball_pos,
                ball_pred_safe * self.obs_scales.ball_pos, # use learned prediction in actor obs
                rel_target_xy,  # 2D relative target base pos
                heading.unsqueeze(-1) * self.obs_scales.projected_gravity,
            ],
            dim=-1,
        )
        alt_critic_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                ball_pos * self.obs_scales.ball_pos,
                ball_linvel * self.obs_scales.ball_linvel,
                # self.ball_prediction * self.obs_scales.ball_pos,
                self.ball_future_pose * self.obs_scales.ball_pos, # use ground-truth future in critic obs
                self.paddle_touch_point* self.obs_scales.ball_pos,
                robot_pos * self.obs_scales.robot_pos,
                heading.unsqueeze(-1) * self.obs_scales.projected_gravity,
                (self.robot_future_pos * self.obs_scales.robot_pos - robot_pos * self.obs_scales.robot_pos),
                self.ball_future_t,  # [N,1]
                (self.ball_episode_length_buf.float() / float(self.max_ball_episode_length)).clamp(0.0, 1.0).unsqueeze(-1),
                # (self.episode_length_buf.float() / float(self.max_episode_length)).clamp(0.0, 1.0).unsqueeze(-1),
                (self.ball_reset_counter/self.max_ball_serve_per_episode).unsqueeze(-1),
                self.has_touch_own_table_prev.unsqueeze(-1) * self.obs_scales.ball_state,
                self.has_touch_paddle.unsqueeze(-1) * self.obs_scales.ball_state,
            ],
            dim=-1,
        )
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [alt_critic_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact],
            dim=-1
        )

        return current_actor_obs, current_critic_obs
    
    def compute_current_observations_perception(self): 
        robot = self.robot
        table = self.table
        ball = self.ball
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        heading = robot.data.heading_w
        # command = self.command_generator.command
        joint_pos = robot.data.joint_pos[:, self.obs_joint_ids] - robot.data.default_joint_pos[:, self.obs_joint_ids]
        joint_vel = robot.data.joint_vel[:, self.obs_joint_ids]
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        ball_pos = ball.data.root_link_pos_w - table.data.root_link_pos_w
        # ball_pos = self.ball.data.root_pos_w - self.scene.env_origins  
        ball_linvel = self.ball.data.root_lin_vel_w
        robot_pos = robot.data.root_link_pos_w - table.data.root_link_pos_w
        # ===== Unified hard validity gate (no-ball idle; from-scratch design) =====
        # On invalid/no-ball, HARD-replace the actor's ball signals with the fixed HOME
        # sentinel — the SAME value the critic's ball_future_pose takes on invalid (see
        # modified_ball_pos in compute_intermediate_values). This makes actor & critic
        # agree on no-ball states and keeps the actor from ever seeing OOD garbage from
        # the predictor MLP (a long no-ball input -> garbage prediction -> divergence).
        mexp = self.mask_invalid.unsqueeze(-1)  # [N,1]; set this step in compute_intermediate_values
        # (a) prediction is in ball_future_pose's frame (robot-table): sentinel == modified_ball_pos
        pred_sentinel = self._prediction_sentinel(dtype=self.ball_prediction.dtype)
        pred_ok = self._ball_prediction_plausible(self.ball_prediction).unsqueeze(-1)
        ball_pred_safe = torch.where(pred_ok, self.ball_prediction, self.ball_future_pose)
        ball_pred_g = torch.where(mexp, pred_sentinel, ball_pred_safe)
        # (b) perception ball slot [0:3] is in (ball - env_origins) frame: convert the same
        #     home point -> world (table + sentinel_rel) -> perception (- env_origins).
        percep_ball_sentinel = (self.table.data.root_link_pos_w + pred_sentinel) - self.scene.env_origins
        perception_g = self.delayed_perception.clone()
        perception_g[:, 0:3] = torch.where(mexp, percep_ball_sentinel, perception_g[:, 0:3])  # robot slot [3:6] untouched

        # Relative target offset (x,y) for actor obs with perception (from GATED prediction)
        ball_target_xy = torch.stack([
            ball_pred_g[:, 0] - 0.1,
            ball_pred_g[:, 1] - self.cfg.robot.paddle_y_offset,
        ], dim=1)
        rel_target_xy = (ball_target_xy - robot_pos[:, :2]) * self.obs_scales.robot_pos

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                perception_g * self.obs_scales.perception,
                # self.paddle_touch_point* self.obs_scales.ball_pos,
                ball_pred_g * self.obs_scales.ball_pos, # GATED learned prediction in actor obs
                rel_target_xy,  # 2D relative target base pos
                heading.unsqueeze(-1) * self.obs_scales.projected_gravity,
            ],
            dim=-1,
        )
        alt_critic_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                ball_pos * self.obs_scales.ball_pos,
                ball_linvel * self.obs_scales.ball_linvel,
                # self.ball_prediction * self.obs_scales.ball_pos,
                self.ball_future_pose * self.obs_scales.ball_pos, # use ground-truth future in critic obs
                self.paddle_touch_point* self.obs_scales.ball_pos,
                robot_pos * self.obs_scales.robot_pos,
                heading.unsqueeze(-1) * self.obs_scales.projected_gravity,
                (self.robot_future_pos * self.obs_scales.robot_pos - robot_pos * self.obs_scales.robot_pos),
                self.ball_future_t,  # [N,1]
                (self.ball_episode_length_buf.float() / float(self.max_ball_episode_length)).clamp(0.0, 1.0).unsqueeze(-1),
                (self.episode_length_buf.float() / float(self.max_episode_length)).clamp(0.0, 1.0).unsqueeze(-1),
                self.has_touch_own_table_prev.unsqueeze(-1) * self.obs_scales.ball_state,
                self.has_touch_paddle.unsqueeze(-1) * self.obs_scales.ball_state,
            ],
            dim=-1,
        )
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [alt_critic_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact],
            dim=-1
        )

        return current_actor_obs, current_critic_obs

    def update_prediction(self, preds: torch.Tensor, valid_mask: torch.Tensor | None = None):
        """Update learned ball prediction and its visualization.

        Args:
            preds: Tensor of shape [N, 3] in env-local coordinates per env.
        """
        try:
            if preds is None:
                return
            # Ensure correct shape and device
            if not isinstance(preds, torch.Tensor):
                preds = torch.as_tensor(preds, dtype=torch.float32)
            preds = preds.to(self.device)
            if preds.shape != (self.num_envs, 3):
                # Attempt to reshape if possible (e.g., [N, H*3] is invalid here)
                if preds.ndim == 2 and preds.shape[1] >= 3:
                    preds = preds[:, :3]
                else:
                    return
            # Store prediction in env-local frame, projected into the same target geometry as
            # the analytic hit point. For fixed-plane tasks this pins x to hit_plane_x, leaving
            # y/z as the learned quantities that matter.
            projected = self._project_prediction_to_target_geometry(preds)
            if valid_mask is not None:
                valid_mask = torch.as_tensor(valid_mask, device=self.device, dtype=torch.bool).reshape(-1)
                if valid_mask.numel() != self.num_envs:
                    return
                sentinel = self._prediction_sentinel(dtype=projected.dtype)
                projected = torch.where(valid_mask.unsqueeze(-1), projected, sentinel)
            self.ball_prediction = projected
            # Visualize only the learned prediction that is valid enough to be a live target.
            # The actor already gates unusable/OOD predictions; drawing raw predictions during
            # invalid flight segments makes the yellow marker look like a drifting trajectory.
            pred_usable = self._ball_prediction_plausible(self.ball_prediction) & (~self.mask_invalid)
            pred_vis = torch.nan_to_num(self.ball_prediction, nan=0.0, posinf=0.0, neginf=0.0).clone()
            pred_vis[:, 2] = torch.where(
                pred_usable,
                torch.clamp(pred_vis[:, 2], min=0.82),
                torch.full_like(pred_vis[:, 2], -10.0),
            )
            self.ball_prediction_vis = pred_vis + self.scene.env_origins
            # debug print once, then sparsely
            if not hasattr(self, "_pred_debug_counter"):
                self._pred_debug_counter = 0
            self._pred_debug_counter += 1
            if self._pred_debug_counter == 1 or self._pred_debug_counter % 50 == 0:
                try:
                    p0 = self.ball_prediction[0].detach().cpu().numpy()
                    f0 = self.ball_future_pose[0].detach().cpu().numpy()
                    err0 = float(torch.norm(self.ball_prediction[0] - self.ball_future_pose[0]).detach().cpu())
                    valid0 = bool((~self.mask_invalid[0]).detach().cpu())
                    usable0 = bool((self._ball_prediction_plausible(self.ball_prediction)[0] & (~self.mask_invalid[0])).detach().cpu())
                    print(
                        "[TTVis] env0 learned_pred="
                        f"{p0} analytic_hit={f0} err={err0:.3f} valid_hit={valid0} pred_usable={usable0}"
                    )
                except Exception:
                    pass
            if not self.headless:
                self.update_ball_pred_visual()
        except Exception:
            # Be robust against any runtime issues; avoid breaking the sim loop
            pass
    
    def compute_observations(self):
        # current_actor_obs, current_critic_obs = self.compute_current_observations()
        current_actor_obs, current_critic_obs = self.compute_current_observations_perception()
        self.current_actor_obs=current_actor_obs
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.perception_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.ball_reset_counter[env_ids] = 0

        self._reset_table(env_ids)
        # Reset ball state
        self.reset_ball(env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()
        self.ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        self.robot_pos = self.robot.data.root_link_pos_w - self.table.data.root_link_pos_w
        self.current_perception = torch.cat([self.ball_pos, self.robot_pos], dim=-1)
        if not self._camera_observation_enable:
            self.delayed_perception[env_ids] = self.current_perception[env_ids]
        self._reset_action_target_limiter(env_ids)
        self._reset_action_response_model(env_ids)

    def _reset_table(self, env_ids):
        """Restore table pose and velocity on full environment resets."""
        if len(env_ids) == 0:
            return

        table_state = self.table.data.default_root_state.clone()[env_ids]
        table_state[:, :3] += self.scene.env_origins[env_ids]
        self.table.write_root_pose_to_sim(table_state[:, :7], env_ids)
        self.table.write_root_velocity_to_sim(table_state[:, 7:], env_ids)

    def _tt_no_ball_now(self):
        """Whether to suppress the ball THIS control step (ball teleported far + mask_invalid).
        Sources: cfg.ball.no_ball_period_s (TRAINING: periodic no-ball so the policy learns the
        no-ball idle), or env vars TT_NO_SERVE / TT_SERVE_PERIOD (DEBUG). sim_step_counter is in
        50 Hz control steps; the ball is active for `ball_active_s` at the start of each period,
        then a no-ball gap for the rest."""
        import os
        if os.environ.get("TT_NO_SERVE"):
            return True
        per = os.environ.get("TT_SERVE_PERIOD")
        # sim_step_counter increments per PHYSICS substep (decimation per control step);
        # convert to 50 Hz control steps so the *50 second->step conversions are correct.
        cs = int(self.sim_step_counter // self.cfg.sim.decimation)
        if per:
            period = max(1, int(float(per) * 50.0))
            active = int(float(os.environ.get("TT_SERVE_ACTIVE", "2.5")) * 50.0)
            return (cs % period) >= active
        nb = float(getattr(self.cfg.ball, "no_ball_period_s", 0.0) or 0.0)
        if nb > 0.0:
            period = max(1, int(nb * 50.0))
            active_target = int(float(getattr(self.cfg.ball, "ball_active_s", nb)) * 50.0)
            cstep = int(getattr(self.cfg.ball, "no_ball_curriculum_steps", 0) or 0)
            if cstep > 0:
                # ramp the no-ball GAP from 0 -> (period-active_target). idle10: the ramp does
                # not start until curriculum_phase1_steps (pure-hitting bootstrap first), so a
                # from-scratch policy learns to HIT before any no-ball appears. Before phase1
                # c=0 -> active=period -> no gap (all ball). Then grow the gap gradually.
                phase1 = int(getattr(self.cfg.ball, "curriculum_phase1_steps", 0) or 0)
                c = min(1.0, max(0.0, float(cs - phase1) / float(cstep)))
                active = int(period - c * (period - active_target))
            else:
                active = active_target
            return (cs % period) >= active
        return False

    def idle_reward_scale(self) -> float:
        """Curriculum factor [0,1] for the idle rewards (reward_idle_pose / reward_idle_stand).
        idle10 (A): 0 during the pure-hitting bootstrap (cs < curriculum_phase1_steps) so the
        policy learns to HIT first — idle9 failed because the idle reward was on from iter 0 and,
        being easy+safe vs hard+risky hitting, the from-scratch policy went couch-potato (never
        hit). After phase1 the factor ramps 0->1 over idle_reward_ramp_steps so idle is layered
        onto a competent hitter, and (ramp_steps < no_ball_curriculum_steps) the idle reference
        rises FASTER than the no-ball difficulty -> it always leads, never lags (anti freeze).
        0 ramp_steps = legacy constant full weight."""
        cs = int(self.sim_step_counter // self.cfg.sim.decimation)
        phase1 = int(getattr(self.cfg.ball, "curriculum_phase1_steps", 0) or 0)
        ramp = int(getattr(self.cfg.ball, "idle_reward_ramp_steps", 0) or 0)
        if ramp <= 0:
            return 1.0 if cs >= phase1 else 0.0
        return min(1.0, max(0.0, float(cs - phase1) / float(ramp)))

    def reset_ball(self, env_ids):
        """Reset only the ball state for specified environments."""
        if len(env_ids) == 0:
            return

        # perf-gated curriculum: record the success-return rate of the FINISHING balls BEFORE
        # their flags are cleared below. success = hit by paddle AND reached the opponent table.
        # NOTE: succ_ema is COMPRESSED vs eval success — training's 1.5s ball-episode timeout often
        # resets the ball before the (hit) return finishes flying to the opponent table, so it is a
        # STRICT "return completed in-window" rate (model_12400: succ_ema~0.33 vs eval 0.98). The
        # gate window is calibrated to THIS signal's scale, not to absolute %. hit_ema/opo_ema are
        # logged for visibility.
        if getattr(self.cfg.ball, "serve_curriculum_perf_gated", False):
            _hit = self.has_touch_paddle[env_ids].float().mean().item()
            _opo = self.has_touch_opo_table_prev[env_ids].float().mean().item()
            _succ = (self.has_touch_paddle[env_ids] & self.has_touch_opo_table_prev[env_ids]).float().mean().item()
            self.succ_ema = 0.98 * self.succ_ema + 0.02 * _succ
            self.hit_ema = 0.98 * self.hit_ema + 0.02 * _hit
            self.opo_ema = 0.98 * self.opo_ema + 0.02 * _opo

        # Reset ball-related buffers for these environments
        self.has_touch_paddle[env_ids] = False
        self.ball_landing_dis_rew[env_ids] = False
        self.has_touch_paddle_rew[env_ids] = False
        self.ball_contact_rew[env_ids] = 0.0
        self.ball_contact_raw_rew[env_ids] = 0.0
        self.paddle_sweet_contact_rew[env_ids] = 0.0
        self.paddle_sweet_contact_latch[env_ids] = 0.0
        self.paddle_hit_plane_latch[env_ids] = 0.0
        self.active_paddle_hit[env_ids] = False
        self.has_first_bounce[env_ids] = False
        self.has_first_bounce_prev[env_ids] = False
        self.has_touch_own_table[env_ids] = False
        self.has_touch_own_table_prev[env_ids] = False
        self.has_touch_opo_table_prev[env_ids] = False
        self.has_return_own_table2_prev[env_ids] = False
        self.has_second_bounce[env_ids] = False
        self.left_after_bounce[env_ids] = False
        self.reward_vel_prev[env_ids] = 0.0
        self.ball_episode_length_buf[env_ids] = 0
        self._reset_prediction_buffers(env_ids)
        generate_new = (self.ball_reset_counter[env_ids] % self.cfg.ball.ball_reset_repeat) == 0
        reuse_old = ~generate_new
        # Bug A fix: a no-ball window parks the ball underground (z=-50), so ball_on_floor (z<0.1)
        # fires every step and reset_ball is called every step. Counting each of those as a "serve"
        # inflated ball_reset_counter past max_serve_per_episode within ~6 steps, tripping the
        # episode time-out in check_reset -> a FULL env reset (robot teleported to spawn) ~every 6
        # steps during no-ball, so the policy could never learn to stand idle. A no-ball re-park is
        # not a serve -> do not count it.
        if not self._tt_no_ball_now():
            self.ball_reset_counter[env_ids] += 1
        self.touch_info = []

        if generate_new.any():
            new_state_env_ids = env_ids[generate_new]
            # Reset ball position and velocity (copied from reset method)
            ball_state = self.ball.data.default_root_state.clone()[new_state_env_ids]
            ball_state[:, :3] += self.scene.env_origins[new_state_env_ids]
            # --- SERVE SAMPLING ---
            n = len(new_state_env_ids)
            _cstep = getattr(self.cfg.ball, "serve_curriculum_steps", 0)
            # idle10 (A): the difficulty curriculum does not start until serve_curriculum_phase_start
            # (RAW sim_step_counter units) -> serves stay at the EASY range during the stage-1
            # fixed-easy hitting bootstrap, then ramp easy->hard over serve_curriculum_steps.
            _cstart = getattr(self.cfg.ball, "serve_curriculum_phase_start", 0) or 0
            if getattr(self.cfg.ball, "serve_curriculum_perf_gated", False):
                c = float(self.serve_c)   # perf-gated: difficulty advanced in step() by success rate
            else:
                c = 0.0 if not _cstep else min(1.0, max(0.0, float(self.sim_step_counter - _cstart) / float(_cstep)))
            if getattr(self.cfg.ball, "serve_bounce_enable", False):
                # RALLY serve: sample a TARGET BOUNCE POINT in the robot's own half and
                # back-compute the launch velocity so the ball ALWAYS bounces in-court
                # (proper table tennis; no volleys). Depth x_b spans mid+deep court;
                # lateral half-width grows with the curriculum (start centered -> spread
                # left/right). Launch from ball default (1.35, 0, 1.03); bounce when ball
                # center z = 0.78 (table top 0.76 + radius 0.02). g = 9.81.
                def _bclerp(b, w, cc):
                    return (b[0] + cc * (w[0] - b[0]), b[1] + cc * (w[1] - b[1]))
                xb_lo, xb_hi = _bclerp(self.cfg.ball.serve_bounce_x_range, getattr(self.cfg.ball, "serve_bounce_x_range_hard", self.cfg.ball.serve_bounce_x_range), c)
                vz_lo, vz_hi = _bclerp(self.cfg.ball.serve_bounce_vz_range, getattr(self.cfg.ball, "serve_bounce_vz_range_hard", self.cfg.ball.serve_bounce_vz_range), c)
                y_half = self.cfg.ball.serve_y_start + c * (self.cfg.ball.serve_y_wide - self.cfg.ball.serve_y_start)
                y_center_easy = getattr(self.cfg.ball, "serve_y_center", 0.0)
                y_center_hard = getattr(self.cfg.ball, "serve_y_center_hard", None)
                if y_center_hard is None:
                    y_center_hard = y_center_easy
                y_center = y_center_easy + c * (y_center_hard - y_center_easy)
                g, Z_LAUNCH, Z_BOUNCE, X_LAUNCH = 9.81, 1.03, 0.78, 1.35
                x_b = torch.empty(n, 1, device=self.device).uniform_(xb_lo, xb_hi)
                y_b = y_center + torch.empty(n, 1, device=self.device).uniform_(-y_half, y_half)
                v_z = torch.empty(n, 1, device=self.device).uniform_(vz_lo, vz_hi)
                t_b = (v_z + torch.sqrt(v_z * v_z + 2.0 * g * (Z_LAUNCH - Z_BOUNCE))) / g
                v_x = (x_b - X_LAUNCH) / t_b   # launch x = 1.35 (env-local)
                v_y = y_b / t_b                # launch y = 0 (env-local); ball_state pos stays centered
            else:
                # legacy speed-curriculum serve (tasks that don't enable bounce mode)
                def _lerp(b, w):
                    return (b[0] + c * (w[0] - b[0]), b[1] + c * (w[1] - b[1]))
                xr = _lerp(self.cfg.ball.ball_speed_x_range, getattr(self.cfg.ball, "ball_speed_x_range_wide", self.cfg.ball.ball_speed_x_range))
                yr = _lerp(self.cfg.ball.ball_speed_y_range, getattr(self.cfg.ball, "ball_speed_y_range_wide", self.cfg.ball.ball_speed_y_range))
                zr = _lerp(self.cfg.ball.ball_speed_z_range, getattr(self.cfg.ball, "ball_speed_z_range_wide", self.cfg.ball.ball_speed_z_range))
                pyr = _lerp(self.cfg.ball.ball_pos_y_range, getattr(self.cfg.ball, "ball_pos_y_range_wide", self.cfg.ball.ball_pos_y_range))
                pzr = _lerp((0.0, 0.0), getattr(self.cfg.ball, "ball_pos_z_delta_wide", (0.0, 0.0)))
                ball_state[:, 1:2] += torch.empty(n, 1, device=self.device).uniform_(*pyr)
                ball_state[:, 2:3] += torch.empty(n, 1, device=self.device).uniform_(*pzr)  # serve-height variation
                v_x = torch.empty(n, 1, device=self.device).uniform_(*xr)
                v_y = torch.empty(n, 1, device=self.device).uniform_(*yr)
                v_z = torch.empty(n, 1, device=self.device).uniform_(*zr)

            # With small probability (≈1%), create zero-velocity, random-position serves
            #disabled for testing
            # try:
            #     special_mask = torch.rand(len(new_state_env_ids), device=self.device) < 0.01
            #     if special_mask.any():
            #         sel = torch.nonzero(special_mask, as_tuple=False).squeeze(-1)
            #         # Sample XYZ uniformly from the given ranges (env-local) and offset by env origins
            #         x_rand = torch.empty(len(sel), 1, device=self.device).uniform_(-2.0, 2.0)
            #         y_rand = torch.empty(len(sel), 1, device=self.device).uniform_(-2.0, 2.0)
            #         z_rand = torch.empty(len(sel), 1, device=self.device).uniform_(0.75, 1.5)
            #         pos_rand = torch.cat((x_rand, y_rand, z_rand), dim=1)
            #         ball_state[sel, :3] = self.scene.env_origins[new_state_env_ids][sel] + pos_rand
            #         # Zero linear velocity for these special cases
            #         v_x[sel] = 0.0
            #         v_y[sel] = 0.0
            #         v_z[sel] = 0.0
            # except Exception:
            #     pass
            lin_vel = torch.cat((v_x, v_y, v_z), dim=1)
            # DEBUG: in a no-ball window (TT_NO_SERVE / TT_SERVE_PERIOD gap) teleport the
            # ball far underground (z=-50) with zero velocity -> completely out of sight,
            # never bounces on any table, cannot affect the scene. mask_invalid is also
            # forced so the policy ignores it -> a true no-ball idle to watch.
            if self._tt_no_ball_now():
                lin_vel = torch.zeros_like(lin_vel)
                ball_state[:, :3] = self.scene.env_origins[new_state_env_ids] + torch.tensor(
                    [0.0, 0.0, -50.0], device=self.device, dtype=ball_state.dtype)
            ang_vel = torch.zeros(len(new_state_env_ids), 3, device=self.device)
            ball_state[:, 7:] = torch.cat((lin_vel, ang_vel), dim=1)
            # Store new states in buffer
            self.reset_ball_state_buf[new_state_env_ids] = ball_state
            # Apply the new state to the simulation
            self.ball.write_root_pose_to_sim(ball_state[:, :7], new_state_env_ids)
            self.ball.write_root_velocity_to_sim(ball_state[:, 7:], new_state_env_ids)

        if reuse_old.any():
            old_state_env_ids = env_ids[reuse_old]
            old_states = self.reset_ball_state_buf[old_state_env_ids]
            # Apply old states to simulation
            self.ball.write_root_pose_to_sim(old_states[:, :7], old_state_env_ids)
            self.ball.write_root_velocity_to_sim(old_states[:, 7:], old_state_env_ids)

        self.ball_linvel_prev[env_ids] = self.reset_ball_state_buf[env_ids, 7:10]
        self._reset_camera_observation(env_ids)

    def _apply_effort_curriculum(self):
        """Anneal the first `num_joints` action joints' effort limit from start_scale -> 1.0 over
        effort_curriculum_steps CONTROL steps, then hold. Lets slow proximal joints be explored early
        (high torque) then forces the policy to adapt to the real torque. Called once per step() (50Hz)."""
        steps = int(getattr(self.cfg.robot, "effort_curriculum_steps", 0) or 0)
        nj = int(getattr(self.cfg.robot, "effort_curriculum_num_joints", 0) or 0)
        if steps <= 0 or nj <= 0:
            return
        cs = self.sim_step_counter // self.cfg.sim.decimation
        start = float(getattr(self.cfg.robot, "effort_curriculum_start_scale", 1.0))
        frac = min(1.0, cs / float(steps))
        scale = start + (1.0 - start) * frac   # start_scale -> 1.0
        self._effort_curr_scale = scale        # for logging
        eff = self._base_action_effort_limit.clone()
        eff[:, :nj] = eff[:, :nj] * scale
        self.robot.write_joint_effort_limit_to_sim(eff, self.action_joint_ids)

    def _reset_action_target_limiter(self, env_ids=None):
        if not hasattr(self, "_last_processed_actions"):
            return
        current_joint_pos = self.robot.data.joint_pos[:, self.action_joint_ids]
        if env_ids is None:
            self._last_processed_actions.copy_(current_joint_pos)
            self.action_target_slew_excess_l2.zero_()
            self.action_target_slew_clip_frac.zero_()
        else:
            self._last_processed_actions[env_ids] = current_joint_pos[env_ids]
            self.action_target_slew_excess_l2[env_ids] = 0.0
            self.action_target_slew_clip_frac[env_ids] = 0.0
        self._randomize_action_target_lowpass(env_ids)

    def _randomize_action_target_lowpass(self, env_ids=None) -> None:
        if self._action_target_lowpass_tau is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self._action_target_lowpass_tau_ranges is None:
            self._action_target_lowpass_tau[env_ids] = self._action_target_lowpass_tau_base
        else:
            lo = self._action_target_lowpass_tau_ranges[:, 0]
            hi = self._action_target_lowpass_tau_ranges[:, 1]
            rand = torch.rand(
                len(env_ids), self.num_actions,
                device=self.device,
                dtype=self._action_target_lowpass_tau.dtype,
            )
            self._action_target_lowpass_tau[env_ids] = lo + rand * (hi - lo)
        if self._action_target_lowpass_vel is not None:
            lo, hi = self._action_target_lowpass_vel_scale_range
            scale = lo + torch.rand(
                len(env_ids), self.num_actions,
                device=self.device,
                dtype=self._action_target_lowpass_vel.dtype,
            ) * (hi - lo)
            self._action_target_lowpass_vel[env_ids] = self._action_target_lowpass_vel_base * scale

    def _apply_action_target_rate_limit(self, processed_actions: torch.Tensor) -> torch.Tensor:
        if self._action_target_max_delta is None:
            self.action_target_slew_excess_l2.zero_()
            self.action_target_slew_clip_frac.zero_()
            return processed_actions
        delta = processed_actions - self._last_processed_actions
        excess = torch.clamp(torch.abs(delta) - self._action_target_max_delta, min=0.0)
        excess_norm = excess / torch.clamp(self._action_target_max_delta, min=1e-6)
        self.action_target_slew_excess_l2.copy_(torch.sum(torch.square(excess_norm), dim=1))
        self.action_target_slew_clip_frac.copy_(torch.mean((excess > 0.0).float(), dim=1))
        limited_delta = torch.minimum(
            torch.maximum(delta, -self._action_target_max_delta),
            self._action_target_max_delta,
        )
        limited_actions = self._last_processed_actions + limited_delta
        self._last_processed_actions.copy_(limited_actions)
        return limited_actions

    def _apply_action_target_lowpass(self, processed_actions: torch.Tensor) -> torch.Tensor:
        if self._action_target_lowpass_tau is None:
            return processed_actions
        # First-order low-pass matching the deploy bridge servo_filter:
        # dq = (target - q_cmd) / tau, clamped to vel_limit, integrated at the policy step_dt.
        desired_dq = (processed_actions - self._last_processed_actions) / self._action_target_lowpass_tau
        if self._action_target_lowpass_vel is not None:
            desired_dq = torch.clamp(
                desired_dq, -self._action_target_lowpass_vel, self._action_target_lowpass_vel
            )
        filtered = self._last_processed_actions + desired_dq * self.step_dt
        self._last_processed_actions.copy_(filtered)
        return filtered

    def _init_action_response_model(self):
        self._action_response_model_enable = bool(
            getattr(self.cfg.robot, "action_response_model_enable", False)
        )
        self.action_response_targets = self._last_processed_actions.clone()
        if not self._action_response_model_enable:
            self._action_response_delay_buffer = None
            return

        def _param_tensor(name: str) -> torch.Tensor:
            values = tuple(getattr(self.cfg.robot, name, ()) or ())
            if len(values) != self.num_actions:
                raise ValueError(f"robot.{name} must have {self.num_actions} values, got {len(values)}")
            return torch.tensor(
                values,
                device=self.device,
                dtype=self.robot.data.default_joint_pos.dtype,
            ).unsqueeze(0)

        self._action_response_fn_hz_base = _param_tensor("action_response_fn_hz")
        self._action_response_zeta_base = _param_tensor("action_response_zeta")
        self._action_response_delay_s_base = _param_tensor("action_response_delay_s")
        self._action_response_gain_base = _param_tensor("action_response_gain")
        self._action_response_bias_base = _param_tensor("action_response_bias_rad")
        self._action_response_u_mean = _param_tensor("action_response_u_mean")
        self._action_response_fn_hz = self._action_response_fn_hz_base.expand(
            self.num_envs, -1
        ).clone()
        self._action_response_zeta = self._action_response_zeta_base.expand(
            self.num_envs, -1
        ).clone()
        self._action_response_gain = self._action_response_gain_base.expand(
            self.num_envs, -1
        ).clone()
        self._action_response_bias = self._action_response_bias_base.expand(
            self.num_envs, -1
        ).clone()

        def _scale_range(name: str) -> tuple[float, float]:
            value = tuple(float(v) for v in getattr(self.cfg.robot, name, (1.0, 1.0)))
            if len(value) != 2 or value[0] <= 0.0 or value[1] < value[0]:
                raise ValueError(f"robot.{name} must be a positive (min,max) pair, got {value}")
            return value

        self._action_response_fn_scale_range = _scale_range("action_response_fn_scale_range")
        self._action_response_zeta_scale_range = _scale_range("action_response_zeta_scale_range")
        self._action_response_gain_scale_range = _scale_range("action_response_gain_scale_range")
        delay_jitter = tuple(
            float(v) for v in (getattr(self.cfg.robot, "action_response_delay_jitter_s", ()) or ())
        )
        if delay_jitter and (len(delay_jitter) != self.num_actions or any(v < 0.0 for v in delay_jitter)):
            raise ValueError(
                "robot.action_response_delay_jitter_s must be empty or contain one "
                f"non-negative value per action joint, got {delay_jitter}"
            )
        self._action_response_delay_jitter = (
            None
            if not delay_jitter
            else torch.tensor(
                delay_jitter,
                device=self.device,
                dtype=self.robot.data.default_joint_pos.dtype,
            ).unsqueeze(0)
        )
        bias_jitter = tuple(
            float(v) for v in (getattr(self.cfg.robot, "action_response_bias_jitter_rad", ()) or ())
        )
        if bias_jitter and (len(bias_jitter) != self.num_actions or any(v < 0.0 for v in bias_jitter)):
            raise ValueError(
                "robot.action_response_bias_jitter_rad must be empty or contain one "
                f"non-negative value per action joint, got {bias_jitter}"
            )
        self._action_response_bias_jitter = (
            None
            if not bias_jitter
            else torch.tensor(
                bias_jitter,
                device=self.device,
                dtype=self.robot.data.default_joint_pos.dtype,
            ).unsqueeze(0)
        )
        self._action_response_omega = torch.zeros_like(self._action_response_fn_hz)
        max_delay_s = self._action_response_delay_s_base
        if self._action_response_delay_jitter is not None:
            max_delay_s = max_delay_s + self._action_response_delay_jitter
        self._action_response_delay_buffer_len = (
            int(torch.ceil(torch.max(max_delay_s) / self.physics_dt).item()) + 1
        )
        self._action_response_delay_steps = torch.zeros(
            self.num_envs,
            self.num_actions,
            device=self.device,
            dtype=torch.long,
        )
        self._action_response_delay_index = 0
        self._action_response_x_rel = torch.zeros_like(self._last_processed_actions)
        self._action_response_v_rel = torch.zeros_like(self._last_processed_actions)
        self._action_response_delay_buffer = torch.zeros(
            self._action_response_delay_buffer_len,
            self.num_envs,
            self.num_actions,
            device=self.device,
            dtype=self.robot.data.default_joint_pos.dtype,
        )
        self._reset_action_response_model()

    def _reset_action_response_model(self, env_ids=None):
        if not getattr(self, "_action_response_model_enable", False):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._randomize_action_response_model(env_ids)
        current_joint_pos = self.robot.data.joint_pos[:, self.action_joint_ids]
        gain = torch.clamp(self._action_response_gain, min=1.0e-6)
        x_rel = (current_joint_pos - self._action_response_u_mean - self._action_response_bias) / gain
        steady_raw_command = self._action_response_u_mean + x_rel
        self._action_response_x_rel[env_ids] = x_rel[env_ids]
        self._action_response_v_rel[env_ids] = 0.0
        self._action_response_delay_buffer[:, env_ids, :] = steady_raw_command[env_ids].unsqueeze(0)
        self.action_response_targets[env_ids] = current_joint_pos[env_ids]

    def _randomize_action_response_model(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)

        def _sample_scale(value_range, dtype):
            lo, hi = value_range
            return lo + torch.rand(count, 1, device=self.device, dtype=dtype) * (hi - lo)

        dtype = self._action_response_fn_hz.dtype
        self._action_response_fn_hz[env_ids] = (
            self._action_response_fn_hz_base
            * _sample_scale(self._action_response_fn_scale_range, dtype)
        )
        self._action_response_zeta[env_ids] = (
            self._action_response_zeta_base
            * _sample_scale(self._action_response_zeta_scale_range, dtype)
        )
        self._action_response_gain[env_ids] = (
            self._action_response_gain_base
            * _sample_scale(self._action_response_gain_scale_range, dtype)
        )
        delay_s = self._action_response_delay_s_base.expand(count, -1)
        if self._action_response_delay_jitter is not None:
            delay_noise = (
                2.0 * torch.rand(count, self.num_actions, device=self.device, dtype=dtype) - 1.0
            ) * self._action_response_delay_jitter
            delay_s = torch.clamp(delay_s + delay_noise, min=0.0)
        self._action_response_delay_steps[env_ids] = torch.round(
            delay_s / self.physics_dt
        ).to(dtype=torch.long)
        self._action_response_bias[env_ids] = self._action_response_bias_base
        if self._action_response_bias_jitter is not None:
            jitter = (2.0 * torch.rand(count, self.num_actions, device=self.device, dtype=dtype) - 1.0)
            self._action_response_bias[env_ids] += jitter * self._action_response_bias_jitter
        self._action_response_omega[env_ids] = (
            2.0 * torch.tensor(np.pi, device=self.device, dtype=dtype)
            * torch.clamp(self._action_response_fn_hz[env_ids], min=1.0e-6)
        )

    def _delayed_action_response_command(self, raw_command: torch.Tensor) -> torch.Tensor:
        write_idx = self._action_response_delay_index
        self._action_response_delay_buffer[write_idx].copy_(raw_command)
        read_idx = (
            write_idx - self._action_response_delay_steps
        ) % self._action_response_delay_buffer_len
        env_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        action_idx = torch.arange(self.num_actions, device=self.device).unsqueeze(0)
        delayed = self._action_response_delay_buffer[read_idx, env_idx, action_idx]
        self._action_response_delay_index = (write_idx + 1) % self._action_response_delay_buffer_len
        return delayed

    def _apply_action_response_model(self, raw_command: torch.Tensor, dt: float) -> torch.Tensor:
        if not getattr(self, "_action_response_model_enable", False):
            self.action_response_targets = raw_command
            return raw_command

        delayed_command = self._delayed_action_response_command(raw_command)
        u_rel = delayed_command - self._action_response_u_mean
        accel = (
            torch.square(self._action_response_omega) * (u_rel - self._action_response_x_rel)
            - 2.0 * self._action_response_zeta * self._action_response_omega * self._action_response_v_rel
        )
        self._action_response_v_rel.add_(accel * dt)
        self._action_response_x_rel.add_(self._action_response_v_rel * dt)
        response = (
            self._action_response_u_mean
            + self._action_response_bias
            + self._action_response_gain * self._action_response_x_rel
        )
        self.action_response_targets = response
        return response

    def _apply_non_policy_joint_targets(self) -> None:
        """Subclass hook for passive joints that must be held outside the policy action set."""
        return

    def step(self, actions: torch.Tensor):

        self._apply_effort_curriculum()
        delayed_actions = self.action_buffer.compute(actions)

        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos[:, self.action_joint_ids]
        processed_actions = self._apply_action_target_rate_limit(processed_actions)
        processed_actions = self._apply_action_target_lowpass(processed_actions)
        self.processed_actions = processed_actions   # commanded joint target (for joint_pos_target_limits reward)

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            action_response_targets = self._apply_action_response_model(processed_actions, self.physics_dt)
            self.robot.set_joint_position_target(action_response_targets, self.action_joint_ids)
            self._apply_non_policy_joint_targets()
            # ! Aerodynamics: Step : BEGIN
            # ! Step before self.scene.write_data_to_sim(), update per decimation step
            self.aero.apply_to_rigid_object(self.ball)
            # ! Aerodynamics: Step : END
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self.compute_perception()
            self.compute_paddle_touch()

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.ball_episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.compute_intermediate_values()

        # perf-gated serve curriculum: advance difficulty only while the success-return rate is
        # in the window. Step sized so a sustained pass takes serve_c_ramp_iters iters for c:0->1.
        # Threshold overridable live via env TT_SUCC_WINDOW (so it can be relaxed if it stalls).
        if getattr(self.cfg.ball, "serve_curriculum_perf_gated", False):
            _win = float(os.environ.get("TT_SUCC_WINDOW", str(getattr(self.cfg.ball, "serve_succ_window", 0.6))))
            _nspe = int(getattr(self.cfg, "num_steps_per_env", 24) or 24)
            if self.serve_c < 1.0 and self.succ_ema >= _win:
                _ramp = max(1, int(getattr(self.cfg.ball, "serve_c_ramp_iters", 10000)))
                self.serve_c = min(1.0, self.serve_c + 1.0 / float(_ramp * _nspe))
            self._curri_log_ctr += 1
            if self._curri_log_ctr % 1200 == 0:   # ~ every 50 iters
                print(f"[curriculum] serve_c={self.serve_c:.3f} succ_ema={self.succ_ema:.3f} (hit={self.hit_ema:.3f} opo={self.opo_ema:.3f}) win={_win:.2f}", flush=True)

        # Check for balls on floor and reset them without resetting the entire environment
        ball_on_floor = self.ball.data.root_pos_w[:, 2] < 0.1  # Adjust threshold as needed
        ball_timeout = self.ball_episode_length_buf >= self.max_ball_episode_length
        ball_reset_condition = ball_on_floor | ball_timeout
        # ball_reset_condition = ball_timeout
        ball_reset_ids = ball_reset_condition.nonzero(as_tuple=False).flatten()
        # expose ball reset ids for downstream modules (e.g., predictor)
        # always set as a tensor on device to avoid attribute errors
        self.ball_reset_ids = ball_reset_ids
        # ball_reset_ids = ball_on_floor.nonzero(as_tuple=False).flatten()
        if len(ball_reset_ids) > 0:
            self.reset_ball(ball_reset_ids)

        self.reset_buf, self.time_out_buf = self.check_reset()

        self.curriculum_manager.compute()
        
        reward_buf = self.reward_manager.compute(self.step_dt)

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        # net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # reset_buf = torch.any(
        #     torch.max(
        #         torch.norm(
        #             net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
        #             dim=-1,
        #         ),
        #         dim=1,
        #     )[0]
        #     > 1.0,
        #     dim=1,
        # )
        
        # reset_buf = (
        #     (self.robot_pos[..., 2] < 0.50) |
        #     (self.robot_pos[..., 0] < -3.6) |
        #     (self.robot_pos[..., 0] > -1.35) |
        #     (self.robot_pos[..., 1] < -2.0) |
        #     (self.robot_pos[..., 1] > 2.0)
        # )
        reset_buf = (
            (self.robot_pos[..., 2] < 0.50) |
            (self.robot_pos[..., 0] < -3.6) |
            (self.robot_pos[..., 0] > -1.35) |
            (self.robot_pos[..., 1] < -1.1) |
            (self.robot_pos[..., 1] > 1.1)
        )       
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        time_out_buf |= self.ball_reset_counter >self.max_ball_serve_per_episode
        # print(self.episode_length_buf,self.max_episode_length)
        # print(self.max_episode_length_s,self.step_dt)
        # print ('time_out_buf',time_out_buf)
        reset_buf |= time_out_buf
        # print ('reset_buf',reset_buf)
        return reset_buf, time_out_buf

    def compute_perception(self):
        self.ball_pos = self.ball.data.root_pos_w - self.scene.env_origins  # Local (offset) position
        self.robot_pos = self.robot.data.root_link_pos_w - self.table.data.root_link_pos_w  # Local (offset) position wrt table
        self.current_perception = torch.cat(
            [
                self.ball_pos,
                # self.ball_linvel,
                self.robot_pos,
            ],
            dim=-1,
        )
        if self._camera_observation_enable:
            camera_ball_pos = self._compute_camera_observation()
            self.delayed_perception = torch.cat([camera_ball_pos, self.robot_pos], dim=-1)
        else:
            self.delayed_perception = self.perception_buffer.compute(self.current_perception)

    def _init_camera_observation_model(self) -> None:
        cfg = self.cfg.domain_rand.camera_observation
        self._camera_observation_enable = bool(getattr(cfg, "enable", False))
        self.camera_track_valid = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.camera_observation_age_s = torch.zeros(self.num_envs, device=self.device)
        self._camera_estimated_ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        if not self._camera_observation_enable:
            return

        if self.cfg.domain_rand.perception_delay.enable:
            raise ValueError(
                "camera_observation and legacy perception_delay are mutually exclusive; "
                "the camera model already owns ball transport latency"
            )
        fps = float(cfg.fps)
        if fps <= 0.0:
            raise ValueError(f"camera_observation.fps must be positive, got {fps}")
        self._camera_period_s = 1.0 / fps
        self._camera_acquire_frames = max(1, int(cfg.acquire_frames))
        self._camera_reset_gap_s = max(self._camera_period_s, float(cfg.reset_gap_s))
        self._camera_coast_max_s = max(0.0, float(cfg.coast_max_s))
        self._camera_dropout_prob = min(1.0, max(0.0, float(cfg.dropout_prob)))
        self._camera_extrapolate = bool(cfg.extrapolate_to_now)
        self._camera_filter_alpha = min(1.0, max(0.0, float(cfg.filter_alpha)))
        self._camera_filter_beta = min(1.0, max(0.0, float(cfg.filter_beta)))
        self._camera_max_extrapolation_s = max(0.0, float(cfg.max_extrapolation_s))
        self._camera_gravity = float(cfg.gravity_mps2)
        self._camera_table_bounce = bool(cfg.table_bounce_enable)
        self._camera_table_z = float(cfg.table_ball_center_z)
        self._camera_table_restitution = float(cfg.table_restitution)
        self._camera_x_range = tuple(float(v) for v in cfg.x_range)
        self._camera_y_range = tuple(float(v) for v in cfg.y_range)
        self._camera_z_range = tuple(float(v) for v in cfg.z_range)
        self._camera_position_noise_std = torch.tensor(
            tuple(float(v) for v in cfg.position_noise_std),
            device=self.device,
            dtype=self.robot.data.default_joint_pos.dtype,
        ).reshape(1, 3)

        ranges = tuple(tuple(float(v) for v in pair) for pair in cfg.latency_ranges_s)
        weights = tuple(float(v) for v in cfg.latency_mode_weights)
        if not ranges or len(ranges) != len(weights):
            raise ValueError("camera latency ranges and weights must have the same non-zero length")
        if any(len(pair) != 2 or pair[0] < 0.0 or pair[1] < pair[0] for pair in ranges):
            raise ValueError(f"invalid camera latency ranges: {ranges}")
        weight_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)
        if bool((weight_tensor < 0.0).any()) or float(weight_tensor.sum()) <= 0.0:
            raise ValueError(f"invalid camera latency mode weights: {weights}")
        self._camera_latency_ranges = torch.tensor(
            ranges, device=self.device, dtype=self.robot.data.default_joint_pos.dtype
        )
        self._camera_latency_weights = weight_tensor / weight_tensor.sum()
        max_latency_s = max(pair[1] for pair in ranges)
        self._camera_history_len = int(np.ceil(max_latency_s / self.physics_dt)) + 3
        state_dtype = self.robot.data.default_joint_pos.dtype
        self._camera_truth_history = torch.zeros(
            self._camera_history_len, self.num_envs, 6, device=self.device, dtype=state_dtype
        )
        self._camera_generation_history = torch.full(
            (self._camera_history_len, self.num_envs),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self._camera_history_index = 0
        self._camera_generation = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._camera_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._camera_latency_s = torch.zeros(self.num_envs, device=self.device, dtype=state_dtype)
        self._camera_phase_s = torch.zeros(self.num_envs, device=self.device, dtype=state_dtype)
        self._camera_raw_gap_s = torch.full(
            (self.num_envs,), self._camera_reset_gap_s, device=self.device, dtype=state_dtype
        )
        self._camera_sample_age_s = torch.zeros(self.num_envs, device=self.device, dtype=state_dtype)
        self._camera_acquire_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._camera_source_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=state_dtype)
        self._camera_source_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=state_dtype)
        self._camera_candidate_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=state_dtype)
        self.camera_track_valid.zero_()

    def _reset_camera_observation(self, env_ids) -> None:
        if not getattr(self, "_camera_observation_enable", False) or len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._camera_generation[env_ids] += 1
        probs = self._camera_latency_weights.expand(len(env_ids), -1)
        self._camera_mode[env_ids] = torch.multinomial(probs, 1).squeeze(-1)
        ranges = self._camera_latency_ranges[self._camera_mode[env_ids]]
        self._camera_latency_s[env_ids] = ranges[:, 0] + torch.rand(
            len(env_ids), device=self.device, dtype=ranges.dtype
        ) * (ranges[:, 1] - ranges[:, 0])
        self._camera_phase_s[env_ids] = torch.rand(
            len(env_ids), device=self.device, dtype=self._camera_phase_s.dtype
        ) * self._camera_period_s
        self._camera_raw_gap_s[env_ids] = self._camera_reset_gap_s
        self._camera_sample_age_s[env_ids] = 0.0
        self._camera_acquire_count[env_ids] = 0
        self.camera_track_valid[env_ids] = False
        self.camera_observation_age_s[env_ids] = 0.0
        self._camera_source_pos[env_ids] = 0.0
        self._camera_source_vel[env_ids] = 0.0
        self._camera_candidate_pos[env_ids] = 0.0
        self._camera_estimated_ball_pos[env_ids] = 0.0

    def _propagate_camera_ball(
        self, pos: torch.Tensor, vel: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match deploy's gravity plus at-most-one table-bounce extrapolator."""
        dt = torch.clamp(dt, min=0.0)
        out_pos = pos.clone()
        out_vel = vel.clone()
        pre = dt
        post = torch.zeros_like(dt)
        if self._camera_table_bounce:
            a = 0.5 * self._camera_gravity
            b = vel[:, 2]
            c = pos[:, 2] - self._camera_table_z
            disc = b.square() - 4.0 * a * c
            root = torch.sqrt(torch.clamp(disc, min=0.0))
            denom = 2.0 * a
            t1 = (-b - root) / denom
            t2 = (-b + root) / denom
            inf = torch.full_like(dt, float("inf"))
            impact = torch.minimum(
                torch.where(t1 >= -1.0e-6, torch.clamp(t1, min=0.0), inf),
                torch.where(t2 >= -1.0e-6, torch.clamp(t2, min=0.0), inf),
            )
            impact_vz = b + self._camera_gravity * impact
            bounced = (
                (disc >= 0.0)
                & torch.isfinite(impact)
                & (impact <= dt)
                & (impact_vz < 0.0)
                & (pos[:, 2] >= self._camera_table_z - 0.02)
            )
            pre = torch.where(bounced, impact, dt)
            post = torch.where(bounced, dt - impact, torch.zeros_like(dt))
        else:
            bounced = torch.zeros_like(dt, dtype=torch.bool)

        out_pos[:, 0] = pos[:, 0] + vel[:, 0] * dt
        out_pos[:, 1] = pos[:, 1] + vel[:, 1] * dt
        out_pos[:, 2] = pos[:, 2] + vel[:, 2] * pre + 0.5 * self._camera_gravity * pre.square()
        out_vel[:, 2] = vel[:, 2] + self._camera_gravity * pre
        if bool(bounced.any()):
            bounce_vz = -self._camera_table_restitution * out_vel[:, 2]
            post_z = self._camera_table_z + bounce_vz * post + 0.5 * self._camera_gravity * post.square()
            post_vz = bounce_vz + self._camera_gravity * post
            out_pos[:, 2] = torch.where(bounced, post_z, out_pos[:, 2])
            out_vel[:, 2] = torch.where(bounced, post_vz, out_vel[:, 2])
        return out_pos, out_vel

    def _compute_camera_observation(self) -> torch.Tensor:
        write_idx = self._camera_history_index
        truth_state = torch.cat([self.ball_pos, self.ball.data.root_lin_vel_w], dim=-1)
        self._camera_truth_history[write_idx].copy_(truth_state)
        self._camera_generation_history[write_idx].copy_(self._camera_generation)

        self._camera_phase_s.sub_(self.physics_dt)
        self._camera_raw_gap_s.add_(self.physics_dt)
        self._camera_sample_age_s.add_(self.physics_dt)
        due = self._camera_phase_s <= 0.0
        if bool(due.any()):
            due_ids = due.nonzero(as_tuple=False).flatten()
            self._camera_phase_s[due_ids] += self._camera_period_s
            # A camera run normally stays in one latency phase (fresh lock or a
            # buffered/backlog phase).  Sample that phase once per rally rather
            # than adding unrealistic frame-to-frame timestamp reordering.
            latency_s = self._camera_latency_s[due_ids]
            delay_steps = torch.round(latency_s / self.physics_dt).long()
            delay_steps = torch.clamp(delay_steps, min=0, max=self._camera_history_len - 2)
            read_idx = (write_idx - delay_steps) % self._camera_history_len
            state = self._camera_truth_history[read_idx, due_ids]
            generation = self._camera_generation_history[read_idx, due_ids]
            source_pos = state[:, :3]
            finite = torch.isfinite(state).all(dim=-1)
            in_view = (
                (source_pos[:, 0] >= self._camera_x_range[0])
                & (source_pos[:, 0] <= self._camera_x_range[1])
                & (source_pos[:, 1] >= self._camera_y_range[0])
                & (source_pos[:, 1] <= self._camera_y_range[1])
                & (source_pos[:, 2] >= self._camera_z_range[0])
                & (source_pos[:, 2] <= self._camera_z_range[1])
            )
            accepted = finite & in_view & (generation == self._camera_generation[due_ids])
            if self._camera_dropout_prob > 0.0:
                accepted &= torch.rand(len(due_ids), device=self.device) >= self._camera_dropout_prob
            accepted_ids = due_ids[accepted]
            if len(accepted_ids) > 0:
                accepted_state = state[accepted]
                measurement = (
                    accepted_state[:, :3]
                    + torch.randn_like(accepted_state[:, :3]) * self._camera_position_noise_std
                )
                previous_count = self._camera_acquire_count[accepted_ids]
                sample_dt = torch.clamp(
                    self._camera_raw_gap_s[accepted_ids], min=self._camera_period_s
                )

                first = previous_count == 0
                if bool(first.any()):
                    first_ids = accepted_ids[first]
                    self._camera_candidate_pos[first_ids] = measurement[first]

                second = previous_count == 1
                if bool(second.any()):
                    second_ids = accepted_ids[second]
                    second_dt = sample_dt[second].unsqueeze(-1)
                    self._camera_source_pos[second_ids] = measurement[second]
                    self._camera_source_vel[second_ids] = (
                        measurement[second] - self._camera_candidate_pos[second_ids]
                    ) / second_dt

                tracking = previous_count >= 2
                if bool(tracking.any()):
                    tracking_ids = accepted_ids[tracking]
                    tracking_dt = sample_dt[tracking]
                    predicted_pos, predicted_vel = self._propagate_camera_ball(
                        self._camera_source_pos[tracking_ids],
                        self._camera_source_vel[tracking_ids],
                        tracking_dt,
                    )
                    innovation = measurement[tracking] - predicted_pos
                    self._camera_source_pos[tracking_ids] = (
                        predicted_pos + self._camera_filter_alpha * innovation
                    )
                    self._camera_source_vel[tracking_ids] = (
                        predicted_vel
                        + self._camera_filter_beta * innovation / tracking_dt.unsqueeze(-1)
                    )
                self._camera_sample_age_s[accepted_ids] = delay_steps[accepted].to(
                    self._camera_sample_age_s.dtype
                ) * self.physics_dt
                self._camera_raw_gap_s[accepted_ids] = 0.0
                self._camera_acquire_count[accepted_ids] = torch.clamp(
                    self._camera_acquire_count[accepted_ids] + 1,
                    max=self._camera_acquire_frames,
                )
                acquired = self._camera_acquire_count[accepted_ids] >= self._camera_acquire_frames
                self.camera_track_valid[accepted_ids[acquired]] = True

        stale = self._camera_raw_gap_s > self._camera_coast_max_s
        self.camera_track_valid[stale] = False
        reset_track = self._camera_raw_gap_s > self._camera_reset_gap_s
        self._camera_acquire_count[reset_track] = 0
        if self._camera_extrapolate:
            estimated, _ = self._propagate_camera_ball(
                self._camera_source_pos,
                self._camera_source_vel,
                torch.clamp(self._camera_sample_age_s, max=self._camera_max_extrapolation_s),
            )
        else:
            estimated = self._camera_source_pos
        self._camera_estimated_ball_pos.copy_(estimated)
        self.camera_observation_age_s.copy_(self._camera_sample_age_s)
        self._camera_history_index = (write_idx + 1) % self._camera_history_len
        return self._camera_estimated_ball_pos

    def get_predictor_ball_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the same timestamped/coasted ball stream used by deployment."""
        if getattr(self, "_camera_observation_enable", False):
            return self._camera_estimated_ball_pos, self.camera_track_valid
        valid = torch.isfinite(self.ball_pos).all(dim=-1) & (self.ball_pos[:, 2] > 0.1)
        return self.ball_pos, valid

    def compute_paddle_touch(self):
        self.ball_global_pos = self.ball.data.root_pos_w 

        # --- Compute Paddle Position and Contact ---
        paddle_index = self._paddle_body_id  # resolved once in __init__ from cfg
        paddle_pos = self.robot.data.body_pos_w[:, paddle_index, :]
        # print("paddle_pos: ", paddle_pos[0, :])
        # print("ball_pos: ", self.ball_global_pos[0,:])

        paddle_quat = self.robot.data.body_quat_w[:, paddle_index, :]
        # 1) Normalize the quaternion (just in case):
        paddle_quat = paddle_quat / paddle_quat.norm(dim=1, keepdim=True)
        # 2) Build the local offset (0, -0.345, 0) and expand to (N,3):
        local_offset = (
            torch.tensor(
                self.cfg.robot.paddle_offset,
                device=paddle_pos.device,
                dtype=paddle_pos.dtype,
            )
            .unsqueeze(0)
            .expand_as(paddle_pos)
        )
        rotated_offset: torch.Tensor = math_utils.quat_apply(paddle_quat, local_offset)
        # 4) Compute your touch point:
        self.paddle_touch_point = paddle_pos + rotated_offset # paddle_position in the world frame.
        self.paddle_touch_point_vel = self.robot.data.body_lin_vel_w[:, paddle_index, :]
        # 5) Compute touch reward:

        distance = torch.norm(self.ball_global_pos - self.paddle_touch_point, dim=1) - 0.02 # corrected for ball radius
        self.paddel_ball_distance = distance
        debug_hx = float(getattr(self.cfg.robot, "hit_plane_x", -1.42))
        if os.environ.get("TT_DEBUG_OFFSET") and abs(float(self.ball_global_pos[0,0]) - debug_hx) < 0.08 and float(self.ball_global_pos[0,2]) > 0.6:
            b = self.ball_global_pos[0].tolist(); tp = self.paddle_touch_point[0].tolist()
            fut = self.ball_future_pose[0].tolist(); pp = self.paddle_pos[0].tolist()
            print(f"[OFFSET] ball_now_z={b[2]:.2f} target_z={fut[2]:.2f} paddle_z={pp[2]:.2f} | paddle-ball={pp[2]-b[2]:+.2f} paddle-target={pp[2]-fut[2]:+.2f}", flush=True)
        contact_score = (
            self.cfg.ball.contact_threshold - distance
        ) / self.cfg.ball.contact_threshold
        sweet_score = self._compute_sweet_contact_score(paddle_quat, contact_score)
        ball_linvel_now = self.ball.data.root_lin_vel_w
        ball_vx_before_step = self.ball_linvel_prev[:, 0]
        self.ball_contact = torch.clamp(contact_score, min=0.0, max=1.0) # determine if in contact region
        self.ball_contact_raw_rew = torch.maximum(self.ball_contact_raw_rew, self.ball_contact)
        if getattr(self.cfg.ball, "require_active_contact", False):
            ball_local = self.ball_global_pos - self.scene.env_origins
            paddle_speed = torch.linalg.norm(self.paddle_touch_point_vel, dim=1)
            paddle_forward_speed = self.paddle_touch_point_vel[:, 0]
            min_speed = float(getattr(self.cfg.ball, "active_contact_min_paddle_speed", 0.0))
            min_forward = float(getattr(self.cfg.ball, "active_contact_min_forward_speed", -100.0))
            speed_score = torch.clamp((paddle_speed - min_speed) / max(min_speed, 0.1), min=0.0, max=1.0)
            forward_score = torch.clamp(
                (paddle_forward_speed - min_forward) / max(abs(min_forward), 0.1),
                min=0.0,
                max=1.0,
            )
            own_bounce_ok = self.has_touch_own_table_prev
            if not getattr(self.cfg.ball, "active_contact_require_own_bounce", False):
                own_bounce_ok = torch.ones_like(own_bounce_ok, dtype=torch.bool)
            hx = self.cfg.robot.hit_plane_x
            plane_margin = float(getattr(self.cfg.ball, "active_contact_hit_plane_margin", 0.28))
            plane_quality = self._compute_hit_plane_contact_score(ball_local[:, 0])
            contact_candidate = contact_score > 0.0
            active_hit = (
                contact_candidate
                & (~self.has_touch_paddle)
                & own_bounce_ok
                & (ball_vx_before_step < -0.05)
                & (torch.abs(ball_local[:, 0] - hx) <= plane_margin)
                & (plane_quality > 0.0)
                & (ball_local[:, 2] > 0.75)
                & (paddle_speed >= min_speed)
                & (paddle_forward_speed >= min_forward)
            )
            active_score = self.ball_contact * torch.minimum(speed_score, forward_score) * plane_quality
            self.ball_contact_rew = torch.maximum(
                self.ball_contact_rew,
                torch.where(active_hit, active_score, torch.zeros_like(active_score)),
            )
            self.active_paddle_hit = active_hit
            self._record_hit_plane_contact(active_hit, plane_quality)
            self._record_sweet_contact(active_hit, sweet_score, plane_quality)
            self._debug_sweet_spot(active_hit, paddle_quat, contact_score, paddle_speed, paddle_forward_speed)
            new_hits = active_hit
        else:
            self.ball_contact_rew = torch.maximum(self.ball_contact_rew, self.ball_contact) # finds reward for closest ball paddle distance
            # self.ball_contact = torch.where(contact_score > 0.0, torch.ones_like(contact_score), torch.zeros_like(contact_score))
            # self.ball_contact = self.ball_contact * ~self.has_touch_paddle # mask invalid if previous ball_contact True
            # new_hits = contact_score > 0  # Tensor[N] bool
            new_hits = (contact_score > 0) & (self.ball_contact < self.ball_contact_rew)  # Tensor[N] bool
            self.active_paddle_hit = new_hits
            neutral_plane_quality = torch.ones_like(sweet_score)
            self._record_hit_plane_contact(new_hits, neutral_plane_quality)
            self._record_sweet_contact(new_hits, sweet_score, neutral_plane_quality)
            self._debug_sweet_spot(new_hits, paddle_quat, contact_score)
        still_false = ~self.has_touch_paddle  # Tensor[N] bool
        self.has_touch_paddle[still_false] = new_hits[still_false] # set has_touch_paddle True for env with ball_contact True
        self.ball_linvel_prev.copy_(ball_linvel_now)

    def _compute_hit_plane_contact_score(self, ball_x_local: torch.Tensor) -> torch.Tensor:
        radius = float(getattr(self.cfg.ball, "hit_plane_contact_radius", 0.0) or 0.0)
        if radius <= 0.0:
            return torch.ones_like(ball_x_local)
        core = float(getattr(self.cfg.ball, "hit_plane_contact_core_radius", 0.0) or 0.0)
        core = min(max(core, 0.0), radius)
        dist = torch.abs(ball_x_local - float(self.cfg.robot.hit_plane_x))
        denom = max(radius - core, 1e-6)
        score = torch.clamp((radius - dist) / denom, min=0.0, max=1.0)
        return torch.where(dist <= core, torch.ones_like(score), score)

    def _compute_sweet_contact_score(self, paddle_quat: torch.Tensor, contact_score: torch.Tensor) -> torch.Tensor:
        radius = float(getattr(self.cfg.ball, "sweet_contact_radius", 0.0) or 0.0)
        if radius <= 0.0:
            return torch.zeros_like(contact_score)
        core = float(getattr(self.cfg.ball, "sweet_contact_core_radius", 0.0) or 0.0)
        core = min(max(core, 0.0), radius)
        face_axis = str(getattr(self.cfg.ball, "sweet_contact_face_axis", "y")).lstrip("-")
        plane_axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}.get(face_axis, (0, 2))
        ball_rel_w = self.ball_global_pos - self.paddle_touch_point
        ball_rel_l = math_utils.quat_apply_inverse(paddle_quat, ball_rel_w)
        plane_dist = torch.linalg.norm(ball_rel_l[:, plane_axes], dim=1)
        denom = max(radius - core, 1e-6)
        score = torch.clamp((radius - plane_dist) / denom, min=0.0, max=1.0)
        score = torch.where(plane_dist <= core, torch.ones_like(score), score)
        return torch.where(contact_score > 0.0, score, torch.zeros_like(score))

    def _record_hit_plane_contact(self, hit_mask: torch.Tensor, plane_score: torch.Tensor):
        first_hits = hit_mask & (~self.has_touch_paddle)
        if not torch.any(first_hits):
            return
        self.paddle_hit_plane_latch = torch.where(first_hits, plane_score, self.paddle_hit_plane_latch)

    def _record_sweet_contact(
        self,
        hit_mask: torch.Tensor,
        sweet_score: torch.Tensor,
        plane_score: torch.Tensor,
    ):
        first_hits = hit_mask & (~self.has_touch_paddle)
        if not torch.any(first_hits):
            return
        gated_sweet = sweet_score * plane_score
        score = torch.where(first_hits, gated_sweet, torch.zeros_like(gated_sweet))
        self.paddle_sweet_contact_rew = torch.maximum(self.paddle_sweet_contact_rew, score)
        self.paddle_sweet_contact_latch = torch.where(first_hits, sweet_score, self.paddle_sweet_contact_latch)

    def _debug_sweet_spot(
        self,
        hit_mask: torch.Tensor,
        paddle_quat: torch.Tensor,
        contact_score: torch.Tensor,
        paddle_speed: torch.Tensor | None = None,
        paddle_forward_speed: torch.Tensor | None = None,
    ):
        if not os.environ.get("TT_DEBUG_SWEET"):
            return
        hit_ids = torch.nonzero(hit_mask, as_tuple=False).flatten()
        if hit_ids.numel() == 0:
            return
        max_print = int(os.environ.get("TT_DEBUG_SWEET_MAX", "4"))
        for env_id_t in hit_ids[:max_print]:
            env_id = int(env_id_t.item())
            ball_rel_w = self.ball_global_pos[env_id : env_id + 1] - self.paddle_touch_point[env_id : env_id + 1]
            ball_rel_l = math_utils.quat_apply_inverse(paddle_quat[env_id : env_id + 1], ball_rel_w)[0]
            r_xy = torch.linalg.norm(ball_rel_l[0:2])
            r_xz = torch.linalg.norm(ball_rel_l[[0, 2]])
            r_yz = torch.linalg.norm(ball_rel_l[1:3])
            center_dist = torch.linalg.norm(ball_rel_w[0])
            speed = float("nan") if paddle_speed is None else float(paddle_speed[env_id])
            forward = float("nan") if paddle_forward_speed is None else float(paddle_forward_speed[env_id])
            print(
                "[SWEET] "
                f"env={env_id} center_dist={float(center_dist):.4f} "
                f"score={float(contact_score[env_id]):.3f} "
                f"local=({float(ball_rel_l[0]):+.4f},{float(ball_rel_l[1]):+.4f},{float(ball_rel_l[2]):+.4f}) "
                f"r_xy={float(r_xy):.4f} r_xz={float(r_xz):.4f} r_yz={float(r_yz):.4f} "
                f"speed={speed:.3f} forward={forward:.3f}",
                flush=True,
            )

    def compute_intermediate_values(self):
        # print("has_touch_paddle", self.has_touch_paddle)
        # self.ball_pos = self.ball.data.root_pos_w - self.table.data.root_pos_w  # Local (offset) position wrt table
        self.ball_quat = self.ball.data.root_quat_w
        self.ball_vel = self.ball.data.root_vel_w
        self.ball_linvel = self.ball.data.root_lin_vel_w
        self.ball_angvel = self.ball.data.root_ang_vel_w
        self.robot_linvel= self.robot.data.root_lin_vel_w
        self.ball_contact_rew = self.ball_contact_rew * (self.has_touch_paddle * ~self.has_touch_paddle_rew) # Mask if previously already gained reward
        self.paddle_sweet_contact_rew = self.paddle_sweet_contact_rew * (self.has_touch_paddle * ~self.has_touch_paddle_rew)
        self.ball_landing_dis_rew = self.has_touch_paddle & ~self.has_touch_paddle_rew # Set True if has_touch_paddle_rew is True, compute landing dis reward once, set False if previously True

        # --- Compute Contact with table ---
        bx, by, bz = (self.ball_pos[:, 0], self.ball_pos[:, 1], self.ball_pos[:, 2])
        # 3) Load your table‐contact bounds from self
        tcx_min, tcx_max = self.cfg.table.table_opponent_contact_x
        tcy_min, tcy_max = self.cfg.table.table_opponent_contact_y
        tcz_min, tcz_max = self.cfg.table.table_opponent_contact_z

        ncx_min, ncx_max = self.cfg.table.table_own_contact_x
        ncy_min, ncy_max = self.cfg.table.table_own_contact_y
        ncz_min, ncz_max = self.cfg.table.table_own_contact_z
        # print(f'bz{bz}')
        # print(f'ncz_max{ncz_max}')
        # 4) Build masks
        self.has_touch_opponent_table_just_now = (
            (bx >= tcx_min)
            & (bx <= tcx_max)
            & (by >= tcy_min)
            & (by <= tcy_max)
            & (bz >= tcz_min)
            & (bz <= tcz_max)
        )
        has_touch_own_table_just_now = (
            (bx >= ncx_min)
            & (bx <= ncx_max)
            & (by >= ncy_min)
            & (by <= ncy_max)
            & (bz >= ncz_min)
            & (bz <= ncz_max)
            # & (~self.has_touch_own_table_prev) # have not touched own table before
            # bz <= ncz_max
        )
        # print(f'env.has_touch_own_table_just_now={has_touch_own_table_just_now}')
        # idle10: capture "ball bounced on own table in a PRIOR step" BEFORE updating the
        # latch, so the first bounce itself is never mistaken for a second bounce.
        bounced_before = self.has_touch_own_table_prev.clone()
        self.has_touch_own_table_prev = (
            self.has_touch_own_table_prev | has_touch_own_table_just_now
        )
        self.has_touch_opo_table_prev = (
            self.has_touch_opo_table_prev | self.has_touch_opponent_table_just_now
        )
        self.has_return_own_table2_prev = (
            (self.has_touch_own_table_prev & self.has_touch_paddle & has_touch_own_table_just_now) | self.has_return_own_table2_prev
        )
        # idle10 dead-ball: a SERVED ball that bounced on own table, rose back above it, then
        # touches own table AGAIN without ever being hit = double bounce = dead (point lost) ->
        # stop chasing it. (has_return_own_table2 above needs a paddle hit first, so it does NOT
        # cover the missed-serve double bounce — this does.) bz/ncz_max are in scope from above.
        self.left_after_bounce = self.left_after_bounce | (bounced_before & (bz > ncz_max))
        self.has_second_bounce = self.has_second_bounce | (
            self.left_after_bounce & has_touch_own_table_just_now & (~self.has_touch_paddle)
        )

        self.touched_paddel_no_bounce_table=(
            self.has_touch_paddle & ~(self.has_return_own_table2_prev | self.has_touch_opo_table_prev)
        )
        # print(f"{self.has_touch_paddle=},{self.has_return_own_table2_prev=},{self.has_touch_opo_table_prev=}")
        # print(f'{self.touched_paddel_no_bounce_table=}')
        # self.has_touch_own_table = has_touch_own_table_just_now
        self.has_first_bounce_prev = self.has_first_bounce.clone()
        still_false = ~self.has_first_bounce
        self.has_first_bounce[still_false] = self.has_touch_own_table[still_false]


        self.touch_info.append(
            {
                "has_touch_own_table": self.has_touch_own_table[0].cpu().numpy(),
                "has_touch_own_table_prev": self.has_touch_own_table_prev[0]
                .cpu()
                .numpy(),
                "has_touch_opponent_table": self.has_touch_opponent_table_just_now[0]
                .cpu()
                .numpy(),
                "has_first_bounce": self.has_first_bounce[0].cpu().numpy(),
                "has_first_bounce_prev": self.has_first_bounce_prev[0].cpu().numpy(),
                "ball_contact": self.ball_contact[0].cpu().numpy(),
                "has_touch_paddle": self.has_touch_paddle[0].cpu().numpy(),
                "has_first_bounce_prev": self.has_first_bounce_prev[0].cpu().numpy(),
            }
        )

        self.paddle_pos = self.paddle_touch_point - self.scene.env_origins 
        has_bounced = self.has_touch_own_table_prev
        vz = self.ball_linvel[:, 2]
        z = self.ball_pos[:, 2]
        x = self.ball_pos[:, 0]
        y = self.ball_pos[:, 1]
        vx = self.ball_linvel[:, 0]
        vy = self.ball_linvel[:, 1]

        # --- serve-arrival probe (guarded, OFF by default) --------------------------------
        # Set TT_SERVE_PROBE=1 to histogram the PHYSICAL ball (y,z) at the instant it crosses
        # the env-local hit plane x=hit_plane_x while flying toward the robot (vx<0). This is the
        # ground truth used to verify a served ball still arrives inside the hit window after a
        # geometry change (e.g. v12 moved hit_plane -1.58 -> -1.62). Reads real physics, so it
        # does NOT depend on the analytic/clamped ball_future_pose. Prints running p5/50/95.
        if os.environ.get("TT_SERVE_PROBE"):
            _hxp = self.cfg.robot.hit_plane_x
            _prev = getattr(self, "_probe_prev_x", None)
            if _prev is None or _prev.numel() != x.numel():
                self._probe_prev_x = x.detach().clone()
                self._probe_y, self._probe_z, self._probe_last = [], [], 0
            else:
                _crossed = (_prev > _hxp) & (x <= _hxp) & (vx < -0.5)
                if _crossed.any():
                    self._probe_y.append(y[_crossed].detach().cpu())
                    self._probe_z.append(z[_crossed].detach().cpu())
                self._probe_prev_x = x.detach().clone()
                _tot = sum(t.numel() for t in self._probe_y)
                if _tot - self._probe_last >= 2000 and _tot > 0:
                    self._probe_last = _tot
                    _ally = torch.cat(self._probe_y); _allz = torch.cat(self._probe_z)
                    _qs = torch.tensor([0.05, 0.5, 0.95])
                    _yq = torch.quantile(_ally, _qs); _zq = torch.quantile(_allz, _qs)
                    _tyl, _tyh = self.cfg.robot.hit_target_y_range
                    _tzl, _tzh = self.cfg.robot.hit_target_z_range
                    print(
                        f"[TT_SERVE_PROBE] n={_tot} @x={_hxp:.3f} | "
                        f"y p5/50/95={_yq[0]:.3f}/{_yq[1]:.3f}/{_yq[2]:.3f} win[{_tyl},{_tyh}] | "
                        f"z p5/50/95={_zq[0]:.3f}/{_zq[1]:.3f}/{_zq[2]:.3f} win[{_tzl},{_tzh}]",
                        flush=True,
                    )
        # ----------------------------------------------------------------------------------

        g=9.81
        body_height = self.cfg.robot.hit_body_height
        vel_max = self.cfg.robot.robot_vel_max
        paddle_y_offset = self.cfg.robot.paddle_y_offset
        hx = self.cfg.robot.hit_plane_x   # env-local hit-guidance plane x. For parked-base A1
                                          # this is the reachable blade x, not the base x.

        self.mask_before = (has_bounced == 0).squeeze(-1)
        self.mask_after = (has_bounced == 1).squeeze(-1)

        h = 0.78
        D = vz.pow(2) + 2.0 * g * (z - h)
        sqrtD = torch.sqrt(torch.clamp(D, min=0.0))
        t_land = (vz + sqrtD) / g
        t_land = torch.clamp(t_land, min=0.0)
        self.valid_before = (D >= 0.0) & self.mask_before
        t1 = torch.where(self.valid_before, t_land, torch.zeros_like(t_land))
        vz_prime = vz - g * t1
        vz_dd = -0.9 * vz_prime
        # Drag-adjusted ascent after first bounce
        k_scalar = torch.tensor(max(1e-8, getattr(self, 'ball_drag_k', 0.13)), device=self.device, dtype=vz.dtype)
        sqrt_gk = torch.sqrt(torch.clamp(torch.tensor(g, device=self.device, dtype=vz.dtype) * k_scalar, min=1e-12))
        sqrt_kg = torch.sqrt(torch.clamp(k_scalar / torch.tensor(g, device=self.device, dtype=vz.dtype), min=0.0))
        vz_up = torch.clamp(vz_dd, min=0.0)
        t2_drag = torch.atan(torch.clamp(vz_up * sqrt_kg, min=0.0)) / torch.clamp(sqrt_gk, min=1e-12)
        delta_z_drag = (0.5 / k_scalar) * torch.log1p(torch.clamp((k_scalar / torch.tensor(g, device=self.device, dtype=vz.dtype)) * vz_up.pow(2), min=0.0))
        self.t_before = t1 + t2_drag
        zpb = delta_z_drag + h

        # Horizontal displacement under quadratic drag: s(T) = (1/k) * ln(1 + k |v0| T) * sign(v0)
        def horiz_disp(v0, T):
            vabs = torch.abs(v0)
            return (1.0 / k_scalar) * torch.sign(v0) * torch.log1p(torch.clamp(k_scalar * vabs * T, min=0.0))

        # Drag-adjusted ascent for current upward motion (case 'a')
        vz_up_a = torch.clamp(vz, min=0.0)
        t_after_drag = torch.atan(torch.clamp(vz_up_a * sqrt_kg, min=0.0)) / torch.clamp(sqrt_gk, min=1e-12)
        delta_z_drag_a = (0.5 / k_scalar) * torch.log1p(torch.clamp((k_scalar / torch.tensor(g, device=self.device, dtype=vz.dtype)) * vz_up_a.pow(2), min=0.0))
        self.t_after = t_after_drag
        zpa = z + delta_z_drag_a
        self.predict_land_t = t_land
        self.predict_x_land = x + horiz_disp(vx, t_land) # acounts for both landing on our side and the opponent's side of table
        self.predict_y_land = y + horiz_disp(vy, t_land)
        predict_ball_land_vis=torch.stack([self.predict_x_land, self.predict_y_land, torch.ones_like(x)*h], dim=-1)


        dx_ascent = horiz_disp(0.7 *vx, t2_drag)
        dy_ascent = horiz_disp(0.7 *vy, t2_drag)
        xpb = self.predict_x_land + dx_ascent
        ypb = self.predict_y_land + dy_ascent
        dx_after = horiz_disp(vx, t_after_drag)
        dy_after = horiz_disp(vy, t_after_drag)
        xpa = x + dx_after
        ypa = y + dy_after

        tx_min, tx_max = getattr(self.cfg.robot, "hit_target_x_range", (-100.0, 100.0))
        ty_min, ty_max = getattr(self.cfg.robot, "hit_target_y_range", (-100.0, 100.0))
        tz_min, tz_max = getattr(self.cfg.robot, "hit_target_z_range", (-100.0, 100.0))
        if tx_min > -99.0 or tx_max < 99.0:
            xpb = torch.clamp(xpb, min=tx_min, max=tx_max)
            xpa = torch.clamp(xpa, min=tx_min, max=tx_max)
        else:
            xpb = torch.clamp(xpb, max=hx)
            xpa = torch.clamp(xpa, max=hx)
        ypb = torch.clamp(ypb, min=ty_min, max=ty_max)
        ypa = torch.clamp(ypa, min=ty_min, max=ty_max)
        zpb = torch.clamp(zpb, min=tz_min, max=tz_max)
        zpa = torch.clamp(zpa, min=tz_min, max=tz_max)

        self.pos_pred_before = torch.stack([xpb, ypb, zpb], dim=-1)
        self.pos_pred_after = torch.stack([xpa, ypa, zpa], dim=-1)

        self.pos_pred_before_ro = torch.stack([xpb - 0.1, ypb - paddle_y_offset, torch.ones_like(xpb) * body_height], dim=-1)
        self.pos_pred_after_ro = torch.stack([xpa - 0.1, ypa - paddle_y_offset, torch.ones_like(xpb) * body_height], dim=-1)

        self.ball_future_pose = torch.where(
            self.mask_before.unsqueeze(-1), self.pos_pred_before, self.pos_pred_after
        )
        # idle10 DECOUPLING — two distinct masks:
        # mask_no_ball = TRUE no-ball (no-ball injection / TT_NO_SERVE): the ball is physically
        #   gone. Drives the IDLE rewards ONLY (hold ready pose), so idle reward never fires on
        #   mid-rally invalid moments (a low/behind ball is NOT "idle, relax").
        self.mask_no_ball = torch.zeros_like(self.has_touch_paddle)
        if self._tt_no_ball_now():
            self.mask_no_ball = torch.ones_like(self.mask_no_ball)
        # mask_invalid = NO valid LIVE hit target right now. Drives the actor ball-gate, the
        #   critic ball_future_pose sentinel, and the approach-reward (ee/body) zeroing.
        #   z<0.75 only catches a ball that dropped BELOW the table (was 0.9, which wrongly
        #   blinded the live low BOUNCE arc at z~0.78 -> "touch but never return"). Dead balls
        #   are caught by has_second_bounce; the old (x<-1.35 & vz<0) term wrongly killed a live
        #   ball descending toward the paddle, so it is removed.
        self.mask_invalid = (
            (self.ball_pos[:, 0] < hx - 0.05)      # ball >5cm behind the robot line -> give up
            | (vx > 0)                             # ball moving away
            | (z < 0.75)                           # ball dropped below the table -> dead/off
            | self.has_second_bounce               # double bounce on own table -> dead
            | self.has_touch_paddle                # already hit this ball
            | self.mask_no_ball                    # true no-ball (injection)
        )
        if self._camera_observation_enable:
            # Match deployment: before two valid camera frames establish a track (or after
            # coast timeout), both raw ball observation and learned target stay at sentinel.
            self.mask_invalid |= ~self.camera_track_valid
        self.mask_terminal = (self.ball_pos[:, 0] > hx + 0.1) | (self.ball_pos[:, 0] < hx - 0.3) | self.has_touch_paddle_rew | (vz < 0.0) | (self.ball_pos[:, 2] < 0.6)
            #mask_terminal: true-> future,mask_terminal: false->distance
        self.has_touch_paddle_rew = self.has_touch_paddle.clone() # finally set mask True for reward computation
        # Expand to match shape (N, 3)
        mask_invalid_expanded = self.mask_invalid.unsqueeze(-1).expand_as(self.ball_future_pose)
        # Zero out those poses
        # FIXED HOME sentinel (env-local), NOT self-referential robot_pos. A robot-relative
        # ready target has rel_target_x == -0.1 constant (no restoring force) -> over a long
        # no-ball gap the robot drifts backward chasing it and falls. Anchoring x,y to the
        # trained home (-1.6, 0) gives a restoring force -> stable idle at home.
        modified_ball_pos = self._prediction_sentinel(dtype=self.ball_future_pose.dtype)
        self.ball_future_pose = torch.where(
            mask_invalid_expanded,
            modified_ball_pos,
            self.ball_future_pose
        )

        # self.ball_future_pose_vis = self.ball_future_pose + self.scene.env_origins
        mask = self.touched_paddel_no_bounce_table.unsqueeze(-1)  # (N, 1)
        self.ball_future_pose_vis = torch.where(
            mask,
            predict_ball_land_vis,
            self.ball_future_pose,
        ) + self.scene.env_origins
        self.robot_future_pos = torch.where(
            self.mask_before.unsqueeze(-1), self.pos_pred_before_ro, self.pos_pred_after_ro
        )
        self.robot_future_pos = torch.where(
            mask_invalid_expanded,      # [N,3] bool
            self.robot_future_pos.new_tensor([hx - 0.27, self.cfg.robot.home_y, body_height]).expand_as(self.robot_future_pos),
            # [-0.9, 0.2, body_height] for all envs
            self.robot_future_pos
        )

        # self.vel_ro_before = torch.clamp((self.pos_pred_before_ro - self.robot_pos ) /torch.clamp(self.t_before.unsqueeze(-1).expand_as(self.ball_future_pose), min=0.2),min=-vel_max, max=vel_max)
        # self.vel_ro_after = torch.clamp((self.pos_pred_after_ro - self.robot_pos ) /torch.clamp(self.t_after.unsqueeze(-1).expand_as(self.ball_future_pose), min=0.2),min=-vel_max, max=vel_max)
        self.vel_ro_before = torch.clamp((self.pos_pred_before_ro - self.robot_pos ) *4,min=-vel_max, max=vel_max)
        self.vel_ro_after = torch.clamp((self.pos_pred_after_ro - self.robot_pos ) *4,min=-vel_max, max=vel_max)

        self.robot_future_vel = torch.where(
            self.mask_before.unsqueeze(-1), self.vel_ro_before, self.vel_ro_after
        )
        self.robot_future_vel = torch.where(
            mask_invalid_expanded,
            torch.zeros_like(self.robot_future_vel),
            self.robot_future_vel
        )

        tb = self.t_before.unsqueeze(-1)  # [N,1]
        ta = self.t_after.unsqueeze(-1)   # [N,1]
        self.ball_future_t = torch.where(self.mask_before.unsqueeze(-1), tb, ta)
        self.ball_future_t = torch.where(
            self.mask_invalid.unsqueeze(-1),
            torch.zeros_like(self.ball_future_t),
            self.ball_future_t
        )

        if not self.headless:
            self.update_ball_future_visual()
            # self.update_robot_future_pos_visual()
            # self.update_robot_future_vel_visual()
            if not hasattr(self, "_hit_vis_debug_counter"):
                self._hit_vis_debug_counter = 0
            self._hit_vis_debug_counter += 1
            if self._hit_vis_debug_counter == 1 or self._hit_vis_debug_counter % 50 == 0:
                try:
                    b0 = self.ball_pos[0].detach().cpu().numpy()
                    f0 = self.ball_future_pose[0].detach().cpu().numpy()
                    p0 = self.ball_prediction[0].detach().cpu().numpy()
                    t0 = float(self.ball_future_t[0, 0].detach().cpu())
                    valid0 = bool((~self.mask_invalid[0]).detach().cpu())
                    print(
                        "[TTVis] env0 ball="
                        f"{b0} analytic_hit={f0} learned_pred={p0} "
                        f"t_hit={t0:.3f} valid_hit={valid0}"
                    )
                except Exception:
                    pass
    def init_obs_buffer(self):
        actor_obs, _ = self.compute_current_observations()
        if self.add_noise:
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6 : 6 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[6 + self.num_actions : 6 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[6 + self.num_actions * 2 : 6 + self.num_actions * 3] = 0.0
            # noise_vec[6 + self.num_actions * 3 : 6 + self.num_actions * 3+3] = noise_scales.ball_pos * self.obs_scales.ball_pos
            # noise_vec[6 + self.num_actions * 3+3 : 6 + self.num_actions * 3+6] = noise_scales.ball_linvel * self.obs_scales.ball_linvel
            # noise_vec[6 + self.num_actions * 3+6 : 6 + self.num_actions * 3+9] = noise_scales.robot_pos * self.obs_scales.robot_pos
            #perception noise
            noise_vec[6 + self.num_actions * 3 : 6 + self.num_actions * 3+6] = noise_scales.perception * self.obs_scales.ball_pos 
            # noise_vec[6 + self.num_actions * 3+9 : 6 + self.num_actions * 3+11] = noise_scales.ball_state * self.obs_scales.ball_state
            #ball prediction noise is set to zero
            noise_vec[6 + self.num_actions * 3 + 6: 6 + self.num_actions * 3 + 9] = 0.0
            noise_vec[6 + self.num_actions * 3 + 9: 6 + self.num_actions * 3 + 11] = noise_scales.perception * self.obs_scales.ball_pos 
            noise_vec[6 + self.num_actions * 3 + 11] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        self.robot_pos = self.robot.data.root_link_pos_w - self.table.data.root_link_pos_w
        self.current_perception = torch.cat([self.ball_pos, self.robot_pos], dim=-1)
        self.perception_buffer.reset()
        self.delayed_perception = self.perception_buffer.compute(self.current_perception)

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # destructor is order-sensitive
            del self.reward_manager
            del self.event_manager
            del self.scene
            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()

            # update closing status
            self._is_closed = True

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
    

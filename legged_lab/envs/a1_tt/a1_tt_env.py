"""A1 table-tennis env: reuse TTEnv, override only the geometric reset thresholds."""
import torch
from isaaclab.utils import math as math_utils

from legged_lab.envs.base.tt_env import TTEnv
from legged_lab.physics.a1_backhand_spin_prior import (
    sample_serve_spin_prior,
    sample_weak_topspin_prior,
)
from legged_lab.physics.a1_backhand_v7_safety import (
    minimum_vertical_capsule_clearance,
)


class A1TTEnv(TTEnv):
    ARM_TABLE_STUCK_TERMINATE_STEPS = 10
    FIXED_LIFT_JOINT_NAMES = ("sj",)

    def _sample_post_bounce_spin_target(
        self,
        count: int,
        *,
        curriculum: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cfg = self.cfg.ball
        mode = str(getattr(cfg, "post_bounce_spin_mode", "correlated_prior"))
        if mode == "weak_topspin":
            return sample_weak_topspin_prior(
                count,
                device=self.device,
                dtype=dtype,
                curriculum=curriculum,
                easy_range_rad_s=tuple(
                    getattr(cfg, "post_bounce_topspin_easy_range_rad_s", (2.0, 4.0))
                ),
                hard_range_rad_s=tuple(
                    getattr(cfg, "post_bounce_topspin_hard_range_rad_s", (2.0, 8.0))
                ),
                tilt_deg=float(getattr(cfg, "post_bounce_topspin_tilt_deg", 10.0)),
            )
        if mode != "correlated_prior":
            raise ValueError(f"Unsupported A1 post-bounce spin mode: {mode!r}")
        return sample_serve_spin_prior(
            count,
            device=self.device,
            dtype=dtype,
            curriculum=curriculum,
            easy_scale=float(getattr(cfg, "post_bounce_spin_easy_scale", 0.0)),
            hard_scale=float(getattr(cfg, "post_bounce_spin_hard_scale", 0.0)),
            magnitude_jitter=float(
                getattr(cfg, "post_bounce_spin_magnitude_jitter", 0.0)
            ),
        )

    def _sample_bounce_candidate_components(
        self,
        launch_pos: torch.Tensor,
        *,
        bounce_x_range: tuple[float, float],
        bounce_vz_range: tuple[float, float],
        y_center: float,
        y_half: float,
        curriculum: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optionally add the v4 real-serve slow-tail component.

        Historical A1 tasks do not define the opt-in attributes and therefore
        execute the unchanged single-uniform sampler in ``TTEnv``.
        """

        x_b, y_b, v_z = super()._sample_bounce_candidate_components(
            launch_pos,
            bounce_x_range=bounce_x_range,
            bounce_vz_range=bounce_vz_range,
            y_center=y_center,
            y_half=y_half,
            curriculum=curriculum,
        )
        cfg = self.cfg.ball
        weight_easy = float(getattr(cfg, "serve_tail_candidate_weight", 0.0) or 0.0)
        weight_hard = float(
            getattr(cfg, "serve_tail_candidate_weight_hard", weight_easy) or 0.0
        )
        tail_weight = weight_easy + float(curriculum) * (weight_hard - weight_easy)
        tail_weight = min(max(tail_weight, 0.0), 1.0)
        if tail_weight <= 0.0:
            return x_b, y_b, v_z

        def _lerp_pair(easy, hard):
            return (
                easy[0] + float(curriculum) * (hard[0] - easy[0]),
                easy[1] + float(curriculum) * (hard[1] - easy[1]),
            )

        tail_x_easy = getattr(cfg, "serve_tail_bounce_x_range", bounce_x_range)
        tail_x_hard = getattr(cfg, "serve_tail_bounce_x_range_hard", tail_x_easy)
        tail_vz_easy = getattr(cfg, "serve_tail_bounce_vz_range", bounce_vz_range)
        tail_vz_hard = getattr(cfg, "serve_tail_bounce_vz_range_hard", tail_vz_easy)
        tail_x_range = _lerp_pair(tail_x_easy, tail_x_hard)
        tail_vz_range = _lerp_pair(tail_vz_easy, tail_vz_hard)

        tail_mask = torch.rand(
            len(launch_pos), device=self.device, dtype=launch_pos.dtype
        ) < tail_weight
        tail_count = int(tail_mask.sum().item())
        if tail_count == 0:
            return x_b, y_b, v_z
        x_b[tail_mask] = torch.empty(
            tail_count, 1, device=self.device, dtype=launch_pos.dtype
        ).uniform_(*tail_x_range)
        v_z[tail_mask] = torch.empty(
            tail_count, 1, device=self.device, dtype=launch_pos.dtype
        ).uniform_(*tail_vz_range)
        return x_b, y_b, v_z

    def reset(self, env_ids):
        super().reset(env_ids)
        self._reset_fixed_lift_joints(env_ids)
        if hasattr(self, "arm_table_stuck_steps") and len(env_ids) > 0:
            self.arm_table_stuck_steps[env_ids] = 0
        if hasattr(self, "paddle_body_intrusion_steps") and len(env_ids) > 0:
            self.paddle_body_intrusion_steps[env_ids] = 0

    def _resolve_fixed_lift_joints(self):
        if hasattr(self, "_fixed_lift_joint_ids"):
            return
        joint_ids = []
        joint_names = []
        for name in self.FIXED_LIFT_JOINT_NAMES:
            if name in self.robot.joint_names:
                joint_ids.append(self.robot.joint_names.index(name))
                joint_names.append(name)
        if 0 < len(joint_ids) != len(self.FIXED_LIFT_JOINT_NAMES):
            raise RuntimeError(
                "A1 fixed lift joint guard could not resolve "
                f"{self.FIXED_LIFT_JOINT_NAMES}; got {joint_names}."
            )
        self._fixed_lift_joint_ids = joint_ids
        self._fixed_lift_joint_names = joint_names

    def _fixed_lift_default_state(self, env_ids=None):
        self._resolve_fixed_lift_joints()
        joint_ids = self._fixed_lift_joint_ids
        if env_ids is None:
            joint_pos = self.robot.data.default_joint_pos[:, joint_ids].clone()
            joint_vel = self.robot.data.default_joint_vel[:, joint_ids].clone()
        else:
            joint_pos = self.robot.data.default_joint_pos[env_ids][:, joint_ids].clone()
            joint_vel = self.robot.data.default_joint_vel[env_ids][:, joint_ids].clone()
        return joint_pos, joint_vel

    def _hold_fixed_lift_targets(self, env_ids=None):
        self._resolve_fixed_lift_joints()
        if len(self._fixed_lift_joint_ids) == 0:
            return
        joint_pos, joint_vel = self._fixed_lift_default_state(env_ids)
        self.robot.set_joint_position_target(joint_pos, self._fixed_lift_joint_ids, env_ids=env_ids)
        self.robot.set_joint_velocity_target(joint_vel, self._fixed_lift_joint_ids, env_ids=env_ids)

    def _reset_fixed_lift_joints(self, env_ids):
        if len(env_ids) == 0:
            return
        self._resolve_fixed_lift_joints()
        if len(self._fixed_lift_joint_ids) == 0:
            return
        joint_pos, joint_vel = self._fixed_lift_default_state(env_ids)
        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids=self._fixed_lift_joint_ids,
            env_ids=env_ids,
        )
        self._hold_fixed_lift_targets(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()

    def _apply_non_policy_joint_targets(self) -> None:
        self._hold_fixed_lift_targets()

    def _compute_arm_table_collision(self):
        """Detect right-arm/paddle intrusion near the tabletop.

        This is intentionally geometry-based instead of contact-sensor based:
        the contact sensor cannot distinguish table contact from a legal ball hit.
        """
        if not hasattr(self, "_arm_table_body_ids"):
            body_ids, _ = self.robot.find_bodies(["Link_r[1-7]", "Link_r_paddle"])
            if len(body_ids) == 0:
                raise RuntimeError("A1 arm/table collision check found no right-arm bodies.")
            self._arm_table_body_ids = body_ids

        body_pos_t = (
            self.robot.data.body_pos_w[:, self._arm_table_body_ids, :]
            - self.table.data.root_link_pos_w[:, None, :]
        )
        x = body_pos_t[..., 0]
        y = body_pos_t[..., 1]
        z = body_pos_t[..., 2]

        warning_margin_xy = 0.08
        warning_xy = (
            (x > (-1.37 - warning_margin_xy))
            & (x < (1.37 + warning_margin_xy))
            & (torch.abs(y) < (0.7625 + warning_margin_xy))
        )
        # Tabletop height is 0.76 m. Keep the warning band tight around the
        # tabletop; normal hit targets are around 0.9-1.25 m and must remain legal.
        warning_z = (z > 0.55) & (z < 0.80)
        warning = warning_xy & warning_z

        terminal_xy = (
            (x > -1.37)
            & (x < 1.37)
            & (torch.abs(y) < 0.7625)
        )
        # Terminate only for deeper intrusion into the actual table footprint.
        terminal_z = (z > 0.55) & (z < 0.78)
        stuck = terminal_xy & terminal_z

        self.arm_table_collision = torch.any(warning, dim=1)
        self.arm_table_collision_count = warning.float().sum(dim=1)
        self.arm_table_stuck_contact = torch.any(stuck, dim=1)
        self.arm_table_stuck_contact_count = stuck.float().sum(dim=1)
        if not hasattr(self, "arm_table_stuck_steps"):
            self.arm_table_stuck_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.arm_table_stuck_steps = torch.where(
            self.arm_table_stuck_contact,
            self.arm_table_stuck_steps + 1,
            torch.zeros_like(self.arm_table_stuck_steps),
        )
        self.arm_table_termination = self.arm_table_stuck_steps >= self.ARM_TABLE_STUCK_TERMINATE_STEPS
        return self.arm_table_collision

    def _compute_paddle_body_safety(self):
        """Compute a conservative v7 paddle-outline/torso-capsule clearance."""

        enabled = bool(getattr(self.cfg.robot, "paddle_body_safety_enable", False))
        if not enabled:
            self.paddle_body_clearance_m = torch.full(
                (self.num_envs,), float("inf"), device=self.device
            )
            self.paddle_body_safety_violation = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self.paddle_body_hard_intrusion = torch.zeros_like(
                self.paddle_body_safety_violation
            )
            self.paddle_body_termination = torch.zeros_like(
                self.paddle_body_safety_violation
            )
            return self.paddle_body_clearance_m

        if not hasattr(self, "_paddle_body_safety_samples_local"):
            samples = tuple(self.cfg.robot.paddle_body_safety_sample_points_local_m)
            if len(samples) == 0:
                raise RuntimeError("paddle body safety is enabled without paddle samples")
            self._paddle_body_safety_samples_local = torch.tensor(
                samples,
                device=self.device,
                dtype=self.robot.data.body_pos_w.dtype,
            ).unsqueeze(0)
            self.paddle_body_intrusion_steps = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self._paddle_body_safety_log_counter = 0
            self._paddle_body_safety_window_min = torch.tensor(
                float("inf"), device=self.device
            )
            self._paddle_body_safety_window_soft = torch.zeros(
                (), device=self.device, dtype=torch.long
            )
            self._paddle_body_safety_window_hard = torch.zeros_like(
                self._paddle_body_safety_window_soft
            )
            self._paddle_body_safety_window_termination = torch.zeros_like(
                self._paddle_body_safety_window_soft
            )
            self._paddle_body_safety_window_samples = 0

        sample_count = self._paddle_body_safety_samples_local.shape[1]
        paddle_quat_w = self.robot.data.body_quat_w[:, self._paddle_body_id, :]
        paddle_pos_w = self.robot.data.body_pos_w[:, self._paddle_body_id, :]
        samples_local = self._paddle_body_safety_samples_local.expand(
            self.num_envs, sample_count, 3
        )
        samples_w = paddle_pos_w.unsqueeze(1) + math_utils.quat_apply(
            paddle_quat_w.unsqueeze(1).expand(-1, sample_count, -1).reshape(-1, 4),
            samples_local.reshape(-1, 3),
        ).reshape(self.num_envs, sample_count, 3)

        root_pos_w = self.robot.data.root_link_pos_w
        root_quat_w = self.robot.data.root_link_quat_w
        samples_base = math_utils.quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, sample_count, -1).reshape(-1, 4),
            (samples_w - root_pos_w.unsqueeze(1)).reshape(-1, 3),
        ).reshape(self.num_envs, sample_count, 3)
        clearance = minimum_vertical_capsule_clearance(
            samples_base,
            center_xy_m=tuple(
                self.cfg.robot.paddle_body_safety_capsule_center_xy_m
            ),
            z_range_m=tuple(self.cfg.robot.paddle_body_safety_capsule_z_range_m),
            radius_m=float(self.cfg.robot.paddle_body_safety_capsule_radius_m),
        )
        self.paddle_body_clearance_m = clearance
        soft_clearance = float(self.cfg.robot.paddle_body_safety_soft_clearance_m)
        terminate_clearance = float(
            self.cfg.robot.paddle_body_safety_termination_clearance_m
        )
        self.paddle_body_safety_violation = clearance < soft_clearance
        self.paddle_body_hard_intrusion = clearance < terminate_clearance
        self.paddle_body_intrusion_steps = torch.where(
            self.paddle_body_hard_intrusion,
            self.paddle_body_intrusion_steps + 1,
            torch.zeros_like(self.paddle_body_intrusion_steps),
        )
        terminate_steps = max(
            1, int(self.cfg.robot.paddle_body_safety_termination_steps)
        )
        self.paddle_body_termination = (
            self.paddle_body_intrusion_steps >= terminate_steps
        )

        log_interval = int(self.cfg.robot.paddle_body_safety_log_interval_steps)
        self._paddle_body_safety_log_counter += 1
        self._paddle_body_safety_window_min = torch.minimum(
            self._paddle_body_safety_window_min, torch.amin(clearance)
        )
        self._paddle_body_safety_window_soft += torch.sum(
            self.paddle_body_safety_violation
        )
        self._paddle_body_safety_window_hard += torch.sum(
            self.paddle_body_hard_intrusion
        )
        self._paddle_body_safety_window_termination += torch.sum(
            self.paddle_body_termination
        )
        self._paddle_body_safety_window_samples += self.num_envs
        should_log = log_interval > 0 and (
            self._paddle_body_safety_log_counter == 1
            or self._paddle_body_safety_log_counter % log_interval == 0
        )
        if should_log:
            quantiles = torch.quantile(
                clearance.float(),
                torch.tensor((0.0, 0.01, 0.05, 0.50), device=self.device),
            )
            sample_count = max(1, self._paddle_body_safety_window_samples)
            print(
                "[A1_BODY_SAFETY] "
                f"clearance_m[min/p01/p05/p50]="
                f"{quantiles[0].item():.4f}/{quantiles[1].item():.4f}/"
                f"{quantiles[2].item():.4f}/{quantiles[3].item():.4f} "
                f"window_min={self._paddle_body_safety_window_min.item():.4f} "
                f"soft={self._paddle_body_safety_window_soft.item() / sample_count:.6f} "
                f"hard={self._paddle_body_safety_window_hard.item() / sample_count:.6f} "
                f"terminate="
                f"{self._paddle_body_safety_window_termination.item() / sample_count:.6f}",
                flush=True,
            )
            self._paddle_body_safety_window_min.fill_(float("inf"))
            self._paddle_body_safety_window_soft.zero_()
            self._paddle_body_safety_window_hard.zero_()
            self._paddle_body_safety_window_termination.zero_()
            self._paddle_body_safety_window_samples = 0
        return clearance

    def check_reset(self):
        # base tilted too far from upright: projected_gravity_b z-component is ~-1 upright,
        # ->0 when tipped 90°. >-0.5 means tilted more than ~60° (flipped/failed).
        # Free heavy chassis rarely flips; this is a safety net.
        tilted = self.robot.data.projected_gravity_b[:, 2] > -0.5
        self._compute_arm_table_collision()
        self._compute_paddle_body_safety()
        reset_buf = (
            tilted |
            self.arm_table_termination |
            self.paddle_body_termination |
            (self.robot_pos[..., 0] < -3.6) |
            (self.robot_pos[..., 0] > -1.35) |
            (self.robot_pos[..., 1] < -1.1) |
            (self.robot_pos[..., 1] > 1.1)
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        time_out_buf |= self.ball_reset_counter > self.max_ball_serve_per_episode
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

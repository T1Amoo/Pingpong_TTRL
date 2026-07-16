"""A1 table-tennis env: reuse TTEnv, override only the geometric reset thresholds."""
import torch
from legged_lab.envs.base.tt_env import TTEnv


class A1TTEnv(TTEnv):
    ARM_TABLE_STUCK_TERMINATE_STEPS = 10
    FIXED_LIFT_JOINT_NAMES = ("sj",)

    def reset(self, env_ids):
        super().reset(env_ids)
        self._reset_fixed_lift_joints(env_ids)
        if hasattr(self, "arm_table_stuck_steps") and len(env_ids) > 0:
            self.arm_table_stuck_steps[env_ids] = 0

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

    def check_reset(self):
        # base tilted too far from upright: projected_gravity_b z-component is ~-1 upright,
        # ->0 when tipped 90°. >-0.5 means tilted more than ~60° (flipped/failed).
        # Free heavy chassis rarely flips; this is a safety net.
        tilted = self.robot.data.projected_gravity_b[:, 2] > -0.5
        self._compute_arm_table_collision()
        reset_buf = (
            tilted |
            self.arm_table_termination |
            (self.robot_pos[..., 0] < -3.6) |
            (self.robot_pos[..., 0] > -1.35) |
            (self.robot_pos[..., 1] < -1.1) |
            (self.robot_pos[..., 1] > 1.1)
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        time_out_buf |= self.ball_reset_counter > self.max_ball_serve_per_episode
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

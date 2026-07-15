"""A1 table-tennis env: reuse TTEnv, override only the geometric reset thresholds."""
import torch
from legged_lab.envs.base.tt_env import TTEnv


class A1TTEnv(TTEnv):
    def _compute_arm_table_collision(self):
        """Detect low right-arm/paddle intrusion into the table volume.

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

        margin_xy = 0.08
        in_table_xy = (
            (x > (-1.37 - margin_xy))
            & (x < (1.37 + margin_xy))
            & (torch.abs(y) < (0.7625 + margin_xy))
        )
        # Only the low tabletop/edge band is forbidden. High swing-through above
        # the table remains legal; actual target z is around 0.9-1.25 m.
        in_low_table_band = (z > 0.55) & (z < 0.95)
        body_collision = in_table_xy & in_low_table_band
        self.arm_table_collision = torch.any(body_collision, dim=1)
        self.arm_table_collision_count = body_collision.float().sum(dim=1)
        return self.arm_table_collision

    def check_reset(self):
        # base tilted too far from upright: projected_gravity_b z-component is ~-1 upright,
        # ->0 when tipped 90°. >-0.5 means tilted more than ~60° (flipped/failed).
        # Free heavy chassis rarely flips; this is a safety net.
        tilted = self.robot.data.projected_gravity_b[:, 2] > -0.5
        arm_table_collision = self._compute_arm_table_collision()
        reset_buf = (
            tilted |
            arm_table_collision |
            (self.robot_pos[..., 0] < -3.6) |
            (self.robot_pos[..., 0] > -1.35) |
            (self.robot_pos[..., 1] < -1.1) |
            (self.robot_pos[..., 1] > 1.1)
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        time_out_buf |= self.ball_reset_counter > self.max_ball_serve_per_episode
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

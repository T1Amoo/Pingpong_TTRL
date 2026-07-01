"""A1 table-tennis env: reuse TTEnv, override only the geometric reset thresholds."""
import torch
from legged_lab.envs.base.tt_env import TTEnv


class A1TTEnv(TTEnv):
    def check_reset(self):
        # base tilted too far from upright: projected_gravity_b z-component is ~-1 upright,
        # ->0 when tipped 90°. >-0.5 means tilted more than ~60° (flipped/failed).
        # Free heavy chassis rarely flips; this is a safety net.
        tilted = self.robot.data.projected_gravity_b[:, 2] > -0.5
        reset_buf = (
            tilted |
            (self.robot_pos[..., 0] < -3.6) |
            (self.robot_pos[..., 0] > -1.35) |
            (self.robot_pos[..., 1] < -1.1) |
            (self.robot_pos[..., 1] > 1.1)
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        time_out_buf |= self.ball_reset_counter > self.max_ball_serve_per_episode
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

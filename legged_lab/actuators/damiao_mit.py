"""Explicit DAMIAO-style MIT actuator model."""

from __future__ import annotations

from collections.abc import Sequence
import torch

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class DamiaoMITActuator(ActuatorBase):
    """Project-local actuator for raw q_des -> MIT torque -> joint effort.

    This class intentionally does not use IsaacLab's IdealPDActuator compute path.
    It keeps the real MIT command form explicit, then adds the actuator-side
    effects we need to identify separately: command delay, optional command slew,
    torque lag, friction, and torque-speed limiting.
    """

    cfg: DamiaoMITActuatorCfg

    def __init__(self, cfg: DamiaoMITActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self._dt = float(cfg.control_dt)
        if self._dt <= 0.0:
            raise ValueError(f"control_dt must be positive, got {self._dt}")

        self._command_delay_s = self._parse_joint_parameter(cfg.command_delay_s, 0.0)
        self._command_lead_s = self._parse_joint_parameter(cfg.command_lead_s, 0.0)
        self._command_bias_rad = self._parse_joint_parameter(cfg.command_bias_rad, 0.0)
        delay_steps = torch.round(self._command_delay_s / self._dt).to(dtype=torch.long)
        self._command_delay_steps = torch.clamp(delay_steps[0], min=0)
        self._command_delay_buffer_len = int(torch.max(self._command_delay_steps).item()) + 1
        self._command_delay_index = 0
        self._command_delay_buffer = torch.zeros(
            self._command_delay_buffer_len,
            self._num_envs,
            self.num_joints,
            device=self._device,
        )
        self._response_model_enable = bool(cfg.response_model_enable)
        self._response_fn_hz = self._parse_joint_parameter(cfg.response_fn_hz, 0.0)
        self._response_zeta = self._parse_joint_parameter(cfg.response_zeta, 1.0)
        self._response_delay_s = self._parse_joint_parameter(cfg.response_delay_s, 0.0)
        self._response_linear_gain = self._parse_joint_parameter(cfg.response_linear_gain, 1.0)
        self._response_intercept = self._parse_joint_parameter(cfg.response_intercept, 0.0)
        self._response_u_mean = self._parse_joint_parameter(cfg.response_u_mean, 0.0)
        self._response_tau_zero_s = self._parse_joint_parameter(cfg.response_tau_zero_s, 0.0)
        response_delay_steps = torch.round(self._response_delay_s / self._dt).to(dtype=torch.long)
        self._response_delay_steps = torch.clamp(response_delay_steps[0], min=0)
        self._response_delay_buffer_len = int(torch.max(self._response_delay_steps).item()) + 1
        self._response_delay_index = 0
        self._response_delay_buffer = torch.zeros(
            self._response_delay_buffer_len,
            self._num_envs,
            self.num_joints,
            device=self._device,
        )

        self._command_velocity_limit = self._parse_joint_parameter(cfg.command_velocity_limit, torch.inf)
        self._use_command_velocity = bool(cfg.use_command_velocity)
        self._velocity_feedback_scale = self._parse_joint_parameter(cfg.velocity_feedback_scale, 1.0)
        self._velocity_filter_time_constant = self._parse_joint_parameter(cfg.velocity_filter_time_constant, 0.0)
        self._torque_time_constant = self._parse_joint_parameter(cfg.torque_time_constant, 0.0)
        self._viscous_friction = self._parse_joint_parameter(cfg.viscous_friction, 0.0)
        self._coulomb_friction = self._parse_joint_parameter(cfg.coulomb_friction, 0.0)
        self._friction_activation_vel = self._parse_joint_parameter(cfg.friction_activation_vel, 0.02)
        self._position_deadband = self._parse_joint_parameter(cfg.position_deadband, 0.0)
        self._torque_deadband = self._parse_joint_parameter(cfg.torque_deadband, 0.0)
        self._torque_scale = self._parse_joint_parameter(cfg.torque_scale, 1.0)
        self._command_velocity_feedforward = self._parse_joint_parameter(cfg.command_velocity_feedforward, 0.0)
        self._command_acceleration_feedforward = self._parse_joint_parameter(
            cfg.command_acceleration_feedforward, 0.0
        )
        self._brake_effort_limit = self._parse_joint_parameter(cfg.brake_effort_limit, self.effort_limit)
        self._torque_speed_limit_enable = bool(cfg.torque_speed_limit_enable)

        self._initialized = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._previous_raw_command = torch.zeros_like(self.computed_effort)
        self._previous_command_velocity = torch.zeros_like(self.computed_effort)
        self._command_velocity = torch.zeros_like(self.computed_effort)
        self._command_acceleration = torch.zeros_like(self.computed_effort)
        self._lead_command_position = torch.zeros_like(self.computed_effort)
        self._command_position = torch.zeros_like(self.computed_effort)
        self._delayed_command_position = torch.zeros_like(self.computed_effort)
        self._response_state_position = torch.zeros_like(self.computed_effort)
        self._response_state_velocity = torch.zeros_like(self.computed_effort)
        self._response_state_acceleration = torch.zeros_like(self.computed_effort)
        self._response_command_position = torch.zeros_like(self.computed_effort)
        self._response_command_velocity = torch.zeros_like(self.computed_effort)
        self._feedback_velocity = torch.zeros_like(self.computed_effort)
        self._torque_state = torch.zeros_like(self.computed_effort)
        self._raw_mit_effort = torch.zeros_like(self.computed_effort)
        self._friction_effort = torch.zeros_like(self.computed_effort)
        self.torque_limit_min = -self.effort_limit.clone()
        self.torque_limit_max = self.effort_limit.clone()

    def reset(self, env_ids: Sequence[int] | None):
        env_ids = self._resolve_env_ids(env_ids)
        self._initialized[env_ids] = False
        self._previous_raw_command[env_ids] = 0.0
        self._previous_command_velocity[env_ids] = 0.0
        self._command_velocity[env_ids] = 0.0
        self._command_acceleration[env_ids] = 0.0
        self._lead_command_position[env_ids] = 0.0
        self._response_state_position[env_ids] = 0.0
        self._response_state_velocity[env_ids] = 0.0
        self._response_state_acceleration[env_ids] = 0.0
        self._feedback_velocity[env_ids] = 0.0
        self._torque_state[env_ids] = 0.0

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            raise ValueError("DamiaoMITActuator requires joint position targets.")

        self._initialize_uninitialized_envs(joint_pos, joint_vel)
        raw_command = control_action.joint_positions
        response_command, response_velocity = self._apply_response_model(raw_command)
        raw_command = response_command
        command = self._apply_command_lead(raw_command)
        command = self._apply_command_slew(command)
        delayed_command = self._delayed_command(command)

        position_error = delayed_command - joint_pos
        if torch.any(self._position_deadband > 0.0):
            position_error = torch.where(
                torch.abs(position_error) <= self._position_deadband,
                torch.zeros_like(position_error),
                position_error - torch.sign(position_error) * self._position_deadband,
            )

        if self._response_model_enable:
            desired_vel = response_velocity
        elif self._use_command_velocity and control_action.joint_velocities is not None:
            desired_vel = control_action.joint_velocities
        else:
            desired_vel = torch.zeros_like(joint_vel)
        feedback_vel = self._filtered_feedback_velocity(joint_vel)
        feedforward = (
            torch.zeros_like(joint_pos)
            if control_action.joint_efforts is None
            else control_action.joint_efforts
        )

        raw_effort = (
            self.stiffness * position_error
            + self.damping * (desired_vel - self._velocity_feedback_scale * feedback_vel)
            + self._command_velocity_feedforward * self._command_velocity
            + self._command_acceleration_feedforward * self._command_acceleration
            + feedforward
        )
        if torch.any(self._torque_deadband > 0.0):
            raw_effort = torch.where(
                torch.abs(raw_effort) <= self._torque_deadband,
                torch.zeros_like(raw_effort),
                raw_effort - torch.sign(raw_effort) * self._torque_deadband,
            )
        raw_effort = self._torque_scale * raw_effort
        self._raw_mit_effort = raw_effort

        alpha = torch.where(
            self._torque_time_constant > 0.0,
            self._dt / (self._torque_time_constant + self._dt),
            torch.ones_like(self._torque_time_constant),
        )
        self._torque_state.add_(alpha * (raw_effort - self._torque_state))

        activation_vel = torch.clamp(self._friction_activation_vel, min=1.0e-6)
        self._friction_effort = (
            self._viscous_friction * joint_vel
            + self._coulomb_friction * torch.tanh(joint_vel / activation_vel)
        )
        self.computed_effort = self._torque_state - self._friction_effort
        self.applied_effort = self._clip_effort(self.computed_effort, joint_vel)

        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _resolve_env_ids(self, env_ids: Sequence[int] | None):
        if env_ids is None or isinstance(env_ids, slice):
            return slice(None)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self._device, dtype=torch.long)
        return torch.tensor(env_ids, device=self._device, dtype=torch.long)

    def _initialize_uninitialized_envs(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        env_ids = torch.nonzero(~self._initialized, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        self._command_position[env_ids] = joint_pos[env_ids]
        self._previous_raw_command[env_ids] = joint_pos[env_ids]
        self._previous_command_velocity[env_ids] = 0.0
        self._command_velocity[env_ids] = 0.0
        self._command_acceleration[env_ids] = 0.0
        self._lead_command_position[env_ids] = joint_pos[env_ids]
        self._response_state_position[env_ids] = joint_pos[env_ids] - self._response_u_mean[env_ids]
        self._response_state_velocity[env_ids] = 0.0
        self._response_state_acceleration[env_ids] = 0.0
        self._response_command_position[env_ids] = joint_pos[env_ids]
        self._response_command_velocity[env_ids] = 0.0
        self._response_delay_buffer[:, env_ids, :] = joint_pos[env_ids].unsqueeze(0)
        self._delayed_command_position[env_ids] = joint_pos[env_ids]
        self._command_delay_buffer[:, env_ids, :] = joint_pos[env_ids].unsqueeze(0)
        self._feedback_velocity[env_ids] = joint_vel[env_ids]
        self._torque_state[env_ids] = 0.0
        self._initialized[env_ids] = True

    def _apply_response_model(self, raw_command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._response_model_enable:
            return raw_command, torch.zeros_like(raw_command)

        write_idx = self._response_delay_index
        self._response_delay_buffer[write_idx].copy_(raw_command)
        delayed_columns = []
        for joint_id, delay_step in enumerate(self._response_delay_steps.tolist()):
            read_idx = (write_idx - int(delay_step)) % self._response_delay_buffer_len
            delayed_columns.append(self._response_delay_buffer[read_idx, :, joint_id])
        self._response_delay_index = (write_idx + 1) % self._response_delay_buffer_len
        delayed_command = torch.stack(delayed_columns, dim=1)

        wn = 2.0 * torch.pi * self._response_fn_hz
        input_state = delayed_command - self._response_u_mean
        self._response_state_acceleration = (
            wn * wn * (input_state - self._response_state_position)
            - 2.0 * self._response_zeta * wn * self._response_state_velocity
        )
        self._response_state_velocity.add_(self._response_state_acceleration * self._dt)
        self._response_state_position.add_(self._response_state_velocity * self._dt)
        self._response_command_position = self._response_intercept + self._response_linear_gain * (
            self._response_state_position + self._response_tau_zero_s * self._response_state_velocity
        )
        self._response_command_velocity = self._response_linear_gain * (
            self._response_state_velocity + self._response_tau_zero_s * self._response_state_acceleration
        )
        return self._response_command_position, self._response_command_velocity

    def _apply_command_lead(self, raw_command: torch.Tensor) -> torch.Tensor:
        self._command_velocity = (raw_command - self._previous_raw_command) / self._dt
        self._command_acceleration = (self._command_velocity - self._previous_command_velocity) / self._dt
        self._lead_command_position = (
            raw_command + self._command_bias_rad + self._command_lead_s * self._command_velocity
        )
        self._previous_raw_command.copy_(raw_command)
        self._previous_command_velocity.copy_(self._command_velocity)
        return self._lead_command_position

    def _apply_command_slew(self, raw_command: torch.Tensor) -> torch.Tensor:
        max_delta = self._command_velocity_limit * self._dt
        delta = raw_command - self._command_position
        finite = torch.isfinite(max_delta)
        limited_delta = torch.where(
            finite,
            torch.clamp(delta, min=-max_delta, max=max_delta),
            delta,
        )
        self._command_position.add_(limited_delta)
        return self._command_position

    def _delayed_command(self, command: torch.Tensor) -> torch.Tensor:
        write_idx = self._command_delay_index
        self._command_delay_buffer[write_idx].copy_(command)
        delayed_columns = []
        for joint_id, delay_step in enumerate(self._command_delay_steps.tolist()):
            read_idx = (write_idx - int(delay_step)) % self._command_delay_buffer_len
            delayed_columns.append(self._command_delay_buffer[read_idx, :, joint_id])
        self._command_delay_index = (write_idx + 1) % self._command_delay_buffer_len
        self._delayed_command_position = torch.stack(delayed_columns, dim=1)
        return self._delayed_command_position

    def _filtered_feedback_velocity(self, joint_vel: torch.Tensor) -> torch.Tensor:
        alpha = torch.where(
            self._velocity_filter_time_constant > 0.0,
            self._dt / (self._velocity_filter_time_constant + self._dt),
            torch.ones_like(self._velocity_filter_time_constant),
        )
        self._feedback_velocity.add_(alpha * (joint_vel - self._feedback_velocity))
        return self._feedback_velocity

    def _clip_effort(self, effort: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        if not self._torque_speed_limit_enable:
            self.torque_limit_min = -self.effort_limit
            self.torque_limit_max = self.effort_limit
            return torch.clip(effort, min=self.torque_limit_min, max=self.torque_limit_max)

        velocity_limit = torch.clamp(self.velocity_limit, min=1.0e-6)
        speed_scale = torch.clamp(1.0 - torch.abs(joint_vel) / velocity_limit, min=0.0, max=1.0)
        accel_limit = self.effort_limit * speed_scale
        brake_limit = torch.maximum(self._brake_effort_limit, self.effort_limit)

        self.torque_limit_max = torch.where(joint_vel > 0.0, accel_limit, brake_limit)
        neg_abs_limit = torch.where(joint_vel < 0.0, accel_limit, brake_limit)
        self.torque_limit_min = -neg_abs_limit
        return torch.clip(effort, min=self.torque_limit_min, max=self.torque_limit_max)


@configclass
class DamiaoMITActuatorCfg(ActuatorBaseCfg):
    """Configuration for :class:`DamiaoMITActuator`."""

    class_type: type = DamiaoMITActuator

    control_dt: float = 0.002
    """Actuator update dt used for delay/slew/torque lag integration."""

    command_delay_s: dict[str, float] | float = 0.0
    """Delay applied to q_des before MIT torque computation."""

    command_lead_s: dict[str, float] | float = 0.0
    """Lead applied as q_des + command_lead_s * dq_des_est before MIT torque."""

    command_bias_rad: dict[str, float] | float = 0.0
    """Constant actuator-side command offset applied before MIT torque."""

    response_model_enable: bool = False
    """Enable an actuator-internal second-order command response state."""

    response_fn_hz: dict[str, float] | float = 0.0
    """Natural frequency of the internal command response model."""

    response_zeta: dict[str, float] | float = 1.0
    """Damping ratio of the internal command response model."""

    response_delay_s: dict[str, float] | float = 0.0
    """Integer-step command delay used by the internal command response model."""

    response_linear_gain: dict[str, float] | float = 1.0
    """Linear gain applied to the internal response state."""

    response_intercept: dict[str, float] | float = 0.0
    """Intercept applied to the internal response state."""

    response_u_mean: dict[str, float] | float = 0.0
    """Mean command subtracted before the internal response state."""

    response_tau_zero_s: dict[str, float] | float = 0.0
    """Optional lead zero applied to the internal response state."""

    command_velocity_limit: dict[str, float] | float = float("inf")
    """Optional actuator-internal q_des slew limit in rad/s."""

    use_command_velocity: bool = False
    """Use incoming velocity targets in the MIT damping term."""

    velocity_feedback_scale: dict[str, float] | float = 1.0
    """Scale applied to measured velocity before the MIT damping term."""

    velocity_filter_time_constant: dict[str, float] | float = 0.0
    """First-order filter time constant for the velocity used by the MIT damping term."""

    torque_time_constant: dict[str, float] | float = 0.0
    """First-order lag on the raw MIT torque command."""

    viscous_friction: dict[str, float] | float = 0.0
    """Viscous friction subtracted from motor output torque."""

    coulomb_friction: dict[str, float] | float = 0.0
    """Smooth Coulomb friction subtracted from motor output torque."""

    friction_activation_vel: dict[str, float] | float = 0.02
    """Velocity scale for smooth Coulomb friction activation."""

    position_deadband: dict[str, float] | float = 0.0
    """Deadband on MIT position error before stiffness is applied."""

    torque_deadband: dict[str, float] | float = 0.0
    """Deadband on raw MIT torque command before torque lag."""

    torque_scale: dict[str, float] | float = 1.0
    """Scale on raw MIT torque before actuator lag, friction, and torque limits."""

    command_velocity_feedforward: dict[str, float] | float = 0.0
    """Torque feedforward gain on internally estimated q_des velocity."""

    command_acceleration_feedforward: dict[str, float] | float = 0.0
    """Torque feedforward gain on internally estimated q_des acceleration."""

    torque_speed_limit_enable: bool = True
    """Enable a simple four-quadrant torque-speed envelope."""

    brake_effort_limit: dict[str, float] | float | None = None
    """Opposing-direction torque limit. Defaults to effort_limit when omitted."""

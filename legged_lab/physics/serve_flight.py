"""Fast, differentiable table-tennis serve preflight checks.

The training serve sampler specifies a nominal no-drag bounce target.  This
module propagates the resulting launch state through the same quadratic drag
law used by :class:`AeroForceField`, checks the net plane, models the first
table bounce, and reports the state at the robot hit plane.  Integration uses
x as the independent variable, so a small fixed number of RK4 steps is enough
and reset-time rejection sampling stays cheap on thousands of GPU envs.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch


class ServeFlightProbe(NamedTuple):
    net_z: torch.Tensor
    bounce_x: torch.Tensor
    hit_y: torch.Tensor
    hit_z: torch.Tensor
    hit_vx: torch.Tensor
    hit_vz: torch.Tensor
    bounced: torch.Tensor


def _flight_rhs_x(state: torch.Tensor, drag_accel_k: float, gravity: float) -> torch.Tensor:
    """Return d[y,z,vx,vy,vz]/dx for quadratic drag plus gravity."""

    _y, _z, vx, vy, vz = state.unbind(dim=-1)
    safe_vx = torch.where(vx < -1.0e-4, vx, torch.full_like(vx, -1.0e-4))
    speed = torch.sqrt(torch.clamp(vx.square() + vy.square() + vz.square(), min=1.0e-12))
    return torch.stack(
        (
            vy / safe_vx,
            vz / safe_vx,
            -drag_accel_k * speed,
            (-drag_accel_k * speed * vy) / safe_vx,
            (-gravity - drag_accel_k * speed * vz) / safe_vx,
        ),
        dim=-1,
    )


def _rk4_x(state: torch.Tensor, dx: torch.Tensor | float, drag_accel_k: float, gravity: float) -> torch.Tensor:
    if not torch.is_tensor(dx):
        dx = state.new_tensor(float(dx))
    while dx.ndim < state.ndim:
        dx = dx.unsqueeze(-1)
    k1 = _flight_rhs_x(state, drag_accel_k, gravity)
    k2 = _flight_rhs_x(state + 0.5 * dx * k1, drag_accel_k, gravity)
    k3 = _flight_rhs_x(state + 0.5 * dx * k2, drag_accel_k, gravity)
    k4 = _flight_rhs_x(state + dx * k3, drag_accel_k, gravity)
    return state + (dx / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _probe_serve_flight_impl(
    launch_pos: torch.Tensor,
    launch_vel: torch.Tensor,
    *,
    hit_plane_x: float,
    net_x: float = 0.0,
    table_ball_center_z: float = 0.78,
    table_restitution: float = 0.95,
    table_dynamic_friction: float = 0.10,
    drag_accel_k: float = 0.09910893224020438,
    gravity: float = 9.81,
    launch_to_net_steps: int = 8,
    net_to_hit_steps: int = 16,
) -> ServeFlightProbe:
    """Probe net clearance, first bounce, and hit-plane arrival for a batch.

    ``launch_pos`` and ``launch_vel`` are ``(N, 3)`` tensors in table-local
    world axes.  All outputs are length-N tensors.  ``bounce_x`` is NaN when no
    descending table-center crossing occurs between the net and hit plane.
    """

    if launch_pos.ndim != 2 or launch_pos.shape[-1] != 3:
        raise ValueError(f"launch_pos must be (N,3), got {tuple(launch_pos.shape)}")
    if launch_vel.shape != launch_pos.shape:
        raise ValueError(f"launch_vel must match launch_pos, got {tuple(launch_vel.shape)}")
    if launch_to_net_steps <= 0 or net_to_hit_steps <= 0:
        raise ValueError("serve flight integration step counts must be positive")

    state = torch.stack(
        (launch_pos[:, 1], launch_pos[:, 2], launch_vel[:, 0], launch_vel[:, 1], launch_vel[:, 2]),
        dim=-1,
    )
    x = launch_pos[:, 0].clone()
    dx_net = (state.new_tensor(float(net_x)) - x) / float(launch_to_net_steps)
    for _ in range(launch_to_net_steps):
        state = _rk4_x(state, dx_net, drag_accel_k, gravity)
        x = x + dx_net
    net_z = state[:, 1].clone()

    bounced = torch.zeros(len(state), dtype=torch.bool, device=state.device)
    bounce_x = torch.full((len(state),), float("nan"), dtype=state.dtype, device=state.device)
    dx_hit = (state.new_tensor(float(hit_plane_x)) - state.new_tensor(float(net_x))) / float(net_to_hit_steps)

    for _ in range(net_to_hit_steps):
        prev_state = state
        next_state = _rk4_x(prev_state, dx_hit, drag_accel_k, gravity)
        crossing = (
            (~bounced)
            & (prev_state[:, 1] > table_ball_center_z)
            & (next_state[:, 1] <= table_ball_center_z)
            & (next_state[:, 4] < 0.0)
        )
        # Keep this branchless on CUDA.  ``if crossing.any()`` forced a device
        # synchronization for every integration step and dominated the reset
        # path for thousands of environments.  The candidate bounce is cheap to
        # evaluate in one vectorized batch and is selected only where ``crossing``
        # is true, preserving the exact trajectory contract.
        denom = torch.clamp(prev_state[:, 1] - next_state[:, 1], min=1.0e-8)
        alpha = torch.clamp((prev_state[:, 1] - table_ball_center_z) / denom, 0.0, 1.0)
        pre_bounce = prev_state + alpha.unsqueeze(-1) * (next_state - prev_state)
        post_bounce = pre_bounce.clone()
        post_bounce[:, 1] = table_ball_center_z
        # A no-spin ball does not preserve its horizontal speed at the
        # first table contact.  Use the conservative Coulomb impulse bound
        # here: PhysX's compliant contact/friction solve removed more
        # horizontal speed from steep serves than the ideal rigid-sphere
        # rolling impulse.  Leaving this out admitted nominally slow balls
        # whose simulated trajectory arrived much lower than preflight.
        tangent = pre_bounce[:, (2, 3)]
        tangent_speed = torch.linalg.norm(tangent, dim=-1)
        coulomb_delta = (
            float(table_dynamic_friction)
            * (1.0 + float(table_restitution))
            * torch.abs(pre_bounce[:, 4])
        )
        delta_speed = torch.minimum(coulomb_delta, tangent_speed)
        tangent_scale = torch.clamp(
            1.0 - delta_speed / torch.clamp(tangent_speed, min=1.0e-8),
            min=0.0,
        )
        post_bounce[:, 2] = pre_bounce[:, 2] * tangent_scale
        post_bounce[:, 3] = pre_bounce[:, 3] * tangent_scale
        post_bounce[:, 4] = -table_restitution * pre_bounce[:, 4]
        remaining_dx = dx_hit * (1.0 - alpha)
        after_bounce = _rk4_x(post_bounce, remaining_dx, drag_accel_k, gravity)
        state = torch.where(crossing.unsqueeze(-1), after_bounce, next_state)
        bx = x + alpha * dx_hit
        bounce_x = torch.where(crossing, bx, bounce_x)
        bounced = bounced | crossing
        x = x + dx_hit

    return ServeFlightProbe(
        net_z=net_z,
        bounce_x=bounce_x,
        hit_y=state[:, 0],
        hit_z=state[:, 1],
        hit_vx=state[:, 2],
        hit_vz=state[:, 4],
        bounced=bounced,
    )


_compiled_cuda_probe = None
_compiled_cuda_probe_failed = False


def probe_serve_flight(
    launch_pos: torch.Tensor,
    launch_vel: torch.Tensor,
    *,
    hit_plane_x: float,
    net_x: float = 0.0,
    table_ball_center_z: float = 0.78,
    table_restitution: float = 0.95,
    table_dynamic_friction: float = 0.10,
    drag_accel_k: float = 0.09910893224020438,
    gravity: float = 9.81,
    launch_to_net_steps: int = 8,
    net_to_hit_steps: int = 16,
) -> ServeFlightProbe:
    """Run the serve probe eagerly on CPU and as one fused graph on CUDA.

    Reset batches are usually small and the eager RK4 implementation launches
    hundreds of tiny CUDA kernels.  Compiling the branchless implementation
    fuses that work and removes the per-step host synchronizations without
    changing any trajectory or acceptance calculation.  The first CUDA call
    pays a one-time compile cost; failures fall back to the verified eager path.
    """

    global _compiled_cuda_probe, _compiled_cuda_probe_failed
    use_compiled = (
        launch_pos.is_cuda
        and os.environ.get("TT_SERVE_FLIGHT_COMPILE", "1") != "0"
        and not _compiled_cuda_probe_failed
    )
    if use_compiled:
        if _compiled_cuda_probe is None:
            _compiled_cuda_probe = torch.compile(
                _probe_serve_flight_impl,
                dynamic=True,
                fullgraph=True,
                mode="reduce-overhead",
            )
        try:
            return _compiled_cuda_probe(
                launch_pos,
                launch_vel,
                hit_plane_x=hit_plane_x,
                net_x=net_x,
                table_ball_center_z=table_ball_center_z,
                table_restitution=table_restitution,
                table_dynamic_friction=table_dynamic_friction,
                drag_accel_k=drag_accel_k,
                gravity=gravity,
                launch_to_net_steps=launch_to_net_steps,
                net_to_hit_steps=net_to_hit_steps,
            )
        except Exception as exc:
            _compiled_cuda_probe_failed = True
            print(
                f"[TT_SERVE_FLIGHT] CUDA compile failed; using eager fallback: {exc}",
                flush=True,
            )

    return _probe_serve_flight_impl(
        launch_pos,
        launch_vel,
        hit_plane_x=hit_plane_x,
        net_x=net_x,
        table_ball_center_z=table_ball_center_z,
        table_restitution=table_restitution,
        table_dynamic_friction=table_dynamic_friction,
        drag_accel_k=drag_accel_k,
        gravity=gravity,
        launch_to_net_steps=launch_to_net_steps,
        net_to_hit_steps=net_to_hit_steps,
    )

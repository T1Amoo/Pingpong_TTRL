"""Compare v7 ready-relative serves against the inherited v6 distribution."""

from __future__ import annotations

import argparse

import torch

from legged_lab.physics import a1_backhand_v3_contract as v3
from legged_lab.physics import a1_backhand_v4_contract as v4
from legged_lab.physics import a1_backhand_v7_contract as v7
from legged_lab.physics.serve_flight import probe_serve_flight


def _lerp_pair(easy, hard, curriculum):
    return (
        easy[0] + curriculum * (hard[0] - easy[0]),
        easy[1] + curriculum * (hard[1] - easy[1]),
    )


def _sample(count: int, curriculum: float, *, translated: bool):
    if translated:
        launch_xyz = v7.BALL_LAUNCH_POS
        hit_plane_x = v7.HIT_PLANE_X
        easy_x = v7.EASY_BOUNCE_X_RANGE
        hard_x = v7.HARD_BOUNCE_X_RANGE
        easy_vz = v7.EASY_BOUNCE_VZ_RANGE
        hard_vz = v7.HARD_BOUNCE_VZ_RANGE
        easy_y = v7.EASY_Y_CENTER
        hard_y = v7.HARD_Y_CENTER
        easy_half = v7.EASY_Y_HALF
        hard_half = v7.HARD_Y_HALF
        tail_weight_hard = v7.TAIL_CANDIDATE_WEIGHT_HARD
        tail_x_hard = v7.TAIL_BOUNCE_X_RANGE_HARD
        tail_vz_hard = v7.TAIL_BOUNCE_VZ_RANGE_HARD
        arrival_y = v7.HIT_TARGET_Y_RANGE
        arrival_z = v7.PREFLIGHT_HIT_Z_RANGE
    else:
        launch_xyz = (1.35, 0.0, 1.03)
        hit_plane_x = v7.V6_HIT_PLANE_X
        easy_x = v4.EASY_BOUNCE_X_RANGE
        hard_x = v4.HARD_BOUNCE_X_RANGE
        easy_vz = v4.EASY_BOUNCE_VZ_RANGE
        hard_vz = v4.HARD_BOUNCE_VZ_RANGE
        easy_y = v4.EASY_Y_CENTER
        hard_y = v4.HARD_Y_CENTER
        easy_half = v4.EASY_Y_HALF
        hard_half = v4.HARD_Y_HALF
        tail_weight_hard = v4.TAIL_CANDIDATE_WEIGHT_HARD
        tail_x_hard = v4.TAIL_BOUNCE_X_RANGE_HARD
        tail_vz_hard = v4.TAIL_BOUNCE_VZ_RANGE_HARD
        arrival_y = v3.HIT_TARGET_Y_RANGE
        arrival_z = v3.PREFLIGHT_HIT_Z_RANGE

    main_x = _lerp_pair(easy_x, hard_x, curriculum)
    main_vz = _lerp_pair(easy_vz, hard_vz, curriculum)
    tail_x = _lerp_pair(easy_x, tail_x_hard, curriculum)
    tail_vz = _lerp_pair(easy_vz, tail_vz_hard, curriculum)
    y_center = easy_y + curriculum * (hard_y - easy_y)
    y_half = easy_half + curriculum * (hard_half - easy_half)
    tail_candidate = torch.rand(count) < curriculum * tail_weight_hard

    launch = torch.tensor(launch_xyz, dtype=torch.float64).repeat(count, 1)
    bounce_x = torch.empty(count, 1, dtype=torch.float64).uniform_(*main_x)
    launch_vz = torch.empty(count, 1, dtype=torch.float64).uniform_(*main_vz)
    tail_count = int(tail_candidate.sum())
    if tail_count:
        bounce_x[tail_candidate] = torch.empty(tail_count, 1, dtype=torch.float64).uniform_(*tail_x)
        launch_vz[tail_candidate] = torch.empty(tail_count, 1, dtype=torch.float64).uniform_(*tail_vz)
    bounce_y = y_center + torch.empty(count, 1, dtype=torch.float64).uniform_(-y_half, y_half)

    t_bounce = (
        launch_vz
        + torch.sqrt(launch_vz.square() + 2.0 * 9.81 * (launch[:, 2:3] - 0.78))
    ) / 9.81
    velocity = torch.cat(
        (
            (bounce_x - launch[:, 0:1]) / t_bounce,
            (bounce_y - launch[:, 1:2]) / t_bounce,
            launch_vz,
        ),
        dim=1,
    )
    probe = probe_serve_flight(
        launch,
        velocity,
        hit_plane_x=hit_plane_x,
        drag_accel_k=v4.DRAG_ACCEL_K,
        table_restitution=v4.TABLE_RESTITUTION,
        table_dynamic_friction=v4.TABLE_DYNAMIC_FRICTION,
    )
    valid = (
        probe.bounced
        & (probe.net_z >= v4.NET_CENTER_Z_MIN + v4.NET_PREDICTION_MARGIN)
        & (probe.bounce_x >= v4.PHYSICAL_BOUNCE_X_RANGE[0])
        & (probe.bounce_x <= v4.PHYSICAL_BOUNCE_X_RANGE[1])
        & (probe.hit_y >= arrival_y[0])
        & (probe.hit_y <= arrival_y[1])
        & (probe.hit_z >= arrival_z[0])
        & (probe.hit_z <= arrival_z[1])
        & (probe.hit_vx.abs() >= v3.HIT_ARRIVAL_ABS_VX_RANGE[0])
        & (probe.hit_vx.abs() <= v3.HIT_ARRIVAL_ABS_VX_RANGE[1])
    )
    return probe, valid, tail_candidate


def _q(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return torch.quantile(
        value[valid],
        torch.tensor([0.05, 0.50, 0.95], dtype=value.dtype),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    delta = torch.tensor(v7.READY_PADDLE_TRANSLATION_M, dtype=torch.float64)
    for curriculum in (0.0, 0.5, 1.0):
        # Reset the seed so the two contracts share proposal quantiles where
        # their normalized uniform random variables still correspond.
        torch.manual_seed(args.seed + int(curriculum * 10))
        old, old_valid, old_tail = _sample(args.samples, curriculum, translated=False)
        torch.manual_seed(args.seed + int(curriculum * 10))
        new, new_valid, new_tail = _sample(args.samples, curriculum, translated=True)

        old_y = _q(old.hit_y, old_valid)
        old_z = _q(old.hit_z, old_valid)
        old_vx = _q(old.hit_vx.abs(), old_valid)
        old_t = _q(old.hit_time, old_valid)
        new_y = _q(new.hit_y, new_valid)
        new_z = _q(new.hit_z, new_valid)
        new_vx = _q(new.hit_vx.abs(), new_valid)
        new_t = _q(new.hit_time, new_valid)
        old_tail_rate = float((old_valid & old_tail).sum() / old_valid.sum())
        new_tail_rate = float((new_valid & new_tail).sum() / new_valid.sum())

        print(
            f"[v7_relative_serve] c={curriculum:.1f} "
            f"accept_old/new={old_valid.float().mean():.4f}/{new_valid.float().mean():.4f} "
            f"tail_old/new={old_tail_rate:.4f}/{new_tail_rate:.4f}"
        )
        print(f"  y old+dy={old_y + delta[1]} new={new_y} error={new_y - old_y - delta[1]}")
        print(f"  z old+dz={old_z + delta[2]} new={new_z} error={new_z - old_z - delta[2]}")
        print(f"  |vx| old={old_vx} new={new_vx} error={new_vx - old_vx}")
        print(f"  time old={old_t} new={new_t} error={new_t - old_t}")

        assert torch.max(torch.abs(new_y - old_y - delta[1])) <= 0.012
        assert torch.max(torch.abs(new_z - old_z - delta[2])) <= 0.030
        assert torch.max(torch.abs(new_vx - old_vx)) <= 0.40
        assert torch.max(torch.abs(new_t - old_t)) <= 0.10
        if curriculum == 1.0:
            assert abs(new_tail_rate - old_tail_rate) <= 0.02


if __name__ == "__main__":
    main()

"""Verify the empirical-mixture A1 backhand-v4 serve contract on CPU."""

from __future__ import annotations

import argparse

import torch

from legged_lab.physics import a1_backhand_v4_contract as contract
from legged_lab.physics.serve_flight import probe_serve_flight


def _lerp_pair(easy, hard, curriculum):
    return (
        easy[0] + curriculum * (hard[0] - easy[0]),
        easy[1] + curriculum * (hard[1] - easy[1]),
    )


def sample_candidates(count: int, curriculum: float):
    main_x_range = _lerp_pair(
        contract.EASY_BOUNCE_X_RANGE,
        contract.HARD_BOUNCE_X_RANGE,
        curriculum,
    )
    main_vz_range = _lerp_pair(
        contract.EASY_BOUNCE_VZ_RANGE,
        contract.HARD_BOUNCE_VZ_RANGE,
        curriculum,
    )
    tail_x_range = _lerp_pair(
        contract.EASY_BOUNCE_X_RANGE,
        contract.TAIL_BOUNCE_X_RANGE_HARD,
        curriculum,
    )
    tail_vz_range = _lerp_pair(
        contract.EASY_BOUNCE_VZ_RANGE,
        contract.TAIL_BOUNCE_VZ_RANGE_HARD,
        curriculum,
    )
    y_center = contract.EASY_Y_CENTER + curriculum * (
        contract.HARD_Y_CENTER - contract.EASY_Y_CENTER
    )
    y_half = contract.EASY_Y_HALF + curriculum * (
        contract.HARD_Y_HALF - contract.EASY_Y_HALF
    )
    tail_candidate = torch.rand(count) < (
        curriculum * contract.TAIL_CANDIDATE_WEIGHT_HARD
    )

    launch = torch.tensor([1.35, 0.0, 1.03], dtype=torch.float64).repeat(count, 1)
    bounce_x = torch.empty(count, 1, dtype=torch.float64).uniform_(*main_x_range)
    launch_vz = torch.empty(count, 1, dtype=torch.float64).uniform_(*main_vz_range)
    tail_count = int(tail_candidate.sum())
    if tail_count:
        bounce_x[tail_candidate] = torch.empty(tail_count, 1, dtype=torch.float64).uniform_(
            *tail_x_range
        )
        launch_vz[tail_candidate] = torch.empty(
            tail_count, 1, dtype=torch.float64
        ).uniform_(*tail_vz_range)
    bounce_y = y_center + torch.empty(count, 1, dtype=torch.float64).uniform_(
        -y_half, y_half
    )

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
        hit_plane_x=-1.243,
        drag_accel_k=contract.DRAG_ACCEL_K,
        table_restitution=contract.TABLE_RESTITUTION,
        table_dynamic_friction=contract.TABLE_DYNAMIC_FRICTION,
    )
    valid = (
        probe.bounced
        & (probe.net_z >= contract.NET_CENTER_Z_MIN + contract.NET_PREDICTION_MARGIN)
        & (probe.bounce_x >= contract.PHYSICAL_BOUNCE_X_RANGE[0])
        & (probe.bounce_x <= contract.PHYSICAL_BOUNCE_X_RANGE[1])
        & (probe.hit_y >= contract.HIT_TARGET_Y_RANGE[0])
        & (probe.hit_y <= contract.HIT_TARGET_Y_RANGE[1])
        & (probe.hit_z >= contract.PREFLIGHT_HIT_Z_RANGE[0])
        & (probe.hit_z <= contract.PREFLIGHT_HIT_Z_RANGE[1])
        & (probe.hit_vx.abs() >= contract.HIT_ARRIVAL_ABS_VX_RANGE[0])
        & (probe.hit_vx.abs() <= contract.HIT_ARRIVAL_ABS_VX_RANGE[1])
    )
    return probe, valid, tail_candidate


def quantiles(value: torch.Tensor, valid: torch.Tensor):
    q = torch.tensor([0.0, 0.05, 0.50, 0.95, 1.0], dtype=value.dtype)
    return [round(float(x), 4) for x in torch.quantile(value[valid], q)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    for curriculum in (0.0, 0.5, 1.0):
        probe, valid, tail_candidate = sample_candidates(args.samples, curriculum)
        accepted = int(valid.sum())
        if accepted == 0:
            raise RuntimeError(f"no accepted serves at curriculum={curriculum}")
        accepted_tail = int((valid & tail_candidate).sum())
        print(
            f"[v4_serve] c={curriculum:.1f} candidates={args.samples} "
            f"acceptance={accepted / args.samples:.4f} accepted={accepted} "
            f"accepted_tail={accepted_tail / accepted:.4f}"
        )
        for name in ("net_z", "bounce_x", "hit_y", "hit_z", "hit_vx", "hit_vz"):
            print(
                f"  {name} q0/q5/q50/q95/q100="
                f"{quantiles(getattr(probe, name), valid)}"
            )

        assert float(probe.net_z[valid].min()) >= (
            contract.NET_CENTER_Z_MIN + contract.NET_PREDICTION_MARGIN - 1.0e-9
        )
        assert float(probe.hit_y[valid].min()) >= contract.HIT_TARGET_Y_RANGE[0] - 1.0e-9
        assert float(probe.hit_y[valid].max()) <= contract.HIT_TARGET_Y_RANGE[1] + 1.0e-9
        assert float(probe.hit_z[valid].min()) >= contract.PREFLIGHT_HIT_Z_RANGE[0] - 1.0e-9
        assert float(probe.hit_z[valid].max()) <= contract.PREFLIGHT_HIT_Z_RANGE[1] + 1.0e-9
        assert float(probe.hit_vx[valid].abs().min()) >= (
            contract.HIT_ARRIVAL_ABS_VX_RANGE[0] - 1.0e-9
        )
        assert float(probe.hit_vx[valid].abs().max()) <= (
            contract.HIT_ARRIVAL_ABS_VX_RANGE[1] + 1.0e-9
        )

        if curriculum == 1.0:
            q = torch.tensor([0.5, 0.95], dtype=probe.hit_z.dtype)
            zq = torch.quantile(probe.hit_z[valid], q)
            vxq = torch.quantile(
                probe.hit_vx[valid].abs(), q
            )
            # The reset-time RK4/Coulomb probe is deliberately conservative;
            # its speed quantiles differ materially from compliant PhysX.
            # Assert its own deterministic regression contract here and use
            # TT_SERVE_PROBE for physical-vs-real validation.
            assert abs(float(zq[0]) - contract.CPU_HARD_HIT_Z_P50) <= 0.03
            assert abs(float(zq[1]) - contract.CPU_HARD_HIT_Z_P95) <= 0.04
            assert abs(float(vxq[0]) - contract.CPU_HARD_ABS_VX_P50) <= 0.12
            assert abs(float(vxq[1]) - contract.CPU_HARD_ABS_VX_P95) <= 0.15
            assert abs(accepted_tail / accepted - contract.CPU_HARD_ACCEPTED_TAIL) <= 0.03


if __name__ == "__main__":
    main()

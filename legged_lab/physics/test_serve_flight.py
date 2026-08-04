import torch

from legged_lab.physics.serve_flight import _flight_rhs_x, probe_serve_flight


def test_time_augmented_flight_preserves_physical_derivatives():
    state5 = torch.tensor([[0.02, 1.10, -3.0, 0.15, 1.2]], dtype=torch.float64)
    state6 = torch.cat((state5, torch.zeros(1, 1, dtype=state5.dtype)), dim=-1)

    derivative5 = _flight_rhs_x(state5, drag_accel_k=0.09910893224020438, gravity=9.81)
    derivative6 = _flight_rhs_x(state6, drag_accel_k=0.09910893224020438, gravity=9.81)

    torch.testing.assert_close(derivative6[:, :5], derivative5)
    torch.testing.assert_close(derivative6[:, 5], 1.0 / state5[:, 2])


def test_serve_probe_returns_direct_hit_plane_time():
    launch = torch.tensor([[1.35, 0.0, 1.03]], dtype=torch.float64)
    bounce_x = torch.tensor([[-0.80]], dtype=torch.float64)
    bounce_y = torch.tensor([[0.041]], dtype=torch.float64)
    launch_vz = torch.tensor([[1.30]], dtype=torch.float64)
    bounce_time = (
        launch_vz
        + torch.sqrt(launch_vz.square() + 2.0 * 9.81 * (launch[:, 2:3] - 0.78))
    ) / 9.81
    velocity = torch.cat(
        (
            (bounce_x - launch[:, 0:1]) / bounce_time,
            (bounce_y - launch[:, 1:2]) / bounce_time,
            launch_vz,
        ),
        dim=-1,
    )

    probe = probe_serve_flight(launch, velocity, hit_plane_x=-1.243)

    assert bool(probe.bounced[0])
    assert 0.0 < float(probe.hit_time[0]) < 2.4
    assert -0.06 <= float(probe.hit_y[0]) <= 0.20
    assert 0.90 <= float(probe.hit_z[0]) <= 1.38

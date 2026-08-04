import torch
import pytest

from rsl_rl.runners.on_policy_predictor_regression_runner import (
    OnPolicyPredictorRegressionRunner,
)


class _CrossingEnv:
    def __init__(self):
        self.targets = torch.tensor(
            [[-1.243, 0.05, 1.15], [-1.243, -0.02, 0.95]], dtype=torch.float32
        )
        self.crossed = torch.tensor([True, False])

    def get_predictor_actual_hit_plane_crossings(self):
        return self.targets, self.crossed

    def get_predictor_serve_ids(self):
        return torch.tensor([7, 9])


def test_actual_crossing_backfills_only_same_serve_pre_plane_rows():
    runner = object.__new__(OnPolicyPredictorRegressionRunner)
    runner.env = _CrossingEnv()
    runner._traj_maxlen = 8
    runner._traj_write_idx = 5
    runner._traj_buf_cpu = torch.zeros(8, 2, 3)
    runner._traj_buf_cpu[:, :, 0] = -0.50
    runner._traj_buf_cpu[5, 0, 0] = -1.25
    runner._sample_valid_buf_cpu = torch.ones(8, 2, dtype=torch.bool)
    runner._serve_id_buf_cpu = torch.zeros(8, 2, dtype=torch.long)
    runner._serve_id_buf_cpu[:6, 0] = 7
    runner._serve_id_buf_cpu[6:, 0] = 6
    runner._serve_id_buf_cpu[:, 1] = 9
    runner._gt_buf_cpu = torch.zeros(8, 2, 3)
    runner._gt_valid_buf_cpu = torch.zeros(8, 2, dtype=torch.bool)

    runner._backfill_actual_hit_plane_targets()

    assert bool(runner._gt_valid_buf_cpu[:5, 0].all())
    assert not bool(runner._gt_valid_buf_cpu[5:, 0].any())
    assert not bool(runner._gt_valid_buf_cpu[:, 1].any())
    torch.testing.assert_close(runner._gt_buf_cpu[0, 0], runner.env.targets[0])


def test_physical_crossing_validation_uses_prior_causal_prediction():
    runner = object.__new__(OnPolicyPredictorRegressionRunner)
    runner.env = _CrossingEnv()
    runner.env.ball_linvel = torch.tensor([[-2.0, 0.0, -0.5], [-3.0, 0.0, -0.2]])
    runner.pred_target_mode = "actual_hit_plane"
    runner._pred_trained = True
    runner._pred_last_prediction_cpu = torch.tensor(
        [[-1.243, 0.15, 1.05], [-1.243, 0.30, 1.30]], dtype=torch.float32
    )
    runner._pred_last_inference_valid_cpu = torch.tensor([True, True])
    runner._pred_last_inference_serve_id_cpu = torch.tensor([7, 8])
    runner.pred_validation_slow_abs_vx_mps = 2.2
    runner.pred_validation_high_z_m = 1.30
    runner.pred_validation_window_samples = 32
    runner._pred_validation_errors_cpu = torch.empty(0, 2)
    runner._pred_validation_slow_cpu = torch.empty(0, dtype=torch.bool)
    runner._pred_validation_high_cpu = torch.empty(0, dtype=torch.bool)
    runner._pred_validation_crossings_total = 0

    runner._accumulate_actual_hit_plane_validation()
    metrics = runner._predictor_validation_metrics()

    assert metrics["predictor_actual_crossing_window_n"] == 1.0
    assert metrics["predictor_actual_crossing_total_n"] == 1.0
    assert metrics["predictor_actual_hard_window_n"] == 1.0
    assert metrics["predictor_actual_y_mae_m"] == pytest.approx(0.10)
    assert metrics["predictor_actual_z_mae_m"] == pytest.approx(0.10)


def test_validation_waits_for_patience_before_sample_shortage_alert(capsys):
    runner = object.__new__(OnPolicyPredictorRegressionRunner)
    runner.pred_validation_interval_iters = 1
    runner.pred_validation_min_samples = 256
    runner.pred_validation_warn_start_iter = 1
    runner.pred_validation_warn_patience = 3
    runner._pred_validation_report_checks = 0
    metrics = {"predictor_actual_crossing_window_n": 12.0}

    runner._maybe_report_predictor_validation(0, metrics)
    runner._maybe_report_predictor_validation(1, metrics)
    assert "PREDICTOR_ALERT" not in capsys.readouterr().out

    runner._maybe_report_predictor_validation(2, metrics)
    assert "PREDICTOR_ALERT" in capsys.readouterr().out


def test_resume_rejects_predictor_target_semantic_mismatch(tmp_path):
    runner = object.__new__(OnPolicyPredictorRegressionRunner)
    runner.pred_target_mode = "actual_hit_plane"
    runner.alg = type(
        "Alg",
        (),
        {
            "policy": type("Policy", (), {"load_state_dict": lambda self, state: True})(),
            "rnd": None,
            "optimizer": type("Optimizer", (), {"load_state_dict": lambda self, state: None})(),
        },
    )()
    runner.empirical_normalization = False
    runner._predictor = torch.nn.Linear(1, 1)
    runner._pred_optim = torch.optim.Adam(runner._predictor.parameters())
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "pred_state_dict": runner._predictor.state_dict(),
            "pred_cfg": {"target_mode": "env_future_pose"},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="cannot resume predictor training across target semantics"):
        runner.load(str(checkpoint), load_optimizer=True)

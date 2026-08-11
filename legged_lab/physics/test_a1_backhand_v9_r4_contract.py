from __future__ import annotations

from pathlib import Path

import pytest

from legged_lab.physics import a1_backhand_v9_contract as v9


ROOT = Path(__file__).resolve().parents[2]
CONFIG_SOURCE = (ROOT / "legged_lab/envs/a1_tt/a1_tt_config.py").read_text()
REGISTRY_SOURCE = (ROOT / "legged_lab/envs/__init__.py").read_text()
ENV_SOURCE = (ROOT / "legged_lab/envs/base/tt_env.py").read_text()
WATCHDOG_SOURCE = (ROOT / "watchdog_a1_tt_backhand_v9.sh").read_text()


def _class_body(source: str, class_name: str, next_class_name: str) -> str:
    return source.split(f"class {class_name}", 1)[1].split(
        f"class {next_class_name}", 1
    )[0]


def test_replace_r4_is_strictly_single_joint():
    baseline = tuple(range(7))
    changed = v9.replace_r4(baseline, 99)
    assert changed == (0, 1, 2, 99, 4, 5, 6)
    with pytest.raises(ValueError):
        v9.replace_r4((0, 1), 99)


def test_v9_frozen_r4_evidence_and_limits():
    assert v9.R4_RESPONSE_FN_HZ == pytest.approx(4.6540465907137785)
    assert v9.R4_RESPONSE_ZETA == pytest.approx(0.1218426552876445)
    assert v9.R4_RESPONSE_DELAY_S == pytest.approx(0.02893988654476473)
    assert v9.R4_RESPONSE_GAIN == pytest.approx(1.0)
    assert v9.R4_RESPONSE_BIAS_RAD == pytest.approx(0.009078698834346307)
    assert v9.R4_EFFORT_LIMIT_NM == pytest.approx(8.0)
    assert v9.R4_TORQUE_KP == pytest.approx(120.0)
    assert v9.R4_TORQUE_KD == pytest.approx(1.0)
    assert v9.R4_TORQUE_SCALE == pytest.approx(0.9932751315748902)
    assert v9.R4_TORQUE_OFFSET_NM == pytest.approx(-0.08238003677908433)
    assert v9.R4_TORQUE_OBSERVER_RMSE_NM < 0.05
    assert v9.R4_RESPONSE_ACCEL_LIMIT_RAD_S2 == float("inf")
    assert v9.HOLDOUT_Q_RMSE_RAD < v9.OLD_RECENTERED_Q_RMSE_RAD
    assert v9.HOLDOUT_IMPROVEMENT_FRACTION >= 0.60
    assert v9.SATURATION_HOLDOUT_TORQUE_PEAK_NM >= 8.0


def test_r4_torque_projection_is_identity_below_limit_and_exact_at_boundary():
    command, raw_torque, projected_torque = v9.project_r4_command(
        command=-1.75, position=-1.76, velocity=0.1
    )
    assert command == pytest.approx(-1.75)
    assert abs(raw_torque) < 8.0
    assert projected_torque == pytest.approx(raw_torque)

    command, raw_torque, projected_torque = v9.project_r4_command(
        command=-1.50, position=-1.76, velocity=0.1
    )
    assert raw_torque > 8.0
    assert projected_torque == pytest.approx(8.0)
    rebuilt_torque = v9.R4_TORQUE_SCALE * (
        v9.R4_TORQUE_KP * (command + 1.76) - v9.R4_TORQUE_KD * 0.1
    ) + v9.R4_TORQUE_OFFSET_NM
    assert rebuilt_torque == pytest.approx(8.0)


def test_v9_changes_only_r4_actuator_chain_on_top_of_v8():
    assert "class A1TableTennisBackhandV9EnvCfg(A1TableTennisBackhandV8EnvCfg)" in CONFIG_SOURCE
    body = _class_body(
        CONFIG_SOURCE,
        "A1TableTennisBackhandV9EnvCfg",
        "A1TableTennisBackhandV9EvalEnvCfg",
    )
    for field in (
        "action_response_fn_hz",
        "action_response_zeta",
        "action_response_delay_s",
        "action_response_gain",
        "action_response_bias_rad",
        "action_response_accel_limit_rad_s2",
    ):
        assert f"self.robot.{field} = backhand_v9.replace_r4" in body
    assert "self.robot.action_response_torque_projection_enable = True" in body
    assert "self.robot.action_response_torque_limit_nm = backhand_v9.replace_r4" in body
    for forbidden in (
        "self.robot.hit_plane_x",
        "self.ball.serve_",
        "self.reward.",
        "self.observations",
    ):
        assert forbidden not in body
    assert "DamiaoMITActuator" not in body
    assert "response_model_enable = False" not in body
    assert "if self._action_response_torque_projection_enable:" in ENV_SOURCE
    assert "delayed_command = self._project_action_response_torque(delayed_command)" in ENV_SOURCE
    assert '"Metrics/r4_torque_clip_frac"' in ENV_SOURCE
    assert '"Metrics/r4_torque_peak_nm"' in ENV_SOURCE


def test_v9_has_isolated_scratch_registry_and_safe_watchdog():
    import_block = REGISTRY_SOURCE.split(
        "from legged_lab.envs.a1_tt.a1_tt_config import (", 1
    )[1].split(")", 1)[0]
    for class_name in (
        "A1TableTennisBackhandV9EnvCfg",
        "A1TableTennisBackhandV9EvalEnvCfg",
        "A1TableTennisBackhandV9AgentCfg",
    ):
        assert class_name in import_block
    assert REGISTRY_SOURCE.count('"a1_tt_backhand_v9",') == 1
    assert REGISTRY_SOURCE.count('"a1_tt_backhand_v9_eval",') == 1
    assert 'experiment_name: str = "a1_tt_backhand_real_v9_r4fit_mit8"' in CONFIG_SOURCE
    assert 'run_name = "scratch_r108_v8reward_r4holdout_mit8_5k10k5k"' in CONFIG_SOURCE
    assert "TASK=a1_tt_backhand_v9" in WATCHDOG_SOURCE
    assert "EXP=logs/a1_tt_backhand_real_v9_r4fit_mit8" in WATCHDOG_SOURCE
    assert "TARGET=${A1_TT_TARGET:-19999}" in WATCHDOG_SOURCE
    assert "--num_envs 4096 --headless --predictor" in WATCHDOG_SOURCE
    assert "pkill" not in WATCHDOG_SOURCE
    assert "rm -" not in WATCHDOG_SOURCE

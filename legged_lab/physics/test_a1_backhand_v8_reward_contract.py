from __future__ import annotations

from pathlib import Path

import pytest

from legged_lab.physics import a1_backhand_v7_contract as v7
from legged_lab.physics import a1_backhand_v8_contract as v8


ROOT = Path(__file__).resolve().parents[2]
CONFIG_SOURCE = (ROOT / "legged_lab/envs/a1_tt/a1_tt_config.py").read_text()
ENV_SOURCE = (ROOT / "legged_lab/envs/base/tt_env.py").read_text()
REGISTRY_SOURCE = (ROOT / "legged_lab/envs/__init__.py").read_text()
WATCHDOG_SOURCE = (ROOT / "watchdog_a1_tt_backhand_v8.sh").read_text()


def _class_body(source: str, class_name: str, next_class_name: str) -> str:
    return source.split(f"class {class_name}", 1)[1].split(
        f"class {next_class_name}", 1
    )[0]


def test_v8_keeps_v7_geometry_physics_and_actor_interface():
    assert v8.MAX_ITERATIONS == v7.MAX_ITERATIONS == 20_000
    assert "class A1TableTennisBackhandV8EnvCfg(A1TableTennisBackhandV7EnvCfg)" in CONFIG_SOURCE
    body = _class_body(
        CONFIG_SOURCE,
        "A1TableTennisBackhandV8EnvCfg",
        "A1TableTennisBackhandV8EvalEnvCfg",
    )
    for forbidden in (
        "self.scene.robot",
        "self.robot.hit_plane_x",
        "self.ball.serve_",
        "action_response_",
        "self.observations",
    ):
        assert forbidden not in body

    # Current perception actor frame is 39 values; five-frame history remains
    # 195.  ball_future_t stays in critic/reward state and is not concatenated
    # into current_actor_obs.
    actor_body = ENV_SOURCE.split("current_actor_obs = torch.cat(", 1)[1].split(
        "alt_critic_obs = torch.cat(", 1
    )[0]
    assert "self.ball_future_t" not in actor_body
    assert "actor_obs_history_length=5" in (
        ROOT / "legged_lab/envs/base/tt_env_config.py"
    ).read_text().replace(" ", "")


def test_v8_timing_and_wrist_cost_are_bounded_and_targeted():
    assert v8.SWING_WINDOW_S == pytest.approx(0.60)
    assert v8.SWING_TRANSITION_S == pytest.approx(0.10)
    assert v8.EARLY_MIN_RETRACTION_M == pytest.approx(0.08)
    assert v8.EARLY_MAX_EXCESS_M == pytest.approx(0.20)
    assert v8.EARLY_FORWARD_PENALTY_WEIGHT == pytest.approx(-0.50)
    assert v8.WRIST_ACTION_RATE_JOINT_WEIGHTS == (
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
    )
    assert v8.WRIST_ACTION_RATE_SWING_FLOOR == pytest.approx(0.25)
    assert v8.WRIST_ACTION_RATE_PENALTY_WEIGHT == pytest.approx(-0.08)
    assert abs(v8.WRIST_ACTION_RATE_PENALTY_WEIGHT) < 0.1


def test_v8_config_wires_reward_only_t_hit_and_windowed_drawdown():
    body = _class_body(
        CONFIG_SOURCE,
        "A1TableTennisBackhandV8EnvCfg",
        "A1TableTennisBackhandV8EvalEnvCfg",
    )
    assert "self.ball.precontact_drawdown_window_s" in body
    assert "mdp.penalty_early_paddle_forward" in body
    assert "self.reward.penalty_phase_wrist_action_rate.weight" in body
    assert "update_windowed_precontact_max_drawdown" in ENV_SOURCE


def test_v8_has_dedicated_scratch_registry_and_watchdog_namespaces():
    import_block = REGISTRY_SOURCE.split(
        "from legged_lab.envs.a1_tt.a1_tt_config import (", 1
    )[1].split(")", 1)[0]
    for class_name in (
        "A1TableTennisBackhandV8EnvCfg",
        "A1TableTennisBackhandV8EvalEnvCfg",
        "A1TableTennisBackhandV8AgentCfg",
    ):
        assert class_name in import_block
    assert REGISTRY_SOURCE.count('"a1_tt_backhand_v8",') == 1
    assert REGISTRY_SOURCE.count('"a1_tt_backhand_v8_eval",') == 1
    assert 'experiment_name: str = "a1_tt_backhand_real_v8_timing_wristquiet"' in CONFIG_SOURCE
    assert 'run_name = "scratch_r108_reward_timing_wristquiet_5k10k5k"' in CONFIG_SOURCE
    assert "TASK=a1_tt_backhand_v8" in WATCHDOG_SOURCE
    assert "EXP=logs/a1_tt_backhand_real_v8_timing_wristquiet" in WATCHDOG_SOURCE
    assert "TARGET=${A1_TT_TARGET:-19999}" in WATCHDOG_SOURCE
    assert "--num_envs 4096 --headless --predictor" in WATCHDOG_SOURCE
    assert "pkill" not in WATCHDOG_SOURCE
    assert "rm -" not in WATCHDOG_SOURCE

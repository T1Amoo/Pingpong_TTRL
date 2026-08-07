from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_SOURCE = (ROOT / "legged_lab/envs/a1_tt/a1_tt_config.py").read_text()
REGISTRY_SOURCE = (ROOT / "legged_lab/envs/__init__.py").read_text()


def test_v6_has_dedicated_reward_env_eval_and_agent_configs():
    for class_name in (
        "A1TableTennisBackhandV6RewardCfg",
        "A1TableTennisBackhandV6EnvCfg",
        "A1TableTennisBackhandV6EvalEnvCfg",
        "A1TableTennisBackhandV6AgentCfg",
    ):
        assert f"class {class_name}" in CONFIG_SOURCE

    v6_source = CONFIG_SOURCE.split(
        "class A1TableTennisBackhandV6EnvCfg", 1
    )[1].split("class A1TableTennisBackhandV6EvalEnvCfg", 1)[0]
    assert "precontact_drawdown_tracking_enable = True" in v6_source
    assert "mdp.reward_a1_future_yz_target" in v6_source
    assert "reward.reward_approach_velocity.weight = 0.0" in v6_source
    assert '"max_speed_mps": backhand_v6.SWING_THROUGH_CAP_MPS' in v6_source


def test_v6_tasks_are_registered_without_repointing_v5():
    assert REGISTRY_SOURCE.count('"a1_tt_backhand_v6",') == 1
    assert REGISTRY_SOURCE.count('"a1_tt_backhand_v6_eval",') == 1
    assert "A1TableTennisBackhandV6EnvCfg()" in REGISTRY_SOURCE
    assert "A1TableTennisBackhandV6EvalEnvCfg()" in REGISTRY_SOURCE
    assert REGISTRY_SOURCE.count('"a1_tt_backhand_v5",') == 1
    assert REGISTRY_SOURCE.count('"a1_tt_backhand_v5_eval",') == 1


def test_v5_env_does_not_enable_v6_tracking_or_rewards():
    v5_source = CONFIG_SOURCE.split(
        "class A1TableTennisBackhandV5EnvCfg", 1
    )[1].split("class A1TableTennisBackhandV5EvalEnvCfg", 1)[0]
    assert "precontact_drawdown_tracking_enable" not in v5_source
    assert "reward_a1_latched_forward_speed_quality" not in v5_source
    assert "penalty_a1_latched_precontact_max_drawdown" not in v5_source

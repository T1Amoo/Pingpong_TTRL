from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2] / "watchdog_a1_tt_backhand_v6.sh"
).read_text()


def test_v6_watchdog_uses_an_independent_task_and_artifact_namespace():
    assert "TASK=a1_tt_backhand_v6" in SCRIPT
    assert "EXP=logs/a1_tt_backhand_real_v6_speedquality_drawdown_yzonly" in SCRIPT
    assert "train_a1_tt_backhand_v6_trainer.pid" in SCRIPT
    assert "train_a1_tt_backhand_v6_watchdog.pid" in SCRIPT
    assert "a1_tt_backhand_real_v5" not in SCRIPT
    assert "train_a1_tt_backhand_v5" not in SCRIPT


def test_v6_watchdog_preserves_the_audited_resume_clock():
    assert 'if [ -n "$iter" ]; then' in SCRIPT
    assert 'if [ "$iter" -gt 0 ]; then' not in SCRIPT
    assert "TT_SIM_STEP_OFFSET=$(((iter + 1) * 240))" in SCRIPT
    assert "unset TT_PPO_NUM_STEPS_PER_ENV TT_PPO_NUM_MINI_BATCHES" in SCRIPT
    assert "TARGET=19999" in SCRIPT
    assert "--task \"$TASK\" --num_envs 4096 --headless --predictor" in SCRIPT

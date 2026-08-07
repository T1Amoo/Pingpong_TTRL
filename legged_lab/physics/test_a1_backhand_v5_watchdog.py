from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2] / "watchdog_a1_tt_backhand_v5.sh"
).read_text()


def test_watchdog_resumes_model_zero_instead_of_restarting_scratch():
    assert 'if [ -n "$iter" ]; then' in SCRIPT
    assert 'if [ "$iter" -gt 0 ]; then' not in SCRIPT


def test_watchdog_resume_clock_counts_completed_checkpoint_iteration():
    assert 'TT_SIM_STEP_OFFSET=$(((iter + 1) * 240))' in SCRIPT


def test_watchdog_pins_the_steps_per_iteration_contract():
    assert "unset TT_PPO_NUM_STEPS_PER_ENV TT_PPO_NUM_MINI_BATCHES" in SCRIPT
    assert "TARGET=19999" in SCRIPT
    assert "TASK=a1_tt_backhand_v5" in SCRIPT

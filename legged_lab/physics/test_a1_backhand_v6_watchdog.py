from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2] / "watchdog_a1_tt_backhand_v6.sh"
).read_text()
TO30K_SCRIPT = (
    Path(__file__).resolve().parents[2] / "watchdog_a1_tt_backhand_v6_to30k.sh"
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
    assert "TARGET=${A1_TT_TARGET:-19999}" in SCRIPT
    assert "--task \"$TASK\" --num_envs 4096 --headless --predictor" in SCRIPT


def test_v6_to30k_wrapper_locks_the_canonical_unchanged_range_resume():
    assert "export A1_TT_TARGET=29999" in TO30K_SCRIPT
    assert (
        "export A1_TT_RESUME_RUN="
        "2026-08-07_10-31-09_scratch_speedquality_drawdown_yzonly_weakspin_5k10k5k"
    ) in TO30K_SCRIPT
    assert "export TT_RUN_NAME=resume_model19999_finalrange_hold10k_to30k" in TO30K_SCRIPT
    assert "export LOAD_OPTIMIZER=1" in TO30K_SCRIPT
    assert "export A1_TT_RUNTIME_PROBES=0" in TO30K_SCRIPT
    assert "exec bash watchdog_a1_tt_backhand_v6.sh" in TO30K_SCRIPT


def test_v6_watchdog_allows_a_separate_continuation_artifact_namespace():
    assert "A1_TT_TRAIN_LOG" in SCRIPT
    assert "A1_TT_WATCHDOG_LOG" in SCRIPT
    assert "A1_TT_TRAINER_PID_FILE" in SCRIPT
    assert "A1_TT_WATCHDOG_PID_FILE" in SCRIPT


def test_v6_to30k_wrapper_binds_the_audited_source_checkpoint():
    assert "export A1_TT_REQUIRED_SOURCE_CKPT=" in TO30K_SCRIPT
    assert "model_19999.pt" in TO30K_SCRIPT
    assert (
        "export A1_TT_REQUIRED_SOURCE_SHA256="
        "413c81cf45cab2186ad71ba5c79f5e54448cf4b0094c67fb6aa5e8267e723faf"
    ) in TO30K_SCRIPT
    assert "A1_TT_REQUIRED_SOURCE_CKPT" in SCRIPT
    assert "A1_TT_REQUIRED_SOURCE_SHA256" in SCRIPT
    assert "sha256sum" in SCRIPT
    assert "A1_TT_EXPECTED_HEAD" in SCRIPT

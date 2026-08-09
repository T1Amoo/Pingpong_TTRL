#!/usr/bin/env bash
set -eu

# Continue the audited v6 canonical model_19999 at the unchanged final hard
# distribution.  This wrapper changes only the target iteration and artifact
# namespace; task, experiment, optimizer, predictor, curriculum endpoint and
# 4096x24x64 PPO contract remain the original v6 values.
export A1_TT_TARGET=29999
export A1_TT_RESUME_RUN=2026-08-07_10-31-09_scratch_speedquality_drawdown_yzonly_weakspin_5k10k5k
export TT_RUN_NAME=resume_model19999_finalrange_hold10k_to30k
export LOAD_OPTIMIZER=1
export A1_TT_RUNTIME_PROBES=0
export A1_TT_REQUIRED_SOURCE_CKPT=logs/a1_tt_backhand_real_v6_speedquality_drawdown_yzonly/2026-08-07_10-31-09_scratch_speedquality_drawdown_yzonly_weakspin_5k10k5k/model_19999.pt
export A1_TT_REQUIRED_SOURCE_SHA256=413c81cf45cab2186ad71ba5c79f5e54448cf4b0094c67fb6aa5e8267e723faf

export A1_TT_TRAIN_LOG=train_a1_tt_backhand_real_v6_finalrange_to30k.log
export A1_TT_WATCHDOG_LOG=train_a1_tt_backhand_real_v6_finalrange_to30k_watchdog.log
export A1_TT_TRAINER_PID_FILE=train_a1_tt_backhand_v6_to30k_trainer.pid
export A1_TT_WATCHDOG_PID_FILE=train_a1_tt_backhand_v6_to30k_watchdog.pid

exec bash watchdog_a1_tt_backhand_v6.sh

#!/usr/bin/env bash
set -u

cd /mnt/workspace/Pingpong_TTRL || exit 1
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate pingpong

export OMNI_KIT_ACCEPT_EULA=YES
export TT_SERVE_PROBE=1

EXP=logs/a1_tt_backhand_real_v1_y055h105
TARGET=29999
TRAIN_LOG=train_a1_tt_backhand_real_v1_y055h105.log
WATCHDOG_LOG=train_a1_tt_backhand_real_v1_y055h105_watchdog.log

max_iter() {
  find "$EXP" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^model_\([0-9][0-9]*\)\.pt$/\1/p' \
    | sort -n \
    | tail -1
}

run_of_iter() {
  find "$EXP" -mindepth 2 -maxdepth 2 -type f -name "model_$1.pt" -printf '%h\n' 2>/dev/null \
    | head -1 \
    | xargs -r basename
}

launch() {
  local iter run remaining
  iter=$(max_iter)
  iter=${iter:-0}
  if [ "$iter" -gt 0 ]; then
    run=$(run_of_iter "$iter")
    if [ "$iter" -ge "$TARGET" ]; then
      echo "[watchdog] $(date --iso-8601=seconds) target=$TARGET already complete at checkpoint=$iter" >> "$WATCHDOG_LOG"
      return 1
    fi
    remaining=$((TARGET + 1 - iter))
    export TT_SIM_STEP_OFFSET=$((iter * 240))
    echo "[watchdog] $(date --iso-8601=seconds) resume run=$run model_$iter.pt remaining=$remaining curriculum_raw_step=$TT_SIM_STEP_OFFSET" >> "$WATCHDOG_LOG"
    python -u -m legged_lab.scripts.train \
      --task a1_tt_backhand --num_envs 4096 --headless --predictor \
      --resume True --load_run "$run" --checkpoint "model_$iter.pt" \
      --max_iterations "$remaining" \
      >> "$TRAIN_LOG" 2>&1 &
  else
    unset TT_SIM_STEP_OFFSET
    echo "[watchdog] $(date --iso-8601=seconds) scratch launch" >> "$WATCHDOG_LOG"
    python -u -m legged_lab.scripts.train \
      --task a1_tt_backhand --num_envs 4096 --headless --predictor \
      >> "$TRAIN_LOG" 2>&1 &
  fi
  TRAIN_PID=$!
  echo "[watchdog] $(date --iso-8601=seconds) trainer_pid=$TRAIN_PID" >> "$WATCHDOG_LOG"
}

if ! launch; then
  exit 0
fi
while true; do
  sleep 60
  iter=$(max_iter)
  iter=${iter:-0}
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    alive=1
  else
    alive=0
  fi
  echo "[watchdog] $(date --iso-8601=seconds) checkpoint=$iter alive=$alive" >> "$WATCHDOG_LOG"

  # max_iterations=30000 owns normal termination.  The watchdog never kills a
  # healthy trainer; after model_29999 exists it only waits for natural exit.
  if [ "$alive" = "0" ] && [ "$iter" -ge "$TARGET" ]; then
    echo "[watchdog] $(date --iso-8601=seconds) target=$TARGET complete" >> "$WATCHDOG_LOG"
    break
  fi
  if [ "$alive" = "0" ]; then
    echo "[watchdog] $(date --iso-8601=seconds) trainer exited at checkpoint=$iter; relaunch" >> "$WATCHDOG_LOG"
    if ! launch; then
      break
    fi
  fi
done

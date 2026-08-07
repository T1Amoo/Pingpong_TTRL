#!/usr/bin/env bash
set -u

cd /mnt/workspace/Pingpong_TTRL || exit 1
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate pingpong

export OMNI_KIT_ACCEPT_EULA=YES
# V5 timing/serve curricula use the audited 24 control steps * 10 physics
# substeps = 240 raw steps per PPO iteration.  Production also uses the
# inherited 64 mini-batches at 4096 envs.  Do not inherit small-smoke overrides
# that would silently invalidate the resume clock or optimizer batch contract.
unset TT_PPO_NUM_STEPS_PER_ENV TT_PPO_NUM_MINI_BATCHES

# Runtime probes synchronize CUDA and are intentionally opt-in.  The physical
# reset-time serve rejection and predictor labels remain active either way.
if [ "${A1_TT_RUNTIME_PROBES:-0}" = "1" ]; then
  export TT_SERVE_PROBE=1
  export TT_SERVE_PROBE_BATCH=${TT_SERVE_PROBE_BATCH:-1000}
  export TT_SPIN_PROBE=1
  export TT_SPIN_PROBE_BATCH=${TT_SPIN_PROBE_BATCH:-1000}
  export TT_POST_IMPACT_PROBE=1
  export TT_POST_IMPACT_PROBE_BATCH=${TT_POST_IMPACT_PROBE_BATCH:-1000}
else
  unset TT_SERVE_PROBE TT_SERVE_PROBE_BATCH
  unset TT_SPIN_PROBE TT_SPIN_PROBE_BATCH
  unset TT_POST_IMPACT_PROBE TT_POST_IMPACT_PROBE_BATCH
fi

EXP=logs/a1_tt_backhand_real_v5_hitfirst_phasegate_weakspin
TARGET=19999
TASK=a1_tt_backhand_v5
TRAIN_LOG=train_a1_tt_backhand_real_v5_hitfirst_phasegate_weakspin.log
WATCHDOG_LOG=train_a1_tt_backhand_real_v5_hitfirst_phasegate_weakspin_watchdog.log

max_iter() {
  find "$EXP" -mindepth 2 -maxdepth 2 -type f -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^model_\([0-9][0-9]*\)\.pt$/\1/p' \
    | sort -n \
    | tail -1
}

run_of_iter() {
  local preferred_run=${A1_TT_RESUME_RUN:-}
  if [ -n "$preferred_run" ] && [ -f "$EXP/$preferred_run/model_$1.pt" ]; then
    echo "$preferred_run"
    return 0
  fi
  find "$EXP" -mindepth 2 -maxdepth 2 -type f -name "model_$1.pt" -printf '%h\n' 2>/dev/null \
    | head -1 \
    | xargs -r basename
}

launch() {
  local iter run remaining
  iter=$(max_iter)
  if [ -n "$iter" ]; then
    run=$(run_of_iter "$iter")
    if [ "$iter" -ge "$TARGET" ]; then
      echo "[watchdog] $(date --iso-8601=seconds) target=$TARGET already complete at checkpoint=$iter" >> "$WATCHDOG_LOG"
      return 1
    fi
    remaining=$((TARGET + 1 - iter))
    # model_N.pt is saved after completing iteration N.  The runner repeats
    # label N on resume, but environment time must continue after N+1 rollouts.
    export TT_SIM_STEP_OFFSET=$(((iter + 1) * 240))
    echo "[watchdog] $(date --iso-8601=seconds) resume task=$TASK run=$run model_$iter.pt remaining=$remaining curriculum_raw_step=$TT_SIM_STEP_OFFSET" >> "$WATCHDOG_LOG"
    python -u -m legged_lab.scripts.train \
      --task "$TASK" --num_envs 4096 --headless --predictor \
      --resume True --load_run "$run" --checkpoint "model_$iter.pt" \
      --max_iterations "$remaining" \
      >> "$TRAIN_LOG" 2>&1 &
  else
    unset TT_SIM_STEP_OFFSET
    echo "[watchdog] $(date --iso-8601=seconds) scratch task=$TASK" >> "$WATCHDOG_LOG"
    python -u -m legged_lab.scripts.train \
      --task "$TASK" --num_envs 4096 --headless --predictor \
      >> "$TRAIN_LOG" 2>&1 &
  fi
  TRAIN_PID=$!
  echo "$TRAIN_PID" > train_a1_tt_backhand_v5_trainer.pid
  echo "[watchdog] $(date --iso-8601=seconds) trainer_pid=$TRAIN_PID" >> "$WATCHDOG_LOG"
}

echo "$$" > train_a1_tt_backhand_v5_watchdog.pid
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

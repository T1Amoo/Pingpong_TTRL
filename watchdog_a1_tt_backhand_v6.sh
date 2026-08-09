#!/usr/bin/env bash
set -u

cd /mnt/workspace/Pingpong_TTRL || exit 1
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate pingpong

export OMNI_KIT_ACCEPT_EULA=YES

if [ -n "${A1_TT_EXPECTED_HEAD:-}" ]; then
  actual_head=$(git rev-parse HEAD 2>/dev/null) || exit 2
  if [ "$actual_head" != "$A1_TT_EXPECTED_HEAD" ]; then
    echo "[watchdog] HEAD mismatch expected=$A1_TT_EXPECTED_HEAD actual=$actual_head" >&2
    exit 2
  fi
fi
if [ -n "${A1_TT_REQUIRED_SOURCE_CKPT:-}" ] || [ -n "${A1_TT_REQUIRED_SOURCE_SHA256:-}" ]; then
  if [ -z "${A1_TT_REQUIRED_SOURCE_CKPT:-}" ] || [ -z "${A1_TT_REQUIRED_SOURCE_SHA256:-}" ]; then
    echo "[watchdog] source checkpoint and SHA256 must be provided together" >&2
    exit 2
  fi
  if [ ! -f "$A1_TT_REQUIRED_SOURCE_CKPT" ]; then
    echo "[watchdog] required source checkpoint missing: $A1_TT_REQUIRED_SOURCE_CKPT" >&2
    exit 2
  fi
  actual_source_sha=$(sha256sum "$A1_TT_REQUIRED_SOURCE_CKPT" | awk '{print $1}')
  if [ "$actual_source_sha" != "$A1_TT_REQUIRED_SOURCE_SHA256" ]; then
    echo "[watchdog] source checkpoint SHA256 mismatch expected=$A1_TT_REQUIRED_SOURCE_SHA256 actual=$actual_source_sha" >&2
    exit 2
  fi
fi
# V6 inherits the audited 24 control steps * 10 physics substeps = 240
# raw steps per PPO iteration and the production 64-mini-batch optimizer.
unset TT_PPO_NUM_STEPS_PER_ENV TT_PPO_NUM_MINI_BATCHES

# Runtime probes synchronize CUDA and remain opt-in.  Physical serve rejection,
# predictor targets, and v6 reward events are active independently of probes.
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

EXP=logs/a1_tt_backhand_real_v6_speedquality_drawdown_yzonly
TARGET=${A1_TT_TARGET:-19999}
TASK=a1_tt_backhand_v6
TRAIN_LOG=${A1_TT_TRAIN_LOG:-train_a1_tt_backhand_real_v6_speedquality_drawdown_yzonly.log}
WATCHDOG_LOG=${A1_TT_WATCHDOG_LOG:-train_a1_tt_backhand_real_v6_speedquality_drawdown_yzonly_watchdog.log}
TRAINER_PID_FILE=${A1_TT_TRAINER_PID_FILE:-train_a1_tt_backhand_v6_trainer.pid}
WATCHDOG_PID_FILE=${A1_TT_WATCHDOG_PID_FILE:-train_a1_tt_backhand_v6_watchdog.pid}

case "$TARGET" in
  ''|*[!0-9]*)
    echo "[watchdog] invalid non-negative A1_TT_TARGET=$TARGET" >&2
    exit 2
    ;;
esac

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
    # model_N.pt is written after iteration N; environment time resumes after
    # N+1 completed rollouts even though the runner repeats label N.
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
  echo "$TRAIN_PID" > "$TRAINER_PID_FILE"
  echo "[watchdog] $(date --iso-8601=seconds) trainer_pid=$TRAIN_PID" >> "$WATCHDOG_LOG"
}

echo "$$" > "$WATCHDOG_PID_FILE"
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

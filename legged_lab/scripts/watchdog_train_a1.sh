#!/usr/bin/env bash
# Watchdog for a1_tt — A1 AGV ping-pong overnight training.
#
# WHY: Runs a new A1 experiment from a known warm-start checkpoint while keeping the task name
#   as a1_tt. By default this seeds a1_tt_v13 from a1_tt_v12/model_13500.pt, then trains to 30000.
#
# Isaac Sim simulation_app.close() busy-spins on teardown after the final ckpt flush.
#   We poll for the target checkpoint landing on disk and kill the hung child immediately
#   (same approach as watchdog_train_v16.sh, locomotion v2 zombie fix).
#
# KILL ORDER: always kill watchdog first, then trainer — kill watchdog PID ($$), not pattern.
#   To stop: kill <watchdog_pid>   (NOT pkill -f watchdog_train_a1 — self-kills the script)
#   To find trainer:  pgrep -f "[a]1_tt"  then kill <trainer_pid>
#
# Usage (local):
#   NUM_ENVS=128 nohup bash legged_lab/scripts/watchdog_train_a1.sh > /tmp/a1_watchdog.log 2>&1 &
#
# Usage (cloud, override python and envs):
#   TARGET=30000 NUM_ENVS=4096 TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python \
#     OMNI_KIT_ACCEPT_EULA=YES nohup bash legged_lab/scripts/watchdog_train_a1.sh > /dev/null 2>&1 &
set -u

export OMNI_KIT_ACCEPT_EULA=YES

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
TASK=${TASK:-a1_tt}
EXP=${EXP:-a1_tt_v13}
SEED_EXP=${SEED_EXP:-a1_tt_v12}
SEED_ITER=${SEED_ITER:-13500}
SEED_DIR_NAME=${SEED_DIR_NAME:-seed_v12_13500}
NUM_ENVS=${NUM_ENVS:-128}
TARGET=${TARGET:-30000}
LOGROOT="$REPO/logs/$EXP"
SEED_DIR="$LOGROOT/$SEED_DIR_NAME"
WLOG="$REPO/train_${EXP}_watchdog.log"
cd "$REPO"

CHILD=""
cleanup() {
  echo "[wd] $(date +%F_%H-%M-%S) signal; killing child $CHILD" | tee -a "$WLOG"
  [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null
  sleep 3
  [ -n "$CHILD" ] && kill -9 "$CHILD" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

latest() {
  local best=-1 bdir="" bfile="" f n
  shopt -s nullglob
  for f in "$LOGROOT"/*/model_*.pt; do
    n=$(basename "$f" | sed -E 's/model_([0-9]+)\.pt/\1/')
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    if [ "$n" -gt "$best" ]; then
      best=$n
      bfile=$(basename "$f")
      bdir=$(basename "$(dirname "$f")")
    fi
  done
  echo "$best|$bdir|$bfile"
}

find_seed() {
  local src="" f
  shopt -s nullglob
  for f in "$REPO/logs/$SEED_EXP"/*/model_${SEED_ITER}.pt; do
    src="$f"
  done
  echo "$src"
}

IFS='|' read -r INIT_N _ _ <<< "$(latest)"
if [ "$INIT_N" -lt 0 ]; then
  SRC="$(find_seed)"
  if [ -z "$SRC" ]; then
    echo "[wd] FATAL: no seed checkpoint logs/$SEED_EXP/*/model_${SEED_ITER}.pt found" | tee -a "$WLOG"
    exit 1
  fi
  mkdir -p "$SEED_DIR"
  cp "$SRC" "$SEED_DIR/model_${SEED_ITER}.pt"
  echo "[wd] $(date +%F_%H-%M-%S) seeded warm-start: $SRC -> $SEED_DIR/model_${SEED_ITER}.pt" | tee -a "$WLOG"
fi

echo "[wd] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK exp=$EXP seed=$SEED_EXP/model_$SEED_ITER" | tee -a "$WLOG"

while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -ge "$((TARGET-1))" ]; then
    echo "[wd] $(date +%F_%H-%M-%S) reached $N >= $TARGET. DONE." | tee -a "$WLOG"
    break
  fi
  REM=$((TARGET - N))

  export TT_SIM_STEP_OFFSET=$((N * 240))
  if [ "$DIR" = "$SEED_DIR_NAME" ] && [ "$N" -eq "$SEED_ITER" ]; then
    export LOAD_OPTIMIZER=0
    echo "[wd] $(date +%F_%H-%M-%S) WARM-START $DIR/$FILE (iter $N) -> +$REM; LOAD_OPTIMIZER=0 TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
  else
    export LOAD_OPTIMIZER=1
    echo "[wd] $(date +%F_%H-%M-%S) resume $DIR/$FILE (iter $N) -> +$REM; LOAD_OPTIMIZER=1 TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
  fi
  "$PY" -u -m legged_lab.scripts.train --task="$TASK" --num_envs="$NUM_ENVS" --headless \
    --logger=tensorboard --predictor --experiment_name="$EXP" --max_iterations="$REM" \
    --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
  CHILD=$!

  # Poll; once target ckpt is on disk, kill the hung child (Isaac shutdown-hang workaround).
  while kill -0 "$CHILD" 2>/dev/null; do
    IFS='|' read -r NC _ _ <<< "$(latest)"
    if [ "$NC" -ge "$((TARGET-1))" ]; then
      echo "[wd] $(date +%F_%H-%M-%S) target ckpt $NC saved; killing child $CHILD (Isaac shutdown-hang workaround)" | tee -a "$WLOG"
      kill -9 "$CHILD" 2>/dev/null
      break
    fi
    sleep 30
  done

  wait "$CHILD" 2>/dev/null; CODE=$?; CHILD=""
  IFS='|' read -r N2 _ _ <<< "$(latest)"
  echo "[wd] $(date +%F_%H-%M-%S) trainer exited code=$CODE; iter=$N2" | tee -a "$WLOG"

  if [ "$N2" -ge "$((TARGET-1))" ]; then
    echo "[wd] completed at $N2." | tee -a "$WLOG"
    break
  fi

  echo "[wd] resume in 20s (to stop: kill $$ first, then pgrep -f '[a]1_tt')" | tee -a "$WLOG"
  sleep 20
done

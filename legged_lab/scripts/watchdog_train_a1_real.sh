#!/usr/bin/env bash
# Watchdog for a1_tt_real - A1 table-tennis real-aligned scratch training.
#
# This run intentionally starts from scratch: no v12/v13 warm-start and no dedicated
# no-ball curriculum. a1_tt_real_v6 uses the seven-joint identified second-order
# arm response model, fixed-sj USD, v4 PPO stability settings, and a 5k easy /
# 5k serve-ramp / 20k hard-hold curriculum.
#
# Usage (local):
#   NUM_ENVS=128 nohup bash legged_lab/scripts/watchdog_train_a1_real.sh > /tmp/a1_real_watchdog.log 2>&1 &
#
# Usage (cloud):
#   TARGET=30000 NUM_ENVS=4096 TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python \
#     OMNI_KIT_ACCEPT_EULA=YES nohup bash legged_lab/scripts/watchdog_train_a1_real.sh >/dev/null 2>&1 &
set -u

export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
TASK=${TASK:-a1_tt_real}
EXP=${EXP:-a1_tt_real_v6}
NUM_ENVS=${NUM_ENVS:-128}
TARGET=${TARGET:-30000}
LOGROOT="$REPO/logs/$EXP"
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

echo "[wd] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK exp=$EXP (scratch, fixed-sj, v4 PPO, paddle-above penalty, 5k easy / 5k ramp / 20k hold)" | tee -a "$WLOG"
while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -ge "$((TARGET-1))" ]; then
    echo "[wd] $(date +%F_%H-%M-%S) reached $N >= $TARGET. DONE." | tee -a "$WLOG"
    break
  fi

  if [ "$N" -lt 0 ]; then
    echo "[wd] $(date +%F_%H-%M-%S) fresh scratch start -> $TARGET" | tee -a "$WLOG"
    export TT_SIM_STEP_OFFSET=0
    "$PY" -u -m legged_lab.scripts.train --task="$TASK" --num_envs="$NUM_ENVS" --headless \
      --logger=tensorboard --predictor --experiment_name="$EXP" --max_iterations="$TARGET" >> "$WLOG" 2>&1 &
  else
    REM=$((TARGET - N))
    export TT_SIM_STEP_OFFSET=$((N * 240))
    echo "[wd] $(date +%F_%H-%M-%S) resume $DIR/$FILE (iter $N) -> +$REM; TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
    "$PY" -u -m legged_lab.scripts.train --task="$TASK" --num_envs="$NUM_ENVS" --headless \
      --logger=tensorboard --predictor --experiment_name="$EXP" --max_iterations="$REM" \
      --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
  fi
  CHILD=$!

  while kill -0 "$CHILD" 2>/dev/null; do
    IFS='|' read -r NC _ _ <<< "$(latest)"
    if [ "$NC" -ge "$((TARGET-1))" ]; then
      echo "[wd] $(date +%F_%H-%M-%S) target ckpt $NC saved; killing child $CHILD (Isaac shutdown-hang workaround)" | tee -a "$WLOG"
      kill -9 "$CHILD" 2>/dev/null
      break
    fi
    sleep 30
  done

  wait "$CHILD" 2>/dev/null
  CODE=$?
  CHILD=""
  IFS='|' read -r N2 _ _ <<< "$(latest)"
  echo "[wd] $(date +%F_%H-%M-%S) trainer exited code=$CODE; iter=$N2" | tee -a "$WLOG"

  if [ "$N2" -ge "$((TARGET-1))" ]; then
    echo "[wd] completed at $N2." | tee -a "$WLOG"
    break
  fi

  echo "[wd] resume in 20s (to stop: kill $$ first, then pgrep -f '[a]1_tt_real')" | tee -a "$WLOG"
  sleep 20
done

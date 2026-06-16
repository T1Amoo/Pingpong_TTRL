#!/usr/bin/env bash
# Watchdog for g1_tt idle11 overnight training.
# idle11 = idle10 sequential curriculum (easy -> difficulty -> idle/no-ball) RE-RUN on the
# BUG-FIXED env: Bug A (no-ball reset_ball no longer inflates ball_reset_counter -> no spurious
# full env reset every ~6 steps) + Bug B (predictor history no longer poisoned by the parked
# z=-50 ball). Clean A/B test: does fixing the two env/predictor bugs make idle trainable?
# - Trains g1_tt (predictor-augmented, 4096 envs) up to TARGET iterations.
# - Auto-resumes from the numerically-latest checkpoint on crash/OOM (curriculum clock kept
#   via TT_SIM_STEP_OFFSET = N*240 so the 3-stage curriculum CONTINUES, not replays).
# - To STOP for good: kill THIS watchdog's PID (forwards kill to the child trainer).
#
# Usage (survives terminal close):
#   TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python nohup bash legged_lab/scripts/watchdog_train_idle11.sh > /dev/null 2>&1 &
#   echo $!            # <- watchdog PID; `kill <PID>` to stop everything
# Watch: tail -f train_idle11_watchdog.log
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
TASK=g1_tt
EXP=g1_tt_idle11
NUM_ENVS=4096
TARGET=30000
LOGROOT="$REPO/logs/$EXP"
WLOG="$REPO/train_idle11_watchdog.log"
cd "$REPO"

CHILD=""
cleanup() { echo "[watchdog] $(date +%F_%H-%M-%S) signal received; killing trainer $CHILD" | tee -a "$WLOG"; [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; sleep 3; [ -n "$CHILD" ] && kill -9 "$CHILD" 2>/dev/null; exit 0; }
trap cleanup INT TERM

latest() {
  local best=-1 bdir="" bfile="" f n
  shopt -s nullglob
  for f in "$LOGROOT"/*/model_*.pt; do
    n=$(basename "$f" | sed -E 's/model_([0-9]+)\.pt/\1/')
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    if [ "$n" -gt "$best" ]; then best=$n; bfile=$(basename "$f"); bdir=$(basename "$(dirname "$f")"); fi
  done
  echo "$best|$bdir|$bfile"
}

echo "[watchdog] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK exp=$EXP" | tee -a "$WLOG"
while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -ge "$((TARGET-1))" ]; then
    echo "[watchdog] $(date +%F_%H-%M-%S) reached iter $N >= $TARGET. DONE." | tee -a "$WLOG"; break
  fi

  if [ "$N" -lt 0 ]; then
    echo "[watchdog] $(date +%F_%H-%M-%S) fresh start -> $TARGET iters" | tee -a "$WLOG"
    export TT_SIM_STEP_OFFSET=0
    "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless \
      --logger=tensorboard --predictor --max_iterations=$TARGET >> "$WLOG" 2>&1 &
  else
    REM=$((TARGET - N))
    export TT_SIM_STEP_OFFSET=$((N * 240))
    echo "[watchdog] $(date +%F_%H-%M-%S) resume from $DIR/$FILE (iter $N); run $REM more -> $TARGET; TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
    "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless \
      --logger=tensorboard --predictor --max_iterations=$REM \
      --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
  fi
  CHILD=$!
  wait "$CHILD"; CODE=$?
  CHILD=""

  IFS='|' read -r N2 _ _ <<< "$(latest)"
  echo "[watchdog] $(date +%F_%H-%M-%S) trainer exited code=$CODE; latest iter=$N2" | tee -a "$WLOG"
  if [ "$CODE" -eq 0 ] && [ "$N2" -ge "$((TARGET-1))" ]; then
    echo "[watchdog] $(date +%F_%H-%M-%S) completed cleanly at $N2." | tee -a "$WLOG"; break
  fi
  echo "[watchdog] interrupted/crashed (code=$CODE, iter=$N2). Resuming in 20s. (kill watchdog PID $$ to stop)" | tee -a "$WLOG"
  sleep 20
done

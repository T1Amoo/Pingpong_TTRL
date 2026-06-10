#!/usr/bin/env bash
# Watchdog for g1_tt overnight training.
# - Trains g1_tt (predictor-augmented, 4096 envs) up to TARGET iterations.
# - If training crashes / OOMs / is interrupted, auto-resumes from the
#   numerically-latest checkpoint (capping total iters at TARGET).
# - To STOP for good: kill THIS watchdog's PID (it forwards the kill to the
#   child trainer). Killing only the python trainer will trigger a resume.
#
# Usage (survives terminal close):
#   nohup bash legged_lab/scripts/watchdog_train_g1.sh > /dev/null 2>&1 &
#   echo $!            # <- this PID is the watchdog; `kill <PID>` to stop everything
# Watch progress:
#   tail -f train_rally_watchdog.log
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"          # legged_lab/scripts/ -> repo root
PY=/home/woan/.conda/envs/pingpong/bin/python
TASK=g1_tt
EXP=g1_tt_rally
NUM_ENVS=4096
TARGET=37500
LOGROOT="$REPO/logs/$EXP"
WLOG="$REPO/train_rally_watchdog.log"
cd "$REPO"

CHILD=""
cleanup() { echo "[watchdog] $(date +%F_%H-%M-%S) signal received; killing trainer $CHILD" | tee -a "$WLOG"; [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; sleep 3; [ -n "$CHILD" ] && kill -9 "$CHILD" 2>/dev/null; exit 0; }
trap cleanup INT TERM

# echo "N|rundir|file" for the numerically-highest model_*.pt across all runs, else "-1||"
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

echo "[watchdog] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK" | tee -a "$WLOG"
while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -ge "$TARGET" ]; then
    echo "[watchdog] $(date +%F_%H-%M-%S) reached iter $N >= $TARGET. DONE." | tee -a "$WLOG"; break
  fi

  if [ "$N" -lt 0 ]; then
    echo "[watchdog] $(date +%F_%H-%M-%S) fresh start -> $TARGET iters" | tee -a "$WLOG"
    "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless \
      --logger=tensorboard --predictor --max_iterations=$TARGET >> "$WLOG" 2>&1 &
  else
    REM=$((TARGET - N))
    echo "[watchdog] $(date +%F_%H-%M-%S) resume from $DIR/$FILE (iter $N); run $REM more -> $TARGET" | tee -a "$WLOG"
    "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless \
      --logger=tensorboard --predictor --max_iterations=$REM \
      --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
  fi
  CHILD=$!
  wait "$CHILD"; CODE=$?
  CHILD=""

  IFS='|' read -r N2 _ _ <<< "$(latest)"
  echo "[watchdog] $(date +%F_%H-%M-%S) trainer exited code=$CODE; latest iter=$N2" | tee -a "$WLOG"
  if [ "$CODE" -eq 0 ] && [ "$N2" -ge "$TARGET" ]; then
    echo "[watchdog] $(date +%F_%H-%M-%S) completed cleanly at $N2." | tee -a "$WLOG"; break
  fi
  echo "[watchdog] interrupted/crashed (code=$CODE, iter=$N2). Resuming in 20s. (kill watchdog PID $$ to stop)" | tee -a "$WLOG"
  sleep 20
done

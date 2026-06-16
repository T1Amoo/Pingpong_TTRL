#!/usr/bin/env bash
# Watchdog for g1_tt idle12 = WARM-START from idle10 model_10000 on the BUG-FIXED env.
# Hypothesis under test: with Bug A/B fixed, idle is FREE from inter-serve gaps (no no-ball
# injection, no_ball_period_s=0). So instead of relearning easy-ball from scratch, warm-start the
# clean easy-ball hitter (model_10000, succ~0.68) and spend the run on HARDER serves via a
# PERFORMANCE-GATED curriculum (serve difficulty advances only while success-return rate >= window).
#
# WARM-START mechanism: seed logs/g1_tt_idle12/<seed>/model_10000.pt (copied from the idle10 run)
# so latest() finds iter 10000 and RESUMES it (LOAD_OPTIMIZER=0 -> fresh optimizer for the shifted
# distribution). TT_SIM_STEP_OFFSET=10000*240 keeps the clock continuous (mostly cosmetic here:
# difficulty is perf-gated by serve_c, not sim_step, and no-ball is off).
#
# Usage:
#   TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python nohup bash legged_lab/scripts/watchdog_train_idle12.sh > /dev/null 2>&1 &
# Watch: tail -f train_idle12_watchdog.log ; grep "\[curriculum\]" train_idle12_watchdog.log
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
TASK=g1_tt
EXP=g1_tt_idle12
NUM_ENVS=4096
TARGET=25000
LOGROOT="$REPO/logs/$EXP"
WLOG="$REPO/train_idle12_watchdog.log"
export LOAD_OPTIMIZER=0          # warm-start into a shifted (harder-serve) distribution -> fresh optimizer
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

echo "[watchdog] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK exp=$EXP (warm-start, LOAD_OPTIMIZER=0)" | tee -a "$WLOG"
while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -lt 0 ]; then
    echo "[watchdog] $(date +%F_%H-%M-%S) ERROR: no seed checkpoint in $LOGROOT (expected model_10000.pt seeded). Aborting." | tee -a "$WLOG"; exit 1
  fi
  if [ "$N" -ge "$((TARGET-1))" ]; then
    echo "[watchdog] $(date +%F_%H-%M-%S) reached iter $N >= $TARGET. DONE." | tee -a "$WLOG"; break
  fi
  REM=$((TARGET - N))
  export TT_SIM_STEP_OFFSET=$((N * 240))
  echo "[watchdog] $(date +%F_%H-%M-%S) resume from $DIR/$FILE (iter $N); run $REM more -> $TARGET; TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET LOAD_OPTIMIZER=$LOAD_OPTIMIZER" | tee -a "$WLOG"
  "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless \
    --logger=tensorboard --predictor --max_iterations=$REM \
    --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
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

#!/usr/bin/env bash
# Watchdog for g1_tt_v7 — FROM-SCRATCH training with the -2.0-corrected serve.
#
# WHY from-scratch (not warm-start): v6 failed because the SERVE was tuned for the old -1.6
#   stance, so at -2.0 the ball dropped to z~0.5-0.7 / never arrived -> returning was physically
#   impossible (contact-only local optimum + divergence). The serve is now re-tuned (flat+fast+
#   deep) so the ball reaches -2.0 at z~1.0 (paddle ready height) — see g1_tt_config serve block.
#   With the real cause fixed, from-scratch is the right recipe (the proven 2026-06-03 approach).
#
# User-specified FIXED-ITER 3-stage curriculum (sim_step-keyed):
#   Stage 1 (iter 0     -> 15000): fixed EASY serve, pure hitting bootstrap.
#   Stage 2 (iter 15000 -> 25000): serve difficulty ramps easy->hard over 10000 iter.
#   Stage 3 (iter 25000 -> 30000): consolidate at full difficulty.
# NO idle / NO no-ball. TARGET=30000.
#
# CURRICULUM-ON-RESUME: serve difficulty is sim_step-keyed -> on RESUME export
#   TT_SIM_STEP_OFFSET=N*240 so the curriculum CONTINUES at the right stage (no rewind to easy).
#   240 = decimation 10 * num_steps_per_env 24.
#
# Usage:
#   TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python nohup bash legged_lab/scripts/watchdog_train_v7.sh > /dev/null 2>&1 &
#   echo $!            # watchdog PID; `kill <PID>` to stop everything
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
TASK=g1_tt
EXP=g1_tt_v7
NUM_ENVS=4096
TARGET=30000
LOGROOT="$REPO/logs/$EXP"
WLOG="$REPO/train_v7_watchdog.log"
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

echo "[watchdog] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK exp=$EXP (FROM-SCRATCH, -2.0 serve)" | tee -a "$WLOG"
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

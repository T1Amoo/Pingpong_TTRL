#!/usr/bin/env bash
# Watchdog for g1_tt_v7 — WARM-START fine-tune (NOT from-scratch).
#
# WHY v7: v6 (from-scratch, stance -2.0) FAILED two ways: (1) never learned to RETURN
#   (contact-only local optimum), (2) diverged ~iter39k. ROOT CAUSE found by serve probe:
#   the serve was tuned for the OLD -1.6 stance; at -2.0 the ball had already dropped to
#   z~0.5-0.7 (or didn't arrive) -> impossible to return. v7 fixes BOTH:
#     * serve re-tuned (flat+fast+deep) so ball reaches -2.0 at z~1.0 (paddle ready height);
#     * WARM-START from idle12 model_24500 (a policy that already RETURNS well at -1.6) so the
#       return skill transfers and only needs to re-aim to the new geometry — avoids the
#       from-scratch contact-only trap. no_ball injection DROPPED (v6 diverged on it).
#
# WARM-START MECHANIC: rsl_rl get_checkpoint_path resolves load_run WITHIN logs/<experiment>.
#   So model_24500.pt is copied into logs/g1_tt_v7/seed_24500/ (done once, below). The FIRST
#   run loads it with LOAD_OPTIMIZER=0 (fresh optimizer) + TT_SIM_STEP_OFFSET=0 (curriculum
#   from easy). latest() EXCLUDES seed_* dirs so the watchdog never mistakes the seed for a
#   real checkpoint. After the first real v7 ckpt is saved, crashes resume from it normally.
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
TARGET=25000
LOGROOT="$REPO/logs/$EXP"
WLOG="$REPO/train_v7_watchdog.log"
SEED_SRC="$REPO/logs/g1_tt_idle12/2026-06-16_11-09-29/model_24500.pt"
SEED_DIR="$LOGROOT/seed_24500"
SEED_CKPT="model_24500.pt"
cd "$REPO"

# --- one-time: stage the warm-start seed checkpoint inside the v7 experiment dir ---
if [ ! -f "$SEED_DIR/$SEED_CKPT" ]; then
  if [ ! -f "$SEED_SRC" ]; then
    echo "[watchdog] FATAL: warm-start source $SEED_SRC not found" | tee -a "$WLOG"; exit 1
  fi
  mkdir -p "$SEED_DIR"
  cp "$SEED_SRC" "$SEED_DIR/$SEED_CKPT"
  echo "[watchdog] staged warm-start seed -> $SEED_DIR/$SEED_CKPT" | tee -a "$WLOG"
fi

CHILD=""
cleanup() { echo "[watchdog] $(date +%F_%H-%M-%S) signal received; killing trainer $CHILD" | tee -a "$WLOG"; [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; sleep 3; [ -n "$CHILD" ] && kill -9 "$CHILD" 2>/dev/null; exit 0; }
trap cleanup INT TERM

# latest REAL v7 ckpt (EXCLUDES seed_* dirs)
latest() {
  local best=-1 bdir="" bfile="" f n d
  shopt -s nullglob
  for f in "$LOGROOT"/*/model_*.pt; do
    d=$(basename "$(dirname "$f")")
    [[ "$d" == seed_* ]] && continue
    n=$(basename "$f" | sed -E 's/model_([0-9]+)\.pt/\1/')
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    if [ "$n" -gt "$best" ]; then best=$n; bfile=$(basename "$f"); bdir="$d"; fi
  done
  echo "$best|$bdir|$bfile"
}

echo "[watchdog] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK exp=$EXP (WARM-START from $SEED_CKPT)" | tee -a "$WLOG"
while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -ge "$((TARGET-1))" ]; then
    echo "[watchdog] $(date +%F_%H-%M-%S) reached iter $N >= $TARGET. DONE." | tee -a "$WLOG"; break
  fi

  if [ "$N" -lt 0 ]; then
    # FIRST run: warm-start from the staged seed, fresh optimizer, curriculum from easy (offset 0)
    echo "[watchdog] $(date +%F_%H-%M-%S) WARM-START from seed_24500/$SEED_CKPT -> $TARGET iters (fresh optimizer, curriculum@0)" | tee -a "$WLOG"
    export TT_SIM_STEP_OFFSET=0
    LOAD_OPTIMIZER=0 "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless \
      --logger=tensorboard --predictor --max_iterations=$TARGET \
      --resume true --load_run seed_24500 --checkpoint "$SEED_CKPT" >> "$WLOG" 2>&1 &
  else
    REM=$((TARGET - N))
    export TT_SIM_STEP_OFFSET=$((N * 240))
    echo "[watchdog] $(date +%F_%H-%M-%S) resume from $DIR/$FILE (iter $N); run $REM more -> $TARGET; TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
    LOAD_OPTIMIZER=1 "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless \
      --logger=tensorboard --predictor --max_iterations=$REM \
      --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
  fi
  CHILD=$!
  wait "$CHILD"; CODE=$?
  CHILD=""

  IFS='|' read -r N2 _ _ <<< "$(latest)"
  echo "[watchdog] $(date +%F_%H-%M-%S) trainer exited code=$CODE; latest real iter=$N2" | tee -a "$WLOG"
  if [ "$CODE" -eq 0 ] && [ "$N2" -ge "$((TARGET-1))" ]; then
    echo "[watchdog] $(date +%F_%H-%M-%S) completed cleanly at $N2." | tee -a "$WLOG"; break
  fi
  echo "[watchdog] interrupted/crashed (code=$CODE, iter=$N2). Resuming in 20s. (kill watchdog PID $$ to stop)" | tee -a "$WLOG"
  sleep 20
done

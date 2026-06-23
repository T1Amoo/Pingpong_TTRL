#!/usr/bin/env bash
# Watchdog for g1_tt_v8 — DE-JITTER fine-tune, WARM-STARTED from v7's model_29999.
#
# WHY warm-start (not from-scratch): v7 already hits well (sim land 77%, deploy hits). The ONLY
#   goal here is to cut the steady-state jitter that was ~2x v5 on the real robot. So we fine-tune
#   the existing v7 policy under a STRONGER joint-acceleration penalty (g1_tt_config dof_acc_l2:
#   -1.25e-7 -> -3.75e-7, 3x). NO no-ball / NO idle (v6 diverged when no-ball ramped). Geometry
#   (serve, -2.0 stance) unchanged from v7.
#
# WARM-START across experiment dirs: v7's ckpt lives in logs/g1_tt_v7/, but train.py resolves
#   --load_run inside logs/<experiment_name>=logs/g1_tt_v8. So we SEED: copy v7's model_29999.pt
#   to logs/g1_tt_v8/seed_v7/ once, then warm-start from it (LOAD_OPTIMIZER=0 -> fresh optimizer
#   for the shifted-reward fine-tune, per train.py:97). New ckpts continue numbering 30000+ into a
#   fresh timestamped run dir under logs/g1_tt_v8/.
#
# CURRICULUM: v7 ended at iter 30000 = full difficulty (stage 3). We stay at full difficulty, so
#   TT_SIM_STEP_OFFSET = N*240 keeps the sim-step-keyed serve curriculum pinned at hard.
#   240 = decimation 10 * num_steps_per_env 24.
#
# TARGET=35000 (= 29999 warm-start + ~5000 fine-tune iters). SHORT on purpose: warm-start is a
#   known DEPLOYMENT KILLER if chased too far (D1 lesson) and TT best ckpts peak early. Do NOT
#   assume model_34999 is best — EVAL-SELECT afterward (1000 balls x 3 seeds + action/jitter
#   metrics, as for v7). v7 model_29999 / BASELINE stay as deploy fallbacks.
#
# Usage:
#   TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python nohup bash legged_lab/scripts/watchdog_train_v8.sh > /dev/null 2>&1 &
#   echo $!            # watchdog PID; `kill <PID>` to stop everything
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
TASK=g1_tt
EXP=g1_tt_v8
SEED_EXP=g1_tt_v7              # warm-start source experiment
SEED_ITER=29999               # warm-start source checkpoint iter
NUM_ENVS=${NUM_ENVS:-4096}
TARGET=${TARGET:-35000}
LOGROOT="$REPO/logs/$EXP"
SEED_DIR="$LOGROOT/seed_v7"
WLOG="$REPO/train_v8_watchdog.log"
cd "$REPO"

# --- one-time seed: copy v7 model_29999 into logs/g1_tt_v8/seed_v7/ so train.py can resolve it.
if [ ! -f "$SEED_DIR/model_${SEED_ITER}.pt" ]; then
  shopt -s nullglob
  src=""
  for f in "$REPO/logs/$SEED_EXP"/*/model_${SEED_ITER}.pt; do src="$f"; done
  if [ -z "$src" ]; then
    echo "[wd] FATAL: no $SEED_EXP/*/model_${SEED_ITER}.pt found to seed warm-start. Aborting." | tee -a "$WLOG"
    exit 1
  fi
  mkdir -p "$SEED_DIR"
  cp "$src" "$SEED_DIR/model_${SEED_ITER}.pt"
  echo "[wd] $(date +%F_%H-%M-%S) seeded warm-start: $src -> $SEED_DIR/model_${SEED_ITER}.pt" | tee -a "$WLOG"
fi

CHILD=""
cleanup() { echo "[wd] $(date +%F_%H-%M-%S) signal; killing $CHILD" | tee -a "$WLOG"; [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; sleep 3; [ -n "$CHILD" ] && kill -9 "$CHILD" 2>/dev/null; exit 0; }
trap cleanup INT TERM

latest() {
  local best=-1 bdir="" bfile="" f n; shopt -s nullglob
  for f in "$LOGROOT"/*/model_*.pt; do
    n=$(basename "$f" | sed -E 's/model_([0-9]+)\.pt/\1/'); [[ "$n" =~ ^[0-9]+$ ]] || continue
    if [ "$n" -gt "$best" ]; then best=$n; bfile=$(basename "$f"); bdir=$(basename "$(dirname "$f")"); fi
  done
  echo "$best|$bdir|$bfile"
}

echo "[wd] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK exp=$EXP (DE-JITTER warm-start from $SEED_EXP/model_$SEED_ITER)" | tee -a "$WLOG"
while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -ge "$((TARGET-1))" ]; then echo "[wd] $(date +%F_%H-%M-%S) reached $N >= $TARGET. DONE." | tee -a "$WLOG"; break; fi
  REM=$((TARGET - N))
  export TT_SIM_STEP_OFFSET=$((N * 240))
  if [ "$N" -eq "$SEED_ITER" ] && [ "$DIR" = "seed_v7" ]; then
    # first launch: warm-start from the seed with a FRESH optimizer (shifted reward)
    export LOAD_OPTIMIZER=0
    echo "[wd] $(date +%F_%H-%M-%S) WARM-START from seed_v7/$FILE (iter $N) -> +$REM; LOAD_OPTIMIZER=0 TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
  else
    # crash-resume of an in-progress v8 run: continue with its optimizer state
    export LOAD_OPTIMIZER=1
    echo "[wd] $(date +%F_%H-%M-%S) resume $DIR/$FILE (iter $N) -> +$REM; LOAD_OPTIMIZER=1 TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
  fi
  "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless \
    --logger=tensorboard --predictor --max_iterations=$REM \
    --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
  CHILD=$!
  # Isaac Sim 4.5's simulation_app.close() busy-spins forever on teardown after the final ckpt is
  # already flushed (locomotion v2 zombie root cause). Poll; once target ckpt is on disk, kill child.
  while kill -0 "$CHILD" 2>/dev/null; do
    IFS='|' read -r NC _ _ <<< "$(latest)"
    if [ "$NC" -ge "$((TARGET-1))" ]; then
      echo "[wd] $(date +%F_%H-%M-%S) target ckpt $NC saved; killing child $CHILD (Isaac shutdown-hang workaround)" | tee -a "$WLOG"
      kill -9 "$CHILD" 2>/dev/null; break
    fi
    sleep 30
  done
  wait "$CHILD" 2>/dev/null; CODE=$?; CHILD=""
  IFS='|' read -r N2 _ _ <<< "$(latest)"
  echo "[wd] $(date +%F_%H-%M-%S) trainer exited code=$CODE; iter=$N2" | tee -a "$WLOG"
  if [ "$N2" -ge "$((TARGET-1))" ]; then echo "[wd] completed at $N2." | tee -a "$WLOG"; break; fi
  echo "[wd] resume in 20s (kill PID $$ to stop)" | tee -a "$WLOG"; sleep 20
done

#!/usr/bin/env bash
# Watchdog for g1_tt_v9_dr — PADDLE-RESTITUTION domain randomization, WARM-STARTED from the
# eval-selected v8 checkpoint (set SEED_ITER to the winner; default 34999).
#
# GOAL: raise the ball-on-table rate on the REAL robot. v7/v8 "kill" the ball long ("杀球出界")
#   because the real rubber is bouncier than the sim's fixed paddle restitution (~0.005). Task
#   g1_tt_dr randomizes ONLY the ball-paddle contact restitution per-env (right_tt_paddle_link),
#   leaving the ball-table bounce realistic, so the policy learns to land the ball across a range
#   of paddle bounciness. See G1TableTennisDREnvCfg + verify_paddle_dr.py (run that FIRST).
#
# WARM-START across experiment dirs (same trick as v8): seed v8's model_<SEED_ITER>.pt into
#   logs/g1_tt_v9_dr/seed_v8/, warm-start with a FRESH optimizer (LOAD_OPTIMIZER=0; the DR is a
#   shifted dynamics distribution). TT_SIM_STEP_OFFSET=N*240 pins the serve curriculum at full
#   hard difficulty (SEED_ITER >= 30000 > phase-end 27000). 240 = decimation 10 * steps_per_env 24.
#
# Usage (cloud):
#   SEED_ITER=33000 TARGET=53000 TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python \
#     nohup bash legged_lab/scripts/watchdog_train_v9_dr.sh > /dev/null 2>&1 &
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
TASK=g1_tt_dr
EXP=${EXP:-g1_tt_v9_dr}
SEED_EXP=g1_tt_v8                       # warm-start source experiment
SEED_ITER=${SEED_ITER:-34999}           # warm-start source checkpoint iter (set to eval winner)
NUM_ENVS=${NUM_ENVS:-4096}
TARGET=${TARGET:-$((SEED_ITER + 20000))}
LOGROOT="$REPO/logs/$EXP"
SEED_DIR="$LOGROOT/seed_v8"
WLOG="$REPO/train_${EXP}_watchdog.log"
cd "$REPO"

# --- one-time seed: copy v8 model_<SEED_ITER> into logs/g1_tt_v9_dr/seed_v8/.
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

echo "[wd] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK exp=$EXP (PADDLE-DR warm-start from $SEED_EXP/model_$SEED_ITER)" | tee -a "$WLOG"
while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -ge "$((TARGET-1))" ]; then echo "[wd] $(date +%F_%H-%M-%S) reached $N >= $TARGET. DONE." | tee -a "$WLOG"; break; fi
  REM=$((TARGET - N))
  export TT_SIM_STEP_OFFSET=$((N * 240))
  if [ "$N" -eq "$SEED_ITER" ] && [ "$DIR" = "seed_v8" ]; then
    export LOAD_OPTIMIZER=0
    echo "[wd] $(date +%F_%H-%M-%S) WARM-START from seed_v8/$FILE (iter $N) -> +$REM; LOAD_OPTIMIZER=0 TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
  else
    export LOAD_OPTIMIZER=1
    echo "[wd] $(date +%F_%H-%M-%S) resume $DIR/$FILE (iter $N) -> +$REM; LOAD_OPTIMIZER=1 TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
  fi
  "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless \
    --logger=tensorboard --predictor --max_iterations=$REM \
    --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
  CHILD=$!
  # Isaac Sim 4.5 simulation_app.close() busy-spins forever on teardown after the final ckpt is
  # flushed. Poll; once target ckpt is on disk, kill the hung child (locomotion v2 zombie fix).
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

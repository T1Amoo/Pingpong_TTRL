#!/usr/bin/env bash
# Watchdog for g1_tt_v14 — SPEED curriculum, WARM-STARTED from v12 model_10000.
#
# WHY: v12 collapsed at its stage-2 LATERAL widening (action_rate exploded -0.86 -> -238). v13 keeps
#   lateral FIXED and instead uses the curriculum to ADD SLOWER balls (model_10000 only ever saw fast
#   ~4.4 m/s balls -> no slow-ball robustness). Serve re-designed to a SHALLOW bounce (-0.80,-0.76)
#   so high-vz SLOW balls still arrive at -1.8 at hittable z (deep bounce overshoots). dof_acc_l2
#   reverted -3.75e-7 -> -1.25e-7 (the 3x value over-stiffened the body, near-frozen in sim2sim).
#   All in g1_tt_config.py (G1TableTennisRewardCfg + serve block + curriculum + DR agent exp name).
#
# WARM-START across experiment dirs (same trick as v9_dr): seed v12's model_10000 into
#   logs/g1_tt_v14/seed_v13/, warm-start with a FRESH optimizer (LOAD_OPTIMIZER=0; reward+curriculum
#   changed). TT_SIM_STEP_OFFSET=N*240 continues the serve curriculum clock (N=10000 -> 2.4M, so
#   c=0 fast-only until iter 15000, then add slow over 15000->25000). 240 = decimation 10 * 24.
#
# Usage (cloud):
#   TARGET=30000 TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python OMNI_KIT_ACCEPT_EULA=YES \
#     nohup bash legged_lab/scripts/watchdog_train_v13.sh > /dev/null 2>&1 &
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
TASK=g1_tt_dr
EXP=${EXP:-g1_tt_v14}
SEED_EXP=g1_tt_v13                       # warm-start source experiment
SEED_ITER=${SEED_ITER:-14900}            # warm-start source checkpoint iter (v13 pre-ramp, c=0 stable)
NUM_ENVS=${NUM_ENVS:-4096}
TARGET=${TARGET:-30000}
LOGROOT="$REPO/logs/$EXP"
SEED_DIR="$LOGROOT/seed_v13"
WLOG="$REPO/train_${EXP}_watchdog.log"
cd "$REPO"

# --- one-time seed: copy v13 model_<SEED_ITER> into logs/g1_tt_v14/seed_v13/.
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

echo "[wd] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK exp=$EXP (SPEED curriculum warm-start from $SEED_EXP/model_$SEED_ITER)" | tee -a "$WLOG"
while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -ge "$((TARGET-1))" ]; then echo "[wd] $(date +%F_%H-%M-%S) reached $N >= $TARGET. DONE." | tee -a "$WLOG"; break; fi
  REM=$((TARGET - N))
  export TT_SIM_STEP_OFFSET=$((N * 240))
  if [ "$N" -eq "$SEED_ITER" ] && [ "$DIR" = "seed_v13" ]; then
    export LOAD_OPTIMIZER=0
    echo "[wd] $(date +%F_%H-%M-%S) WARM-START from seed_v13/$FILE (iter $N) -> +$REM; LOAD_OPTIMIZER=0 TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
  else
    export LOAD_OPTIMIZER=1
    echo "[wd] $(date +%F_%H-%M-%S) resume $DIR/$FILE (iter $N) -> +$REM; LOAD_OPTIMIZER=1 TT_SIM_STEP_OFFSET=$TT_SIM_STEP_OFFSET" | tee -a "$WLOG"
  fi
  "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless \
    --logger=tensorboard --predictor --max_iterations=$REM \
    --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
  CHILD=$!
  # Isaac Sim simulation_app.close() busy-spins forever on teardown after the final ckpt flush.
  # Poll; once target ckpt is on disk, kill the hung child (locomotion v2 zombie fix).
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

#!/usr/bin/env bash
# Auto-consolidation chainer for g1_tt_v14.
#
# Phase 1 (the running watchdog_train_v14.sh, TARGET=30000) ramps the serve curriculum
# easy->hard over iter 15000->25000 then trains c=1 to 30000, and EXITS at 30000.
# This script WAITS for phase 1 to truly finish (its watchdog process gone + no trainer +
# model_30000 on disk), then RESUMES the same run to 40000 — i.e. +10000 iters of
# CONSOLIDATION at the fully-learned range. The serve curriculum clamps c=1.0 for
# sim_step > 6M (iter > 25000), and resume sets TT_SIM_STEP_OFFSET=30000*240=7.2M, so
# the consolidation trains ENTIRELY on the learned full (c=1) serve distribution — no ramp.
#
# Mechanism: watchdog_train_v14.sh's latest() finds model_30000 in the run dir (DIR != seed_v13,
# N != 14900) -> NORMAL resume with LOAD_OPTIMIZER=1 (NOT a fresh seed warm-start). It resumes
# 30000 -> 40000 in the SAME timestamped run dir (ckpts continue model_30100..40000).
#
# Usage (cloud, launch in background NOW; it sleeps until phase 1 ends then fires phase 2):
#   TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python OMNI_KIT_ACCEPT_EULA=YES \
#     setsid bash legged_lab/scripts/auto_consolidate_v14.sh >/dev/null 2>&1 < /dev/null &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/root/miniconda3/envs/pingpong/bin/python}
EXP=g1_tt_v14
LOGROOT="$REPO/logs/$EXP"
PHASE1=30000          # phase-1 target (already set in the running v14 watchdog)
PHASE2=40000          # consolidation target (+10000 at the learned c=1 range)
LOG="$REPO/auto_consolidate_v14.log"
cd "$REPO"

latest() { ls "$LOGROOT"/*/model_*.pt 2>/dev/null | grep -oE 'model_[0-9]+' | sed 's/model_//' | sort -n | tail -1; }

echo "[auto] $(date +%F_%H-%M-%S) armed; waiting for v14 phase-1 ($PHASE1) to finish..." | tee -a "$LOG"
while true; do
  N=$(latest); N=${N:-0}
  WD=$(pgrep -f watchdog_train_v14 | grep -v $$ | wc -l)   # phase-1 watchdog still alive?
  TR=$(pgrep -f 'legged_lab.scripts.train' | wc -l)        # any trainer alive?
  # phase 1 is DONE only when target ckpt is on disk AND nothing is training/supervising
  if [ "$N" -ge "$((PHASE1-1))" ] && [ "$WD" -eq 0 ] && [ "$TR" -eq 0 ]; then
    echo "[auto] $(date +%F_%H-%M-%S) phase-1 DONE at iter $N (no watchdog/trainer). Launching consolidation -> $PHASE2." | tee -a "$LOG"
    break
  fi
  sleep 120
done

# Phase 2: resume the v14 run to 40000 (c=1 consolidation). Reuse the v14 watchdog with TARGET=40000.
EXP=$EXP TARGET=$PHASE2 NUM_ENVS=4096 TRAIN_PY=$PY OMNI_KIT_ACCEPT_EULA=YES \
  setsid bash "$HERE/watchdog_train_v14.sh" >> "$LOG" 2>&1 < /dev/null &
echo "[auto] $(date +%F_%H-%M-%S) consolidation watchdog launched (PID groups detached). It resumes model_$PHASE1 -> $PHASE2." | tee -a "$LOG"

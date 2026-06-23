#!/usr/bin/env bash
# Watchdog for LOCAL g1_locomotion velocity training (4060). Auto-resumes from the latest
# checkpoint on crash/OOM. Stop with: pkill -f watchdog_train_locomotion.sh
#   nohup bash legged_lab/scripts/watchdog_train_locomotion.sh > /dev/null 2>&1 &
#   tail -f train_locomotion_watchdog.log
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
TASK=g1_locomotion
EXP=g1_locomotion_v3   # must match G1LocomotionAgentCfg.experiment_name (train.py writes there)
NUM_ENVS=${NUM_ENVS:-512}
TARGET=${TARGET:-15000}
LOGROOT="$REPO/logs/$EXP"
WLOG="$REPO/train_locomotion_watchdog.log"
cd "$REPO"

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

echo "[wd] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK" | tee -a "$WLOG"
while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -ge "$((TARGET-1))" ]; then echo "[wd] $(date +%F_%H-%M-%S) reached $N >= $TARGET. DONE." | tee -a "$WLOG"; break; fi
  if [ "$N" -lt 0 ]; then
    echo "[wd] $(date +%F_%H-%M-%S) fresh start -> $TARGET" | tee -a "$WLOG"
    "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless --logger=tensorboard --max_iterations=$TARGET >> "$WLOG" 2>&1 &
  else
    REM=$((TARGET - N))
    echo "[wd] $(date +%F_%H-%M-%S) resume $DIR/$FILE (iter $N) -> +$REM" | tee -a "$WLOG"
    "$PY" -m legged_lab.scripts.train --task=$TASK --num_envs=$NUM_ENVS --headless --logger=tensorboard --max_iterations=$REM --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
  fi
  CHILD=$!
  # Isaac Sim 4.5's simulation_app.close() busy-spins forever on teardown after the
  # final checkpoint is already flushed by runner.learn (root cause of v1's 3-day zombie).
  # So don't blindly wait: poll, and once the target ckpt is on disk, kill the hung child.
  while kill -0 "$CHILD" 2>/dev/null; do
    IFS='|' read -r NC _ _ <<< "$(latest)"
    if [ "$NC" -ge "$((TARGET-1))" ]; then
      echo "[wd] $(date +%F_%H-%M-%S) target ckpt $NC saved; killing child $CHILD (Isaac shutdown-hang workaround)" | tee -a "$WLOG"
      kill -9 "$CHILD" 2>/dev/null
      break
    fi
    sleep 30
  done
  wait "$CHILD" 2>/dev/null; CODE=$?; CHILD=""
  IFS='|' read -r N2 _ _ <<< "$(latest)"
  echo "[wd] $(date +%F_%H-%M-%S) trainer exited code=$CODE; iter=$N2" | tee -a "$WLOG"
  if [ "$N2" -ge "$((TARGET-1))" ]; then echo "[wd] completed at $N2." | tee -a "$WLOG"; break; fi
  echo "[wd] resume in 20s (kill PID $$ to stop)" | tee -a "$WLOG"; sleep 20
done

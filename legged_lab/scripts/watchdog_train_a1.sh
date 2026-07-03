#!/usr/bin/env bash
# Watchdog for a1_tt — A1 AGV ping-pong, local laptop (RTX 4060 8 GB) reduced-env convergence validation.
#
# WHY: Validates the a1_tt task/reward/serve pipeline converges from scratch on local hardware
#   before cloud scale-out. Chosen NUM_ENVS=128 is the largest power-of-two that fits the 8 GB
#   GPU (256 crashes during scene construction; 128 peaks at ~5094 MiB leaving ~500 MB headroom).
#   Full 4096-env cloud run is separate. TARGET=5000 is for local convergence validation only.
#
# Isaac Sim simulation_app.close() busy-spins on teardown after the final ckpt flush.
#   We poll for the target checkpoint landing on disk and kill the hung child immediately
#   (same approach as watchdog_train_v16.sh, locomotion v2 zombie fix).
#
# KILL ORDER: always kill watchdog first, then trainer — kill watchdog PID ($$), not pattern.
#   To stop: kill <watchdog_pid>   (NOT pkill -f watchdog_train_a1 — self-kills the script)
#   To find trainer:  pgrep -f "[a]1_tt"  then kill <trainer_pid>
#
# Usage (local):
#   TARGET=5000 nohup bash legged_lab/scripts/watchdog_train_a1.sh > /tmp/a1_watchdog.log 2>&1 &
#
# Usage (cloud, override python and envs):
#   TARGET=5000 NUM_ENVS=4096 TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python \
#     OMNI_KIT_ACCEPT_EULA=YES nohup bash legged_lab/scripts/watchdog_train_a1.sh > /dev/null 2>&1 &
set -u

export OMNI_KIT_ACCEPT_EULA=YES

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
TASK=a1_tt
EXP=${EXP:-a1_tt_v3}
NUM_ENVS=${NUM_ENVS:-128}
TARGET=${TARGET:-5000}
LOGROOT="$REPO/logs/$EXP"
WLOG="$REPO/train_${EXP}_watchdog.log"
cd "$REPO"

CHILD=""
cleanup() {
  echo "[wd] $(date +%F_%H-%M-%S) signal; killing child $CHILD" | tee -a "$WLOG"
  [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null
  sleep 3
  [ -n "$CHILD" ] && kill -9 "$CHILD" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

latest() {
  local best=-1 bdir="" bfile="" f n
  shopt -s nullglob
  for f in "$LOGROOT"/*/model_*.pt; do
    n=$(basename "$f" | sed -E 's/model_([0-9]+)\.pt/\1/')
    [[ "$n" =~ ^[0-9]+$ ]] || continue
    if [ "$n" -gt "$best" ]; then
      best=$n
      bfile=$(basename "$f")
      bdir=$(basename "$(dirname "$f")")
    fi
  done
  echo "$best|$bdir|$bfile"
}

echo "[wd] $(date +%F_%H-%M-%S) start; target=$TARGET envs=$NUM_ENVS task=$TASK exp=$EXP" | tee -a "$WLOG"

while true; do
  IFS='|' read -r N DIR FILE <<< "$(latest)"
  if [ "$N" -ge "$((TARGET-1))" ]; then
    echo "[wd] $(date +%F_%H-%M-%S) reached $N >= $TARGET. DONE." | tee -a "$WLOG"
    break
  fi
  REM=$((TARGET - N))

  if [ "$N" -lt 0 ]; then
    # fresh start, no checkpoint yet
    echo "[wd] $(date +%F_%H-%M-%S) fresh start -> $REM iters" | tee -a "$WLOG"
    "$PY" -u -m legged_lab.scripts.train --task="$TASK" --num_envs="$NUM_ENVS" --headless \
      --logger=tensorboard --predictor --max_iterations="$REM" >> "$WLOG" 2>&1 &
  else
    echo "[wd] $(date +%F_%H-%M-%S) resume $DIR/$FILE (iter $N) -> +$REM" | tee -a "$WLOG"
    "$PY" -u -m legged_lab.scripts.train --task="$TASK" --num_envs="$NUM_ENVS" --headless \
      --logger=tensorboard --predictor --max_iterations="$REM" \
      --resume true --load_run "$DIR" --checkpoint "$FILE" >> "$WLOG" 2>&1 &
  fi
  CHILD=$!

  # Poll; once target ckpt is on disk, kill the hung child (Isaac shutdown-hang workaround).
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

  if [ "$N2" -ge "$((TARGET-1))" ]; then
    echo "[wd] completed at $N2." | tee -a "$WLOG"
    break
  fi

  echo "[wd] resume in 20s (to stop: kill $$ first, then pgrep -f '[a]1_tt')" | tee -a "$WLOG"
  sleep 20
done

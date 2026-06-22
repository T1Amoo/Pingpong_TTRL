#!/usr/bin/env bash
# Overnight orchestrator for LOCAL g1_locomotion_v2 velocity training.
# Waits for the current watchdog run to finish, evaluates velocity-tracking quality,
# and AUTO-EXTENDS training (resumes from latest ckpt at a higher TARGET) if the
# policy is still improving but below the deployable bar. Stops when GOOD / plateaued
# / cap. Detached via nohup -> survives Claude session exit.
#
#   cd <repo>; export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
#   nohup bash legged_lab/scripts/overnight_locomotion_finish.sh >/dev/null 2>&1 &
#   tail -f overnight_locomotion.log ; cat logs/g1_locomotion_v2/OVERNIGHT_RESULT.txt
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY=${TRAIN_PY:-/home/woan/.conda/envs/pingpong/bin/python}
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
STEP=10000          # iters to add per extension
RESULT="$REPO/logs/g1_locomotion_v2/OVERNIGHT_RESULT.txt"
OLOG="$REPO/overnight_locomotion.log"
log(){ echo "[$(date +%F_%H-%M-%S)] $*" | tee -a "$OLOG"; }

log "orchestrator start (waits for watchdog, evals, auto-extends if still improving)"
while true; do
  # 1) wait until the locomotion watchdog has fully exited (its TARGET reached)
  while pgrep -f watchdog_train_locomotion >/dev/null 2>&1; do sleep 120; done
  sleep 15  # let final ckpt/tfevents flush

  # 2) evaluate
  EV="$($PY legged_lab/scripts/eval_locomotion.py 2>/dev/null | grep -E '^ITER=' | tail -1)"
  log "eval: $EV"
  DEC=$(echo "$EV" | sed -E 's/.*DECISION=([A-Z]+).*/\1/')
  ITER=$(echo "$EV" | sed -E 's/ITER=([0-9]+).*/\1/')

  if [ "$DEC" = "EXTEND" ]; then
    NEWT=$(( ITER + STEP ))
    log "still improving below bar -> extend to TARGET=$NEWT (resume from latest ckpt)"
    TARGET=$NEWT nohup bash legged_lab/scripts/watchdog_train_locomotion.sh >/dev/null 2>&1 &
    sleep 180  # let the new watchdog boot + spawn trainer before re-checking pgrep
    continue
  fi

  # 3) terminal decision -> write result and exit
  {
    echo "=== g1_locomotion_v2 overnight result ($(date +%F_%H-%M-%S)) ==="
    echo "$EV"
    case "$DEC" in
      GOOD)      echo "VERDICT: GOOD - velocity tracking reached deployable quality (>=0.85).";;
      CONVERGED) echo "VERDICT: CONVERGED - plateaued below 0.85; more rounds won't help, needs reward/obs tuning, not more iters.";;
      CAP)       echo "VERDICT: CAP - hit 35000-iter safety cap; tracking=$( echo "$EV" | sed -E 's/.*TRACK=([0-9.]+).*/\1/').";;
      *)         echo "VERDICT: UNKNOWN ($DEC) - inspect manually.";;
    esac
    echo "latest run dir: $(ls -dt logs/g1_locomotion_v2/2026-* 2>/dev/null | head -1)"
    echo "NEXT: export onnx + write velocity deploy.yaml + wire into unitree_rl_lab"
    echo "      (see memory g1_tt_v6_retrain_and_locomotion_2026-06-18 Task2, 7 steps)."
  } > "$RESULT"
  log "DONE ($DEC) -> wrote $RESULT"
  break
done

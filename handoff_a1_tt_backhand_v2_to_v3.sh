#!/usr/bin/env bash
set -u

cd /mnt/workspace/Pingpong_TTRL || exit 1

V2_EXP=logs/a1_tt_backhand_real_v2_r115_netclear_highslow_paddle075
V2_FINAL=model_19999.pt
V2_WATCHDOG_PID=${V2_WATCHDOG_PID:-}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}
HANDOFF_LOG=handoff_a1_tt_backhand_v2_to_v3.log
V3_NOHUP=watchdog_a1_tt_backhand_v3_nohup.log

log() {
  echo "[handoff] $(date --iso-8601=seconds) $*" >> "$HANDOFF_LOG"
}

if ! [[ "$V2_WATCHDOG_PID" =~ ^[0-9]+$ ]]; then
  log "invalid V2_WATCHDOG_PID=$V2_WATCHDOG_PID"
  exit 2
fi

log "armed v2_watchdog_pid=$V2_WATCHDOG_PID expected_commit=$EXPECTED_COMMIT"

# Wait for the final checkpoint and the matching final-iteration log.  The
# checkpoint must remain the same size for two minutes before any process is
# touched, so a partially written torch archive cannot trigger the handoff.
stable_size=-1
stable_checks=0
while true; do
  final_path=$(find "$V2_EXP" -mindepth 2 -maxdepth 2 -type f -name "$V2_FINAL" -print -quit 2>/dev/null)
  if [ -n "$final_path" ] && grep -q 'Learning iteration 19999/20000' \
      train_a1_tt_backhand_real_v2_r115_netclear_highslow_paddle075.log 2>/dev/null; then
    size=$(stat -c %s "$final_path" 2>/dev/null || echo 0)
    if [ "$size" -gt 1000000 ] && [ "$size" -eq "$stable_size" ]; then
      stable_checks=$((stable_checks + 1))
    else
      stable_size=$size
      stable_checks=0
    fi
    if [ "$stable_checks" -ge 2 ]; then
      log "v2 complete final=$final_path size=$size"
      break
    fi
  fi
  sleep 60
done

# §0 process order: stop the watchdog first, then its current explicit trainer
# child.  At this point v2 has already produced and logged its final checkpoint.
trainer_pid=""
for child in $(ps -o pid= --ppid "$V2_WATCHDOG_PID" 2>/dev/null); do
  args=$(ps -p "$child" -o args= 2>/dev/null || true)
  case "$args" in
    *"legged_lab.scripts.train --task a1_tt_backhand_v2"*) trainer_pid=$child ;;
  esac
done

if kill -0 "$V2_WATCHDOG_PID" 2>/dev/null; then
  kill "$V2_WATCHDOG_PID"
  log "stopped v2 watchdog pid=$V2_WATCHDOG_PID"
fi
sleep 3
if [ -n "$trainer_pid" ] && kill -0 "$trainer_pid" 2>/dev/null; then
  kill "$trainer_pid"
  log "sent SIGTERM to completed v2 trainer pid=$trainer_pid"
  for _ in $(seq 1 12); do
    kill -0 "$trainer_pid" 2>/dev/null || break
    sleep 5
  done
  if kill -0 "$trainer_pid" 2>/dev/null; then
    kill -9 "$trainer_pid"
    log "sent SIGKILL to completed v2 trainer pid=$trainer_pid after 60s grace"
  fi
fi

head=$(git rev-parse HEAD)
if [ -n "$EXPECTED_COMMIT" ] && [ "$head" != "$EXPECTED_COMMIT" ]; then
  log "wrong checkout head=$head expected=$EXPECTED_COMMIT; refusing to start v3"
  exit 3
fi

source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate pingpong
if ! python -m legged_lab.scripts.verify_a1_backhand_v3_serve --samples 50000 >> "$HANDOFF_LOG" 2>&1; then
  log "v3 serve verification failed; refusing to start trainer"
  exit 4
fi

setsid bash watchdog_a1_tt_backhand_v3.sh > "$V3_NOHUP" 2>&1 < /dev/null &
v3_watchdog_pid=$!
echo "$v3_watchdog_pid" > train_a1_tt_backhand_v3_watchdog_launcher.pid
log "launched v3 watchdog pid=$v3_watchdog_pid"

sleep 180
if kill -0 "$v3_watchdog_pid" 2>/dev/null; then
  trainer=$(cat train_a1_tt_backhand_v3_trainer.pid 2>/dev/null || true)
  log "v3 startup check watchdog_alive=1 trainer_pid=$trainer"
else
  log "v3 startup check watchdog_alive=0; inspect $V3_NOHUP"
  exit 5
fi

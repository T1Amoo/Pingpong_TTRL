#!/usr/bin/env bash
# Supervisor: the running locomotion watchdog has TARGET=20000 and STOPS there.
# This waits until model_20000.pt is on disk AND the old watchdog+trainer have fully
# exited, then relaunches the watchdog with TARGET=40000 to auto-continue overnight.
# Start with:  nohup bash legged_lab/scripts/supervisor_locomotion_to40k.sh >/dev/null 2>&1 &
#   log: train_locomotion_supervisor.log
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LOGROOT="$REPO/logs/g1_locomotion_v3"
SLOG="$REPO/train_locomotion_supervisor.log"
cd "$REPO"

latest() {
  local best=-1 f n; shopt -s nullglob
  for f in "$LOGROOT"/*/model_*.pt; do
    n=$(basename "$f" | sed -E 's/model_([0-9]+)\.pt/\1/'); [[ "$n" =~ ^[0-9]+$ ]] || continue
    [ "$n" -gt "$best" ] && best=$n
  done
  echo "$best"
}

echo "[sup] $(date +%F_%H-%M-%S) start; waiting for iter>=20000 then will relaunch wd TARGET=40000" | tee -a "$SLOG"

# 1) wait until the TARGET=20000 run has produced model_20000.pt
while [ "$(latest)" -lt 20000 ]; do sleep 120; done
echo "[sup] $(date +%F_%H-%M-%S) iter $(latest) reached >=20000" | tee -a "$SLOG"

# 2) wait for the old watchdog + its trainer child to fully exit (avoid 2 trainers on GPU).
#    NOTE: these grep patterns are inside this FILE, not in this process's cmdline
#    (cmdline is just `bash .../supervisor_locomotion_to40k.sh`), so no self-match.
while pgrep -f "watchdog_train_locomotion.sh" >/dev/null \
   || pgrep -f "scripts.train.*g1_locomotion" >/dev/null; do
  sleep 20
done
echo "[sup] $(date +%F_%H-%M-%S) old watchdog/trainer gone; GPU free" | tee -a "$SLOG"
sleep 10

# 3) relaunch watchdog to 40000 (its latest() resumes from model_20000 -> +20000)
export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
TARGET=40000 NUM_ENVS=512 nohup bash "$HERE/watchdog_train_locomotion.sh" >/dev/null 2>&1 &
echo "[sup] $(date +%F_%H-%M-%S) relaunched watchdog TARGET=40000 (pid $!). done." | tee -a "$SLOG"

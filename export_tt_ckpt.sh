#!/usr/bin/env bash
# Export one g1_tt_v8 checkpoint to the deploy ONNX bundle for sim2sim jitter testing.
#   play.py exports policy.onnx (then loops forever -> we poll+kill); predictor.onnx via
#   export_predictor_onnx.py. Bundle copied into the deploy policy dir as v8_<iter>.
# Usage: bash export_tt_ckpt.sh <iter>
set -u
IT="$1"
REPO=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
DEPLOY=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/g1_23dof/config/policy/table_tennis
RUN=2026-06-23_12-21-52
PY=/home/woan/.conda/envs/pingpong/bin/python
EXP_DIR="$REPO/logs/g1_tt_v8/$RUN/exported"
CKPT="$REPO/logs/g1_tt_v8/$RUN/model_$IT.pt"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json PYTHONUNBUFFERED=1

[ -f "$CKPT" ] || { echo "FATAL: missing $CKPT"; exit 1; }
rm -f "$EXP_DIR/policy.onnx"
echo "[exp] $(date +%T) play.py export for model_$IT ..."
"$PY" -m legged_lab.scripts.play --task g1_tt --predictor --headless --num_envs 1 \
  --load_run "$RUN" --checkpoint "model_$IT.pt" >/tmp/play_$IT.log 2>&1 &
PID=$!
# wait up to 5 min for policy.onnx, then kill the (looping) play process
for i in $(seq 1 150); do
  [ -f "$EXP_DIR/policy.onnx" ] && break
  kill -0 "$PID" 2>/dev/null || { echo "[exp] play.py died early; see /tmp/play_$IT.log"; tail -5 /tmp/play_$IT.log; exit 1; }
  sleep 2
done
sleep 2  # let the file finish flushing
kill -9 "$PID" 2>/dev/null
[ -f "$EXP_DIR/policy.onnx" ] || { echo "[exp] FAIL: no policy.onnx"; tail -8 /tmp/play_$IT.log; exit 1; }
echo "[exp] $(date +%T) predictor.onnx ..."
"$PY" legged_lab/scripts/export_predictor_onnx.py --ckpt "$CKPT" --out "$EXP_DIR/predictor.onnx" --history_len 5 2>&1 | tail -1

OUT="$DEPLOY/v8_$IT"
mkdir -p "$OUT/exported" "$OUT/params"
cp "$EXP_DIR/policy.onnx" "$EXP_DIR/predictor.onnx" "$OUT/exported/"
cp "$DEPLOY/v7_29999/params/deploy.yaml" "$OUT/params/deploy.yaml"
echo "[exp] DONE -> $OUT  ($(ls -la "$OUT/exported"/*.onnx | awk '{print $5,$9}' | tr '\n' ' '))"

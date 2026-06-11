#!/usr/bin/env bash
# =============================================================================
# Pingpong_TTRL — 云端 GPU 实例一键安装（Isaac Sim 4.5 + IsaacLab 2.1.1 + 项目）
# 实测于阿里云 PAI-DSW：Ubuntu 24.04 / py3.12 基础镜像 / NVIDIA L20 /
# 实例创建时设 NVIDIA_DRIVER_CAPABILITIES=all（图形/Vulkan 能力，必须！）。
#
# 幂等：可重复运行。成功结尾打印 "SETUP_OK"。
# 用法：  bash setup_cloud.sh    （在云端实例上跑；建议 nohup）
# =============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/mnt/workspace}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$WORKSPACE/IsaacLab}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE/Pingpong_TTRL}"
CONDA_DIR="${CONDA_DIR:-/root/miniconda3}"
ENV_NAME="${ENV_NAME:-pingpong}"

echo "===== [0/6] 系统图形库 + Vulkan ICD（headless 也要 Vulkan）====="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq cmake build-essential git wget \
  vulkan-tools libvulkan1 libglu1-mesa libgl1 libegl1 \
  libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 >/dev/null
# caps=all 会挂 libGLX_nvidia.so 但常不挂 ICD json -> 手动补
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'JSON'
{ "file_format_version" : "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version" : "1.3.277" } }
JSON
fi
if ! vulkaninfo --summary 2>/dev/null | grep -q "NVIDIA"; then
  echo "FATAL: Vulkan 看不到 NVIDIA GPU。实例缺图形能力。"
  echo "  -> 重建实例并设环境变量 NVIDIA_DRIVER_CAPABILITIES=all（或用 isaaclab 镜像底座）。"
  exit 1
fi
echo "  Vulkan 看到 NVIDIA GPU ✓"

echo "===== [1/6] miniconda + py3.10 env ($ENV_NAME) ====="
if [ ! -d "$CONDA_DIR" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda create -n "$ENV_NAME" python=3.10 -y -q 2>/dev/null || true
conda activate "$ENV_NAME"
pip install -q --upgrade pip

echo "===== [2/6] Isaac Sim 4.5 + 钉死 torch 2.5.1+cu118 ====="
pip install -q 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
# isaaclab 依赖是 torch>=2.5.1（宽松）；强制降到验证过的 2.5.1+cu118
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

echo "===== [3/6] flatdict(--no-build-isolation)+ IsaacLab v2.1.1 核心 ====="
pip install -q flatdict==4.0.1 --no-build-isolation
if [ ! -d "$ISAACLAB_DIR" ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
cd "$ISAACLAB_DIR"
git fetch --tags -q && git checkout v2.1.1 -q     # 必须 2.1.1（is_global API）；不要 2.1.0/2.2+
pip install -q -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_rl -e source/isaaclab_tasks
# isaaclab_mimic 不装（要 egl_probe/robomimic，本项目用不到）

echo "===== [4/6] 项目 legged_lab + rsl_rl fork ====="
if [ ! -d "$PROJECT_DIR" ]; then
  git clone https://github.com/T1Amoo/Pingpong_TTRL.git "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"
git fetch -q && git checkout g1-tt-deploy -q && git pull -q
pip install -q -e .
pip install -q -e rsl_rl          # 改过的 fork（含 OnPolicyPredictorRegressionRunner），放最后覆盖
pip install -q "numpy==1.26.4"    # 必须 <2

echo "===== [5/6] 验证 ====="
python -c "import isaaclab, torch; print('  isaaclab', isaaclab.__version__, '| torch', torch.__version__, torch.version.cuda)"
python -c "import torch; assert torch.__version__.startswith('2.5.1') and torch.version.cuda=='11.8', 'torch 版本不对!'; print('  torch 版本正确 ✓')"

echo "===== [6/6] done ====="
echo "SETUP_OK"

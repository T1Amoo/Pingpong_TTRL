#!/usr/bin/env bash
# =============================================================================
# Pingpong_TTRL — 云端 GPU 实例一键安装（Isaac Sim 4.5 + IsaacLab 2.1.1 + 项目）
# 实测于阿里云 PAI-DSW：Ubuntu 24.04 / py3.12 基础镜像 / NVIDIA L20 /
# 实例创建时设 NVIDIA_DRIVER_CAPABILITIES=all（图形/Vulkan 能力，必须！）。
#
# 幂等：可重复运行。成功结尾打印 "SETUP_OK"。已内置 5 个 PAI-DSW 实测坑的修复：
#   #1 conda ToS  #2 空 miniconda 目录  #3 URDF importer headless 挂死
#   #4 重复 Vulkan ICD  #5 watchdog PY 路径(见 watchdog 的 TRAIN_PY)
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
# 坑#4: 不要造重复 ICD。基础镜像(caps=all)的 /etc 里常已有 nvidia_icd.json;
# 若再在 /usr/share 造一个 -> 同 GPU 枚举两次 -> "Multiple ICDs" -> GPU foundation 失败。
# 优先用 /etc 的;两者都无才造 /usr/share。最后导出 VK_ICD_FILENAMES 指向唯一那个。
ICD_ETC=/etc/vulkan/icd.d/nvidia_icd.json
ICD_SHARE=/usr/share/vulkan/icd.d/nvidia_icd.json
if [ -f "$ICD_ETC" ]; then
  rm -f "$ICD_SHARE"                       # 去掉我们可能造的重复
  VK_ICD="$ICD_ETC"
elif [ -f "$ICD_SHARE" ]; then
  VK_ICD="$ICD_SHARE"
else
  cat > "$ICD_SHARE" <<'JSON'
{ "file_format_version" : "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version" : "1.3.277" } }
JSON
  VK_ICD="$ICD_SHARE"
fi
export VK_ICD_FILENAMES="$VK_ICD"
echo "$VK_ICD" > "$WORKSPACE/.vk_icd"       # 供 watchdog/launch 读取(启动训练也要设)
if ! vulkaninfo --summary 2>/dev/null | grep -q "NVIDIA"; then
  echo "FATAL: Vulkan 看不到 NVIDIA GPU。实例缺图形能力。"
  echo "  -> 重建实例并设环境变量 NVIDIA_DRIVER_CAPABILITIES=all（或用 isaaclab 镜像底座）。"
  exit 1
fi
echo "  Vulkan 看到 NVIDIA GPU ✓  (VK_ICD_FILENAMES=$VK_ICD)"

echo "===== [1/6] miniconda + py3.10 env ($ENV_NAME) ====="
# 坑#2: 守卫看 conda 可执行文件,不看目录(残留空目录会让 [ -d ] 跳过安装然后 source 失败)
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  rmdir "$CONDA_DIR" 2>/dev/null || true
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
# 坑#1: 新版 conda 建 env 前要接受频道 ToS,否则 CondaToSNonInteractiveError(老版无此命令 -> ||true)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda create -n "$ENV_NAME" python=3.10 -y -q || true
conda activate "$ENV_NAME"
pip install -q --upgrade pip

echo "===== [2/6] Isaac Sim 4.5 + 钉死 torch 2.5.1+cu118 ====="
pip install -q 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
# isaaclab 依赖是 torch>=2.5.1（宽松）；强制降到验证过的 2.5.1+cu118
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# 坑#3: URDF importer 扩展在 headless 启动时无条件 build UI(ui.StringField)->GIL 死锁。
# G1 用 UrdfFileCfg,每次 spawn 都 enable 它 -> 卡死。patch build_ui:headless 时早返回。
# (幂等;每次重装 isaacsim 都会重置 site-packages,故每次 setup 都重打。)
echo "  patch URDF importer (headless build_ui early-return) ..."
python - <<'PYEOF'
import glob, os
sp = os.path.join(os.environ["CONDA_PREFIX"], "lib", "python3.10", "site-packages")
pat = os.path.join(sp, "isaacsim/extscache/isaacsim.asset.importer.urdf*/isaacsim/asset/importer/urdf/scripts/extension.py")
marker = "    def build_ui(self):\n"
ins = marker + (
'        # PATCH(headless): ui.StringField hangs the GIL forever with no renderer\n'
'        # (Isaac Sim --headless). URDF->USD uses omni.kit.commands only, not this window.\n'
'        if not carb.settings.get_settings().get("/app/window/enabled"):\n'
'            self._scrolling_frame = None\n'
'            self.options_frame = None\n'
'            self.extra_frames["extra"] = None\n'
'            return\n'
)
n = 0
for f in glob.glob(pat):
    s = open(f).read()
    if "PATCH(headless)" in s or marker not in s:
        continue
    open(f, "w").write(s.replace(marker, ins, 1)); n += 1
    print("    patched:", f)
print("    URDF patch applied to", n, "file(s)" if n != 1 else "file")
PYEOF

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
echo "提示: 启动训练时务必 export VK_ICD_FILENAMES=$VK_ICD 和 OMNI_KIT_ACCEPT_EULA=YES,"
echo "      并用 TRAIN_PY=$CONDA_DIR/envs/$ENV_NAME/bin/python 传给 watchdog。"
echo "SETUP_OK"

# 云端环境安装手册（Pingpong_TTRL / G1 23DoF 乒乓 RL）

> 逐命令复现训练环境。针对**没有现成 IsaacLab 镜像**的云 GPU 实例（如阿里云 PAI-DSW），从纯 Python 3.10 基础镜像自建。
> 本手册基于本地已跑通环境的实测版本整理，照抄即可。

---

## 0. 必须匹配的版本（硬约束）

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | **3.10** | Isaac Sim 4.5 只支持 3.10，3.11/3.12/3.13 装不上 |
| PyTorch | **2.5.1+cu118** | 由 isaacsim 4.5 pip 自动带，**锁死 cu118**，不要手动升级 |
| CUDA | **11.8**（torch 自带运行库） | 只需 GPU 驱动 ≥ 520（云上 A10/A100 均满足）|
| Isaac Sim | **4.5.0.0**（pip 安装） | ⚠️ 升到 5.0+ 成功率明显下降，**务必锁 4.5.0** |
| IsaacLab | **v2.1.1**（git tag，commit 90b79bb） | ⚠️ README 写 2.1.0，但代码需 2.1.1 的 `is_global` API，**必须 2.1.1** |
| rsl_rl | **2.3.1**，项目内**改过的 fork**（`Pingpong_TTRL/rsl_rl`） | 含 `OnPolicyPredictorRegressionRunner`，**不能用 pip 版** |
| numpy | **1.26.4** | 必须 < 2 |
| flatdict | **4.0.1** | 装不上时加 `--no-build-isolation` |

---

## 1. 创建实例 / 选镜像

- **镜像**：没有 IsaacLab 镜像，选 **Python 3.10 的 GPU PyTorch 镜像**（如 PAI 的
  `2.6.0-gpu-py310-cu124-ubuntu22.04-accl`，Ubuntu 22.04 最稳；24.04 也行）。
  - ❌ 不要选 py3.11/3.12 镜像（Isaac Sim 4.5 装不上）。
  - ❌ 不要选 modelscope py3.11 镜像。
- **GPU**：A10(24GB) 或 A100 均可；headless RL 不需要 RTX 光追。本地 4060(8GB) 跑 4096 env 占 ~7GB，24GB 很富裕。
- **系统盘**：≥ **80–100 GiB**（Isaac Sim pip 包 + 缓存约 25GB，IsaacLab + 项目另算）。
- **网络**：需访问公网（拉 `pypi.nvidia.com` 和 GitHub）。

---

## 2. 建独立 Python 3.10 环境（别污染镜像自带 torch）

```bash
# 若镜像带 conda：
conda create -n pingpong python=3.10 -y
conda activate pingpong

# 若无 conda，用 venv 亦可：
# python3.10 -m venv ~/pingpong && source ~/pingpong/bin/activate

pip install --upgrade pip
```

---

## 3. 安装 Isaac Sim 4.5.0（pip，会自动带 torch 2.5.1+cu118）

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

验证 torch 版本正确（必须是 `2.5.1+cu118`）：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 期望: 2.5.1+cu118 11.8
```

> ⚠️ 之后任何 pip 安装都**不要**把 torch 升级成 cu12 版。若被带升级，回退：
> `pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118`

接受 EULA（headless 运行 Isaac Sim 必需）：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> ~/.bashrc
```

---

## 4. 安装 IsaacLab v2.1.1（必须这个 tag）

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.1.1          # commit 90b79bb；不要用 main / 2.1.0 / 2.2+
# Isaac Sim 已 pip 装好，isaaclab.sh 会自动识别当前 python 里的 isaacsim
./isaaclab.sh -i             # 安装 isaaclab / isaaclab_assets / isaaclab_rl / isaaclab_tasks
```

验证 IsaacLab：

```bash
python -c "import isaaclab; print('isaaclab', isaaclab.__version__)"   # 期望 0.41.3
```

---

## 5. 安装本项目（legged_lab + 改过的 rsl_rl fork）

```bash
cd ~
git clone https://github.com/T1Amoo/Pingpong_TTRL.git
cd Pingpong_TTRL
git checkout g1-tt-deploy          # 含 G1 乒乓 + 无球 idle 最新代码

# (a) 装 legged_lab（项目根 setup.py）
pip install -e .

# (b) 装改过的 rsl_rl fork —— 必须放在最后，覆盖 isaaclab.sh 可能装的 pip rsl_rl
pip install -e rsl_rl
```

验证 rsl_rl 是 fork 版（路径指向项目目录）：

```bash
python -c "import rsl_rl, os; print(os.path.dirname(rsl_rl.__file__))"
# 期望路径在 .../Pingpong_TTRL/rsl_rl 下
```

> ✅ **资产无需单独传**：G1 球拍 URDF（`legged_lab/assets/unitree/g1_description/g1_23dof_tt_paddle.urdf`）、
> 球桌/球、meshes 等全部随仓库 git 跟踪，clone 即带。

---

## 6. 收尾依赖 + 已知坑

```bash
# numpy 锁 <2（被别的包升上去会崩）
pip install "numpy==1.26.4"

# flatdict：正常装；若编译失败，用 --no-build-isolation
pip install flatdict==4.0.1 || pip install flatdict==4.0.1 --no-build-isolation
```

**已知坑清单：**
- **IsaacLab 必须 v2.1.1**（2.1.0 的 `is_global` 会崩）。
- **Isaac Sim 锁 4.5.0**（5.0+ 同配置成功率明显下降，README 已注明）。
- **torch 不要升级**（保持 2.5.1+cu118）。
- **numpy < 2**；**rsl_rl 用 fork**（不是 pip 版）。
- 首次启动 Isaac Sim 会编译 shader、拉 kit 缓存，**第一次很慢**（几分钟），属正常。
- 云上无显示器：训练一律加 `--headless`。

---

## 7. 冒烟验证（小规模，确认能跑）

```bash
cd ~/Pingpong_TTRL
export OMNI_KIT_ACCEPT_EULA=YES
python -m legged_lab.scripts.train --task=g1_tt --num_envs=64 --headless \
  --logger=tensorboard --predictor --max_iterations=3
```

看到 `Learning iteration 0/3 … 2/3`、无 Traceback / 无 NaN 即环境 OK。

---

## 8. 正式训练（可断点续训的 watchdog）

```bash
cd ~/Pingpong_TTRL
nohup bash legged_lab/scripts/watchdog_train_idle5.sh > /dev/null 2>&1 &
echo $! > /tmp/idle5_watchdog.pid     # 这个 PID 是 watchdog；kill 它停全部
tail -f train_idle5_watchdog.log      # 看进度
```

- 当前实验：`g1_tt_idle5`（无球 idle 冻结修复版，含 `reward_idle_stand`）。
- 停止训练：`kill $(cat /tmp/idle5_watchdog.pid)`（先杀 watchdog，否则它会自动重启）。
- ckpt / tensorboard 在 `logs/g1_tt_idle5/<时间戳>/`。

---

## 9. 评估（挑 ckpt）

```bash
# 固定中等难度 bounce 发球，看真命中率（别用训练 reward 选 ckpt）
python -m legged_lab.scripts.eval --task=g1_tt_eval --predictor --headless \
  --load_run <run时间戳> --checkpoint model_<N>.pt --num_envs 100 --seed 0

# 测无球站稳（间断发球）：加环境变量
TT_SERVE_PERIOD=10 python -m legged_lab.scripts.eval --task=g1_tt_eval --predictor --headless \
  --load_run <run时间戳> --checkpoint model_<N>.pt --num_envs 100
```

挑 ckpt 用 **eval 真成功率 + 跨 seed 复测**，不要假设最后一个最好。

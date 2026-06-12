---
name: g1-tt-cloud-train
description: 通过 SSH 在云端 GPU 实例上安装 Pingpong_TTRL 的 G1 乒乓训练环境(Isaac Sim 4.5 + IsaacLab 2.1.1)并自动启动 idle9(无球 ready-pose 站稳版)训练通宵跑。当用户要在云端搭建/启动 G1 乒乓训练时使用。
---

# G1 乒乓 云端安装 + 训练（idle9 / 无球 ready-pose 站稳版）

自动在云端 GPU 实例上:**装环境 → 冒烟验证 → 起 idle9 训练(nohup,通宵跑)**。
**全程不要停下来问用户,按步骤执行到底**(除非遇到无法解决的 FATAL 才报告并停)。

## SSH 目标（实例变了就改这里的 HOST/PORT）

```bash
HOST=39.101.75.133 ; PORT=1021 ; USER=root
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 $USER@$HOST"
```
先测连通:`$SSH 'echo OK; nvidia-smi -L'`。连不上(密钥/IP 变了)→ 报告用户并停。

## 环境变量(每次起 python 训练/冒烟都要带,经 setup 验证过的 5 个坑已内置在 setup_cloud.sh)

```bash
ENVSET='source /root/miniconda3/etc/profile.d/conda.sh && conda activate pingpong \
  && export OMNI_KIT_ACCEPT_EULA=YES \
  && export VK_ICD_FILENAMES=$(cat /mnt/workspace/.vk_icd 2>/dev/null || echo /etc/vulkan/icd.d/nvidia_icd.json) \
  && export TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python'
```

## 步骤

### 1. 拉最新代码 + 跑安装脚本（nohup,幂等,含 ~20GB 下载;5 个坑已内置）
```bash
$SSH 'set -e; cd /mnt/workspace; [ -d Pingpong_TTRL ] || git clone https://github.com/T1Amoo/Pingpong_TTRL.git; cd Pingpong_TTRL && git fetch -q && git checkout g1-tt-deploy -q && git pull -q && nohup bash setup_cloud.sh > /mnt/workspace/setup.log 2>&1 & echo started'
```

### 2. 轮询安装日志（每 ~60s,可能 20–40 分钟;若环境已装好会很快到 SETUP_OK）
```bash
$SSH 'tail -8 /mnt/workspace/setup.log'
```
- 出现 **`SETUP_OK`** → 进下一步。
- 出现 **`FATAL`**(Vulkan 看不到 NVIDIA)→ 实例缺图形能力。报告用户:**重建实例加 `NVIDIA_DRIVER_CAPABILITIES=all`,或换 isaaclab 镜像底座**。停。
- 其它报错 → 读日志修后重跑 setup_cloud.sh(5 个已知坑脚本已处理:conda ToS / 空 miniconda 目录 / URDF headless patch / 重复 ICD / torch 版本 / flatdict)。

### 3. 冒烟验证（64 env / 3 iter）
```bash
$SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL && timeout 600 \$TRAIN_PY -m legged_lab.scripts.train --task=g1_tt --num_envs=64 --headless --logger=tensorboard --predictor --max_iterations=3 2>&1 | tail -25"
```
看到 `Learning iteration 0/3 … 2/3`、无 Traceback/NaN/`Multiple ICD`/`Failed to create GPU` → OK。

### 4. 起 idle9 训练（nohup,通宵;先清旧 run 防 watchdog 捡残留 ckpt）
```bash
$SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL \
  && : > train_idle9_watchdog.log \
  && nohup bash legged_lab/scripts/watchdog_train_idle9.sh > /dev/null 2>&1 & echo \$! > /tmp/idle9_watchdog.pid; echo launched pid=\$(cat /tmp/idle9_watchdog.pid)"
```
(watchdog 用 `TRAIN_PY` 覆盖 python 路径;`logs/g1_tt_idle9/` 是新实验,本就空,无需清。)

### 5. 等 ~2-3 分钟(IsaacSim 启动)后确认训练在跑
```bash
$SSH 'grep -E "Learning iteration" /mnt/workspace/Pingpong_TTRL/train_idle9_watchdog.log | tail -1'
```
- 看到 `Learning iteration N/40000`(N 在涨)→ ✓ 报告用户:idle9 已起,通宵跑,早上 eval。
- 没有 → `$SSH 'tail -40 .../train_idle9_watchdog.log'` 排错(看 GPU/Vulkan)。

## idle9 是什么
- **无球冻结的真正修复**:`reward_idle_pose` —— 无球(mask_invalid)时奖励**上半身(waist+双臂)跟踪一个 ready 待击姿态**(取自 model_36000 打球时的 median 上身姿态,击球臂是抬起的,不是放下的默认姿态→idle↔hit 平滑);下半身(腿)留给平衡。这是之前 idle3-8 缺的"明确该摆哪个姿势"。
- 几何用已验证的 **-1.6**(50cm/idle6 暂缓,在 git 历史里)。从零训,TARGET=40000(~大半天)。
- watchdog nohup 挂着:Claude 窗口/笔记本断了训练照跑。

## 判健康 / 注意
- 从零训**早期 ep_len 低正常**;看趋势上升。
- 真冻结信号:`Mean episode length` 持续塌到 ~5(深塌)。偶尔单次浅 dip 不算。idle9 若过 ~iter 5k-13k 不深塌 = 修复有效。
- ⚠️ 挑 ckpt 用 **eval 真成功率**,别假设最后一个最好:
  ```bash
  $SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL && \$TRAIN_PY -m legged_lab.scripts.eval --task=g1_tt_eval --predictor --headless --load_run <时间戳> --checkpoint model_<N>.pt --num_envs 100 --seed 0 2>&1 | tail -20"
  ```

## 不要做
- 不要中途反复问用户;按步骤跑完再报告。
- 不要去推送/改动 `unitree_rl_lab`(上游公开 fork,之前泄露过)。

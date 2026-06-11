---
name: g1-tt-cloud-train
description: 通过 SSH 在云端 GPU 实例上安装 Pingpong_TTRL 的 G1 乒乓训练环境(Isaac Sim 4.5 + IsaacLab 2.1.1)并自动启动 idle6(离桌50cm)训练通宵跑。当用户要在云端搭建/启动 G1 乒乓训练时使用。
---

# G1 乒乓 云端安装 + 训练（idle6 / 离桌边 50cm 版）

自动在云端 GPU 实例上完成:**装环境 → 冒烟验证 → 起 idle6 训练(nohup,通宵跑)**。
**全程不要停下来问用户,按步骤执行到底**(除非遇到无法解决的 FATAL 才报告并停)。

## SSH 目标（实例变了就改这里的 HOST/PORT）

```bash
HOST=39.101.75.133 ; PORT=1021 ; USER=root
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $USER@$HOST"
```
先测连通:`$SSH 'echo OK; nvidia-smi -L'`。连不上(密钥/IP 变了)→ 报告用户并停。

## 步骤

### 1. 拉最新代码 + 跑安装脚本（nohup,幂等,含 ~20GB 下载）
```bash
$SSH 'set -e; cd /mnt/workspace; [ -d Pingpong_TTRL ] || git clone https://github.com/T1Amoo/Pingpong_TTRL.git; cd Pingpong_TTRL && git fetch -q && git checkout g1-tt-deploy -q && git pull -q && nohup bash setup_cloud.sh > /mnt/workspace/setup.log 2>&1 & echo started'
```

### 2. 轮询安装日志（每 ~60s,可能 20–40 分钟）
```bash
$SSH 'tail -6 /mnt/workspace/setup.log'
```
- 出现 **`SETUP_OK`** → 装好,进下一步。
- 出现 **`FATAL`**(Vulkan 看不到 NVIDIA)→ 实例缺图形能力。报告用户:**重建实例加环境变量 `NVIDIA_DRIVER_CAPABILITIES=all`,或换 isaaclab 镜像底座**。停。
- 出现 Traceback/error → 读日志按这些常见坑修后重跑 setup_cloud.sh:
  - torch 被别的包升成 cu12 → `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118`
  - flatdict 构建失败 → `pip install flatdict==4.0.1 --no-build-isolation`
  - cmake 缺 → `apt-get install -y cmake build-essential`

### 3. 冒烟验证（64 env / 3 iter）
```bash
$SSH 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate pingpong && cd /mnt/workspace/Pingpong_TTRL && export OMNI_KIT_ACCEPT_EULA=YES && timeout 600 python -m legged_lab.scripts.train --task=g1_tt --num_envs=64 --headless --logger=tensorboard --predictor --max_iterations=3 2>&1 | tail -25'
```
看到 `Learning iteration 0/3 … 2/3`、无 Traceback/NaN → OK。报 Vulkan/GPU 错 → 回步骤2 的 FATAL 处理。

### 4. 起 idle6 训练（nohup,通宵）
```bash
$SSH 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate pingpong && cd /mnt/workspace/Pingpong_TTRL && export OMNI_KIT_ACCEPT_EULA=YES && nohup bash legged_lab/scripts/watchdog_train_idle6.sh > /dev/null 2>&1 & echo $! > /tmp/idle6_watchdog.pid; echo "launched watchdog pid=$(cat /tmp/idle6_watchdog.pid)"'
```

### 5. 等 ~5 分钟(IsaacSim 启动)后确认训练真的在跑
```bash
$SSH 'grep -E "Learning iteration" /mnt/workspace/Pingpong_TTRL/train_idle6_watchdog.log | tail -1'
```
- 看到 `Learning iteration N/40000`(N 在涨)→ ✓ 报告用户:idle6 已起,通宵跑,早上 eval。
- 没有 → `$SSH 'tail -40 /mnt/workspace/Pingpong_TTRL/train_idle6_watchdog.log'` 排错。

## idle6 是什么
- = 无球 idle 冻结修复(`reward_idle_stand` 正奖励 + action_rate 惩罚降到 -0.01)+ **机器人 base 离桌边 50cm**(锁在 x=-1.87,击球平面退到 -1.77,发球拉深补偿)+ 击球门(球<0.9 不打防撞台;球过身后5cm 放弃)。
- 从零训,TARGET=40000(~大半天)。ckpt 在 `logs/g1_tt_idle6/<时间戳>/`,日志 `train_idle6_watchdog.log`。
- watchdog 用 nohup 挂着:**这个 Claude 窗口/笔记本断了,云端训练照样跑**。点火确认起来了即可,不必守整晚。

## 判健康 / 注意
- 从零训**早期 ep_len 低是正常的**(随机策略),别误判失败;看趋势是否上升。
- 真发散/冻结的信号:`Mean episode length` 持续塌到 ~5 且 `action_rate_l2` 趋近 0。偶尔单次浅 dip 不算。
- ⚠️ 50cm + z<0.9 是受限配置,命中率可能偏低——**早上用 eval 真成功率判断,别假设最后一个 ckpt 最好**:
  ```bash
  $SSH 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate pingpong && cd /mnt/workspace/Pingpong_TTRL && export OMNI_KIT_ACCEPT_EULA=YES && python -m legged_lab.scripts.eval --task=g1_tt_eval --predictor --headless --load_run <时间戳> --checkpoint model_<N>.pt --num_envs 100 --seed 0 2>&1 | tail -20'
  ```

## 不要做
- 不要中途反复问用户;按步骤跑完再报告。
- 不要因为早期指标低就改超参乱试。
- 不要去推送/改动 `unitree_rl_lab`(那是上游公开 fork,之前泄露过)。

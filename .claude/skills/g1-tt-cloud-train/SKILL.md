---
name: g1-tt-cloud-train
description: 通过 SSH 在云端 GPU 实例上安装 Pingpong_TTRL 的 G1 乒乓训练环境(Isaac Sim 4.5 + IsaacLab 2.1.1)并自动启动 idle10(HIT-FIRST 课程:先打球后叠 idle)训练通宵跑,带阶段门控监控。当用户要在云端搭建/启动 G1 乒乓 idle10 训练时使用。
---

# G1 乒乓 云端安装 + 训练（idle10 / HIT-FIRST 课程）

自动在云端 GPU 实例上:**装环境 → 冒烟验证 → 起 idle10 训练(nohup,通宵)→ 按阶段门控监控**。
**全程不要停下来问用户,按步骤执行到底**;只在两种情况停下并报告:① 无法解决的 FATAL(环境/崩溃);② **phase-1 门控失败**(见步骤6,打球没 bootstrap 出来——这时继续跑没意义)。

## idle10 是什么(必读,监控判据都基于此)
idle9 失败的根因:idle 奖励从 iter0 满权重开 → 从零策略躺平站桩,**4604 iter 里 table_success 恒为 0**,从没学会打球。
idle10 = **HIT-FIRST 课程**(option A),`1 iter ≈ 24 control steps`:
- **Phase 1(iter 0 → ~5000):球一直在 + idle 奖励=0** → 纯打球 bootstrap(就是已验证能从零学会打球的周末配方)。**这一段必须练出打球**。
- **Phase 2(~iter 5000 之后):idle 奖励 0→1 用 ~6000 iter 爬满(~iter 11000),无球间隙 0→满 用 ~12000 iter 爬满(~iter 17000)**。idle 爬得比无球快 2× → 稳定的 ready-pose 参考永远领先无球难度(避免 idle3 的 freeze-collapse)。
- 之后到 TARGET=30000 巩固。idle 权重已压低(idle_pose 3.0→1.0, idle_stand 1.0→0.5)防止侵蚀打球。
- ⚠️ 课程绑 `sim_step_counter`,崩溃 resume 会从 phase1 重来(策略接着 ckpt 跑,无害,只是 idle 巩固变慢)。

## SSH 目标（实例变了就改这里的 HOST/PORT）
```bash
HOST=39.101.75.133 ; PORT=1021 ; USER=root
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 $USER@$HOST"
```
先测连通:`$SSH 'echo OK; nvidia-smi -L'`。连不上(密钥/IP 变了)→ 报告用户并停。

## 环境变量(每次起 python 训练/冒烟/eval 都要带;5 个坑已内置在 setup_cloud.sh)
```bash
ENVSET='source /root/miniconda3/etc/profile.d/conda.sh && conda activate pingpong \
  && export OMNI_KIT_ACCEPT_EULA=YES \
  && export VK_ICD_FILENAMES=$(cat /mnt/workspace/.vk_icd 2>/dev/null || echo /etc/vulkan/icd.d/nvidia_icd.json) \
  && export TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python'
```

## 步骤

### 1. 拉最新代码 + 跑安装脚本（nohup,幂等;首次 ~20GB 下载,已装好则很快）
```bash
$SSH 'set -e; cd /mnt/workspace; [ -d Pingpong_TTRL ] || git clone https://github.com/T1Amoo/Pingpong_TTRL.git; cd Pingpong_TTRL && git fetch -q && git checkout g1-tt-deploy -q && git pull -q && nohup bash setup_cloud.sh > /mnt/workspace/setup.log 2>&1 & echo started'
```

### 2. 轮询安装日志（每 ~60s;首次 20–40 分钟;已装好会很快到 SETUP_OK）
```bash
$SSH 'tail -8 /mnt/workspace/setup.log'
```
- 出现 **`SETUP_OK`** → 进下一步。
- 出现 **`FATAL`**(Vulkan 看不到 NVIDIA)→ 实例缺图形能力。报告用户:**重建实例加 `NVIDIA_DRIVER_CAPABILITIES=all`,或换 isaaclab 镜像底座**。停。
- 其它报错 → 读日志修后重跑 setup_cloud.sh(5 个已知坑脚本已处理)。

### 3. 冒烟验证（64 env / 3 iter）
```bash
$SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL && timeout 600 \$TRAIN_PY -m legged_lab.scripts.train --task=g1_tt --num_envs=64 --headless --logger=tensorboard --predictor --max_iterations=3 2>&1 | tail -25"
```
看到 `Learning iteration 0/3 … 2/3`、`reward_idle_pose/stand: 0.0000`(phase1 内 idle 本就为 0,正常)、无 Traceback/NaN/`Multiple ICD`/`Failed to create GPU` → OK。

### 4. 起 idle10 训练（nohup,通宵;新实验名,日志本就空,无需清旧 run）
```bash
$SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL \
  && : > train_idle10_watchdog.log \
  && nohup bash legged_lab/scripts/watchdog_train_idle10.sh > /dev/null 2>&1 & echo \$! > /tmp/idle10_watchdog.pid; echo launched pid=\$(cat /tmp/idle10_watchdog.pid)"
```

### 5. 等 ~2-3 分钟(IsaacSim 启动)后确认在跑
```bash
$SSH 'grep -E "Learning iteration" /mnt/workspace/Pingpong_TTRL/train_idle10_watchdog.log | tail -1'
```
看到 `Learning iteration N/30000`(N 在涨)→ ✓。没有 → `$SSH 'tail -40 .../train_idle10_watchdog.log'` 排错。

### 6. ⭐⭐ PHASE-1 门控（最关键;到 iter ~5500 时查一次)
phase1(iter<5000)idle 关、无球关 = 纯打球,**就是已验证能从零学会打球的配方**。所以到 iter ~5500 时 **table_success 必须已经涨起来**;若还是 ~0,说明打球没 bootstrap 出来(配置错/发散),**继续跑 phase2 叠 idle 毫无意义** → 停 + 报告用户。
```bash
$SSH 'L=/mnt/workspace/Pingpong_TTRL/train_idle10_watchdog.log
grep "reward_table_success:" $L | awk "{print \$NF}" | tail -40 | sort -rn | head -3
grep -E "Learning iteration" $L | tail -1'
```
- **判据(到达 iter ≥5000 后)**:最近 table_success 的高点 **> 0.30** → ✓ 打球 bootstrap 成功,放心进 phase2。继续监控(步骤7)。
  - 参照:原始成功 run iter4000 时 table_success≈0.72;idle9 全程 0.000。门槛 0.30 足以区分"学会了"和"躺平"。
- **table_success 仍 ~0(<0.05)且 iter 已过 6000** → ❌ **STOP**:`kill $(cat /tmp/idle10_watchdog.pid)`,报告用户"phase-1 打球未 bootstrap,与预期(纯打球配方)矛盾,需排查"。不要硬着头皮往下跑。

### 7. PHASE-2 监控(过了门控之后,通宵盯)
关注两类失败 + 两个里程碑:
```bash
# 取最近 iter 与 ep_len / 关键 reward
$SSH 'L=/mnt/workspace/Pingpong_TTRL/train_idle10_watchdog.log
grep -E "Learning iteration|Mean episode length:|reward_table_success:|action_rate" $L | tail -12'
```
- **冻结(idle3 式)**:`Mean episode length` **持续**塌到 ~5(连续多 iter,非单点)。单 iter dip 到 30-50 随即回弹是正常 PPO 噪声,不报。
- **发散**:`action_rate_l2` 每个 iter 都 -1e5 量级(持续非偶发)+ ep_len 持续 ~40。(偶发尖峰是固有噪声,看中位/趋势。)
- **里程碑 ~iter 11000**(idle 爬满):无球此时 ~50%,看 ep_len 仍稳(250-300)。
- **里程碑 ~iter 17000**(无球爬满 70%)+ 之后:**双指标都要好**——① ep_len 稳(没被满额无球冻住);② `reward_table_success` 仍维持 phase1 末的水平(idle 没把打球侵蚀掉)。若 table_success 明显跌 → idle 侵蚀打球,记录并报告(可能需降 idle 权重重训)。

### 8. 跑满 TARGET=30000 后 eval（挑 ckpt;别假设 last=best)
两项都要测,选**两者都好**的 ckpt(打球 ~0.7 量级 且 无球能站稳):
```bash
# (a) 纯打球成功率(无球关,固定中等 bounce);对标 model_36000 的 0.74
$SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL && \$TRAIN_PY -m legged_lab.scripts.eval --task=g1_tt_eval --predictor --headless --load_run <时间戳> --checkpoint model_<N>.pt --num_envs 100 --seed 0 2>&1 | tail -20"
# (b) 无球站稳(间断发球,10s 周期):看无球段不摔不抽
$SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL && TT_SERVE_PERIOD=10 \$TRAIN_PY -m legged_lab.scripts.eval --task=g1_tt_eval --predictor --headless --load_run <时间戳> --checkpoint model_<N>.pt --num_envs 100 2>&1 | tail -20"
```
跨 seed(0 / 7777)复测 top ckpt。报告用户:各 ckpt 的 (a)(b) 成绩 + 推荐导出哪个到 sim2sim。⚠️ 8GB 卡同时只能跑一个 eval;云端 L20(24GB)可放宽但仍建议串行。

## 不要做
- 不要中途反复问用户;按步骤跑完再报告(phase-1 门控失败除外,那个必须停+报)。
- 不要去推送/改动 `unitree_rl_lab`(上游公开 fork,之前泄露过)。
- 不要删 ckpt/run(用户铁律:训练成果永不删)。

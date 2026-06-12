---
name: g1-tt-cloud-train
description: 通过 SSH 在云端 GPU 实例上安装 Pingpong_TTRL 的 G1 乒乓训练环境(Isaac Sim 4.5 + IsaacLab 2.1.1)并自动启动 idle10(三阶段从零课程:先固定易球学打,再加难度,最后叠 idle)通宵/周末跑,带阶段门控监控。当用户要在云端搭建/启动 G1 乒乓 idle10 训练时使用。
---

# G1 乒乓 云端安装 + 训练（idle10 / 三阶段从零课程 option A）

自动在云端 GPU 实例上:**装环境 → 冒烟 → 清旧 idle10 → 起 idle10 训练(nohup,周末跑)→ 按阶段门控监控 → 跑满后 eval**。
**全程不要停下来问用户,按步骤执行到底**;只在两种情况停下并报告:① 无法解决的 FATAL(环境/崩溃);② **stage-1 门控失败**(易球都没学会打 = 有 bug,见步骤6)。

## idle10 是什么(必读,监控判据基于此)
从零学会打球需要 ~15-20k iter(不是几千)。idle9 失败是 idle 奖励从 iter0 就开 → 躺平不打球。idle10 = **三阶段从零课程**,`1 iter ≈ 24 控制步(cs)`,发球难度用 raw 计数(240/iter):
- **Stage 1(iter 0 → ~15000):固定 EASY 发球**(无难度课程)+ idle 关 + 无球关 → 纯学打简单球(已验证的 2026-06-03 配方)。
- **Stage 2(iter ~15000 → ~27000):发球难度 easy→hard 渐进** → 啃难球。
- **Stage 3(iter ~27000 → ~39000):idle 奖励 + 无球渐进**(此时已会打)。idle 比无球快 2× 爬满(idle 满~33000,无球满~39000)→ 稳定 ready-pose 参考永远领先无球难度,避免 idle3 freeze。
- 巩固到 **TARGET=42000**。idle 权重已压低(pose1.0/stand0.5)。
- ⚠️ 课程绑 sim_step_counter,但 watchdog resume 会注入 `TT_SIM_STEP_OFFSET=N*240` 让课程接着 N 走,**不回退到 stage1**(已在 watchdog 内处理)。
- ⏱ 42000 iter 在 L20 上可能 2-3 天,**周末未必跑满**——没关系:idle 在 ~iter33000 就训上了,跑到 ~36000+ 即有可用结果。

## SSH 目标（实例变了就改 HOST/PORT）
```bash
HOST=39.101.75.133 ; PORT=1021 ; USER=root
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 $USER@$HOST"
```
先测连通:`$SSH 'echo OK; nvidia-smi -L'`。连不上 → 报告用户并停。SSH 偶发 exit 255(网络抖),重试 1-2 次即可。

## 环境变量(每次起 python 训练/冒烟/eval 都带;5 个坑已内置在 setup_cloud.sh)
```bash
ENVSET='source /root/miniconda3/etc/profile.d/conda.sh && conda activate pingpong \
  && export OMNI_KIT_ACCEPT_EULA=YES \
  && export VK_ICD_FILENAMES=$(cat /mnt/workspace/.vk_icd 2>/dev/null || echo /etc/vulkan/icd.d/nvidia_icd.json) \
  && export TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python'
```

## 步骤

### 1. 拉最新代码 + 跑安装脚本（nohup,幂等;已装好则很快）
```bash
$SSH 'set -e; cd /mnt/workspace; [ -d Pingpong_TTRL ] || git clone https://github.com/T1Amoo/Pingpong_TTRL.git; cd Pingpong_TTRL && git fetch -q && git checkout g1-tt-deploy -q && git pull -q && nohup bash setup_cloud.sh > /mnt/workspace/setup.log 2>&1 & echo started'
```

### 2. 轮询安装日志（每 ~60s;已装好会很快 SETUP_OK）
```bash
$SSH 'tail -8 /mnt/workspace/setup.log'
```
- `SETUP_OK` → 下一步。`FATAL`(Vulkan 看不到 NVIDIA)→ 报告用户重建实例加 `NVIDIA_DRIVER_CAPABILITIES=all`,停。其它报错读日志修后重跑 setup_cloud.sh。

### 3. 冒烟验证（64 env / 3 iter）
```bash
$SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL && timeout 600 \$TRAIN_PY -m legged_lab.scripts.train --task=g1_tt --num_envs=64 --headless --logger=tensorboard --predictor --max_iterations=3 2>&1 | tail -25"
```
看到 `Learning iteration 0/3 … 2/3`、`reward_idle_pose/stand: 0.0000`(stage1 内 idle 本就 0,正常)、无 Traceback/NaN/`Multiple ICD` → OK。

### 4. ⭐清掉旧 idle10 run(含冒烟产物)+ 起 idle10 训练（nohup,周末跑）
**必须先清** `logs/g1_tt_idle10/`,否则 watchdog 会从冒烟 ckpt(model_2)误 resume(用户要求覆盖旧 idle10)。此清空只在本次首发执行一次;watchdog 内部 resume 不重跑本步。
```bash
$SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL \
  && rm -rf logs/g1_tt_idle10 \
  && : > train_idle10_watchdog.log \
  && nohup bash legged_lab/scripts/watchdog_train_idle10.sh > /dev/null 2>&1 & echo \$! > /tmp/idle10_watchdog.pid; echo launched pid=\$(cat /tmp/idle10_watchdog.pid)"
```

### 5. 等 ~2-3 分钟后确认在跑
```bash
$SSH 'grep -E "Learning iteration|fresh start" /mnt/workspace/Pingpong_TTRL/train_idle10_watchdog.log | tail -2'
```
看到 `fresh start -> 42000` 且 `Learning iteration N/42000`(N 涨)→ ✓。

### 6. ⭐⭐ STAGE-1 门控(到 iter ~13000-15000 时查一次,最关键)
stage1(iter<15000)固定易球、无 idle、无无球 = 已验证能学会打简单球的配方。所以到 iter ~13000 时 **shaped table_success 必须已涨起来**(老固定易球 run iter4000 就到 shaped 0.7);若仍 ~0,说明易球都学不会(配置错/发散),**继续往后(加难度/idle)毫无意义** → 停 + 报告。
```bash
$SSH 'L=/mnt/workspace/Pingpong_TTRL/train_idle10_watchdog.log
grep "reward_table_success:" $L | awk "{print \$NF}" | tail -60 | sort -rn | head -3
grep -E "Learning iteration" $L | tail -1'
```
- iter ≥13000 且 shaped table_success 高点 **> 0.40** → ✓ 易球打球已 bootstrap,放行进 stage2。
- iter 已过 15000 但仍 **< 0.05** → ❌ STOP:`kill $(cat /tmp/idle10_watchdog.pid)`,报告"stage-1 易球未学会打,需排查"。
- 在 0.05~0.40 之间且还在涨 → 继续观察,别停。

### 7. STAGE-2/3 监控(周末盯)
```bash
$SSH 'L=/mnt/workspace/Pingpong_TTRL/train_idle10_watchdog.log
grep -E "Learning iteration|Mean episode length:|reward_table_success:|action_rate" $L | tail -14'
```
- **Stage 2(iter 15000→27000,加难度)**:`table_success` 随发球变难可能下探,应部分回升;ep_len 别持续塌。
- **Stage 3(iter 27000+,加 idle/无球)**:
  - 里程碑 ~iter33000(idle 满)、~iter39000(无球满 70%)。
  - **冻结(idle3 式)**:`Mean episode length` 持续塌 ~5(连续多 iter,非单点)。单点 dip 回弹是正常噪声。
  - **发散**:`action_rate_l2` 每 iter 都 -1e5 量级(持续)+ ep_len 持续 ~40。
  - **双指标(iter39000+)**:① ep_len 稳(满额无球没冻住);② `table_success` 仍维持 stage2 末水平(idle 没侵蚀打球)。table_success 明显跌 → idle 侵蚀,记录并报告。

### 8. 跑满(或周末到点)后 eval(挑 ckpt;别假设 last=best)
两项都测,选两者都好的(打球 ~0.7 量级 且 无球能站稳):
```bash
# (a) 纯打球成功率(无球关,固定中等 bounce);对标 model_36000 的 0.74
$SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL && \$TRAIN_PY -m legged_lab.scripts.eval --task=g1_tt_eval --predictor --headless --load_run <时间戳> --checkpoint model_<N>.pt --num_envs 100 --seed 0 2>&1 | tail -20"
# (b) 无球站稳(间断发球 10s 周期)
$SSH "$ENVSET && cd /mnt/workspace/Pingpong_TTRL && TT_SERVE_PERIOD=10 \$TRAIN_PY -m legged_lab.scripts.eval --task=g1_tt_eval --predictor --headless --load_run <时间戳> --checkpoint model_<N>.pt --num_envs 100 2>&1 | tail -20"
```
跨 seed(0/7777)复测 top ckpt。报告各 ckpt 的 (a)(b) 成绩 + 推荐导出哪个。⚠️ 同时只跑一个 eval。

## 不要做
- 不要中途反复问用户(stage-1 门控失败除外,那个必须停+报)。
- 不要推送/改动 `unitree_rl_lab`(上游公开 fork,泄露过)。
- 不要删 ckpt/run(用户铁律);步骤4 的 `rm -rf logs/g1_tt_idle10` 是用户明确要求覆盖的旧冒烟 run,仅此一处、仅首发执行。

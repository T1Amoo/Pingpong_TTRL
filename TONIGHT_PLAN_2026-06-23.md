# 今晚训练计划 (2026-06-23 临走时执行) — 待 29999 sim2real 反馈后微调

> 现在只准备,不执行。用户先用 model_29999 做 sim2real;有问题再据此调权重。
> 切换时机:用户临走。两条都启动后挂监控过夜。

---
## A. locomotion 移到本地续训(从云端 v3 resume,别从零)

云端 v3 = `g1_locomotion_v3`(4096env,加了 gait/clearance/slide/fly,健康)。停云端前先拉 ckpt:

```bash
# 1) 云端查最新 ckpt
ssh -p 1021 root@39.101.75.133 'ls -t /mnt/workspace/Pingpong_TTRL/logs/g1_locomotion_v3/*/ | head; \
  R=$(ls -dt /mnt/workspace/Pingpong_TTRL/logs/g1_locomotion_v3/*/|head -1); ls $R/model_*.pt|tail -3'
# 2) 拉 run 目录(params + 最新 model_*.pt)到本地同路径
RUN=<云端run时间戳>
mkdir -p logs/g1_locomotion_v3/$RUN/params
scp -P 1021 "root@39.101.75.133:/mnt/workspace/Pingpong_TTRL/logs/g1_locomotion_v3/$RUN/params/*" logs/g1_locomotion_v3/$RUN/params/
scp -P 1021 "root@39.101.75.133:/mnt/workspace/Pingpong_TTRL/logs/g1_locomotion_v3/$RUN/model_<最新>.pt" logs/g1_locomotion_v3/$RUN/
# 3) 停云端 v3(腾 GPU 给 TT 微调)
ssh -p 1021 root@39.101.75.133 'pkill -f watchdog_train_locomotion; pkill -9 -f "scripts.train.*g1_locomotion"'
# 4) 本地 resume:watchdog_train_locomotion.sh 的 latest() 会自动找到 model_<最新> 并 --resume 续训到 TARGET=20000
cd /media/woan/.../Pingpong_TTRL
export OMNI_KIT_ACCEPT_EULA=YES VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
TARGET=20000 NUM_ENVS=512 nohup bash legged_lab/scripts/watchdog_train_locomotion.sh >/dev/null 2>&1 &
# (注意:watchdog EXP 现=g1_locomotion_v3 ✓;本地512env比云端4096慢~8倍,过夜跑)
```

---
## B. 云端微调 29999 去抖(reward shaping,先微调带兜底)

### B1. 加奖励(改 `legged_lab/envs/g1_tt/g1_tt_config.py` 的 G1TableTennisRewardCfg)
**去抖不伤打球**——两招(权重待 sim2real 后定,先给保守起点):

1) **jerk(动作二阶差)惩罚** —— 需先在 `legged_lab/mdp/rewards.py` 加函数(高频嗡嗡=高jerk,流畅快挥=低jerk):
```python
def action_jerk_l2(env: "BaseEnv") -> torch.Tensor:
    # 二阶差分: a_t - 2 a_{t-1} + a_{t-2}; env 需存 last_action / last_last_action
    return torch.sum(torch.square(env.action - 2*env.last_action + env.last_last_action), dim=1)
# ⚠️ 确认 tt_env 是否已存 last_last_action;没有则在 step 里维护(prev2=prev; prev=action)
```
配置加:`action_jerk = RewTerm(func=mdp.action_jerk_l2, weight=-0.005)` (保守起步,按成功率 ramp)
**或更省事**:先把现有 `dof_acc_l2` 从 -1.25e-7 调到 -3e-7~-5e-7 试(关节加速度≈jerk-ish)。

2) **idle 门控 stillness** —— 用已有 `env.mask_invalid`(无可打球时 True,reward_idle_stand 同款门控)。等球时罚动作幅度/速率,击球放开:
```python
def idle_action_rate(env: "TTEnv") -> torch.Tensor:  # 仅 idle 时罚动作变化
    r = torch.sum(torch.square(env.action - env.last_action), dim=1)
    return r * env.mask_invalid.float()
```
配置加:`idle_still = RewTerm(func=mdp.idle_action_rate, weight=-0.05)`
(⚠️ v7 no_ball_period_s=0 一直发球,mask_invalid 只在两球间触发 → 这招管"等球抖";主动跟踪抖靠 jerk)

### B2. 微调启动(从 v7/model_29999 warm-start,新实验名隔离)
- 新 EXP=`g1_tt_v7_ft`,改 `G1TableTennisAgentCfg.experiment_name`(或复制 watchdog 改 EXP)。
- warm-start:把 `logs/g1_tt_v7/2026-06-22_02-39-44/model_29999.pt` 作种子放进新 run,`--resume true --load_run ... --checkpoint model_29999.pt`,`LOAD_OPTIMIZER=0`(fresh opt,避免 resume 失稳)。
- 云端跑,~3-5k iter 即可见效。watchdog 仿 watchdog_train_v7.sh,TARGET 设 35000(29999+~5k)。

### B3. 监控判据(关键)
- **盯难球成功率**(跑 `eval --task g1_tt_eval_hard --predictor --load_run g1_tt_v7_ft/<run> --checkpoint model_X.pt TT_EVAL_MAX_SERVES=1000`):
  - 抖动降(sim2sim 稳态 maxqvel 往 v5 的 0.07-0.12 靠)+ 难球成功率守住 ~0.77 → **成功**。
  - 抖动不降 或 难球成功率塌(<0.6) → 惩罚过重/warm-start 失稳 → 退**从零重训**(带去抖奖励)。

---
## 备注
- 部署选定:**model_29999**(config `policy_dir: config/policy/table_tennis/v7_29999`)。27000(v7/)、25000(v7_25000/)留。
- sim2sim 几何已修(-2.0/serve);eval.py 已加 TT_EVAL_MAX_SERVES+动作指标;tt_sim 已加 WASD+TT_JOINT_DIAG。
- pkill 自杀坑:手动杀进程用 pgrep 取 PID 再 kill -9,别在命令里 `pkill -f <含本命令字串>`。

# A1 乒乓 v12：前探击球几何（击球面移到 ready 拍面前方）

日期：2026-07-30
分支：a1-tt-migration
配置类：`A1TableTennisV12EnvCfg`（继承 `A1TableTennisV11EnvCfg`）

## 1. 背景与问题

v11 用 V10 的 NEW_POSE 前手姿态（`r1=-0.505 … r7=-1.3`）+ `hit_plane_x=-1.58`。用 URDF FK（`X1_URDF_V1_1_sj_fixed.urdf`，base=`(-1.8,0.76,0.028)`，identity 朝向）核算，default/ready 姿态下：

- ready `paddle_touch_point = (-1.578, 0.016, 1.163)`
- `hit_plane_x = -1.58`
- **Δ(ready_x − hit_plane_x) = +1.5 mm ≈ 0**

即 **ready 拍面几乎完全贴在击球面上等球，零送拍**。这与早期设计草案（Lark 图：*"hit plane 从准备位前方 0.12m 改成 0.16m，要求更明显的送拍距离"*）的本意相反——那份草案里的送拍意图从未落地。

FK 交叉验证：算出的 ready `y=0.016`、base 配置姿态算出的 `y=0.104` 与配置注释 *"paddle_touch_point y≈0.10 with home_y=0.76"* 吻合，确认 FK 正确。

## 2. 目标

让**击球面位于 ready 拍面前方 ~0.16m（朝球网 +x）**，逼出主动前探/送拍击球，同时避免手臂前探时蹭到球桌。

成功判据：
- v12 生效几何满足 `hit_plane_x − ready_paddle_x ≈ 0.16m`（±0.02）。
- `hit_plane_x` 离球桌近端边缘（x=-1.37）留 ≥0.20m 余量。
- 发球分布在 `x=hit_plane_x` 处的 (y,z) 落在击球窗 `y∈[-0.12,0.43]`、`z∈[0.92,1.27]` 内（probe 直方图验证）。
- 训练可跑通：过 iter 20000 后 `value_loss` 守住 < 100（沿用 critic 稳定判据 [[g1_tt_critic_divergence_termination_penalty_2026-06-30]]）。

## 3. 设计

### 3.1 几何（核心改动）

手臂姿态固定时，base 后移多少 ready 拍面就后移多少：`ready_paddle_x ≈ base_x + 0.222`。

| 量 | v11 | **v12** |
|---|---|---|
| `base.init_state.pos` x | -1.8 | **-2.0** |
| `hit_plane_x` | -1.58 | **-1.62** |
| `hit_target_x_range` | (-1.58,-1.58) | **(-1.62,-1.62)** |
| ready 拍面 x（FK 推算）| -1.578 | ≈ **-1.778** |
| 送拍距离 hit−ready | ~0 | **+0.158 ≈ 0.16m** ✓ |
| hit_plane 离桌沿(-1.37) | 0.21m | **0.25m** ✓ |

- base=-2.0 与 G1 验证过的站位一致（`g1.py:40` *"站位≥60cm 离桌、匹配 hit_plane"*）。
- y/z 不受 x 后移影响：`hit_target_y_range=(-0.12,0.43)`、`hit_target_z_range=(0.92,1.27)` 不变（继承 v11）。
- base y 仍为 0.76、`home_y=0.76` 不变。

### 3.2 发球重算（解算 + probe 验证）

球在自侧球台弹跳（`serve_bounce_x∈(-1.24,-0.96)`）后朝机器人飞（x 递减），在 `hit_plane_x` 处被拦。v12 拦截点从 -1.58 移到 -1.62（**仅多走 4cm、低几 cm**），改动很小。

步骤：
1. 用已验证的直球飞行模型（Cd≈0.4，见 [[g1_tt_findings_reward_truncation_aero_latency_2026-06-25]]）解算当前发球分布在 x=-1.62 处的 (y,z)。
2. 跑一个短 sim probe（headless），记录 4096 env 发球落到 x=-1.62 平面时的 (y,z) 直方图。
3. 若 z 中位/分位掉出 [0.92,1.27]，微调 `serve_bounce_vz_range`（vz 是速度旋钮，见 [[g1_tt_v13_speed_height_curriculum_2026-06-29]]）把 z 抬回窗内；y 目标窗不变、`serve_y_center` 大概率不动。
4. 硬球分布（`*_hard`）同法核验。

### 3.3 predictor 默认开启

`train.py` 的 `--predictor`（`action="store_true"`）改为**默认 True**，新增 `--no-predictor`（`action="store_false", dest="predictor"`）反向开关。

⚠️ **Blast radius（已确认接受）**：这会让 G1/T1 等**所有任务默认也训 predictor**，除非显式传 `--no-predictor`。用户已选择此全局默认方案。

### 3.4 送拍奖励：不新增（已确认）

Lark 图里的 `reward_flat_push_through`、`reward_paddle_forward_pre_hit`、`paddle_wait_*`/`joint_wait_*`（4 约束）、前向速度阈值 0.35 —— **全仓 grep 确认代码里均不存在**（rewards.py 无定义、任何 config 无注册）。仅 `reward_swing_through`(权重 0.25) 落地。

决策：**v12 不从零实现这些送拍/等待奖励**。前探由几何 + `reward_future_dis_ee`（把拍拉向 hit_plane 目标）自然驱动，`reward_swing_through`(0.25) 辅助。理由：YAGNI + 避免新增 reward shaping 触发 critic 发散（历史教训 [[g1_tt_critic_divergence_termination_penalty_2026-06-30]]）。若 v12 跑出"够不到/前探不足"再补写。

### 3.5 其余全部继承 v11 不动

NEW_POSE 前手姿态、2026-07-29 执行器 fit（DamiaoMIT response fn/zeta/delay/gain）、一阶低通、`reward_arm_ready_idle`+`penalty_arm_vel_idle` 等待整形、PPO 设置（lr 5e-4、entropy 0.006、adaptive）、课程（**从零 scratch，easy→ramp→hold，`max_iterations=30000`**）。

## 4. 启动流程

1. v12 代码就绪 + probe 验证发球在窗内。
2. 停 v11：云端先 `pgrep` 拿 watchdog PID 再 kill，再 kill 训练进程 PID（避免 pkill 自杀 [[pkill_self_kill_on_ssh_use_explicit_pid]]）。
3. 启动 v12：`python -m legged_lab.scripts.train --task a1_tt_v12 --num_envs 4096 --headless`（predictor 已默认开），watchdog 带 `OMNI_KIT_ACCEPT_EULA=YES`，轮询到目标 ckpt 落盘即 kill 子进程（[[g1_tt_v6_retrain_and_locomotion_2026-06-18]] 修复）。
4. 代码走 git push、云端 pull（[[code_sync_via_git_not_scp]]）。

## 5. 风险

- **发球出窗**：4cm 位移看似小，但球在下落，z 可能掉出下界。→ 3.2 的 probe 强制验证，出窗必调 vz。
- **critic 发散**：几何变了是新分布，从零训。→ 盯 `value_loss`（最早最准信号），过 20000 守 < 100。
- **predictor 全局默认波及 G1/T1**：其它机器人下次训练会默认带 predictor。→ 已加 `--no-predictor` opt-out；spec 记录在案。
- **前探不足**：若几何+future_dis_ee 不足以逼出明显送拍。→ 事后按需补 `reward_swing_through` 权重或新写 pre_hit 奖励，不在 v12 首版。

## 6. 验证清单

- [ ] FK 复核 v12 生效几何：ready 拍面 x ≈ -1.778，hit_plane -1.62，Δ≈0.16。
- [ ] probe：发球 (y,z) @ x=-1.62 落在击球窗内（易球+硬球）。
- [ ] `--no-predictor` 反向开关生效（G1/T1 可 opt-out）。
- [ ] v12 从零起训，value_loss 过 20000 守 < 100。
- [ ] 交叉筛选 ckpt 时按 [[tt_ckpt_cross_selection_procedure]]（hard eval 上台率 × sim2sim 稳态），选型前上真机。

# A1 反手 v8：奖励侧时序与腕部稳定性设计

日期：2026-08-11

训练任务：`a1_tt_backhand_v8`

实验目录：`a1_tt_backhand_real_v8_timing_wristquiet`

训练方式：从零 `5k easy + 10k ramp + 5k hold`

## 1. 结论

v8 只修正 v7 真机/仿真都存在的两个策略行为，不改几何、发球、接触、相机、预测器、执行器或部署接口：

1. `ball_future_t` 只作为训练奖励的 privileged state，不进入 actor。actor 仍为每帧 39 维、5 帧历史，共 195 维；现有 ONNX 和真机部署接口不需要修改。
2. pre-contact max-drawdown 从 `t_hit <= 0.60 s` 的挥拍窗口开启时才开始统计。窗口开启时以当前拍心 x 为零回撤基线，窗口前的准备/回位不再被误算成一次性回撤惩罚。
3. 恢复轻量 early-forward cost：`t_hit >= 0.70 s` 时全强度，`0.70 -> 0.60 s` 平滑释放，进入挥拍窗口后为零。击球平面后方保留 8 cm，不禁止 y/z 预定位，也不禁止短 ETA 球立即启动。
4. 在已有全关节 action-rate cost 之外，仅对 r5/r7 加 phase-aware action-rate cost。挥拍窗口外全强度，窗口内保留 25%，不对 r1--r4/r6 加额外约束，不增加部署侧滤波。
5. r4 保持 v7 nominal 和新 ready-pose 重锚点。本版不混入尚未通过独立留出验证的 r4 residual、方向分支或额外 effort cap。

## 2. 参数

| 项目 | v8 值 | 目的 |
|---|---:|---|
| swing window | `0.60 s` | 覆盖真机 gate ETA P95 约 `0.630 s` 的主体，同时保留紧急球立即运动 |
| transition | `0.10 s` | 避免奖励在单个控制 tick 上跳变 |
| early x retraction | `0.08 m` | 保留最后约 8 cm 的主动前挥，不要求完整 19.9 cm 行程一直等待 |
| early x cost weight | `-0.50` | 轻量 dense cost，不能压过接触/回球结果阶梯 |
| extra action-rate joints | `r5, r7` | 真机与 MuJoCo 都确认是 actor target 层的主要 3--5 Hz 抖动通道 |
| wrist action-rate weight | `-0.08` | 在原全关节 `-0.015` 基础上定向抑制 raw target 翻转 |
| in-swing wrist cost floor | `0.25` | 保留必要的拍面调整和击球角速度 |

## 3. 数据依据

真机记录：`系统辨识/sim2real/20260811/a1_backhand_v7_model9200_r108_newpose_trial1_20260811_104028.csv`。

- 130 次 gate 事件；gate 时匀速 ETA 到击球平面 P05/P50/P95 约 `0.157/0.356/0.630 s`。
- post-tau 目标前伸 2 cm 的中位 onset 约 `20 ms`，实际拍心约 `100 ms`；MuJoCo clean-ready 事件也出现立即前伸，说明不是单纯真机电机滞后。
- 真机 gate 段 0.2 s 高通 RMS：post-tau qdes 的 r5/r7 为 `0.080/0.072 rad`，实际 q 为 `0.072/0.074 rad`；主峰约 `3.52/3.83 Hz`。post-tau 到反馈相关系数为 `0.969/0.983`，最佳滞后约 `60 ms`。
- MuJoCo 同样存在 r5/r7 qdes/feedback 抖动，说明轻量训练侧 action-rate 约束有必要；继续加重部署 qdes 滤波会同时损失击球所需带宽。

## 4. r4 独立判断

按 robot-clock 对齐并对 `joint_seq` 去重后，本次策略有效段的 r4 反馈力矩统计为：

| 区间 | `|tau4| >= 7.6` | `|tau4| >= 8.0` | P95 / max |
|---|---:|---:|---:|
| 全策略有效段 | `5.74%` | `4.93%` | `7.98 / 8.86` |
| gate engaged | `14.00%` | `11.42%` | `8.35 / 8.81` |
| gate live | `8.90%` | `6.92%` | `8.21 / 8.59` |

全策略段 `|tau4| >= 7.6` 共 353 段，持续时间 P50/P95/max 约 `0.070/0.184/0.270 s`。因此 r4 不是偶发单点触边，而是在见球/挥拍阶段频繁进入饱和区。

现有新姿态重锚定二阶模型的动态段 r4 RMSE/P95 为 `0.0793/0.1780 rad`，模型相对真机仍有约 `10 ms` 额外形状滞后。旧动态曲线可继续作为 nominal，但单一 `fn/zeta/delay/gain` 参数组不能解释方向、姿态和连续高需求下的饱和残差。

8 月 8 日的多工作点 fit/select 已否决以下方案：

- 在黑箱二阶 response 后叠 6/8/10 Nm stiff-PD cap：重复建模，低负载 RMSE 恶化 4--6 倍。
- residual v2：select 高负载 RMSE/P95 只改善 `13.7%/16.8%`，低于 20% 准入线，并在 stress 中失败。
- 方向分支：teacher-forced 改善明显，但 closed-loop 仅约 2.3%，gate recall 塌为 0。

当前能确定的“更好模型”是统一灰盒闭环结构，而不是一组可立即替换的标量参数：从 post-tau qdes 直接预测 r4 q/dq，在同一状态方程中联合表示 nominal 小信号动态、约 20 ms 外加形状滞后、方向/姿态相关的连续 torque-speed/加速度饱和以及库仑/粘性摩擦。它仍需新的独立 WP-A holdout；在通过前，状态保持 `KEEP_V6_NOMINAL`。

## 5. 验证门槛

v8 不以训练总 reward 单独判定，至少做以下交叉验证：

1. Isaac play 与 MuJoCo 同一批分层发球，比较长 ETA 球的前伸 onset；`t_hit > 0.70 s` 时不应系统性提前越过 hit-plane 后 8 cm 的准备边界。
2. 对同一 checkpoint 统计 r5/r7 raw/post-tau qdes 的 0.2 s 高通 RMS 和 3--5 Hz 峰值。目标是相对 v7 MuJoCo 基线下降至少 30%，且接触率和拍速质量不退化。
3. 检查 contact、table success、真实落点质量和 paddle/body clearance，确保时序/平滑奖励没有换来“等得太久”“腕部不动”或机身碰撞。
4. 真机先做无球和小样本分层发球，再扩到完整发球范围；继续记录 raw qdes、post-tau qdes、q/dq/effort 和 gate/timing。

## 6. 已完成验证

- 纯逻辑单元测试：window 前动作不计 drawdown、window 开启正确重置基线、接触后统计冻结。
- 配置契约测试：v8 只继承 v7 并增量改奖励，r5/r7 定向权重和独立 task/watchdog 命名锁定。
- 本地 Isaac `2 env x 24 steps x 1 iteration` PPO/predictor 烟测通过；Actor MLP 明确打印 `in_features=195`，Critic 为 320，v8 新奖励项被 RewardManager 正常注册。

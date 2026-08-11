# A1 反手 v9：R4 独立留出响应与 8 Nm MIT 命令边界

日期：2026-08-11

候选任务：`a1_tt_backhand_v9`

实验目录：`a1_tt_backhand_real_v9_r4fit_mit8`

状态：**代码与云端最小烟测通过；正式训练课程已按用户指令改为 30k**

训练课程：`5000 easy + 10000 linear ramp + 15000 full-range hold`，总计
`30000` iterations，最终目标 checkpoint 为零基编号的 `model_29999.pt`。
前 15k 的发球分布与 v8 完全相同，只延长 full-range 巩固阶段，不改变
R4 以外的环境、奖励、观测或执行器语义。

## 1. 单变量边界

v9 完整继承 v8 的 V2/108 cm 资产、ready pose、击球平面、发球分布及
前 15k 课程轨迹、相机、预测器、接触材质、自碰撞/球拍机身安全、奖励和
195 维 actor 接口；训练侧只延长 full-range hold，物理侧只修改 R4 执行链：

1. 将 R4 二阶响应 nominal 更新为独立 robot-clock 留出通过的参数。
2. 将 v8 的 `90 rad/s^2` R4 response-acceleration 代理上限替换为 robot-clock 验证的 MIT torque-observer 8 Nm 命令投影；不是在代理上限后再叠一层限制。
3. 二阶响应仍只在 `TTEnv` 中执行，high-bandwidth implicit tracker 继续保持无限 effort，未切换到 `DamiaoMITActuator` 或给 tracker 直接加 8 Nm。此前回放证明后一种做法会让低负载 probe 也长期顶到 8 Nm，属于重复建模。
4. R1--R3、R5--R7 的响应、加速度上限、执行器和全部 DR 保持 v8。

## 2. 冻结参数与证据

| 项目 | v9 R4 nominal |
|---|---:|
| `fn_hz` | `4.6540465907` |
| `zeta` | `0.1218426553` |
| `delay_s` | `0.0289398865` |
| `gain` | `1.0` |
| `bias_rad` | `+0.0090786988` |
| torque-observer hard boundary | `8.0 Nm` |

投影使用真机固定配置 `Kp=120, Kd=1`，以及独立留出得到的 torque scale=`0.993275`、offset=`-0.08238 Nm`。observer 在独立低负载留出上的 RMSE 为 `0.0463 Nm`。每个 physics tick 使用已经过响应延迟的 R4 q_des 和当前 q/dq 计算需求力矩；需求在 `[-8,8] Nm` 内时命令完全不变，越界时才把 q_des 投影回边界。

独立 0.02 rad / 0.2--2.0 Hz robot-clock 留出中，冻结模型 R4 q RMSE 为 `0.000827 rad`；旧 v6/v8 响应只重定中心后为 `0.002349 rad`，改善 `64.8%`。该低负载留出峰值只有 `1.810 Nm`，所以 8 Nm 边界继续由独立 0.05 rad / 3 Hz 记录（峰值 `8.007 Nm`）约束，不能宣称已经唯一辨识出饱和区动力学。

证据目录：`../系统辨识/joint4/20260811/r4_newpose_mit8_fit_v2_robot_clock_holdout/`。

## 3. 启动前准入

准入评审时云端 v8 继续运行；以下检查通过后，仍需用户单独明确授权才能
停止 v8、启动 v9。该授权已于 2026-08-11 取得，正式切换保留 v8 全部产物：

1. 配置隔离测试确认 v8 对象未被修改、v9 只有 R4 链路变化。
2. 冻结低负载和已有自然挥拍轨迹回放确认 R4 低负载误差没有因 torque projection 显著退化。
3. 高需求回放必须看到 observer 投影后的需求不超过 8 Nm，并统计触边比例/持续时间；implicit tracker 的 `computed/applied_torque` 不是 SDK MIT 力矩，不能拿来判真机饱和。
4. 本地 Isaac 2 env、24 steps、1 iteration PPO/predictor smoke 通过，actor 仍为 195 维。
5. 启训后仍需按 eval、play、MuJoCo 和真机交叉判断，不能只看训练 reward。

## 4. 已完成验证

1. 直接把 high-bandwidth implicit tracker 的 R4 effort 改成 8 Nm 已否决。低负载 probe 中它有 `47.88%` 的 physics target 行落在 8 Nm，computed demand 峰值约 `330 Nm`，而真机同一留出峰值只有 `1.81 Nm`。这是 tracker 为追随“已经包含电机动态的响应目标”产生的假饱和，不是 SDK MIT 力矩。
2. 最终 torque-projection 链在原始 robot-clock 100 Hz 独立留出上，去掉前 0.5 s 初始化后 R4 q RMSE=`0.000818 rad`，与离线冻结模型的 `0.000827 rad` 一致；observer 峰值=`1.861 Nm`、projection 触发 `0` 次。
3. 在旧 v7 真机高需求 12 s 冻结窗口上，raw observer demand P95/peak=`11.71/22.27 Nm`；projection 输出峰值严格为 `8.0 Nm`，输出行触发率=`6.83%`，连续触发 P50/P95/max=`10/137/150 ms`。该窗口 R4 sim-real q RMSE=`0.0752 rad`，没有劣于此前同类新位姿 nominal 动态段约 `0.0793 rad` 的量级。
4. 关闭 projection 的同窗 A/B 在高需求段发生 PhysX CUDA launch failure，没有生成有效结果；该失败只作为“不允许无边界高需求外推”的诊断，不作为定量改善证据，并导致本机 CUDA context 需重启后恢复。
5. 云端 L20 以 `2 env x 24 steps x 1 iteration` 完成 PPO/predictor smoke，退出码 `0`；Actor 明确为 `in_features=195`，Critic 为 `320`，`Metrics/r4_torque_clip_frac` 和 `Metrics/r4_torque_peak_nm` 已进入训练日志。静态/契约测试 `9 passed`。

回放产物：

- `../系统辨识/joint4/20260811/r4_newpose_mit8_fit_v2_robot_clock_holdout/v9_isaac_response_torque_projection_holdout.csv`
- `../系统辨识/joint4/20260811/r4_newpose_mit8_fit_v2_robot_clock_holdout/v9_isaac_response_torque_projection_v7_highdemand.csv`

代码提交：`68bcae81377ff337084cb802226afbd2d82ed382`（后续文档修订提交见 Git 历史）。

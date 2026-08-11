# A1 反手 v9：R4 独立留出响应与 8 Nm 物理边界

日期：2026-08-11

候选任务：`a1_tt_backhand_v9`

实验目录：`a1_tt_backhand_real_v9_r4fit_mit8`

状态：**本地候选，尚未替换或停止云端 v8**

## 1. 单变量边界

v9 完整继承 v8 的 V2/108 cm 资产、ready pose、击球平面、发球课程、相机、预测器、接触材质、自碰撞/球拍机身安全、奖励和 195 维 actor 接口。只修改 R4 执行链：

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

云端 v8 继续运行。v9 只有在以下本地检查通过后才具备候选资格；即使通过，停止 v8、启动 v9 仍需用户单独明确授权：

1. 配置隔离测试确认 v8 对象未被修改、v9 只有 R4 链路变化。
2. 冻结低负载和已有自然挥拍轨迹回放确认 R4 低负载误差没有因 effort cap 显著退化。
3. 高需求回放必须看到 observer 投影后的需求不超过 8 Nm，并统计触边比例/持续时间；implicit tracker 的 `computed/applied_torque` 不是 SDK MIT 力矩，不能拿来判真机饱和。
4. 本地 Isaac 2 env、24 steps、1 iteration PPO/predictor smoke 通过，actor 仍为 195 维。
5. 启训后仍需按 eval、play、MuJoCo 和真机交叉判断，不能只看训练 reward。

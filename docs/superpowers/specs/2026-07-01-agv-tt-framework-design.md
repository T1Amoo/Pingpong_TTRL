# A1 AGV 机器人乒乓训练框架 —— 设计文档

日期：2026-07-01 · 分支：`a1-tt-migration`（从 `g1-tt-deploy` 切出）

## 0. 目标与范围

把自研 AGV 底盘机器人 **A1** 接入 Pingpong_TTRL，搭起训练框架并**当天跑通**（4096 env、易球课程、日志/critic 正常不发散即算跑通）。

范围（本版）：
- **只训右臂 7 DoF**（`joint_yb_1..7`）。
- 底盘、腰部升降、左臂、头 **不训练**。
- 底盘 = **自由重刚体**（见 §3），非刚性焊死。
- home 点 x = **-1.8 m**。
- 选型、去抖、真机部署 = 后续版本，不在本版。

硬件来源文件：`/home/woan/下载/a1.usd`、`/home/woan/下载/A1_fullBody`（URDF+mesh）、`/home/woan/下载/a1.py`（`A1_TABLE_TENNIS_CFG`）。

## 1. 资产落位

- 新建 `legged_lab/assets/a1/`：
  - `a1.usd`（拷自 `/home/woan/下载/a1.usd`，28.8 MB）。
  - `a1.py`：`A1_TT_CFG = ArticulationCfg(...)`，基于硬件 `A1_TABLE_TENNIS_CFG` 改（路径、帧、关节、actuator）。
- 复用现成 `BALL_CFG`（`legged_lab/assets/table_tennis/`）与 `table.py`（桌面几何 2.74×1.525×0.76）。

**缺项核查（原始"是否还缺"）**：
1. 拍面 `Link_yb_paddle` 的 **collider + restitution** 必须在 USD 里存在（G1 靠拍碰撞体 + e=0.75 打球）。首启在 sim 里验证；若缺则在 USD/config 补 collider 与 restitution。
2. 球、桌复用现成资产，无缺。
3. `data/robots/a1/a1.usd` 这类相对路径需改为 TTRL 资产路径。

## 2. 坐标系（M → W → R）

- **M 动捕**：乒乓桌**桌面中心**，**+x 指向对手半场**（已确认）。
- **W 训练/世界**：桌面中心**地面**，+z 上、+x 对手、机器人在 -x 侧朝 +x，桌面 z=0.76。
- **M → W**：轴对齐、无旋转，`p_W = p_M + (0, 0, 0.76)`。
- **R base_link**：a1.**urdf** 读出是 z-up（轮子在 +z 0.035、地面在下方），但实际 **a1.usd 的帧与 urdf 不一致**（用户告知：轮子有高度、-x 朝地）。**以 USD 为准**：首启 spawn 后在 sim 里实测 base 朝向 + 轮子触地偏移，据此设：
  - `init_state.rot`：摆正机器人 + 使右臂/拍朝 +x_W。
  - `init_state.pos.z`：加轮子高度偏移，使轮子触地 z=0。
  - `init_state.pos.x = -1.8`，`y = 0`。
  - ⚠️ 硬件 a1.py 原 `pos=(1.7,0,0)` 是他们场景值，本项目改 -1.8。

## 3. 机器人 config（`A1_TT_CFG`）

**底盘（B 方案，自由重刚体）**：
- 去掉 `fix_root_link=True` → 底盘作真实质量/惯量的自由刚体。
- 轮子 `joint`：锁死（stiffness 1e4、velocity~0、不驱动）→ 底盘大致原地，但挥臂反作用力可晃/倾它 → 重力投影与摇晃惩罚真实起效，建模真机 AGV 晃动。

**关节训练/固定**：
- **训练**：`joint_yb_1..7`（右臂 7 DoF），`action_scale=0.25`，`preserve_order=True`。
- **固定**（不入训练表 → target 恒为 init、刚度保持）：左臂 `joint_zb_*`、头 `joint_head_*`、`joint_lift`、轮子。

**升降 lift 高度（按 G1 先验）**：
- 调 `joint_lift`（axis +z、travel [-0.8,-0.05]）使**拍 ready 世界高度 ≈ G1 ready（z≈1.0–1.1）**。
- 起点 = a1.py 的 init `joint_lift=-0.28`；首启用 FK/sim 核对拍高后定值。

**ready 位姿**：沿用 a1.py "whip_high3"：
`joint_yb_1..7 = 1.769, -0.762, -1.863, 1.445, 0.206, -0.827, 1.043`。

**右臂 actuator（用户实测电机规格，DAMIAO wiki）**：
- 依据：`ω_n = 2π·10 = 62.83 rad/s`；`kp = armature·ω_n²`；`kd ≈ 2·2·armature·ω_n`（ζ≈2 过阻尼）；`armature = 转子惯量 × 减速比²`。

| 关节 | 电机 | 减速比 | armature | stiffness(kp) | damping(kd) | effort |
|---|---|---|---|---|---|---|
| joint_yb_1/2/3 | 4340(48V) | 40 | **0.032** | **126.33094** | **8.042478** | 28 |
| joint_yb_4/5/6/7 | 4310(48V) | 10 | **0.0018** | **7.106115** | **0.452389** | 8 |

- velocity limit 沿用 a1.py：{8,8,8,20,20,20,20}。
- armature 需加进 `ImplicitActuatorCfg`（原 a1.py 未含）。

## 4. 观测（精简 + 重力 + base 角速度）

每帧 **38 维**，历史 5 帧 → 190：

| 分量 | 维度 | 说明 |
|---|---|---|
| 右臂 joint_pos | 7 | 训练关节 |
| 右臂 joint_vel | 7 | |
| last_action | 7 | |
| projected_gravity | 3 | 感知 base 倾斜（底盘会晃） |
| base_ang_vel | 3 | 感知晃动速率，配合摇晃惩罚 |
| ball perception | 6 | 球 pos+vel（动捕语义） |
| predictor 输出 | 3 | 学习式弹道预测器（ball-only，复用 G1） |
| rel_target | 2 | 相对目标（沿用 TTEnv） |

- critic obs 保留 root_lin_vel/ang_vel 通道（底盘自由，正好有意义）。

## 5. 奖励

- **保留**：ready-pose 正则化；击球 / 落台 / predictor 跟踪等打球核心项（沿用 TTEnv）。
- **剥离**：腿/脚/步态 / base 移动类 locomotion 项（A1 无脚）。
- **新增 —— 基体摇晃惩罚**：`base 角速度 L2 + base 倾角（projected_gravity 偏离竖直）`。**不含 base 线加速度**（用户指定）。权重先给小值，防一上来压死挥拍，后续调。
- `termination_penalty`：沿用 G1 修复后的 **-100**（避免 critic value_loss 被巨值炸，见 g1 critic 发散经验）。

## 6. 环境架构（Approach 1：config 复用 + 薄子类）

- 复用 `legged_lab/envs/base/tt_env.py` 的 `TTEnv`。
- 写薄子类 **`A1TTEnv`**（`legged_lab/envs/a1_tt/`）：中和 `tt_env.py` 里硬编码的 feet/contact 依赖——
  - `feet_body_names` / `termination_contact_body_names` / feet contact sensor：A1 有轮无脚 → 置空或改为轮子/底盘 body，摔倒判据改用 base 倾角/高度阈值。
  - critic obs 里 `feet_contact` 通道：移除或置零。
  - 保留 root_lin/ang_vel 通道（底盘自由）。
- config：`A1TableTennisEnvCfg(TTEnvCfg)`（`legged_lab/envs/a1_tt/a1_tt_config.py`）：
  - `self.robot.hit_plane_x = -1.8`。
  - 导入 `A1_TT_CFG`、`BALL_CFG`、`table`。
  - actions/observations 的 `joint_names` = 右臂 7 关节。
  - 注册训练/eval 任务名（`a1_tt` 等），参照 g1_tt 注册方式。

## 7. 发球 / home

- home = x **-1.8**。
- **直接复用 G1 v11+ 已调好的 -1.8 发球几何**（bounce_serve，bounce_x/vz 使球到 -1.8 时 z≈1.0=拍 ready 高度），省重调。
- 课程：先易球单段跑通；速度/难度课程后续。

## 8. 今天跑通判据

- 4096 env 起训不崩。
- 日志 reward 正常增长、**critic value_function loss 稳定 <100**（不发散）。
- sim 里机器人摆正、拍在 ready 高度、球能发到 -1.8。
- 选型/去抖/部署 = 后续。

## 9. 风险与首启验证清单

1. **USD 帧朝向**：spawn 后确认 base 摆正、臂朝 +x、轮子触地（据实测调 rot/z）。
2. **拍 collider/restitution**：确认存在，否则补。
3. **底盘自由 + 轮锁**：确认底盘不乱漂、不翻，挥臂能观察到小幅晃动（摇晃惩罚有信号）。
4. **lift 高度**：FK 核对拍 ready 高度 ≈ G1。
5. **actuator armature/kp/kd**：确认 4340/4310 分组正确、无数值发散。
6. **TTEnv feet/contact 中和**：确认无因缺 feet body 报错。

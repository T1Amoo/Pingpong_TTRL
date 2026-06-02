# Booster T1 → Unitree G1 23DoF 乒乓训练框架迁移设计

- 日期: 2026-06-02
- 状态: 设计已与用户逐节确认，待 spec 审阅 → writing-plans
- 范围: 仅"训练框架"。相机 SDK 接入、机器人本机部署留到训练跑通后另议。

## 1. 背景与目标

`Pingpong_TTRL`（ICRA2026 PACE，基于 LeggedLab + 定制 RSL-RL）原生支持 **Booster T1（本仓库实际为 21 DoF 动作集）** 的端到端乒乓 RL（预测增强 PPO）。
目标：在保留原 T1 任务作对照基线的前提下，新增一套 **Unitree G1 23DoF** 的乒乓训练任务 `g1_tt`，能加载、站立、发球、物理击球、跑通训练（含 predictor），为后续真机（动捕 + 本机部署）打基础。

G1 资产来源：`lgy/whole_body_tracking`（同机工作区）已总结好的 G1 描述与可信配置。

## 2. 关键决策（已与用户确认）

- **A. 代码组织**：新建并行 `g1_tt` 任务，T1 完整保留（surgical，不污染原项目）。
- **A. 资产管线**：改 G1 URDF 加固定球拍 link，沿用 `whole_body_tracking` 的 `UrdfFileCfg` spawn（纯文本可 diff；同一份参数几何既喂仿真又能导出 STL）。
- **球拍挂点**：`right_wrist_roll_rubber_hand`（换掉橡胶手），**保留全部 23 DoF**（含 wrist_roll）。不采用论文 T1 的"挂肘、锁腕"做法（那会变 22 DoF，与需求矛盾），但物理形式仍仿照 T1（3D 打印管 + 摩擦套筒 + 真乒乓拍）。
- **击球触点用"偏移法"**：不新增独立 articulation body；`paddle_touch_point = wrist_pos + R·offset`，最稳、最像 T1。
- **打印件只交付结构尺寸**：管长/外径/套筒内径/拍面直径/相对腕部偏移与朝向。与 G1 腕部法兰的机械接口由硬件团队负责，结构件留空白安装面参数。
- **拍面仿真尺寸 ~150mm**（略小于真拍 ~160-165mm）：安全方向——真机拍更大只会更易击中，真机表现 ≥ 仿真。

## 3. 设计详述

### ① G1 机器人配置（`g1_tt` 用）
- 复用 `whole_body_tracking/robots/g1.py` 的 `G1_CYLINDER_CFG`（URDF spawn + 真实电机 armature/刚度模型；legs/feet/waist_yaw/arms 四组 `ImplicitActuatorCfg`）。
- 在 `Pingpong_TTRL` 新建 `legged_lab/assets/unitree/g1.py` 落 TT 专用 cfg，三处改动：
  1. URDF 换成带球拍副本 `g1_23dof_tt.urdf`（见 ②）。
  2. 初始姿态改成乒乓预备站姿（右臂抬肘正手预备，左臂收贴），`pos=(-1.6, 0.0, 0.76)`。
  3. 执行器用 g1.py 的 implicit 模型；`action_scale` 先全局 0.25，列入标定项。
- 取舍：T1 的 `DelayedPDActuatorCfg`（0~3 步执行延迟）暂不引入，留到 sim2real 阶段。

### ② 球拍适配件（仿真件 + 打印件，同一份参数几何）
- 挂点 `right_wrist_roll_rubber_hand`，加 fixed-joint 子 link `right_paddle`。
- 参数化几何（初值，后续 FK 标定）：
  - 管(tube)：圆柱外径 ~32mm，长 ~80mm，沿前臂轴延伸
  - 拍面(blade)：薄圆盘直径 ~150mm，厚 ~6mm（碰撞用），法线朝击球方向
  - 套筒(sleeve)：管内摩擦配合槽（仅打印件需要，仿真用实心管近似）
  - 腕→拍面中心偏移：~0.20m（替代 T1 基于肘的 0.345m，实际值 FK 标定）
- 仿真要点：拍面+管给 collision+visual 几何，物理材质 **restitution=0.8**（球靠物理碰撞弹回，对齐原 T1）。
- 打印件交付：同套参数后续用脚本导出 STL（管+套筒+卡口+空白安装面），硬件团队 mate 到 G1 腕部法兰。
- 限制说明：精确打印 CAD 需 G1 腕部机械接口尺寸（孔距/法兰），URDF 无此信息；本设计提供参数化结构件近似。

### ③ 关节映射与维度
- `num_actions / num_joints`：21 → **23**。
- obs/action `joint_names` 整表替换为 G1 命名：
  - 腿(12): `{left,right}_{hip_pitch,hip_roll,hip_yaw,knee,ankle_pitch,ankle_roll}_joint`
  - 腰(1): `waist_yaw_joint`
  - 臂(10): `{left,right}_{shoulder_pitch,shoulder_roll,shoulder_yaw,elbow,wrist_roll}_joint`
- `joint_deviation_*` 按手性重映射：左臂收紧(~-0.2)保持贴身；右臂(含 wrist_roll)放松(~-0.05)允许甩拍；髋 yaw/roll -0.2；`waist_yaw` -0.2（对应原 torso 项）。
- 用关节名/ body 名查索引，去掉 `paddle_index=15` 这类魔法下标。

### ④ 奖励/几何重标定清单
**A. tt_env.py 与 T1 形态绑定的硬编码：**
| 位置 | T1 值 | G1 改法 |
|---|---|---|
| `paddle_index=15` (`:846`) | right_hand_link | 按名查 `right_wrist_roll_rubber_hand` body index |
| `local_offset=[0,-0.345,0]` (`:857`) | 肘基 0.345m | 腕基新偏移(~0.20m)+正确本地轴，FK 标定 |
| `body_height=0.69` (`:974`) | T1 击球站高 | G1 站姿身体目标高，重算(~0.70-0.73) |
| `paddle_y_offset=-0.60` (`:976`) | T1 臂展横偏 | 按 G1 预备站姿"拍面−基座"横偏重算 |
| 复位边界 `robot.z<0.50` (`:814`) | T1 摔倒阈值 | G1 pelvis 0.76 → 阈值 ~0.5-0.55 复核 |

**B. 不改（球桌/球几何，与机器人无关）**：`mask_terminal`/`mask_invalid` 球 x 阈值、`reward_future_landing_dis` 落点目标 (1.15,0)、`reward_future_pass_net` 过网高 0.76+0.35、发球速度范围。原则：奖励里只有"机器人/拍"相关几何要改，"球/桌"相关不动。

**C. config body 名引用**：`Trunk`→`torso_link`（终止接触 + add_base_mass + height_scanner）；`.*_foot_link`→`.*_ankle_roll_link`（足）。

**D. 标定方法**：写单环境 FK 探针脚本——加载 G1+球拍、摆预备站姿，读 `right_wrist_roll` 位姿与拍面触点，量"拍面−基座" (x,y,z) 偏移，回填 `body_height`/`paddle_y_offset`/`local_offset`，使预备站姿拍面落在合理正手区（几何关系对齐 T1）。

### ⑤ 场景 / 站位 / 终止
- 球桌/球/可视化球（绿黄蓝）：资产不变，复用。
- G1 站位：与 T1 同相对球桌位置 `x≈-1.6`，仅站高 `z:0.72→0.76`，发球范围不变。
- 预备站姿：右臂正手预备，左臂收贴。
- 两级终止保留：球级软复位（落地/1.5s 超时只重发球）+ episode 级（摔倒/越界/发够球数/超时）。摔倒沿用几何阈值 `pelvis.z<~0.5`（接触力终止保持注释，不引入）。
- 自碰撞：先 `enabled_self_collisions=False`（与 T1 TT cfg 一致，更稳更快），可回调项。
- height scanner：保持关闭（平地）。

### ⑥ 验证标准（分级 goal-driven）
| # | 检查项 | 通过判据 |
|---|---|---|
| 1 | 资产加载 | G1+球拍 URDF spawn 成功，关节数==23，拍面 body 带碰撞几何 |
| 2 | 站立 | 预备站姿零/默认动作，pelvis.z 保持 >0.6 持续 2s 不倒 |
| 3 | FK 标定 | 探针输出"拍面−基座"偏移，回填三常量，拍面落正手区 |
| 4 | 球能弹回 | 球射向拍面物理弹回（restitution 生效） |
| 5 | 训练跑通 | `train.py --task=g1_tt --num_envs=64 --headless` ≥100 iter 无 NaN/崩溃，reward 有限 |
| 6 | predictor 兼容 | 加 `--predictor` 挂上，`predictor_mse` 有日志，不崩 |
| 7 | 学习信号 | 数百 iter 内 `reward_contact`/`TT_hit_rate` 出现非零上升 |

**"框架搭好" = 1-6 全过 + 7 非零击球信号**。完全收敛（论文 96% 命中）属后续训练调参，不计入本次。

## 4. 文件改动清单（预估）

**新增：**
- `legged_lab/assets/unitree/g1.py` — G1 TT ArticulationCfg（移植 + TT 站姿）
- `legged_lab/assets/unitree/g1_description/g1_23dof_tt.urdf` — G1 23dof + 球拍 fixed link（从 wbt 副本改）
- `legged_lab/envs/g1_tt/g1_tt_config.py` — G1 任务/奖励/agent cfg（仿 t1_tt）
- `legged_lab/envs/g1_tt/__init__.py` — 注册 `g1_tt` / `g1_tt_eval` 任务
- `legged_lab/scripts/fk_probe.py`（或临时脚本）— FK 标定探针
- （后续）STL 导出脚本 — 球拍结构件

**修改：**
- 任务注册入口（gym register 处）增加 `g1_tt`
- `tt_env.py`：去硬编码（`paddle_index`/`local_offset`/`body_height`/`paddle_y_offset` 改为可配置/按名解析），保持 T1 行为不变（默认值回落 T1）。优先用"参数化 + 配置项"而非分支 if，避免污染。

> 注：`tt_env.py` 已 1188 行、承担过多职责；本次仅做"把 T1 写死的形态常量外提为配置"的针对性改造，不做无关重构。

## 5. 延后事项（不在本次范围）
- 相机/动捕（Nokov Mars 1.3H，9 相机/240fps，已初判满足）SDK 接入。
- 机器人本机部署回路（Nokov → perception 向量 → 50Hz 推理 → G1 SDK 下发；本体感知走 IMU+编码器）。
- 球拍结构件精确打印 CAD（需 G1 腕部法兰实测尺寸）。
- 旋转球（Magnus）建模——原项目仅建阻力（drag=0.4378, magnus=0），保持不变。
- 正手以外的击球、完全收敛调参。

# G1 23DoF 乒乓训练框架迁移 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Pingpong_TTRL 中新增一套可加载、可站立、可物理击球、可跑通训练（含 predictor）的 Unitree G1 23DoF 乒乓任务 `g1_tt`，T1 任务完整保留作基线。

**Architecture:** 新建并行 `g1_tt` 任务；复用 whole_body_tracking 的 G1 配置 + URDF（vendoring 进本仓库），在右腕 `right_wrist_roll_rubber_hand` 上加固定球拍几何（偏移法击球触点）；把 `tt_env.py` 中与 T1 形态绑定的硬编码常量外提为 cfg 字段（默认值=T1 当前值，保证 T1 行为不变），G1 cfg 覆盖之。

**Tech Stack:** IsaacSim 4.5.0 + IsaacLab 2.1.1 + LeggedLab + 定制 RSL-RL；conda 环境 `pingpong`；Python 3.10。

**关于"测试"：** 本项目是 IsaacSim 物理仿真 RL，无快速单元测试框架。本计划以**可运行的验证探针脚本 + 预期输出**作为 TDD 的等价物：每个任务"先写检查→运行看失败→实现→运行看通过→提交"。所有 python 命令用 `/home/woan/.conda/envs/pingpong/bin/python` 执行；仓库根目录 `/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL`。已在分支 `g1-tt-migration` 上。

---

## File Structure

**新增：**
- `legged_lab/assets/unitree/__init__.py` — 空包标记
- `legged_lab/assets/unitree/g1_description/` — vendoring 的 G1 URDF + meshes（从 wbt 拷贝）
- `legged_lab/assets/unitree/g1_description/g1_23dof_tt.urdf` — G1 23dof + 球拍固定件
- `legged_lab/assets/unitree/g1.py` — `G1_TT_CFG`（ArticulationCfg，TT 站姿）
- `legged_lab/envs/g1_tt/__init__.py` — 空
- `legged_lab/envs/g1_tt/g1_tt_config.py` — `G1TableTennisEnvCfg / G1TableTennisAgentCfg / G1TT_EvalEnvCfg`
- `legged_lab/scripts/fk_probe.py` — FK 标定/站立/弹球探针

**修改：**
- `legged_lab/envs/base/tt_config.py` — `RobotCfg` 增加 5 个球拍/几何字段（带 T1 默认值）
- `legged_lab/envs/base/tt_env.py` — 4 处硬编码改读 `self.cfg.robot.*`（默认值不变）
- `legged_lab/envs/__init__.py` — 注册 `g1_tt` / `g1_tt_eval`

---

## Task 1: 外提 tt_env 硬编码几何常量（先做，保证 T1 不变）

先把形态相关常量外提为 cfg 字段、默认值=T1 现值，这样 G1 cfg 只需覆盖。先验证 T1 回归不变。

**Files:**
- Modify: `legged_lab/envs/base/tt_config.py`（`RobotCfg`）
- Modify: `legged_lab/envs/base/tt_env.py:846-866`（paddle touch）、`:974-976`（body_height/paddle_y_offset）

- [ ] **Step 1: 在 `RobotCfg` 增加字段（默认=T1 值）**

打开 `legged_lab/envs/base/tt_config.py`，在 `class RobotCfg:` 内（`effort_limit_scale` 之后）追加：

```python
    # --- Table-tennis paddle / hitting geometry (defaults match Booster T1) ---
    paddle_body_name: str = "right_hand_link"   # body the paddle is rigidly attached to
    paddle_offset: tuple = (0.0, -0.345, 0.0)   # paddle face center offset in that body's local frame
    hit_body_height: float = 0.69               # target body height for robot_future_pos
    paddle_y_offset: float = -0.60              # lateral base->paddle offset in ready stance
    robot_vel_max: float = 7.0                  # clamp for robot_future_vel target
```

- [ ] **Step 2: 改 `compute_paddle_touch` 读 cfg（tt_env.py:846, 857）**

把 `legged_lab/envs/base/tt_env.py` 中：
```python
        paddle_index = 15  # paddle belongs to 'right_hand_link'
        paddle_pos = self.robot.data.body_pos_w[:, paddle_index, :]
```
改为：
```python
        paddle_index = self._paddle_body_id  # resolved once in __init__ from cfg
        paddle_pos = self.robot.data.body_pos_w[:, paddle_index, :]
```
并把：
```python
        local_offset = (
            torch.tensor(
                [0.0, -0.345, 0.0], # Good to double check.
                device=paddle_pos.device,
                dtype=paddle_pos.dtype,
            )
            .unsqueeze(0)
            .expand_as(paddle_pos)
        )
```
改为：
```python
        local_offset = (
            torch.tensor(
                self.cfg.robot.paddle_offset,
                device=paddle_pos.device,
                dtype=paddle_pos.dtype,
            )
            .unsqueeze(0)
            .expand_as(paddle_pos)
        )
```

- [ ] **Step 3: 在 __init__ 解析 paddle body id（tt_env.py ~283 之后）**

在 `self.obs_joint_ids, self.obs_joint_names = self.robot.find_joints(...)` 之后追加：
```python
        _paddle_ids, _ = self.robot.find_bodies(self.cfg.robot.paddle_body_name)
        assert len(_paddle_ids) == 1, f"paddle_body_name resolved to {len(_paddle_ids)} bodies"
        self._paddle_body_id = _paddle_ids[0]
```

- [ ] **Step 4: 改 body_height/paddle_y_offset/vel_max 读 cfg（tt_env.py:974-976）**

把：
```python
        body_height=0.69
        vel_max=7.0
        paddle_y_offset = -0.60
```
改为：
```python
        body_height = self.cfg.robot.hit_body_height
        vel_max = self.cfg.robot.robot_vel_max
        paddle_y_offset = self.cfg.robot.paddle_y_offset
```

- [ ] **Step 5: 运行 T1 回归检查——确认行为不变**

Run:
```bash
/home/woan/.conda/envs/pingpong/bin/python -c "
from legged_lab.envs.base.tt_config import RobotCfg
c = RobotCfg()
assert c.paddle_body_name == 'right_hand_link'
assert c.paddle_offset == (0.0, -0.345, 0.0)
assert c.hit_body_height == 0.69 and c.paddle_y_offset == -0.60 and c.robot_vel_max == 7.0
print('RobotCfg defaults OK')
"
```
Expected: `RobotCfg defaults OK`（不启 sim，纯导入校验默认值与 T1 一致）

- [ ] **Step 6: Commit**

```bash
git add legged_lab/envs/base/tt_config.py legged_lab/envs/base/tt_env.py
git commit -m "refactor(tt): externalize T1 paddle/hit geometry constants into RobotCfg (defaults unchanged)"
```

---

## Task 2: Vendoring G1 描述文件 + 生成带球拍的 URDF

**Files:**
- Create: `legged_lab/assets/unitree/__init__.py`
- Create: `legged_lab/assets/unitree/g1_description/`（拷贝）
- Create: `legged_lab/assets/unitree/g1_description/g1_23dof_tt.urdf`

- [ ] **Step 1: 拷贝 G1 描述（urdf + meshes）进本仓库**

Run:
```bash
SRC=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/g1_description
DST=legged_lab/assets/unitree/g1_description
mkdir -p legged_lab/assets/unitree "$DST"
touch legged_lab/assets/unitree/__init__.py
cp "$SRC/g1_23dof_rev_1_0.urdf" "$DST/"
cp -r "$SRC/meshes" "$DST/"
ls "$DST" && ls "$DST/meshes" | wc -l
```
Expected: 列出 `g1_23dof_rev_1_0.urdf` 与 `meshes`，mesh 文件数 > 30。

- [ ] **Step 2: 生成带球拍的 URDF（脚本式编辑右腕 link）**

创建 `g1_23dof_tt.urdf`：复制基础 urdf，然后在 `right_wrist_roll_rubber_hand` link 内**替换 `<inertial>`** 并**追加球拍 collision+visual**。先复制：
```bash
cp legged_lab/assets/unitree/g1_description/g1_23dof_rev_1_0.urdf \
   legged_lab/assets/unitree/g1_description/g1_23dof_tt.urdf
```

手工编辑 `g1_23dof_tt.urdf`，找到 `<link name="right_wrist_roll_rubber_hand">`，把其 `<inertial>` 块整体替换为（适配件+真拍，杆+盘模型，质量 0.25kg，质心沿 +X 0.12m）：
```xml
    <inertial>
      <origin xyz="0.12 0.0 0.0" rpy="0 0 0"/>
      <mass value="0.25"/>
      <inertia ixx="0.00035" ixy="0.0" ixz="0.0" iyy="0.0016" iyz="0.0" izz="0.0016"/>
    </inertial>
```
然后在该 link 的 `</link>` 之前追加球拍管（圆柱，沿 +X）与拍面（薄圆盘，法线沿 +X，置于 0.20m 处）的 collision+visual：
```xml
    <!-- TT paddle adapter tube (PLA), extends along +X from wrist -->
    <visual>
      <origin xyz="0.06 0 0" rpy="0 1.5708 0"/>
      <geometry><cylinder radius="0.016" length="0.12"/></geometry>
    </visual>
    <collision>
      <origin xyz="0.06 0 0" rpy="0 1.5708 0"/>
      <geometry><cylinder radius="0.016" length="0.12"/></geometry>
    </collision>
    <!-- TT paddle blade: thin disk, face normal along +X, at 0.20m -->
    <visual>
      <origin xyz="0.20 0 0" rpy="0 1.5708 0"/>
      <geometry><cylinder radius="0.075" length="0.006"/></geometry>
    </visual>
    <collision>
      <origin xyz="0.20 0 0" rpy="0 1.5708 0"/>
      <geometry><cylinder radius="0.075" length="0.006"/></geometry>
    </collision>
```
（`rpy="0 1.5708 0"` 把圆柱默认 +Z 轴转到 +X；拍面半径 0.075m=直径 150mm；管长 0.12m 居中于 0.06m。）

- [ ] **Step 3: 验证 URDF 良构 + 球拍 link 含碰撞**

Run:
```bash
/home/woan/.conda/envs/pingpong/bin/python -c "
import xml.etree.ElementTree as ET
r = ET.parse('legged_lab/assets/unitree/g1_description/g1_23dof_tt.urdf').getroot()
joints=[j.get('name') for j in r.findall('joint') if j.get('type') in ('revolute','continuous')]
assert len(joints)==23, f'expected 23 joints, got {len(joints)}'
hand=[l for l in r.findall('link') if l.get('name')=='right_wrist_roll_rubber_hand'][0]
ncol=len(hand.findall('collision'))
m=float(hand.find('inertial/mass').get('value'))
print('joints',len(joints),'hand collisions',ncol,'hand mass',m)
assert ncol>=2 and abs(m-0.25)<1e-6
print('URDF OK')
"
```
Expected: `joints 23 hand collisions ... hand mass 0.25` + `URDF OK`

- [ ] **Step 4: Commit**

```bash
git add legged_lab/assets/unitree/
git commit -m "feat(assets): vendor G1 23dof description and add paddle-augmented URDF"
```

---

## Task 3: G1 TT ArticulationCfg

**Files:**
- Create: `legged_lab/assets/unitree/g1.py`

- [ ] **Step 1: 写 `G1_TT_CFG`（移植 wbt g1.py + TT 站姿）**

创建 `legged_lab/assets/unitree/g1.py`，内容（actuator 增益沿用 wbt 的 armature/自然频率模型；URDF 指向带球拍件；self_collisions=False；TT 预备站姿右臂抬起）：

```python
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ

G1_TT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,  # keep paddle disk as a true cylinder
        asset_path=f"{ISAAC_ASSET_DIR}/unitree/g1_description/g1_23dof_tt.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False,
            linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8, solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.6, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            "waist_yaw_joint": 0.0,
            # left arm tucked
            "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.6, "left_wrist_roll_joint": 0.0,
            # right arm forehand-ready
            "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2,
            "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.6, "right_wrist_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim={".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0,
                              ".*_hip_pitch_joint": 88.0, ".*_knee_joint": 139.0},
            velocity_limit_sim={".*_hip_yaw_joint": 32.0, ".*_hip_roll_joint": 20.0,
                                ".*_hip_pitch_joint": 32.0, ".*_knee_joint": 20.0},
            stiffness={".*_hip_pitch_joint": STIFFNESS_7520_14, ".*_hip_roll_joint": STIFFNESS_7520_22,
                       ".*_hip_yaw_joint": STIFFNESS_7520_14, ".*_knee_joint": STIFFNESS_7520_22},
            damping={".*_hip_pitch_joint": DAMPING_7520_14, ".*_hip_roll_joint": DAMPING_7520_22,
                     ".*_hip_yaw_joint": DAMPING_7520_14, ".*_knee_joint": DAMPING_7520_22},
            armature={".*_hip_pitch_joint": ARMATURE_7520_14, ".*_hip_roll_joint": ARMATURE_7520_22,
                      ".*_hip_yaw_joint": ARMATURE_7520_14, ".*_knee_joint": ARMATURE_7520_22},
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0, velocity_limit_sim=37.0,
            stiffness=2.0 * STIFFNESS_5020, damping=2.0 * DAMPING_5020, armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=139, velocity_limit_sim=20.0,
            stiffness=STIFFNESS_7520_22, damping=DAMPING_7520_22, armature=ARMATURE_7520_22,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                              ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=25.0, velocity_limit_sim=37.0,
            stiffness=STIFFNESS_5020, damping=DAMPING_5020, armature=ARMATURE_5020,
        ),
    },
)
```

- [ ] **Step 2: 导入校验（不启 sim）**

Run:
```bash
/home/woan/.conda/envs/pingpong/bin/python -c "
from legged_lab.assets.unitree.g1 import G1_TT_CFG
print('actuator groups:', list(G1_TT_CFG.actuators.keys()))
print('urdf:', G1_TT_CFG.spawn.asset_path.split('/')[-1])
print('G1_TT_CFG OK')
"
```
Expected: `actuator groups: ['legs', 'feet', 'waist_yaw', 'arms']` / `urdf: g1_23dof_tt.urdf` / `G1_TT_CFG OK`

- [ ] **Step 3: Commit**

```bash
git add legged_lab/assets/unitree/g1.py
git commit -m "feat(assets): add G1_TT ArticulationCfg (URDF spawn, TT ready stance)"
```

---

## Task 4: G1 任务配置 + 注册

**Files:**
- Create: `legged_lab/envs/g1_tt/__init__.py`（空）
- Create: `legged_lab/envs/g1_tt/g1_tt_config.py`
- Modify: `legged_lab/envs/__init__.py`

- [ ] **Step 1: 写 `g1_tt_config.py`（仿 t1_tt，改关节名表/body 名/几何/球拍 cfg）**

创建 `legged_lab/envs/g1_tt/__init__.py`（空文件）与 `legged_lab/envs/g1_tt/g1_tt_config.py`。后者继承 T1 的 reward/env 结构，仅覆盖机器人相关项。先读 `legged_lab/envs/t1_tt/t1_tt_config.py` 全文作为模板，复制其 `T1TableTennisRewardCfg / T1TableTennisEnvCfg / T1TT_EvalEnvCfg / T1TableTennisAgentCfg` 四个类为 `G1*`，并按下表改动：

```python
# 在 __post_init__ 中：
self.scene.robot = G1_TT_CFG                       # 来自 legged_lab.assets.unitree.g1
self.scene.height_scanner.prim_body_name = "torso_link"
self.robot.terminate_contacts_body_names = ["torso_link"]
self.robot.feet_body_names = [".*_ankle_roll_link"]
self.robot.num_actions = 23
self.robot.num_joints = 23
self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]

# 球拍/击球几何（覆盖 RobotCfg 默认；数值在 Task 6 FK 标定后回填）：
self.robot.paddle_body_name = "right_wrist_roll_rubber_hand"
self.robot.paddle_offset = (0.20, 0.0, 0.0)        # +X，FK 标定后更新
self.robot.hit_body_height = 0.72                  # FK 标定后更新
self.robot.paddle_y_offset = -0.30                 # FK 标定后更新

# 23 个关节名（obs 与 action 同序）：
G1_JOINT_NAMES = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
    "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint",
    "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint",
]
self.observations.joint_names = G1_JOINT_NAMES
self.actions.joint_names = G1_JOINT_NAMES
```

`joint_deviation_*` reward 项按手性重映射（在 `G1TableTennisRewardCfg` 内，把 Booster 关节名换成 G1，权重沿用 T1 的左紧右松）：
```python
joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])})
joint_deviation_left_arm = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_.*", "left_elbow_joint", "left_wrist_roll_joint"])})
joint_deviation_right_arm = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_shoulder_.*", "right_elbow_joint", "right_wrist_roll_joint"])})
joint_deviation_torso = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])})
```
> 注意：删除 T1 原有的 `joint_deviation_left_shoulder_roll` 等 Booster 专属项；`AgentCfg.experiment_name` 改为 `"g1_table_tennis"`。`T1TT_EvalEnvCfg` 的发球范围/复位 pose_range 原样保留（球桌不变）。

- [ ] **Step 2: 注册任务（envs/__init__.py）**

在 `legged_lab/envs/__init__.py` 末尾追加：
```python
from legged_lab.envs.g1_tt.g1_tt_config import (
    G1TableTennisEnvCfg,
    G1TableTennisAgentCfg,
    G1TT_EvalEnvCfg,
)
task_registry.register("g1_tt", TTEnv, G1TableTennisEnvCfg(), G1TableTennisAgentCfg())
task_registry.register("g1_tt_eval", TTEnv, G1TT_EvalEnvCfg(), G1TableTennisAgentCfg())
```

- [ ] **Step 3: 注册校验（不启 sim）**

Run:
```bash
/home/woan/.conda/envs/pingpong/bin/python -c "
import legged_lab.envs  # triggers registration
from legged_lab.utils.task_registry import task_registry
cfg, agent = task_registry.get_cfgs('g1_tt')
assert cfg.robot.num_actions == 23
assert len(cfg.observations.joint_names) == 23
assert cfg.robot.paddle_body_name == 'right_wrist_roll_rubber_hand'
assert agent.experiment_name == 'g1_table_tennis'
print('g1_tt registered, dims OK')
"
```
Expected: `g1_tt registered, dims OK`

- [ ] **Step 4: Commit**

```bash
git add legged_lab/envs/g1_tt/ legged_lab/envs/__init__.py
git commit -m "feat(envs): add g1_tt / g1_tt_eval task config and registration"
```

---

## Task 5: 资产加载 + 站立烟测（验证 1-2）

**Files:**
- Create: `legged_lab/scripts/fk_probe.py`

- [ ] **Step 1: 写探针脚本（加载 g1_tt、报关节数/拍面 body、零动作站立 2s）**

创建 `legged_lab/scripts/fk_probe.py`：
```python
"""G1 TT probe: load env, report joints/paddle body, hold zero action, print paddle vs base geometry."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="g1_tt")
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import torch
from legged_lab.envs import *  # noqa
from legged_lab.utils.task_registry import task_registry

env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
env_cfg.scene.num_envs = 1
env_class = task_registry.get_task_class(args.task)
env = env_class(env_cfg, headless=True)

robot = env.robot
print("[probe] num joints:", robot.num_joints)
print("[probe] joint names:", robot.joint_names)
pid = env._paddle_body_id
print("[probe] paddle body:", robot.body_names[pid], "idx", pid)

# zero-action hold
act = torch.zeros(env.num_envs, env.num_actions, device=env.device)
zmin = 1e9
for i in range(args.steps):
    env.step(act)
    z = float(env.robot.data.root_link_pos_w[0, 2].item())
    zmin = min(zmin, z)
print("[probe] min pelvis z over hold:", round(zmin, 3))

# paddle vs base geometry (ready stance)
import isaaclab.utils.math as mu
paddle_pos = robot.data.body_pos_w[:, pid, :]
paddle_quat = robot.data.body_quat_w[:, pid, :]
off = torch.tensor(env_cfg.robot.paddle_offset, device=env.device).unsqueeze(0).expand_as(paddle_pos)
face = paddle_pos + mu.quat_apply(paddle_quat, off)
base = robot.data.root_link_pos_w
rel = (face - base)[0]
print("[probe] paddle_face - base (x,y,z):", [round(float(v),3) for v in rel])
env.close()
app.close()
```

- [ ] **Step 2: 运行——预期首跑可能报缺字段/几何不当（看是否加载成功）**

Run:
```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
/home/woan/.conda/envs/pingpong/bin/python legged_lab/scripts/fk_probe.py --task g1_tt 2>&1 | tail -25
```
Expected（验证1-2 通过判据）：打印 `num joints: 23`、`paddle body: right_wrist_roll_rubber_hand`、`min pelvis z over hold:` 应 **> 0.6**（站立不倒）。若 < 0.6 说明站姿/增益不稳 → 调 init joint_pos 或 self_collisions，重跑。

- [ ] **Step 3: Commit**

```bash
git add legged_lab/scripts/fk_probe.py
git commit -m "feat(scripts): add G1 TT load/stand/FK probe"
```

---

## Task 6: FK 标定 → 回填几何常量（验证 3）

**Files:**
- Modify: `legged_lab/envs/g1_tt/g1_tt_config.py`

- [ ] **Step 1: 读探针输出的 `paddle_face - base (x,y,z)`**

用 Task 5 跑出的 `[probe] paddle_face - base (x,y,z): [X, Y, Z]`。这给出预备站姿下拍面相对基座的真实偏移。

- [ ] **Step 2: 回填几何常量**

在 `g1_tt_config.py` 的 `__post_init__` 中按实测更新：
- `self.robot.paddle_y_offset = Y`（拍面相对基座横偏，取探针的 Y）
- `self.robot.hit_body_height = Z + 0.76`（拍面世界高 = 相对高 Z + 站高；如探针已是世界高则直接取）
- 若 `paddle_offset` 的 0.20m 使拍面落点不合理（拍面没在身体右前正手区），调 `paddle_offset` 的 X 后重跑 Task 5 探针。

> 判据：调到 `paddle_face - base` 的 (x,y) 落在身体右前方合理正手击球区（x 略负/前伸、y 负/右侧），与 T1 几何关系同向。

- [ ] **Step 3: 重跑探针确认几何合理**

Run:
```bash
/home/woan/.conda/envs/pingpong/bin/python legged_lab/scripts/fk_probe.py --task g1_tt 2>&1 | grep probe
```
Expected: `paddle_face - base` 落在正手区；`min pelvis z` 仍 > 0.6。

- [ ] **Step 4: Commit**

```bash
git add legged_lab/envs/g1_tt/g1_tt_config.py
git commit -m "calib(g1_tt): backfill paddle/hit geometry from FK probe"
```

---

## Task 7: 物理击球（弹球）验证（验证 4）

**Files:**
- Modify: `legged_lab/scripts/fk_probe.py`（加 `--bounce` 模式）

- [ ] **Step 1: 给探针加弹球测试**

在 `fk_probe.py` 的 `env.close()` 之前追加（朝拍面射一颗球，看法向速度反号）：
```python
if "--bounce" in __import__("sys").argv:
    pid = env._paddle_body_id
    face = (env.robot.data.body_pos_w[:, pid, :]
            + mu.quat_apply(env.robot.data.body_quat_w[:, pid, :],
                            torch.tensor(env_cfg.robot.paddle_offset, device=env.device)
                            .unsqueeze(0).expand(env.num_envs, 3)))
    # place ball 0.15m in front of paddle face (+X side) moving toward it (-X)
    bstate = env.ball.data.default_root_state.clone()
    bstate[:, :3] = face + torch.tensor([0.15, 0.0, 0.0], device=env.device)
    bstate[:, 7:10] = torch.tensor([-3.0, 0.0, 0.0], device=env.device)
    env.ball.write_root_pose_to_sim(bstate[:, :7])
    env.ball.write_root_velocity_to_sim(bstate[:, 7:])
    vx0 = float(env.ball.data.root_lin_vel_w[0, 0].item())
    for _ in range(30):
        env.step(act)
    vx1 = float(env.ball.data.root_lin_vel_w[0, 0].item())
    print("[bounce] ball vx before:", round(vx0,2), "after:", round(vx1,2))
    assert vx1 > 0.5, "ball did not rebound off paddle"
    print("[bounce] REBOUND OK")
```

- [ ] **Step 2: 运行弹球测试**

Run:
```bash
/home/woan/.conda/envs/pingpong/bin/python legged_lab/scripts/fk_probe.py --task g1_tt --bounce 2>&1 | grep -E "bounce"
```
Expected: `ball vx before: -3.0 after: <正数>` + `REBOUND OK`（球从拍面物理弹回，证明碰撞几何+全局 0.8 恢复材质生效）。若 after 仍为负 → 球拍碰撞几何未生效，检查 URDF collision / `replace_cylinders_with_capsules`。

- [ ] **Step 3: Commit**

```bash
git add legged_lab/scripts/fk_probe.py
git commit -m "test(g1_tt): add paddle rebound probe (verifies collision geometry)"
```

---

## Task 8: 训练烟测（无 predictor，验证 5）

- [ ] **Step 1: 跑 100 iter 烟测**

Run:
```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
timeout 900 /home/woan/.conda/envs/pingpong/bin/python legged_lab/scripts/train.py \
  --task=g1_tt --logger=tensorboard --num_envs=64 --headless --max_iterations=100 2>&1 | tail -30
```
Expected: 出现 `Learning iteration 0..` 递增到接近 100，`Mean reward` 为有限数（非 nan/inf），无 Traceback/CUDA error。
> 若 `--max_iterations` 不被 train.py 识别，改用 AgentCfg 里临时设 `max_iterations=100`，或跑约 1-2 分钟手动 Ctrl-C 看是否稳定迭代。

- [ ] **Step 2: 检查无 NaN**

Run:
```bash
ls -t logs/g1_table_tennis/ | head -1
```
Expected: 生成了带时间戳的 run 目录（训练确实落盘）。

- [ ] **Step 3: Commit（若 Task 1-7 中有微调）**

```bash
git add -A && git commit -m "chore(g1_tt): training smoke test passes (no predictor)" --allow-empty
```

---

## Task 9: predictor 兼容 + 学习信号（验证 6-7）

- [ ] **Step 1: 跑带 predictor 的烟测**

Run:
```bash
timeout 1200 /home/woan/.conda/envs/pingpong/bin/python legged_lab/scripts/train.py \
  --task=g1_tt --logger=tensorboard --num_envs=64 --headless --predictor --max_iterations=300 2>&1 | tail -40
```
Expected（验证 6）：`predictor_mse` 出现在日志里、随迭代下降趋势；无崩溃。

- [ ] **Step 2: 检查学习信号（验证 7）**

Run:
```bash
RUN=$(ls -t logs/g1_table_tennis/ | head -1)
/home/woan/.conda/envs/pingpong/bin/python -c "
from tensorboard.backend.event_processing import event_accumulator as ea
import glob,os
f=sorted(glob.glob('logs/g1_table_tennis/$RUN/**/events*',recursive=True))[0]
a=ea.EventAccumulator(f); a.Reload()
tags=a.Tags()['scalars']
hit=[s.value for s in a.Scalars('Train/TT_hit_rate')] if 'Train/TT_hit_rate' in tags else []
print('TT_hit_rate samples:', hit[-3:] if hit else 'none')
"
```
Expected: `TT_hit_rate` 出现非零值（哪怕很小），证明几何/奖励链路通、机器人开始偶尔击到球。

- [ ] **Step 3: 最终提交 + 更新记忆**

```bash
git add -A && git commit -m "feat(g1_tt): predictor-augmented training runs; non-zero hit signal" --allow-empty
```

---

## 完成定义

验证 1-6 全过、验证 7 出现非零击球信号，即"G1 训练框架搭好"。完全收敛（论文 96% 命中级）属后续训练调参，不在本计划。后续：相机/动捕 SDK、本机部署、球拍精确打印 CAD（见 spec 第 5 节延后事项）。

**显式延后（spec 中的可选项，不在本计划任务内）：** 球拍质量/质心 ±20% 域随机化——属鲁棒性增强，待框架跑通且开始正式训练时，作为 sim2real 加固再加（在 `domain_rand.events` 增 randomize_rigid_body_mass 项，body=`right_wrist_roll_rubber_hand`）。

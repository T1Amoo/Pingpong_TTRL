# A1 AGV 乒乓训练框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把自研 AGV 机器人 A1 接入 Pingpong_TTRL，用薄子类复用现有 `TTEnv`，只训右臂 7 DoF、底盘自由重刚体、home=-1.8，当天在**本地降 env** 上跑通不发散（本地 RTX 4060 8GB；全量 4096 env 云端 L20 后续，用户自行开实例）。

**Architecture:** 复用 `legged_lab/envs/base/tt_env.py::TTEnv`（obs 布局硬编码、随 `num_actions` 自动缩放，已含 base_ang_vel+projected_gravity）。新增薄子类 `A1TTEnv` 只重写 `check_reset` 的几何阈值（G1 阈值按 pelvis 0.76 调过，A1 base 高度不同）。新增 `A1_TT_CFG`（自由底盘、DAMIAO 电机 kp/kd/armature、非训关节锁刚度）与 `A1TableTennisEnvCfg`（剥离 feet/ankle 奖励、保留 ball+arm+ready-pose+摇晃惩罚）。

**Tech Stack:** IsaacLab (ArticulationCfg/ImplicitActuatorCfg)、rsl_rl PPO、Python 3.10、legged_lab task_registry。

## Global Constraints

- 分支 `a1-tt-migration`，从 `g1-tt-deploy` 切出。代码同步走 git push/pull，不 scp。
- **绝不停正在跑的训练**，除非用户直接允许。绝不删训练成果/ckpt/脚本（要挪先归档 + 明说去向 + 先问）。
- 只训 `joint_yb_1..7`（右臂 7 DoF）；左臂 `joint_zb_*`、头 `joint_head_*`、`joint_lift`、轮子 = 不训练，靠 actuator 刚度保持在 init。
- 底盘 = 自由重刚体（**去掉** `fix_root_link=True`），轮子锁死不驱动。
- home x = **-1.8**。obs 每帧 39 维（`18 + 3×num_actions`，num_actions=7）× 5 历史 = 195，已含 ang_vel(3)+projected_gravity(3)，无需改 obs 代码。
- 右臂 actuator（DAMIAO，ω_n=2π·10=62.83）：`joint_yb_1/2/3`(4340,减速比40) stiffness=126.33094 / damping=8.042478 / armature=0.032 / effort=28；`joint_yb_4/5/6/7`(4310,减速比10) stiffness=7.106115 / damping=0.452389 / armature=0.0018 / effort=8。velocity limit={8,8,8,20,20,20,20}。
- `termination_penalty=-100`（不用 -1000，避免 critic value_loss 被巨值炸）。
- 本项目**验证=sim 冒烟**（此 env 无 pytest 单测；"测试"= 用 Isaac headless 跑并核对输出/落盘）。判据：起训不崩、`Mean value_function loss` 稳定 `<100`、reward 增长、机器人摆正拍在 ready 高度。
- **本地执行**：Isaac 用 `/home/woan/.conda/envs/pingpong/bin/python`。本地 GPU=RTX 4060 8GB → 全程降 `num_envs`（先试 64-128，能跑再往上加到显存上限，不追 4096）。全量 4096 云端 L20 后续（用户自行开实例，本计划不含）。
- 运行 Isaac 需 `OMNI_KIT_ACCEPT_EULA=YES`。

---

### Task 1: 建分支 + 落资产 + 实测 articulation 事实

USD 帧朝向、base_link 高度、拍所在（合并）body 名与偏移、轮子 body 名 —— 这些只能在 sim 里实测。本任务产出一份"事实清单"，后续任务据此填数。

**Files:**
- Create: `legged_lab/assets/a1/a1.usd`（拷自 `/home/woan/下载/a1.usd`）
- Create: `legged_lab/assets/a1/__init__.py`（空）
- Create: `legged_lab/scripts/inspect_a1.py`（一次性实测脚本）
- Create: `docs/superpowers/plans/a1_facts.md`（记录实测值，供后续任务引用）

**Interfaces:**
- Produces: `a1_facts.md` 含字段 —— `A1_JOINT_ORDER`（articulation 全关节名顺序）、`RIGHT_ARM_JOINTS=[joint_yb_1..7]` 是否存在、`WHEEL_BODIES`（正则）、`BASE_BODY`（root body 名）、`PADDLE_BODY`（Link_yb_paddle 合并后的父 body 名）、`BASE_Z_UPRIGHT`（摆正时 base_link 世界 z）、`INIT_ROT`（使机器人摆正+臂朝+x 的四元数 wxyz）、`PADDLE_WORLD_POS`（ready 位姿下拍世界坐标）。

- [ ] **Step 1: 建分支**

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
git checkout -b a1-tt-migration
```

- [ ] **Step 2: 落 USD 资产**

```bash
mkdir -p legged_lab/assets/a1
cp /home/woan/下载/a1.usd legged_lab/assets/a1/a1.usd
touch legged_lab/assets/a1/__init__.py
ls -la legged_lab/assets/a1/
```
Expected: `a1.usd` 约 28.8 MB 存在。

- [ ] **Step 3: 写实测脚本 `inspect_a1.py`**

参考已有 `legged_lab/scripts/view_robot.py` 的 AppLauncher 起法。脚本 spawn A1（先按 a1.py 的 init：`fix_root_link=True`、`rot=(0,0,0,1)`）、`sim.reset()` 后打印：

```python
"""One-shot: spawn A1, print joint/body names, base height, paddle body pose."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch, isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from legged_lab.assets.a1.a1 import A1_TT_CFG  # created in Task 2; for Step 3 use a temp inline cfg (see note)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
cfg = A1_TT_CFG.replace(prim_path="/World/Robot")
robot = Articulation(cfg)
sim.reset()
print("JOINTS:", robot.joint_names)
print("BODIES:", robot.body_names)
print("BASE root body:", robot.body_names[0])
zpos = robot.data.body_pos_w[0, :, 2]
for n, z in zip(robot.body_names, zpos.tolist()):
    print(f"  body {n:40s} z={z:.4f}")
app.close()
```

注：Task 1 Step 3 先于 Task 2 需要 `A1_TT_CFG`。执行顺序上，先做 Task 2 Step 1（写最小 a1.py 能 import），再回来跑本脚本；或本脚本内联一个临时 `ArticulationCfg` 指向 `legged_lab/assets/a1/a1.usd`。**推荐：先建临时内联 cfg 跑通实测，Task 2 再写正式 a1.py。**

- [ ] **Step 4: 跑实测，记录事实**

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/inspect_a1.py --headless 2>&1 | tee /tmp/a1_inspect.log
grep -E "JOINTS|BODIES|body " /tmp/a1_inspect.log
```
Expected: 打印出全关节名（含 `joint_yb_1..7`）、全 body 名（含轮子 body、`Link_yb_*`、base root body）、各 body 世界 z。

- [ ] **Step 5: 判定帧朝向 + 摆正四元数**

从 Step 4 的 body z 值判断：若 base root body 的 z 明显低于 `Link_yb_paddle` 且轮子 body z 最低（贴地）→ 机器人已摆正（z-up）。若不是（用户提示 USD 可能 -x 朝地），据 body 的 x/z 分布推 `INIT_ROT`（绕某轴 90°）使摆正 + 臂朝 +x。把结论（`BASE_Z_UPRIGHT`、`INIT_ROT`、`PADDLE_BODY`= `Link_yb_7`/`Link_yb_paddle` 合并后实际名、`WHEEL_BODIES` 正则、拍世界 pos）写进 `docs/superpowers/plans/a1_facts.md`。

- [ ] **Step 6: Commit**

```bash
git add legged_lab/assets/a1/ legged_lab/scripts/inspect_a1.py docs/superpowers/plans/a1_facts.md
git commit -m "feat(a1): add A1 USD asset + articulation inspection + recorded facts"
```

---

### Task 2: 写 `A1_TT_CFG`（自由底盘 + DAMIAO 电机 + 锁非训关节）

**Files:**
- Create: `legged_lab/assets/a1/a1.py`
- Test: 复用 `legged_lab/scripts/inspect_a1.py`

**Interfaces:**
- Consumes: `a1_facts.md` 的 `INIT_ROT`、`BASE_Z_UPRIGHT`。
- Produces: `A1_TT_CFG: ArticulationCfg`（供 `a1_tt_config.py` 导入）。

- [ ] **Step 1: 写 `a1.py`**

```python
"""A1 AGV arm robot config for table tennis (free chassis, right-arm-only training)."""
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

A1_USD_PATH = os.path.join(os.path.dirname(__file__), "a1.usd")

# DAMIAO motors (ω_n=2π·10=62.83): kp=armature·ω_n², kd≈2·2·armature·ω_n
_KP = {"joint_yb_1": 126.33094, "joint_yb_2": 126.33094, "joint_yb_3": 126.33094,
       "joint_yb_4": 7.106115, "joint_yb_5": 7.106115, "joint_yb_6": 7.106115, "joint_yb_7": 7.106115}
_KD = {"joint_yb_1": 8.042478, "joint_yb_2": 8.042478, "joint_yb_3": 8.042478,
       "joint_yb_4": 0.452389, "joint_yb_5": 0.452389, "joint_yb_6": 0.452389, "joint_yb_7": 0.452389}
_ARM = {"joint_yb_1": 0.032, "joint_yb_2": 0.032, "joint_yb_3": 0.032,
        "joint_yb_4": 0.0018, "joint_yb_5": 0.0018, "joint_yb_6": 0.0018, "joint_yb_7": 0.0018}
_EFFORT = {"joint_yb_1": 28.0, "joint_yb_2": 28.0, "joint_yb_3": 28.0,
           "joint_yb_4": 8.0, "joint_yb_5": 8.0, "joint_yb_6": 8.0, "joint_yb_7": 8.0}
_VEL = {"joint_yb_1": 8.0, "joint_yb_2": 8.0, "joint_yb_3": 8.0,
        "joint_yb_4": 20.0, "joint_yb_5": 20.0, "joint_yb_6": 20.0, "joint_yb_7": 20.0}

# from a1_facts.md (Task 1): quaternion (w,x,y,z) that stands the robot upright + arm toward +x
A1_INIT_ROT = (1.0, 0.0, 0.0, 0.0)   # REPLACE with measured INIT_ROT
A1_INIT_Z = 0.0                      # REPLACE with measured BASE_Z_UPRIGHT so wheels touch z=0

A1_TT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=A1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=10.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,   # B: free chassis; arm reaction can wobble it
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.8, 0.0, A1_INIT_Z),
        rot=A1_INIT_ROT,
        joint_pos={
            "joint_lift": -0.28,   # tuned in Task 4 for G1-prior paddle-ready height
            "joint_yb_1": 1.769, "joint_yb_2": -0.762, "joint_yb_3": -1.863,
            "joint_yb_4": 1.445, "joint_yb_5": 0.206, "joint_yb_6": -0.827, "joint_yb_7": 1.043,
            "joint_zb_1": 0.0, "joint_zb_2": 0.0, "joint_zb_3": 0.0, "joint_zb_4": 0.0,
            "joint_zb_5": 0.0, "joint_zb_6": 0.0, "joint_zb_7": 0.0,
            "joint_head_lr": 0.0, "joint_head_ud": 0.0,
            "joint_left_wheel": 0.0, "joint_right_wheel": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_yb_[1-7]"],
            effort_limit_sim=_EFFORT, velocity_limit_sim=_VEL,
            stiffness=_KP, damping=_KD, armature=_ARM,
        ),
        "left_arm": ImplicitActuatorCfg(joint_names_expr=["joint_zb_[1-7]"],
            effort_limit_sim=200.0, velocity_limit_sim=0.1, stiffness=10000.0, damping=1000.0),
        "lift": ImplicitActuatorCfg(joint_names_expr=["joint_lift"],
            effort_limit_sim=1000.0, velocity_limit_sim=0.0, stiffness=5000.0, damping=500.0),
        "head": ImplicitActuatorCfg(joint_names_expr=["joint_head_.*"],
            effort_limit_sim=10.0, velocity_limit_sim=0.1, stiffness=10000.0, damping=1000.0),
        "wheels": ImplicitActuatorCfg(joint_names_expr=["joint_.*_wheel"],
            effort_limit_sim=10.0, velocity_limit_sim=0.0, stiffness=10000.0, damping=1000.0),
    },
)
```

- [ ] **Step 2: 填实测值**

用 `a1_facts.md` 的 `INIT_ROT`、`BASE_Z_UPRIGHT` 替换 `A1_INIT_ROT`、`A1_INIT_Z`。`A1_INIT_Z` 取"使轮子 body 世界 z≈0"的 base pos.z（= 摆正后 base 相对轮子的高度）。

- [ ] **Step 3: 跑冒烟：spawn 摆正、无 NaN**

改 `inspect_a1.py` 用正式 `A1_TT_CFG`，跑：
```bash
OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/inspect_a1.py --headless 2>&1 | tee /tmp/a1_spawn.log
grep -E "body Link_yb|body .*wheel|nan|NaN|Error" /tmp/a1_spawn.log
```
Expected: 轮子 body z≈0（贴地）、`Link_yb_paddle`/拍 body z>0.8、无 NaN/Error。若机器人歪/穿地，回 Step 2 调 `A1_INIT_ROT`/`A1_INIT_Z`。

- [ ] **Step 4: Commit**

```bash
git add legged_lab/assets/a1/a1.py legged_lab/scripts/inspect_a1.py
git commit -m "feat(a1): A1_TT_CFG with free chassis + DAMIAO kp/kd/armature + locked non-arm joints"
```

---

### Task 3: 写 `a1_tt_config.py` + `A1TTEnv` 子类 + 注册任务

**Files:**
- Create: `legged_lab/envs/a1_tt/__init__.py`（空）
- Create: `legged_lab/envs/a1_tt/a1_tt_config.py`
- Create: `legged_lab/envs/a1_tt/a1_tt_env.py`（`A1TTEnv(TTEnv)`）
- Modify: `legged_lab/envs/__init__.py`（import + register）

**Interfaces:**
- Consumes: `A1_TT_CFG`；`a1_facts.md` 的 `BASE_Z_UPRIGHT`、`PADDLE_BODY`、`WHEEL_BODIES`、`BASE_BODY`。
- Produces: task 名 `a1_tt`（训练）、`a1_tt_eval`（易球固定分布）。

- [ ] **Step 1: 写 `A1TTEnv` 子类（重写 `check_reset` 几何阈值）**

A1 base 高度 ≠ G1 pelvis 0.76，`TTEnv.check_reset` 的 `z<0.50` 会误杀。用实测 `BASE_Z_UPRIGHT` 定阈值 = `BASE_Z_UPRIGHT - 0.25`（倾倒/塌陷才触发）。

```python
"""A1 table-tennis env: reuse TTEnv, override only the geometric reset thresholds."""
import torch
from legged_lab.envs.base.tt_env import TTEnv

# from a1_facts.md (Task 1)
A1_BASE_Z_UPRIGHT = 0.42        # REPLACE with measured base_link upright world z
A1_RESET_Z_MIN = A1_BASE_Z_UPRIGHT - 0.25   # tip/collapse threshold

class A1TTEnv(TTEnv):
    def check_reset(self):
        reset_buf = (
            (self.robot_pos[..., 2] < A1_RESET_Z_MIN) |
            (self.robot_pos[..., 0] < -3.6) |
            (self.robot_pos[..., 0] > -1.35) |
            (self.robot_pos[..., 1] < -1.1) |
            (self.robot_pos[..., 1] > 1.1)
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        time_out_buf |= self.ball_reset_counter > self.max_ball_serve_per_episode
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf
```

- [ ] **Step 2: 写 `a1_tt_config.py` —— 奖励(剥离 feet、保留 ball+arm+ready+摇晃)**

```python
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
import legged_lab.mdp as mdp
from legged_lab.assets.a1.a1 import A1_TT_CFG
from legged_lab.assets.table_tennis.table import TABLE_CFG
from legged_lab.assets.table_tennis.ball import BALL_CFG
from legged_lab.envs.base.tt_env_config import TTAgentCfg, TTEnvCfg, RewardCfg  # noqa:F401

A1_ARM_JOINTS = ["joint_yb_1","joint_yb_2","joint_yb_3","joint_yb_4","joint_yb_5","joint_yb_6","joint_yb_7"]

@configclass
class A1TableTennisRewardCfg(RewardCfg):
    # --- base-shake penalty (arm reaction wobbles the free chassis) ---
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.02)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)   # tilt from projected_gravity
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # --- arm smoothness / limits ---
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.025)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.002)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_pos_target_limits = RewTerm(func=mdp.joint_pos_target_limits, weight=-1.0)
    joint_deviation_right_arm = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_ARM_JOINTS)})
    # --- don't crash into table ---
    penalty_robot_table_proximity_x = RewTerm(func=mdp.penalty_robot_table_proximity_x,
        weight=-20.0, params={"min_distance": 0.15, "std": 0.07})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    # --- ready-pose regularization when no playable ball (A) ---
    reward_idle_stand = RewTerm(func=mdp.reward_idle_stand, weight=0.5)
    reward_idle_pose = RewTerm(func=mdp.reward_idle_pose, weight=1.0, params={"k": 1.0})
    # --- ball / hitting core (robot-agnostic) ---
    reward_contact = RewTerm(func=mdp.reward_contact, weight=150.0)
    reward_future_dis_ee = RewTerm(func=mdp.reward_future_ee_target, weight=2.0,
        params={"std_ee": 0.5, "threshold": 0.15})
    reward_future_dis_ro = RewTerm(func=mdp.reward_future_body_target, weight=5.0,
        params={"std_ro": 0.5, "threshold": 0.05})
    reward_future_vel_base = RewTerm(func=mdp.reward_future_vel_target, weight=5.0,
        params={"vel_std": 1.2, "threshold": 0.1})
    reward_future_landing_dis = RewTerm(func=mdp.reward_future_landing_dis, weight=60.0,
        params={"threshold": 3.0})
    reward_future_pass_net = RewTerm(func=mdp.reward_future_pass_net, weight=100.0,
        params={"std_h": 0.4, "z_target": 0.76 + 0.35})
    reward_table_success = RewTerm(func=mdp.reward_table_success, weight=100.0)
```

- [ ] **Step 3: 写 `A1TableTennisEnvCfg` + eval + agent**

```python
@configclass
class A1TableTennisEnvCfg(TTEnvCfg):
    reward = A1TableTennisRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.002
        self.sim.decimation = 10  # 50 Hz
        self.scene.height_scanner.enable_height_scan = False
        self.scene.height_scanner.prim_body_name = "base_link"   # REPLACE with a1_facts BASE_BODY
        self.scene.robot = A1_TT_CFG
        self.scene.table = TABLE_CFG
        self.scene.ball = BALL_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        # feet/terminate must RESOLVE to real A1 bodies (used only for critic feet_contact + __init__)
        self.robot.terminate_contacts_body_names = ["base_link"]      # REPLACE with BASE_BODY
        self.robot.feet_body_names = [".*wheel.*"]                    # REPLACE with WHEEL_BODIES regex
        self.robot.num_actions = 7
        self.robot.num_joints = 7
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]  # REPLACE BASE_BODY
        # reset-joint DR groups -> only the right arm is "manipulation"; nothing to randomize as locomotion.
        self.domain_rand.events.reset_locomotion_joints.params["asset_cfg"].joint_names = A1_ARM_JOINTS[:1]
        self.domain_rand.events.reset_manipulation_joints.params["asset_cfg"].joint_names = A1_ARM_JOINTS
        # base is free but should NOT be reset-scattered like a walking robot: keep it near home.
        self.domain_rand.events.reset_base.params["pose_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.05, 0.05)}
        # paddle geometry (from a1_facts.md; tuned in Task 4)
        self.robot.paddle_body_name = "Link_yb_7"        # REPLACE with PADDLE_BODY (merged)
        self.robot.paddle_offset = (0.0, 0.0, 0.172)     # REPLACE from FK (Task 4)
        self.robot.hit_body_height = 0.42                # REPLACE with A1 base_link upright z
        self.robot.paddle_y_offset = -0.55               # tuned in Task 4
        self.robot.hit_plane_x = -1.8
        self.observations.joint_names = A1_ARM_JOINTS
        self.actions.joint_names = A1_ARM_JOINTS
        # serve: reuse the G1 v11+ tuned -1.8 easy distribution
        self.ball.serve_bounce_enable = True
        self.ball.serve_bounce_x_range = (-0.90, -0.76)
        self.ball.serve_bounce_vz_range = (1.2, 1.6)
        self.ball.serve_y_start = 0.5
        self.ball.serve_curriculum_steps = 0     # easy-only for first run (no ramp)
        self.ball.no_ball_period_s = 0.0

@configclass
class A1TT_EvalEnvCfg(A1TableTennisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.max_episode_length_s = 99999999999
        self.ball.serve_curriculum_steps = 0

@configclass
class A1TableTennisAgentCfg(TTAgentCfg):
    experiment_name: str = "a1_tt"
    logger = "tensorboard"
    save_interval = 100
    max_iterations = 30000
    predictor = {"history_len": 5, "traj_max_len": 128, "hidden_sizes": [64, 64],
                 "lr": 0.5e-3, "epochs_per_update": 1, "batch_size": 1024, "train_until_iters": 20}
```

- [ ] **Step 4: 填实测值**

用 `a1_facts.md` 替换所有标 `REPLACE` 处：`BASE_BODY`（height_scanner/terminate/add_base_mass）、`WHEEL_BODIES`（feet_body_names）、`PADDLE_BODY`、`hit_body_height`/`A1_BASE_Z_UPRIGHT`。

- [ ] **Step 5: 注册任务**

在 `legged_lab/envs/__init__.py` 加：
```python
from legged_lab.envs.a1_tt.a1_tt_config import (
    A1TableTennisEnvCfg, A1TableTennisAgentCfg, A1TT_EvalEnvCfg,
)
from legged_lab.envs.a1_tt.a1_tt_env import A1TTEnv
task_registry.register("a1_tt", A1TTEnv, A1TableTennisEnvCfg(), A1TableTennisAgentCfg())
task_registry.register("a1_tt_eval", A1TTEnv, A1TT_EvalEnvCfg(), A1TableTennisAgentCfg())
```

- [ ] **Step 6: 冒烟：env 能构造（少量 env、几步）**

```bash
OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/train.py --task a1_tt --headless \
  --num_envs 16 --max_iterations 2 2>&1 | tee /tmp/a1_env_smoke.log
grep -E "value_function|Error|Traceback|resolve|find_joints|nan" /tmp/a1_env_smoke.log | head
```
Expected: 无 Traceback（尤其无 feet_cfg/termination resolve 失败、无 joint 找不到）；能打印 iteration 0/1 日志。若报 body/joint 名不存在 → 回 Step 4 用实测名修正。

- [ ] **Step 7: Commit**

```bash
git add legged_lab/envs/a1_tt/ legged_lab/envs/__init__.py
git commit -m "feat(a1): A1TTEnv subclass + a1_tt env/eval config + task registration"
```

---

### Task 4: 调 lift 高度 + 拍偏移，使拍 ready 高度 ≈ G1（z≈1.0–1.1）

**Files:**
- Modify: `legged_lab/assets/a1/a1.py`（`joint_lift` init）
- Modify: `legged_lab/envs/a1_tt/a1_tt_config.py`（`paddle_offset`/`paddle_y_offset`/`hit_body_height`）
- Test: `legged_lab/scripts/fk_probe.py`（已有；或 `inspect_a1.py`）

**Interfaces:**
- Consumes: `PADDLE_BODY`；G1 目标 = 拍世界 z≈1.0–1.1、在机器人身前朝 +x。

- [ ] **Step 1: 量当前拍 ready 世界高度**

```bash
OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/inspect_a1.py --headless 2>&1 | grep -E "Link_yb_paddle|Link_yb_7"
```
记录拍 body 世界 z。

- [ ] **Step 2: 调 `joint_lift` init 使拍 z∈[1.0,1.1]**

`joint_lift` travel [-0.8,-0.05]，更负 = 升更高（axis +z、mount z=1.273）。若拍偏低，减小 `joint_lift`（更负）；偏高则增大。改 `a1.py` 的 `"joint_lift"` init 值，重跑 Step 1 直到拍 z∈[1.0,1.1]。

- [ ] **Step 3: 定 `paddle_offset` / `paddle_y_offset` / `hit_body_height`**

`paddle_offset` = 从 `PADDLE_BODY` 帧到拍面中心的偏移（FK：`joint_yb_paddle` origin +z0.172 起，按实测微调，使 obs 里的拍触点≈真实拍面）。`hit_body_height` = base_link 摆正世界 z。`paddle_y_offset` 先按 -0.55，Task 5 训练观察击球间隙再调。

- [ ] **Step 4: 验证**

```bash
OMNI_KIT_ACCEPT_EULA=YES python -u legged_lab/scripts/inspect_a1.py --headless 2>&1 | grep -E "paddle|Link_yb_7"
```
Expected: 拍世界 z∈[1.0,1.1]、x 在 base 身前(朝 +x 方向)。

- [ ] **Step 5: Commit**

```bash
git add legged_lab/assets/a1/a1.py legged_lab/envs/a1_tt/a1_tt_config.py
git commit -m "feat(a1): tune lift height + paddle offset to G1-prior ready height"
```

---

### Task 5: 冒烟训练 → 本地降 env 验证收敛

**Files:**
- Create: `legged_lab/scripts/watchdog_train_a1.sh`（参考已有 `watchdog_train_v16.sh`）

**Interfaces:**
- Consumes: task `a1_tt`。
- Produces: `logs/a1_tt/<run>/` ckpt + tensorboard。

- [ ] **Step 1: 短冒烟（64 env / 200 iter），核对不发散 + 显存**

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/woan/.conda/envs/pingpong/bin/python -u legged_lab/scripts/train.py --task a1_tt --headless \
  --num_envs 64 --max_iterations 200 2>&1 | tee /tmp/a1_train200.log
grep -E "value_function loss|Mean reward|Traceback|nan|CUDA out of memory" /tmp/a1_train200.log | tail -20
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```
Expected: 无 OOM/NaN/Traceback；`Mean value_function loss` 全程 `<100`（不飙 1e6）；`Mean reward` 有增长趋势；显存留有余量。若 value_loss 爆 → 检查 termination_penalty=-100、检查 reset 阈值是否误杀（大量 iter-1 reset）。

- [ ] **Step 2: 写 watchdog 脚本（本地 pingpong env）**

参考 `watchdog_train_v16.sh`：`PY=/home/woan/.conda/envs/pingpong/bin/python`、`--task a1_tt`、`OMNI_KIT_ACCEPT_EULA=YES`、`--num_envs` 取显存能承受值（Step 1 定）、目标 iter 先设 5000（本地验证收敛，非满程）。`kill_training_also_kill_watchdog` 经验：先杀 watchdog 再杀训练；`pkill` 用 `[a]1` bracket 防自杀。

- [ ] **Step 3: 起本地降 env 训练，验证收敛趋势**

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
# num_envs 取 Step 1 验证能装下的最大值（先试 256，OOM 则降到 128/64）
nohup bash legged_lab/scripts/watchdog_train_a1.sh > /tmp/a1_watchdog.log 2>&1 &
sleep 90; grep -E "iteration|value_function|reward" /tmp/a1_watchdog.log 2>/dev/null | tail
```
Expected: 训练起来、iter 递增、value_loss<100、reward 上升。**起训后不擅自停**（遵守铁律）。这验证框架/奖励/发球在本地收敛；全量 4096 云端 L20 后续（用户开实例后把分支 push、云端 pull 起训）。

- [ ] **Step 4: Commit + push（分支备份到 GitHub）**

```bash
git add legged_lab/scripts/watchdog_train_a1.sh
git commit -m "feat(a1): local watchdog train script + reduced-env convergence run"
git push -u origin a1-tt-migration
```

---

## Self-Review

**Spec coverage:**
- §1 资产落位 → Task 1（USD/assets）✓；缺项核查(拍 collider)→ Task 1 Step 4/5 实测 body 存在 + Task 2 Step 3 spawn 验证 ✓
- §2 坐标系(M→W→R + 轮子高度) → Task 1 Step 5 实测帧 + Task 2 `A1_INIT_ROT`/`A1_INIT_Z` ✓
- §3 config(自由底盘/锁关节/lift/actuator DAMIAO) → Task 2 ✓；lift 高度 → Task 4 ✓
- §4 obs(39维含 gravity+ang_vel) → 无需改代码（tt_env 自动缩放），Global Constraints 记录 + Task 3 Step 6 验证维度 ✓
- §5 奖励(保留 ready-pose、剥离 feet、加摇晃惩罚) → Task 3 Step 2（ang_vel_xy/z + flat_orientation = 摇晃；剥离所有 feet/ankle）✓
- §6 架构(薄子类) → Task 3 Step 1（A1TTEnv 重写 check_reset）✓
- §7 发球/home(-1.8 复用 G1) → Task 3 Step 3 serve 段 ✓
- §8 跑通判据 → Task 5 ✓

**Placeholder scan:** 所有 `REPLACE` 均为 Task 1 实测值填入点（非占位符——有明确产出来源与命令），且在消费任务的 "填实测值" step 显式处理。无 TBD/TODO。

**Type consistency:** `A1_ARM_JOINTS`（7 关节名）在 reward/obs/actions 一致；`num_actions=num_joints=7` 与 obs 39维(`18+3×7`)一致；`A1_TT_CFG` 导出名在 Task 2 定义、Task 3 消费一致；`check_reset` 签名与 `TTEnv` 一致。

**已知风险(Task 内已置验证):** ①USD 帧朝向未知 → Task 1/2 实测调 rot；②拍合并 body 名 → Task 1 实测；③reset z 阈值误杀 → Task 3 用实测 base 高度 + Task 5 Step 1 查 iter-1 reset 率；④自由底盘可能漂移/翻 → Task 2 Step 3 + Task 5 观察 ang_vel 惩罚是否有信号。

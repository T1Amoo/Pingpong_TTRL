# A1 AGV Robot Articulation Facts

Empirically measured by running `legged_lab/scripts/inspect_a1.py` headless with
`fix_root_link=True`, identity rotation `(1,0,0,0)` wxyz, pos `(0,0,0)`.

USD source: `legged_lab/assets/a1/a1.usd` (28.8 MB, copied from `/home/woan/下载/a1.usd`)

---

## A1_JOINT_ORDER

Full articulation joint name list in Isaac articulation order (19 joints):

```python
A1_JOINT_ORDER = [
    'joint_right_wheel',  # 0
    'joint_left_wheel',   # 1
    'joint_head_lr',      # 2
    'joint_lift',         # 3
    'joint_head_ud',      # 4
    'joint_yb_1',         # 5  right arm
    'joint_zb_1',         # 6  left arm
    'joint_yb_2',         # 7
    'joint_zb_2',         # 8
    'joint_yb_3',         # 9
    'joint_zb_3',         # 10
    'joint_yb_4',         # 11
    'joint_zb_4',         # 12
    'joint_yb_5',         # 13
    'joint_zb_5',         # 14
    'joint_yb_6',         # 15
    'joint_zb_6',         # 16
    'joint_yb_7',         # 17
    'joint_zb_7',         # 18
]
```

Note: yb = right arm, zb = left arm. Arms interleaved in joint order.

## RIGHT_ARM_JOINTS

`joint_yb_1` through `joint_yb_7` are all present. Confirmed.

## BODY_ORDER

Full body name list in Isaac articulation order (21 bodies):

```python
A1_BODY_ORDER = [
    'base_link',        # 0  root
    'link_right_wheel', # 1
    'link_left_wheel',  # 2
    'link_head_lr',     # 3
    'link_lift',        # 4
    'link_head_ud',     # 5
    'link_yb_1',        # 6  right arm link 1
    'link_zb_1',        # 7  left arm link 1
    'link_yb_2',        # 8
    'link_zb_2',        # 9
    'link_yb_3',        # 10
    'link_zb_3',        # 11
    'link_yb_4',        # 12
    'link_zb_4',        # 13
    'link_yb_5',        # 14
    'link_zb_5',        # 15
    'link_yb_6',        # 16
    'link_zb_6',        # 17
    'link_yb_7',        # 18
    'link_zb_7',        # 19
    'Link_yb_paddle',   # 20  paddle (note capital L)
]
```

## WHEEL_BODIES

```python
WHEEL_BODIES = r"link_(right|left)_wheel"
```

Both wheel bodies: `link_right_wheel`, `link_left_wheel`

## BASE_BODY

```python
BASE_BODY = "base_link"
```

Root body (index 0 in body_names).

## PADDLE_BODY

```python
PADDLE_BODY = "Link_yb_paddle"
```

`Link_yb_paddle` is a distinct body in the articulation (index 20). It is the child of the
`joint_yb_7` / `link_yb_7` chain — the fixed-joint import kept it as a separate rigid body
(no merging occurred). Note the capital `L` unlike other lowercase link names.

---

## Frame Orientation Analysis

**The USD is already z-up.** With identity rotation (1,0,0,0) wxyz:

- `base_link` at z = 0.000 (root, anchored by fix_root_link)
- Both wheels at z = 0.035 (wheel centers 3.5 cm above base_link frame origin)
- Arm chain starts at z ≈ 1.18 (shoulder), descends to paddle at z = 0.444
- Head at z ≈ 1.44 (topmost point)

The robot stands upright with identity rotation. The arms extend along the **Y axis**:
- Right arm (yb): y ≈ -0.24 (negative y side)
- Left arm (zb): y ≈ +0.24 (positive y side)
- Paddle: y ≈ -0.24, z ≈ 0.44

To make the **right arm / paddle face +x** (standard TT deployment orientation),
rotate the robot **+90° around z-axis**: q = (0.7071, 0, 0, 0.7071) wxyz.

---

## BASE_Z_UPRIGHT

```python
BASE_Z_UPRIGHT = 0.0
```

`base_link` world z when upright with fix_root_link=True. In free simulation the base will
sit higher; the wheel radius ≈ 0.035 m sets the ground clearance.

Note: for free-standing (fix_root_link=False), set `pos=(0, 0, 0.035)` so wheels touch ground.

## INIT_ROT

```python
INIT_ROT = (0.7071, 0.0, 0.0, 0.7071)  # wxyz, +90° around z
```

Rotates the robot so the right arm (paddle) faces +x.
- Identity (1,0,0,0) → arm faces -y
- +90° around z → arm faces +x

## PADDLE_WORLD_POS

In ready pose with `INIT_ROT` applied (arm in hardware whip_high3 pose):

```python
PADDLE_WORLD_POS = (0.2425, 0.0400, 0.4442)  # (x, y, z) in meters
```

Derived by rotating original paddle pos (x=0.04, y=-0.2425, z=0.4442) by +90° around z.

---

## Raw Per-Body World Positions (identity rot, fix_root_link=True)

From `inspect_a1.py` output with `pos=(0,0,0)`, `rot=(1,0,0,0)`:

```
[00] base_link                                           x= -0.0000  y=  0.0000  z= -0.0000
[01] link_right_wheel                                    x=  0.0001  y= -0.1461  z=  0.0350
[02] link_left_wheel                                     x=  0.0001  y=  0.1461  z=  0.0350
[03] link_head_lr                                        x=  0.0400  y=  0.0000  z=  1.4057
[04] link_lift                                           x= -0.0175  y=  0.0000  z=  1.2209
[05] link_head_ud                                        x=  0.0400  y=  0.0390  z=  1.4365
[06] link_yb_1                                           x=  0.0400  y= -0.1150  z=  1.1827
[07] link_zb_1                                           x=  0.0400  y=  0.1150  z=  1.1827
[08] link_yb_2                                           x=  0.0400  y= -0.2424  z=  1.1827
[09] link_zb_2                                           x=  0.0400  y=  0.2424  z=  1.1827
[10] link_yb_3                                           x=  0.0400  y= -0.2425  z=  1.0894
[11] link_zb_3                                           x=  0.0399  y=  0.2425  z=  1.0894
[12] link_yb_4                                           x=  0.0400  y= -0.2435  z=  0.9492
[13] link_zb_4                                           x=  0.0399  y=  0.2435  z=  0.9492
[14] link_yb_5                                           x=  0.0400  y= -0.2425  z=  0.8612
[15] link_zb_5                                           x=  0.0399  y=  0.2425  z=  0.8612
[16] link_yb_6                                           x=  0.0400  y= -0.2420  z=  0.7282
[17] link_zb_6                                           x=  0.0399  y=  0.2420  z=  0.7282
[18] link_yb_7                                           x=  0.0400  y= -0.2425  z=  0.6162
[19] link_zb_7                                           x=  0.0399  y=  0.2417  z=  0.6162
[20] Link_yb_paddle                                      x=  0.0400  y= -0.2425  z=  0.4442
```

Z sorted (lowest to highest):
```
z= -0.0000  base_link
z=  0.0350  link_left_wheel
z=  0.0350  link_right_wheel
z=  0.4442  Link_yb_paddle
z=  0.6162  link_yb_7
z=  0.6162  link_zb_7
z=  0.7282  link_yb_6
z=  0.7282  link_zb_6
z=  0.8612  link_yb_5
z=  0.8612  link_zb_5
z=  0.9492  link_yb_4
z=  0.9492  link_zb_4
z=  1.0894  link_yb_3
z=  1.0894  link_zb_3
z=  1.1827  link_yb_1
z=  1.1827  link_yb_2
z=  1.1827  link_zb_1
z=  1.1827  link_zb_2
z=  1.2209  link_lift
z=  1.4057  link_head_lr
z=  1.4365  link_head_ud
```

---

## Quick Reference for Task 2+

| Field | Value |
|-------|-------|
| USD path | `legged_lab/assets/a1/a1.usd` |
| num_joints | 19 |
| num_bodies | 21 |
| BASE_BODY | `base_link` |
| PADDLE_BODY | `Link_yb_paddle` (capital L, index 20) |
| WHEEL_BODIES regex | `link_(right\|left)_wheel` |
| RIGHT_ARM joint names | `joint_yb_[1-7]` (7 joints) |
| LEFT_ARM joint names | `joint_zb_[1-7]` (7 joints) |
| LIFT joint | `joint_lift` |
| HEAD joints | `joint_head_lr`, `joint_head_ud` |
| WHEEL joints | `joint_right_wheel`, `joint_left_wheel` |
| BASE_Z_UPRIGHT | `0.0` (fix_root_link), wheel centers at z=0.035 |
| INIT_ROT (wxyz) | `(0.7071, 0.0, 0.0, 0.7071)` — +90° around z, arm faces +x |
| PADDLE_WORLD_POS | `(0.2425, 0.040, 0.444)` in ready pose |
| Arm shoulder z | ≈ 1.18 m (link_yb_1) |
| Arm z range | 0.44 m (paddle) to 1.18 m (shoulder) |
| Lift link z | 1.22 m |
| Head z | ≈ 1.44 m |

---

## BASE_Z_SETTLED

Measured by `inspect_a1.py --settle`: drop robot free-base from z=0.3, step 2 s (400 steps @ dt=0.005),
read settled world positions. Config: `fix_root_link=False`, `rot=(0.7071,0,0,0.7071)`, whip_high3 arm pose.

```python
BASE_Z_SETTLED = 0.0282  # base_link world z after settling on wheels
```

| Body | Settled world z |
|------|----------------|
| `base_link` | 0.0282 m |
| `link_right_wheel` | 0.0629 m |
| `link_left_wheel` | 0.0629 m |
| `Link_yb_paddle` | 0.9094 m |

**A1_INIT_Z = 0.0282** (spawn base_link at this height so wheels rest on ground plane).

Note: Wheel centers at 0.0629 m (= base_link 0.0282 + wheel offset 0.035 + slight ground contact
compression ~0.007 m).

**Real r1 centerline calibration — 2026-07-08:** current hardware maximum usable `r1` joint
centerline height is **1.15 m** above floor. For `X1_URDF_V1_1`, the approximate chain is:

```python
r1_z ~= base_link_z + sj_origin_z + r0_origin_z + sj
     ~= 0.0282 + 1.2107 + 0.025 + sj
```

So the training lift value for the real envelope is:

```python
sj = 1.15 - (0.0282 + 1.2107 + 0.025) = -0.1139
```

Do not use the URDF `sj=0` maximum for sim2real training; it implies `r1_z ~= 1.264 m`,
which is outside the current measured hardware envelope.

**Paddle ready height — IMPORTANT (controller correction):** paddle settles at **0.909 m** AFTER
stepping physics ~2 s, vs 0.444 m read in Task 1 (identity rot, pose read BEFORE stepping). The
+90° z-rotation does NOT change height — that earlier explanation was wrong. Real cause: with the
soft real-motor distal stiffness (kp=7.106 on joint_yb_4..7), the arm does NOT rigidly hold the
commanded whip_high3 pose; under gravity it relaxes to a spring/gravity EQUILIBRIUM (~0.909 paddle z).
Realistic motor behavior. The OPERATIVE ready height for training is the settled ~0.909 (training
steps physics), conveniently near G1's ~1.0. Consequences:
- Task 4: tune from the SETTLED pose (~0.909), not commanded 0.444; verify arm reaches the hit zone;
  expect notable gravity droop on soft distal joints (real hardware limit, cannot change kp).
- Task 3: base_link settles at ~0.028 m → G1-style `robot_pos.z < 0.50` fall-detection is USELESS
  for A1 (base already near floor). Use TILT-based termination (projected_gravity z-comp) + x/y bounds.
NaN check: OK (no NaN in body positions after settling).

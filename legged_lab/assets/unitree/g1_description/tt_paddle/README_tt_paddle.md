# G1 23DoF table-tennis paddle end-effector

Fixed (PACE-style) ping-pong paddle fixture for the **right arm** of the Unitree
G1 23DoF. It replaces the right rubber-hand geometry with a 3D-printed
adapter + handle slot + paddle. **No active joints are added** — every new
joint is `fixed`, so the robot stays at **23 DoF**.

## Files

| File | What it is |
|------|------------|
| `../g1_tt_paddle_adapter.xacro` | Parametric xacro macro (source of truth for dimensions). |
| `build_tt_urdf.py` | Pure-stdlib generator → produces the two URDFs below (no xacro/ROS needed). |
| `../g1_23dof_tt.urdf` | Base with the right rubber-hand visual/collision removed, inertia zeroed. |
| `../g1_23dof_tt_paddle.urdf` | **Ready-to-load**: base + paddle fixture. |
| `paddle_holder_mesh.py` | OPTIONAL trimesh script for a true hollow slot + drilled screw holes. |

## Geometry & frames

Local convention (matches the wrist_roll link, whose +X already points along
the palm direction):

- **+X** flange → paddle (extension-tube length axis)
- **+Y** handle-slot lateral width
- **+Z** paddle blade face normal (the hitting surface normal)

Chain off `right_wrist_roll_rubber_hand` (cumulative +X, default params):

```
mount(0.012) → extension(0.16) → holder(slot 0.075) → handle(0.095) → blade(r0.075)
paddle_contact_frame  @ x≈0.302 m, +Z = outward hitting normal
```

Named frames for RL reward / contact / IK:
`right_adapter_mount_frame`, `right_paddle_handle_frame`, `right_paddle_contact_frame`.

## How to (re)generate the URDFs

```bash
cd .../assets/g1_description/tt_paddle
python3 build_tt_urdf.py        # writes ../g1_23dof_tt.urdf and ../g1_23dof_tt_paddle.urdf
```

To change dimensions: edit the `PARAMS` dict in `build_tt_urdf.py` (and mirror
them in the xacro), then re-run.

## Using the xacro instead (if you have the `xacro` tool)

Add to a top-level robot xacro that also pulls in the base description:

```xml
<robot name="g1_23dof_tt" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- ... include / paste the base g1_23dof links+joints here ... -->
  <xacro:include filename="g1_tt_paddle_adapter.xacro"/>
  <xacro:g1_tt_paddle_adapter parent_link="right_wrist_roll_rubber_hand"/>
</robot>
```

The macro attaches **after** `right_wrist_roll_rubber_hand` (the 23rd-DoF
wrist_roll child). Do **not** touch any upstream G1 joints. Expand:

```bash
xacro g1_23dof_tt.xacro -o g1_23dof_tt_paddle.urdf
```

> Note: the base `g1_23dof.urdf` is plain URDF (no xacro), and `xacro` is not on
> this machine's PATH — that's why `build_tt_urdf.py` is the primary path. The
> xacro is for parametric tweaking once you have `pip install xacro` or ROS.

## Validate

```bash
# stdlib structural check is built into the generator (link/joint refs,
# single root, DoF count). For the ROS validator:
check_urdf ../g1_23dof_tt_paddle.urdf      # needs liburdfdom-tools

# RViz: load the URDF, enable TF, confirm right_paddle_contact_frame sits at
# the blade face with +Z (blue) pointing out of the hitting surface, and that
# the blade plane is perpendicular to the extension tube.
```

## IsaacLab

Point your articulation/USD-conversion config at `g1_23dof_tt_paddle.urdf`
(or convert it to USD). Because all paddle joints are `fixed`, the actuator /
joint config is unchanged — the 23 DoF and their names are identical to the
base. Query the paddle pose by the body name `right_paddle_contact_frame`.

## Contact / physics properties (set in the sim backend, not in URDF)

Plain URDF cannot express restitution. On `right_tt_paddle_blade_link`:

- **Isaac Sim**: RigidBodyMaterial — `restitution ≈ 0.85`, `dynamic_friction ≈ 0.6`,
  `restitution_combine_mode = max`.
- **MuJoCo**: `<geom>` `solref`/`solimp`/`friction`, `condim=4`.
- **Gazebo**: `<surface><bounce restitution_coefficient="0.85"/>` + `<friction>`.

## MEASURE-ME parameters (override after measuring the real part)

`mount_xyz`, `mount_rpy` (flange offset/orientation if it differs from the
wrist frame), `insertion_depth`, `blade_center_x`, `blade_radius`,
`blade_thickness`, all `*_mass`, and `screw_x1`/`screw_x2`.

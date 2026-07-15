# A1 v13 Night Training Plan

Date: 2026-07-08

## Goal

Train a sim2real-aligned A1 table-tennis policy to 30000 iterations using the current real hardware geometry.

## Changes

1. Set the A1 lift default from real `r1` centerline height:
   - Target `r1_z = 1.15 m`.
   - `r1_z ~= 0.0282 + 1.2107 + 0.025 + sj`.
   - Training uses `sj = -0.1139`.

2. Move the fixed hit plane backward:
   - Previous `hit_plane_x = -1.55`.
   - New `hit_plane_x = -1.60`.
   - `hit_target_x_range` remains fixed to the same plane.

3. Do not add a new robot-table collision reward in this run.
   - Existing config already has a base-table proximity penalty.
   - The current observed table grazing is more likely caused by target geometry being too far forward than by a missing sparse collision term.
   - Adding arm/table contact penalties requires contact-sensor/body filtering and can easily suppress swing exploration.

## Training

Experiment:

```bash
a1_tt_v13
```

Warm start:

```bash
logs/a1_tt_v12/*/model_13500.pt
```

Target:

```bash
30000
```

Default watchdog:

```bash
TARGET=30000 NUM_ENVS=4096 TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python \
  OMNI_KIT_ACCEPT_EULA=YES nohup bash legged_lab/scripts/watchdog_train_a1.sh > /dev/null 2>&1 &
```

## Morning Checks

Primary:

- `reward_contact`
- `reward_sweet_contact`
- `reward_table_success`
- `reward_future_pass_net`
- `reward_future_landing_dis`

Risk checks:

- `penalty_robot_table_proximity_x`
- `penalty_ball_body_block`
- `joint_computed_torque_limit`
- `action_rate_l2`
- `dof_acc_l2`

Decision rule:

- If table grazing persists with `hit_plane_x=-1.60`, add an explicit arm/paddle-table proximity or contact penalty next.
- If hit rate drops sharply, inspect whether `hit_target_z_range` should be lifted/narrowed after the new `r1_z=1.15` geometry.

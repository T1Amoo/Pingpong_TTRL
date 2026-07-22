# A1 TT Real V1 Config Summary

This note summarizes the current reusable `a1_tt_real` / `a1_tt_real_v1` experiment configuration as of 2026-07-14.

For the current `a1_tt_real_v4` reproduction reference for the second-order response
model and high-stiffness implicit actuator, use
`docs/superpowers/plans/2026-07-16-a1-real-response-actuator-repro.md`.

## Entry Point

- Task registry: `a1_tt_real`
- Environment cfg: `A1TableTennisDeployEnvCfg`
- Agent cfg: `A1TableTennisDeployAgentCfg`
- Experiment name: `a1_tt_real_v1`
- Run name: `scratch_identified_second_order`
- Training style: scratch, headless cloud run, predictor enabled.

Cloud launch:

```bash
cd /mnt/workspace/Pingpong_TTRL
TARGET=100000 NUM_ENVS=4096 TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python \
  OMNI_KIT_ACCEPT_EULA=YES nohup bash legged_lab/scripts/watchdog_train_a1_real.sh >/dev/null 2>&1 &
```

Local low-env launch:

```bash
cd Pingpong_TTRL
NUM_ENVS=128 nohup bash legged_lab/scripts/watchdog_train_a1_real.sh > /tmp/a1_real_watchdog.log 2>&1 &
```

## Control And Action Path

The control/action pipeline is:

```text
policy raw action
  -> delayed action buffer
  -> clip to [-10, 10]
  -> q_des target = default_joint_pos + 0.25 * clipped_action
  -> deploy-style q_des slew clamp at 50 Hz
  -> identified second-order real-arm response model at physics dt=0.002 s
  -> high-bandwidth ImplicitActuator tracks the filtered target
```

Simulation timing:

- Physics dt: `0.002 s`
- Decimation: `10`
- Policy/control step in `TTEnv`: `50 Hz`
- `num_steps_per_env`: `24`

Raw action scaling:

- `normalization.clip_actions = 10.0`
- `robot.action_scale = 0.25`
- Unclamped target envelope from action alone is `default_joint_pos +/- 2.5 rad`.

## Conservative Exploration Envelope

Current `a1_tt_real` uses a hard q_des slew clamp before the second-order response model:

```python
A1_REAL_DEPLOY_MAX_DELTA_PER_TRAIN_TICK = (
    0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10
)
```

This is per `TTEnv` 50 Hz policy step, so the intended exploration velocity envelope is:

```text
r1-r4: 2.5 rad/s
r5-r7: 5.0 rad/s
```

This is intentionally more conservative than the unloaded real arm node's 100 Hz `max_delta_per_cycle` envelope, because the loaded arm may not reliably achieve the higher speed.

Important: the older constant `A1_DEPLOY_QDES_MAX_DELTA_PER_TICK = (0.020, 0.024, 0.036, 0.032, 0.080, 0.064, 0.160)` is not the active `a1_tt_real` limiter. The active limiter is `A1_REAL_DEPLOY_MAX_DELTA_PER_TRAIN_TICK`.

The reward term `action_target_slew_limit` reads the limiter's excess statistic. A nonzero value in logs confirms the limiter is active.

## Real-Arm Response Model

The real-arm dynamics are represented in `TTEnv`, not by the IsaacLab actuator itself.

Enabled flags:

```python
robot.action_target_rate_limit_enable = True
robot.action_response_model_enable = True
```

The response model is a per-joint identified second-order model with delay, gain, and bias:

```text
joint order: r1, r2, r3, r4, r5, r6, r7
u_mean:
  0.5683523842, -0.6892612481, 0.7196804488, 1.1293556606,
 -1.2407980020, 0.0304735249, 0.7714033876
fn_hz:
  4.9746366278, 3.4161507055, 4.9676413070, 4.3413362835,
 15.3097168220, 8.2237599756, 18.6114086518
zeta:
  0.4941346531, 0.3701409159, 0.4758729367, 0.2298104187,
  0.7941795710, 0.5658324982, 1.3208910166
delay_s:
  0.0020000000, 0.0053529088, 0.0081168432, 0.0180064201,
  0.0175631046, 0.0139939308, 0.0149971247
gain:
  0.9898595326, 0.9609492557, 1.0032869566, 1.0027218617,
  0.9998451720, 1.0020846023, 1.0005302867
bias_rad:
 -0.0151680349, 0.0181176610, -0.0014760354, -0.0315261933,
 -0.0003882480, 0.0010492924, -0.0002311704
```

The model integrates at physics rate (`0.002 s`) inside the decimation loop. Delay is implemented by a physics-step delay buffer.

## Implicit Actuator Used In Sim

`A1_TT_REAL_FITTED_CFG` replaces only the right-arm actuator with a high-bandwidth `ImplicitActuatorCfg`.

This actuator is not intended to model the motor response. It is intentionally stiff so PhysX tracks `action_response_targets`, while the identified response model above provides the motor dynamics.

Right-arm actuator:

```text
joint order: r1-r7
stiffness:
  r1 20000
  r2 30000
  r3 20000
  r4 20000
  r5 20000
  r6 20000
  r7 20000
damping:
  all 100
effort_limit_sim:
  all 1.0e9
velocity_limit_sim:
  all 1.0e9
```

The fitted real robot node gains that produced the response fit were:

```text
kp: 300, 300, 300, 120, 120, 120, 120
kd: 3.5, 3.5, 3.5, 1.0, 1.0, 1.0, 1.0
```

## Robot Initial State

Robot spawn:

```text
position: (-1.8, 0.76, 0.0282)
rotation: identity wxyz (1, 0, 0, 0), robot faces +x toward the table
root link: fixed in articulation
```

Lift height:

```text
r1 centerline target height: 1.15 m
sj = 1.15 - (0.0282 + 1.2107 + 0.025) ~= -0.1139
```

Right-arm ready pose:

```text
r1 0.569
r2 -0.692
r3 0.717
r4 1.13
r5 -1.24
r6 0.0314
r7 0.772
```

Only the right arm is controlled:

```text
num_actions = 7
num_joints = 7
actions/observations joint_names = r1-r7
```

## Task Geometry

Paddle:

```text
paddle body: Link_r_paddle
paddle offset: (0.0, 0.0, 0.085)
paddle_y_offset: -0.66
home_y: 0.76
hit_body_height: 0.028
```

Hit target:

```text
hit_plane_x = -1.60
hit_target_x_range = (-1.60, -1.60)
hit_target_y_range = (0.0, 0.55)
hit_target_z_range = (0.90, 1.25)
```

The learned predictor may output 3 values, but the task projects/uses x as the fixed hit plane anchor; y and z are the meaningful target coordinates.

Serve:

```text
serve_bounce_enable = True
serve_bounce_x_range = (-1.24, -0.96)
serve_bounce_vz_range = (1.60, 2.10)
serve_y_center = 0.12
serve_y_start = 0.04
serve_y_wide = 0.12
serve_curriculum_steps = 0
```

No no-ball/idle curriculum in this run:

```text
no_ball_period_s = 0.0
ball_active_s = 0.0
no_ball_curriculum_steps = 0
idle_reward_ramp_steps = 0
curriculum_phase1_steps = 0
```

## Reset Behavior

`TTEnv.reset()` now explicitly resets the table on every full environment reset:

```text
table_state = table.default_root_state[env_ids]
table_state.position += scene.env_origins[env_ids]
write table pose and velocity back to sim
```

Reset order relevant to this experiment:

```text
scene.reset(env_ids)
reset event manager
reset reward/log buffers
reset command/obs/action/perception buffers
reset episode counters
reset table pose and velocity
reset ball
write scene data and forward sim
reset action-target limiter from current joint positions
reset second-order action-response model from current joint positions
```

This prevents table pose drift after contacts or bad rollouts.

## Domain Randomization And Base

The parked A1 task removes most startup disturbance:

- `fix_root_link=True` in the articulation.
- Base reset is deterministic: x/y/yaw and all root velocity ranges are zero.
- `push_robot = None`.
- `reset_locomotion_joints = None`.
- Right-arm ready-pose jitter only: `reset_manipulation_joints.position_range = (-0.03, 0.03)`.
- Action and perception delay randomization are disabled.
- Observation noise remains enabled.
- Wheel/contact material is deterministic: static friction `4.0`, dynamic friction `3.0`, restitution `0.0`.

## Agent And Predictor

PPO:

```text
seed = 42
device = cuda:0
actor hidden dims = [512, 512, 128]
critic hidden dims = [512, 512, 128]
activation = elu
init_noise_std = 1.0
learning_rate = 5e-4
entropy_coef = 0.006
gamma = 0.95
lambda = 0.95
num_learning_epochs = 5
num_mini_batches = 4
save_interval = 100
max_iterations = 100000
empirical_normalization = True
```

Predictor:

```text
history_len = 5
traj_max_len = 128
hidden_sizes = [64, 64]
lr = 5e-4
epochs_per_update = 1
batch_size = 1024
train_until_iters = 200
```

## Operational Notes

- If reusing `a1_tt_real_v1`, move or delete the existing `logs/a1_tt_real_v1` first if a fresh scratch run is desired; the watchdog auto-resumes from the numerically latest checkpoint under that experiment name.
- A live run should show `Episode_Reward/action_target_slew_limit` nonzero early in training if the q_des limiter is active.
- Current cloud scratch command uses `NUM_ENVS=4096` on L20 and logs to `train_a1_tt_real_v1_watchdog.log`.

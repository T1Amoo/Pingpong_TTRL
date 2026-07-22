# A1 Real Response And High-Stiffness Implicit Actuator Reproduction

This note is the current reproduction reference for the `a1_tt_real` training path. It
separates the measured real-arm response model from the IsaacLab actuator used to track
that response target.

Current default entry point:

```text
task: a1_tt_real
env cfg: A1TableTennisDeployEnvCfg
agent cfg: A1TableTennisDeployAgentCfg
experiment_name: a1_tt_real_v4
run_name: scratch_4096_mb64_sparseboost
```

Relevant code:

```text
legged_lab/envs/__init__.py
  task_registry.register("a1_tt_real", A1TTEnv, A1TableTennisDeployEnvCfg(), A1TableTennisDeployAgentCfg())

legged_lab/envs/a1_tt/a1_tt_config.py
  A1_REAL_FITTED_* constants
  A1TableTennisDeployEnvCfg.__post_init__()
  A1TableTennisDeployAgentCfg

legged_lab/envs/base/tt_env.py
  _init_action_response_model()
  _reset_action_response_model()
  _apply_action_response_model()
  step()

legged_lab/assets/a1/a1.py
  A1_TT_REAL_FITTED_CFG
```

## Action Path

The training action path is:

```text
policy action
  -> action delay buffer
  -> clip to [-10, 10]
  -> q_des_raw = default_joint_pos + 0.25 * clipped_action
  -> deploy-style per-control-step q_des slew clamp at 50 Hz
  -> measured second-order real-arm response model at physics dt = 0.002 s
  -> high-stiffness ImplicitActuatorCfg tracks the filtered q target
```

Timing:

```text
physics dt: 0.002 s
decimation: 10
policy/control rate: 50 Hz
num_actions: 7
joint order: r1, r2, r3, r4, r5, r6, r7
```

Active q_des slew clamp before the response model:

```python
A1_REAL_DEPLOY_MAX_DELTA_PER_TRAIN_TICK = (
    0.05,  # r1, 2.5 rad/s at 50 Hz
    0.05,  # r2, 2.5 rad/s at 50 Hz
    0.05,  # r3, 2.5 rad/s at 50 Hz
    0.10,  # r4, 5.0 rad/s at 50 Hz
    0.10,  # r5, 5.0 rad/s at 50 Hz
    0.10,  # r6, 5.0 rad/s at 50 Hz
    0.10,  # r7, 5.0 rad/s at 50 Hz
)
```

Do not use the older `A1_DEPLOY_QDES_MAX_DELTA_PER_TICK` tuple for this training path.
`a1_tt_real` uses `A1_REAL_DEPLOY_MAX_DELTA_PER_TRAIN_TICK`.

## Second-Order Response Model

The measured motor/arm response is implemented in `TTEnv`, not in the IsaacLab actuator.
The policy still emits a raw position target. After clipping and slew limiting, `TTEnv`
filters that target through the identified closed-loop response.

For each joint:

```text
u(t): raw q_des after slew clamp
u_d(t): delayed u(t - delay_s), with delay_steps = round(delay_s / physics_dt)
x(t): relative second-order state
v(t): relative velocity state
omega = 2*pi*fn_hz

u_rel = u_d - u_mean
v_dot = omega^2 * (u_rel - x) - 2*zeta*omega*v
x_dot = v
q_target = u_mean + bias_rad + gain*x
```

The implementation uses explicit Euler at every physics substep:

```text
v += v_dot * dt
x += v * dt
```

Reset behavior matters for reproduction. On reset, the model is initialized from the
current simulated joint position:

```text
x = (current_q - u_mean - bias_rad) / gain
v = 0
delay buffer = steady raw command
action_response_targets = current_q
```

Active flags in `A1TableTennisDeployEnvCfg`:

```python
self.robot.action_target_rate_limit_enable = True
self.robot.action_target_max_delta_per_tick = A1_REAL_DEPLOY_MAX_DELTA_PER_TRAIN_TICK
self.robot.action_response_model_enable = True
self.robot.action_response_u_mean = A1_REAL_FITTED_U_MEAN
self.robot.action_response_fn_hz = A1_REAL_FITTED_FN_HZ
self.robot.action_response_zeta = A1_REAL_FITTED_ZETA
self.robot.action_response_delay_s = A1_REAL_FITTED_DELAY_S
self.robot.action_response_gain = A1_REAL_FITTED_GAIN
self.robot.action_response_bias_rad = A1_REAL_FITTED_BIAS_RAD
```

### Response Parameters

Joint order is always `r1..r7`.

| joint | node kp | node kd | u_mean | fn_hz | zeta | delay_s | gain | bias_rad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 300.0 | 3.5 | 0.5683523842 | 4.9746366278 | 0.4941346531 | 0.0020000000 | 0.9898595326 | -0.0151680349 |
| r2 | 300.0 | 3.5 | -0.6892612481 | 3.4161507055 | 0.3701409159 | 0.0053529088 | 0.9609492557 | 0.0181176610 |
| r3 | 300.0 | 3.5 | 0.7196804488 | 4.9676413070 | 0.4758729367 | 0.0081168432 | 1.0032869566 | -0.0014760354 |
| r4 | 120.0 | 1.0 | 1.1293556606 | 4.3413362835 | 0.2298104187 | 0.0180064201 | 1.0027218617 | -0.0315261933 |
| r5 | 120.0 | 1.0 | -1.2407980020 | 15.3097168220 | 0.7941795710 | 0.0175631046 | 0.9998451720 | -0.0003882480 |
| r6 | 120.0 | 1.0 | 0.0304735249 | 8.2237599756 | 0.5658324982 | 0.0139939308 | 1.0020846023 | 0.0010492924 |
| r7 | 120.0 | 1.0 | 0.7714033876 | 18.6114086518 | 1.3208910166 | 0.0149971247 | 1.0005302867 | -0.0002311704 |

`node kp/kd` are the real arm-node gains used during the system-ID data collection.
They are not the IsaacLab tracking actuator gains.

## High-Stiffness Implicit Actuator

`A1_TT_REAL_FITTED_CFG` replaces only the right-arm actuator. The actuator is intentionally
stiff so PhysX tracks `action_response_targets`; it is not the real motor response model.

```python
A1_TT_REAL_FITTED_CFG.actuators["right_arm"] = ImplicitActuatorCfg(
    joint_names_expr=A1_RIGHT_ARM_JOINTS,
    effort_limit_sim={
        "r1": 1.0e9, "r2": 1.0e9, "r3": 1.0e9, "r4": 1.0e9,
        "r5": 1.0e9, "r6": 1.0e9, "r7": 1.0e9,
    },
    velocity_limit_sim={
        "r1": 1.0e9, "r2": 1.0e9, "r3": 1.0e9, "r4": 1.0e9,
        "r5": 1.0e9, "r6": 1.0e9, "r7": 1.0e9,
    },
    stiffness={
        "r1": 20000.0, "r2": 30000.0, "r3": 20000.0, "r4": 20000.0,
        "r5": 20000.0, "r6": 20000.0, "r7": 20000.0,
    },
    damping={
        "r1": 100.0, "r2": 100.0, "r3": 100.0, "r4": 100.0,
        "r5": 100.0, "r6": 100.0, "r7": 100.0,
    },
)
```

Important distinction:

```text
real fitted response: TTEnv target-space filter, per-joint fn/zeta/delay/gain/bias
implicit actuator: high-bandwidth PhysX position servo tracking the filtered q target
```

Do not replace this with `IdealPDActuatorCfg` or `DelayedPDActuatorCfg` when reproducing
the current `a1_tt_real` run.

## Training Commands

Cloud scratch run:

```bash
cd /mnt/workspace/Pingpong_TTRL
TARGET=100000 NUM_ENVS=4096 TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python \
  OMNI_KIT_ACCEPT_EULA=YES nohup bash legged_lab/scripts/watchdog_train_a1_real.sh >/dev/null 2>&1 &
```

Local low-env watchdog run:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
NUM_ENVS=128 OMNI_KIT_ACCEPT_EULA=YES \
  nohup bash legged_lab/scripts/watchdog_train_a1_real.sh > /tmp/a1_real_watchdog.log 2>&1 &
```

One-iteration smoke run without watchdog:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
OMNI_KIT_ACCEPT_EULA=YES /home/woan/.conda/envs/pingpong/bin/python -u \
  -m legged_lab.scripts.train \
  --task=a1_tt_real \
  --num_envs=2 \
  --headless \
  --logger=tensorboard \
  --predictor \
  --experiment_name=a1_tt_real_v4_smoke \
  --max_iterations=1
```

Watchdog defaults at the time of this note:

```text
TASK=a1_tt_real
EXP=a1_tt_real_v4
NUM_ENVS=128
TARGET=100000
watchdog log: train_a1_tt_real_v4_watchdog.log
checkpoint root: logs/a1_tt_real_v4
```

If a fresh scratch run is required, archive or remove `logs/a1_tt_real_v4` and
`train_a1_tt_real_v4_watchdog.log` first. The watchdog intentionally resumes from the
latest checkpoint under the experiment directory.

## Verification Commands

The verification script name still contains `idealpd`, but the current reproduction check
uses `--actuator-type implicit`.

Track the fitted 0.1-2 Hz second-order targets:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
OMNI_KIT_ACCEPT_EULA=YES /home/woan/.conda/envs/pingpong/bin/python -u \
  legged_lab/scripts/verify_a1_real_fitted_idealpd.py \
  --headless \
  --actuator-type implicit \
  --ideal-kp-list 20000,30000,20000,20000,20000,20000,20000 \
  --ideal-kd 100 \
  --ideal-effort 1e9 \
  --ideal-velocity 1e9 \
  --max-rmse-rad 0.006 \
  --max-abs-rad 0.02 \
  --output-dir ../系统辨识/reports/a1_real_response_actuator_repro/implicit_fit_0p1_2hz
```

Stress the high-stiffness implicit actuator with a generated 2-5 Hz chirp:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
OMNI_KIT_ACCEPT_EULA=YES /home/woan/.conda/envs/pingpong/bin/python -u \
  legged_lab/scripts/verify_a1_real_fitted_idealpd.py \
  --headless \
  --actuator-type implicit \
  --generated-chirp \
  --chirp-start-hz 2 \
  --chirp-end-hz 5 \
  --chirp-duration-s 20 \
  --chirp-amplitude-rad 0.08 \
  --chirp-ramp-s 1.0 \
  --ideal-kp-list 20000,30000,20000,20000,20000,20000,20000 \
  --ideal-kd 100 \
  --ideal-effort 1e9 \
  --ideal-velocity 1e9 \
  --ignore-s 1.0 \
  --max-rmse-rad 0.02 \
  --max-abs-rad 0.08 \
  --output-dir ../系统辨识/reports/a1_real_response_actuator_repro/implicit_generated_chirp_2_5hz
```

Previous validation artifacts with the same actuator settings are under:

```text
../系统辨识/reports/a1_tt_real_v1_implicit_verify_per_joint_kp_20_30_20_kd100_unlimited_alljoints
../系统辨识/reports/a1_tt_real_v1_implicit_verify_generated_chirp_2-5hz_amp0.08_alljoints
```

The old directory names contain `v1`, but the actuator parameter set is the same
high-stiffness implicit tracking actuator documented above.

Observed validation summary from the generated 2-5 Hz chirp:

```text
all seven joints passed
4-5 Hz larger joints: about 8-10 ms lag, about 5-7% amplitude attenuation
4-5 Hz smaller joints: about 4 ms lag, near-unity amplitude ratio
no divergence observed
```

## Minimal Checklist For Reproduction

1. Confirm `task_registry` maps `a1_tt_real` to `A1TableTennisDeployEnvCfg`.
2. Confirm `A1TableTennisDeployEnvCfg` sets `self.scene.robot = A1_TT_REAL_FITTED_CFG`.
3. Confirm `action_target_rate_limit_enable` and `action_response_model_enable` are both true.
4. Confirm the seven response parameter tuples match the table above.
5. Confirm `A1_TT_REAL_FITTED_CFG.actuators["right_arm"]` is `ImplicitActuatorCfg` with `kp=[20000,30000,20000,20000,20000,20000,20000]`, `kd=100`, `effort_limit_sim=1e9`, and `velocity_limit_sim=1e9`.
6. Run the implicit actuator verification before launching a long training run.
7. For scratch training, archive the old `logs/a1_tt_real_v4` directory so the watchdog does not auto-resume.

# A1/H1 Table-Tennis Sim2sim

This folder is the sim2sim half of the custom A1/H1 ping-pong deployment path.
It uses the same DAMIAO MIT-control shape that `h1_pingpong/src/armcontrol`
uses on hardware:

```cpp
control_mit(motor, kp, kd, q_des, dq_des, tau_ff)
```

The default MuJoCo runner uses a stable position-servo actuator for visual
sim2sim, with the same action target and joint ranges as the training config.
Its velocity cap is calibrated from an Isaac play trace under the real 28/8 Nm
effort limits; the training velocity limits are hard caps, not the speed the
Isaac implicit actuator actually reaches. The servo also uses a first-order
tracking time constant (`--servo-tau`, default `0.25 s`) so it does not move at
the cap whenever the target changes. The equivalent MIT torque is still
estimated and logged:

```text
tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
```

Use `--dynamic-pd` to test the explicit MIT-PD torque path. The runner now
builds a generated MJCF scene from the active profile instead of using the raw
URDF directly. The MJCF keeps the URDF geometry/inertias, and adds
MuJoCo-only metadata that URDF did not carry: joint armature, damping,
frictionloss, DAMIAO actuator force ranges, motor actuators, sensors, and
table-tennis contact pairs. Those values are still initial estimates until
hardware logs are used to calibrate them.

## Run

Headless smoke test:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda run -n g1tt_sim2sim python sim2sim/build_a1_mjcf.py
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py --headless-steps 200
```

Viewer:

```bash
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py
```

Current low-arm `a1_tt_backhand_real_v7` candidate, visual sanity mode:

```bash
A1_SIM2SIM_PROFILE=v1_3_backhand_low_arm_2100 \
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py \
  --policy /home/woan/桌面/isolated_worktrees/Pingpong_TTRL_mentor_damiao_v7/logs/a1_tt_backhand_real_v7_low_arm_ready_push_run1/2026-07-30_14-56-45_scratch_backhand_damiao_mit_3k_easy_contact_curriculum/exported/policy.onnx
```

Current low-arm candidate, training/real-response matched mode:

```bash
A1_SIM2SIM_PROFILE=v1_3_backhand_low_arm_2100 \
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py \
  --policy /home/woan/桌面/isolated_worktrees/Pingpong_TTRL_mentor_damiao_v7/logs/a1_tt_backhand_real_v7_low_arm_ready_push_run1/2026-07-30_14-56-45_scratch_backhand_damiao_mit_3k_easy_contact_curriculum/exported/policy.onnx \
  --actuator-mode damiao_mit \
  --real-response-model
```

Dead/out balls are re-served on the next 50 Hz control tick by default
(`--serve-pause-steps 1`). This matches the training setup more closely than
leaving the policy in a long no-ball window. Increase `--serve-pause-steps`
only when you explicitly want to inspect post-hit recovery without a new ball.

Explicit torque diagnostic:

```bash
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py --dynamic-pd --diag-every 20
```

Fast servo diagnostic using the training hard velocity limits:

```bash
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py --fast-servo --diag-every 20
```

Zero-action scene test:

```bash
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py --headless-steps 50 --no-policy
```

Asset/profile selection is controlled at process start with
`A1_SIM2SIM_PROFILE` because `policy_io.py` freezes the observation constants
when it is imported:

```bash
# Old reference path for 10999 / old-URDF videos.
A1_SIM2SIM_PROFILE=model_10999 \
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py

# Current V1_3 backhand training profile.
A1_SIM2SIM_PROFILE=v1_3_backhand \
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py

# Current low-arm V1_3 candidate trained from 2026-07-30 model_2100.
A1_SIM2SIM_PROFILE=v1_3_backhand_low_arm_2100 \
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py

# Mentor isolation test: current backhand policy geometry, but old V1_1 URDF.
A1_SIM2SIM_PROFILE=old_v1_1_backhand \
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py
```

## Defaults

- Scene MJCF: `sim2sim/scene/a1_tt_scene.xml`
- Source URDF: `/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/legged_lab/assets/a1/X1_URDF_V1_1/urdf/X1_URDF_V1_1.urdf`
- Policy ONNX: `/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_v13/2026-07-08_12-40-15/exported/policy.onnx`
- Control: 50 Hz policy, 500 Hz MuJoCo physics
- Action: `q_des = default_q + clip(raw_action, +/-10) * 0.25`
- Training torque limits: r1-r3 `28 Nm`, r4-r7 `8 Nm`
- MJCF DAMIAO actuator ranges: r1-r3 `28 Nm`, r4-r7 `8 Nm`

If `predictor.onnx` exists next to `policy.onnx`, the runner uses it for the
5-frame ball prediction path and falls back to the analytic hit-plane predictor
when the predictor output is outside the training gate.

`--actuator-mode isaac_approx` is the default visual sanity mode. It is useful
for checking scene/profile/policy geometry, but it is not the 0729 real motor
fit. For the current DamiaoMIT-trained policy, use `--actuator-mode damiao_mit`
with `--real-response-model` when judging sim2sim against the training actuator
and real-response chain.

## Ball Gate And Hit-Point Visuals

The runner uses a hard ball-validity gate before building the policy
observation. When the gate is invalid, the actor's ball slot and hit prediction
are both replaced with the profile's fixed home sentinel. For the current
`v1_3_backhand` profile this is the training target center:

```text
(-1.159, 0.0675, 1.12)
```

The gate rejects balls that are behind the hit plane, moving away, below table
height, double-bounced, already hit, outside the play volume, too fast, or
predicted to miss the robot half on first table contact. `--gate-confirm-frames`
and `--gate-coast-frames` can add hysteresis for noisy perception; defaults are
`1/1` to match the current deterministic sim/training mask.

In viewer mode the predicted hit point is drawn as a marker. Green means the
gate is engaged and the policy sees the live ball; gray means invalid and the
marker is the home sentinel. Disable it with `--no-hit-viz`.

## Sim2real boundary

The reusable boundary is:

```text
policy obs -> policy.onnx -> raw action -> q_des/kp/kd/dq_des/tau_ff
```

Sim2sim consumes this command by writing MuJoCo joint torques. Sim2real should
consume the same command by sending DAMIAO MIT commands:

```cpp
control_mit(motor, kp, kd, q_des, dq_des, tau_ff)
```

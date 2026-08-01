# A1 backhand 9700 unified deployment contract

This file is the reviewed, table-frame contract for the frozen mentor policy.
Values below agree with the delivered `training_params/env.yaml`, the matching
MuJoCo profile, and the exported ONNX interfaces.

## Coordinate frame and geometry

- World/table frame: table center is `(0, 0, 0)`, table top is `z=0.76 m`.
- Robot base pose: position `(-1.8, 0.0, 0.0282) m`, quaternion
  `(w, x, y, z)=(1, 0, 0, 0)`. The root is fixed in training.
- Robot asset: `X1_URDF_V1_3/urdf/X1_URDF_V1_3.urdf` (matching paddle USD in
  IsaacLab).
- Hit plane: `x=-1.243 m`, which is `0.557 m` forward of the base origin and
  `0.127 m` inside the robot-side table edge (`x=-1.37 m`).
- Actor-facing target box: `y=[-0.025, 0.107] m`, `z=[0.86, 0.98] m`.
- No-ball/predictor sentinel: `(-1.243, 0.041, 0.92) m`.
- Ready-pose paddle center in the current MuJoCo scene is approximately
  `(-1.441, 0.039, 1.059) m`. The nominal target center is therefore about
  `0.198 m` forward and `0.139 m` lower than the ready paddle center.

## Joint order and ready pose

All arrays use the strict order `r1, r2, r3, r4, r5, r6, r7` and radians.

| Joint | Ready q | URDF hard range | Motor family |
|---|---:|---:|---|
| r1 | 1.450 | [-1.05, 3.14] | DM4340_48V |
| r2 | -0.762 | [-3.14, 0.262] | DM4340_48V |
| r3 | -2.050 | [-2.76, 2.76] | DM4340_48V |
| r4 | 1.445 | [-1.92, 1.92] | DM4310_48V |
| r5 | 0.206 | [-2.76, 2.76] | DM4310_48V |
| r6 | -0.827 | [-1.57, 1.57] | DM4310_48V |
| r7 | 1.043 | [-2.76, 2.76] | DM4310_48V |

The policy output is an offset around this ready pose:

```text
q_des = ready_q + clip(action, -10, 10) * 0.25
```

The result must then be clipped to 95% soft joint limits. Training reset jitter
was only `+/-0.03 rad`; that is not a replacement for a safe MoveJ transition
from the measured real-arm state.

## Timing, observations, and predictor

- Physics/actuator step: `0.002 s` (500 Hz).
- Policy decimation: `10`; policy/control observation rate is 50 Hz.
- Actor interface: `195 -> 7`, comprising five frames of 39 observations.
- Predictor interface: 5 ball-position samples (`15 -> 3`).
- Actor normalization is embedded in the exported policy; external observation
  clipping is `+/-100`.
- Invalid-ball action is zero, which means return/hold the ready pose. Keep
  `predictor.onnx` beside `policy.onnx`.

## Ball and serve domain

- Launch point: `(1.35, 0.0, 1.03) m`.
- First-bounce x: `[-1.053, -0.773] m`.
- First-bounce y: center `0.041 m`, half width `0.020 m`.
- Post-bounce vertical speed: `[0.0, 0.45] m/s`.
- Ball radius/mass: `0.02 m / 0.0034 kg`; the sampler includes quadratic air
  drag matching the delivered sim2sim implementation.
- Viewer pause is 100 policy ticks (about 2 s). Training-style reset uses one
  tick, so the pause changes presentation only, not policy geometry.

## Required actuator chain

Use `damiao_mit` together with the fitted real-response model. Per-joint arrays
remain in `r1..r7` order:

```text
KP       = [300, 300, 300, 120, 120, 120, 60]
KD       = [3.5, 3.5, 3.5, 1.0, 1.0, 1.0, 0.5]
effort   = [28, 28, 28, 8, 8, 8, 8] Nm
velocity = [4, 4, 5, 6, 8, 6, 9] rad/s
50Hz q_des delta limit = [0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10] rad
```

The fitted response constants (`fn_hz`, damping ratio, delay, linear gain,
intercept and `u_mean`) are preserved verbatim in `training_params/env.yaml`
and the namespaced `policy_io.py`. Their fitted center is near the older
backhand acquisition pose (`r1~=1.77`, `r3~=-1.86`); runtime reset performs the
required backsolve. Do not use `u_mean` as the policy ready pose.

## Real-robot SDK contract

Real deployment is a direct DAMIAO SDK path, not the workspace's RBDL-based
gravity-feedforward controller. The control call must use:

```text
control_mit(motor, KP[i], KD[i], q_des[i], 0.0, 0.0)
```

That is, desired velocity and feedforward torque are both zero. Therefore the
URDF is a training/sim2sim geometry asset and is not used to compute the real
motor command. In particular, do not add model-based gravity compensation: the
delivered training-matched `damiao_mit` path also applies no `qfrc_bias` or
other torque feedforward.

The SDK deployment still has to preserve the strict `r1..r7` motor mapping,
50 Hz policy updates, 500 Hz command holding/slew, ready pose, SDK gains,
velocity/effort envelope, table-frame observation transform, ball-validity
gate, predictor history, and hit-plane geometry recorded above. Before enabling
torque, run a servo-disabled observation/action trace and a low-speed MoveJ to
the ready pose.

The current real bridge shapes the raw 50 Hz policy target with the first-order
command filter below before publishing to the SDK:

```text
tau_s = [0.10, 0.10, 0.08, 0.10, 0.05, 0.05, 0.10] s
velocity_limit = [1.0, 1.2, 1.8, 1.6, 4.0, 3.2, 8.0] rad/s
```

The hard `qdes_slew/max_delta_per_tick` branch is disabled. The SDK node keeps
only a higher per-cycle discontinuity guard; under the velocity limits above it
does not shape the normal trajectory.

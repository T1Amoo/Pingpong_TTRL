# A1 backhand hitplane020 deployment package

This package is for the current 9700 checkpoint deployment/sim2sim check.

## Contents

- `policy/policy.onnx`: PPO policy exported from `model_9700.pt`.
- `policy/predictor.onnx`: learned ball predictor. Keep it in the same directory as `policy.onnx`.
- `policy/model_9700.pt`: original IsaacLab/RSL checkpoint for traceability.
- `training_params/env.yaml`, `training_params/agent.yaml`: training config snapshot.
- `sim2sim_files/`: current MuJoCo sim2sim files used on this machine.
- `patches/current_sim2sim_changes.diff`: patch against the local sim2sim repo for review/application.

## Policy identity

- Training repo: `Pingpong_TTRL_mentor_damiao_v7`
- Experiment: `a1_tt_backhand_real_v7_hitplane020_scratch_run1`
- Run: `2026-07-31_19-13-02_scratch_backhand_damiao_mit_success_first_slow_curriculum`
- Checkpoint: `model_9700.pt`
- Exported files: `policy.onnx` + `predictor.onnx`

## Required sim2sim profile

Use this profile for the 9700 policy:

```bash
A1_SIM2SIM_PROFILE=v1_3_backhand_low_arm_hitplane020
```

Do not use `v1_3_backhand_low_arm_2100` for this checkpoint. That older profile uses the old hit plane.

Important profile values:

```text
URDF: X1_URDF_V1_3/urdf/X1_URDF_V1_3.urdf
ready q: r1=1.450, r2=-0.762, r3=-2.050, r4=1.445, r5=0.206, r6=-0.827, r7=1.043
hit_plane_x: -1.243
pred_sentinel: [-1.243, 0.041, 0.92]
serve_bounce_x_range: [-1.053, -0.773]
serve_bounce_vz_range: [0.0, 0.45]
serve_y_center: 0.041
serve_y_half: 0.02
```

## MuJoCo viewer command

From the sim2sim repo root:

```bash
A1_SIM2SIM_PROFILE=v1_3_backhand_low_arm_hitplane020 \
/home/woan/miniforge3/envs/g1tt_sim2sim/bin/python \
  deploy/robots/a1_h1/sim2sim/run_a1_tt_sim2sim.py \
  --policy /path/to/this_package/policy/policy.onnx \
  --actuator-mode damiao_mit \
  --real-response-model \
  --serve-pause-steps 100 \
  --diag-every 50
```

`--serve-pause-steps 100` means about 2 seconds between a dead/parked ball and the next serve at 50 Hz control rate. For training-style immediate reset, use `--serve-pause-steps 1`.

## Notes

- `predictor.onnx` is loaded automatically from the same directory as `policy.onnx`.
- The deployment check should use `--actuator-mode damiao_mit --real-response-model`.
- The copied sim2sim files include the current ball gate, serve sampler, predictor observation path, Damiao MIT actuator path, and hitplane020 profile.

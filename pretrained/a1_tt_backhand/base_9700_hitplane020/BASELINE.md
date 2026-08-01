# a1_tt_backhand base 9700 hitplane020

This directory is the frozen import of
`a1_backhand_hitplane020_9700_deploy_for_mentor_20260801.tar.gz`.

## Identity

- Source training repo: `Pingpong_TTRL_mentor_damiao_v7`
- Experiment: `a1_tt_backhand_real_v7_hitplane020_scratch_run1`
- Run: `2026-07-31_19-13-02_scratch_backhand_damiao_mit_success_first_slow_curriculum`
- Checkpoint: `policy/model_9700.pt`
- Actor interface: 195 observations to 7 actions
- Predictor interface: 15 history values to 3D hit prediction

## Frozen task contract

- Robot asset: `X1_URDF_V1_3/urdf/X1_URDF_V1_3.urdf`
- Base pose: `[-1.8, 0.0, 0.0282]`
- Ready joints r1-r7: `[1.450, -0.762, -2.050, 1.445, 0.206, -0.827, 1.043]`
- Hit plane x: `-1.243`
- Hit target y: `[-0.025, 0.107]`
- Hit target z: `[0.86, 0.98]`
- Predictor sentinel: `[-1.243, 0.041, 0.92]`
- Serve bounce x: `[-1.053, -0.773]`
- Serve bounce vz: `[0.0, 0.45]`
- Serve y center/half-width: `0.041 / 0.02`
- Required sim2sim actuator path: `damiao_mit` with the fitted real-response model

`training_params/env.yaml` and `training_params/agent.yaml` are the
authoritative training snapshots. The delivery package does not contain the
mentor training source changes for all custom reward functions, so this import
is an exact policy/deployment baseline, not yet a claim that the current
IsaacLab task can resume training bit-for-bit.

See `DEPLOY_CONTRACT.md` for the unified base/joint/hit-plane/actuator analysis
and the explicit real-robot integration gate.

## Integrity

The original package contents are retained in this directory. `SHA256SUMS`
validates the delivered files. The source archive SHA256 is recorded in
`SOURCE_ARCHIVE.sha256`.

Use the matching launcher:

```bash
unitree_rl_lab/deploy/robots/a1_h1/a1_tt_backhand/run_sim2sim.sh
```

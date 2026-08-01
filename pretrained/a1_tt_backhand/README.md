# A1 TT Backhand Pretrained Baselines

This directory stores immutable, externally trained A1 backhand baselines.
Do not overwrite an existing baseline when importing a newer checkpoint; add a
new versioned directory instead.

Current baseline:

- `base_9700_hitplane020/`: mentor-provided V1.3 low-arm backhand policy,
  checkpoint `model_9700.pt`, imported on 2026-08-01.

The matching MuJoCo entry point lives in
`unitree_rl_lab/deploy/robots/a1_h1/a1_tt_backhand/`.

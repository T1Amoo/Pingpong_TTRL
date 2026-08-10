# X1_URDF_V2

This asset is an isolated derivative of `X1_URDF_V1_3` for the A1 backhand-v7
training task. It keeps `sj` fixed and lowers only its baked URDF origin by
`0.07 m`:

```text
base_z 0.0282 + sj_origin_z 1.02680245585163 + r0_origin_z 0.025
= r1 centerline_z 1.08000245585163 m
```

The original `X1_URDF_V1_3` asset remains the independent 1.15 m version. The
`meshes` symlink intentionally reuses its unchanged CAD meshes; the generated
`X1_URDF_V2_paddle.usd` and configuration layers live entirely in this
directory.

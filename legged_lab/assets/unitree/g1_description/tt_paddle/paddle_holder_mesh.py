#!/usr/bin/env python3
"""
paddle_holder_mesh.py -- OPTIONAL. Generate a TRUE hollow U-slot holder mesh
with drilled screw holes (boolean difference), for when the 3-box primitive
approximation in the URDF is not accurate enough.

Requires trimesh + a boolean backend (manifold3d or blender):
    pip install trimesh manifold3d

Outputs (into ../meshes/, so the URDF "meshes/..." path resolves):
    paddle_holder_visual.stl     hollow U-channel with screw holes
    paddle_holder_collision.stl  simplified solid outer box (stable in sim)

The mesh is built in the SAME local frame as right_tt_paddle_holder_link:
  origin at the holder proximal face, slot cavity centered on Y=0 / Z=0,
  slot runs along +X, U opens toward +Z.

To use it, replace the three <box> visuals (and optionally the box collision)
of the holder link with:
    <visual>  <geometry><mesh filename="meshes/paddle_holder_visual.stl"/></geometry></visual>
    <collision><geometry><mesh filename="meshes/paddle_holder_collision.stl"/></geometry></collision>
keeping <origin xyz="0 0 0" rpy="0 0 0"/> (the mesh already carries the offset).
"""
import os
import trimesh
import numpy as np

# --- params: keep in sync with build_tt_urdf.py / the xacro ---
slot_length = 0.075
slot_width = 0.030
slot_height = 0.016
wall_thickness = 0.004
screw_diameter = 0.004
screw_x = [0.025, 0.055]

EPS = 1e-3  # noqa: F841 (kept for tweaking boolean breach margins)
outer_y = slot_width + 2 * wall_thickness
outer_z = slot_height + wall_thickness
center_z = -wall_thickness / 2.0  # outer box center (top open, bottom walled)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meshes")


def boxc(size, center):
    b = trimesh.creation.box(extents=size)
    b.apply_translation(center)
    return b


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # solid outer envelope
    outer = boxc([slot_length, outer_y, outer_z], [slot_length / 2.0, 0.0, center_z])

    # cavity: open on +-X (insertion) and +Z (top) -> leaves bottom + 2 side walls.
    # Sized to breach both X ends and the top; bottom sits at z = -slot_height/2.
    cavity = boxc([slot_length + 0.04, slot_width, slot_height + 0.02],
                  [slot_length / 2.0, 0.0, (slot_height + 0.02) / 2.0 - slot_height / 2.0])

    holder = outer.difference(cavity)

    # drill screw holes along +-Y
    for sx in screw_x:
        cyl = trimesh.creation.cylinder(radius=screw_diameter / 2.0, height=outer_y + 0.02)
        # cylinder default axis = Z; rotate to Y
        cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        cyl.apply_translation([sx, 0.0, 0.0])
        holder = holder.difference(cyl)

    holder.export(os.path.join(OUT_DIR, "paddle_holder_visual.stl"))
    # simplified collision: solid outer box (convex, cheap, stable)
    outer.export(os.path.join(OUT_DIR, "paddle_holder_collision.stl"))
    print("wrote:")
    print("  ", os.path.join(OUT_DIR, "paddle_holder_visual.stl"))
    print("  ", os.path.join(OUT_DIR, "paddle_holder_collision.stl"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_review_stl.py -- export the paddle end-effector for visual review
(numpy-only). The CONNECTOR (mount + extension + U-holder + screw pins) is drawn
from primitives; the PADDLE is the real imported mesh (../meshes/tt_paddle.stl),
transformed with the SAME placement as g1_23dof_tt_paddle.urdf.

Outputs into this folder:
  mechanism.stl  连接机构 = adapter mount + extension tube + holder
  paddle.stl     球拍     = real paddle STL, rotated/scaled/seated into the slot
  assembly.stl   两者合并 (open this to check fit)

Frame: +X flange->paddle, +Z blade face normal.
"""
import os
import struct
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- params (mirror build_tt_urdf.py PARAMS) ----
mount_r, mount_L = 0.025, 0.012
ext_r, ext_L = 0.018, 0.16
slot_len, slot_w, slot_h, wall = 0.075, 0.032, 0.022, 0.004
screw_d, screw_x = 0.004, [0.025, 0.055]

# cumulative +X origins (mount_xyz = 0)
x_mount = 0.0
x_ext = mount_L                      # 0.012
x_holder = x_ext + ext_L             # 0.172

# real paddle mesh placement (mirror build_tt_urdf.py PARAMS)
PADDLE_SRC = os.path.join(os.path.dirname(OUT), "meshes", "tt_paddle.stl")
paddle_scale = 0.001
paddle_y = -0.00351
paddle_z = -0.00149

outer_y = slot_w + 2 * wall
side_y = slot_w / 2 + wall / 2
bottom_z = -(slot_h / 2 + wall / 2)


def box_tris(size, center):
    sx, sy, sz = (s / 2 for s in size)
    cx, cy, cz = center
    v = np.array([[cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
                  [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
                  [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
                  [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz]])
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
             (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    return [v[list(f)] for f in faces]


def cyl_tris(radius, length, center, axis="z", seg=48):
    """cylinder centered at origin along `axis`, then translated to center."""
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    top = np.stack([radius * np.cos(th), radius * np.sin(th),
                    np.full(seg, length / 2)], axis=1)
    bot = np.stack([radius * np.cos(th), radius * np.sin(th),
                    np.full(seg, -length / 2)], axis=1)
    if axis == "x":
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]])
    elif axis == "y":
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    else:
        R = np.eye(3)
    top = top @ R.T + center
    bot = bot @ R.T + center
    ctop = np.array(center) + R @ np.array([0, 0, length / 2.0])
    cbot = np.array(center) + R @ np.array([0, 0, -length / 2.0])
    tris = []
    for i in range(seg):
        j = (i + 1) % seg
        tris.append(np.array([top[i], top[j], ctop]))      # top cap
        tris.append(np.array([bot[j], bot[i], cbot]))      # bottom cap
        tris.append(np.array([top[i], bot[i], bot[j]]))    # side
        tris.append(np.array([top[i], bot[j], top[j]]))
    return tris


def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            ln = np.linalg.norm(n)
            n = n / ln if ln > 1e-12 else np.zeros(3)
            f.write(struct.pack("<3f", *n))
            for v in t:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def mechanism():
    tris = []
    tris += cyl_tris(mount_r, mount_L, [x_mount + mount_L / 2, 0, 0], "x")
    tris += cyl_tris(ext_r, ext_L, [x_ext + ext_L / 2, 0, 0], "x")
    # holder U-channel
    tris += box_tris([slot_len, outer_y, wall], [x_holder + slot_len / 2, 0, bottom_z])
    tris += box_tris([slot_len, wall, slot_h], [x_holder + slot_len / 2, side_y, 0])
    tris += box_tris([slot_len, wall, slot_h], [x_holder + slot_len / 2, -side_y, 0])
    # screw pins
    for sx in screw_x:
        tris += cyl_tris(screw_d / 2, outer_y + 0.002, [x_holder + sx, 0, 0], "y")
    return tris


def paddle():
    """Load the real paddle STL (mm, ASCII), apply the SAME transform as the
    URDF (scale, rotate -90deg about Z so handle->blade runs +X, recenter on the
    slot axis), and seat it at the holder (x_holder)."""
    vs = [[float(x) for x in l.split()[1:4]]
          for l in open(PADDLE_SRC) if l.split()[:1] == ['vertex']]
    v = np.array(vs).reshape(-1, 3, 3) * paddle_scale     # mm -> m, grouped per tri
    # rotate -90deg about Z: (x,y,z) -> (y, -x, z)
    rot = np.empty_like(v)
    rot[..., 0] = v[..., 1]
    rot[..., 1] = -v[..., 0]
    rot[..., 2] = v[..., 2]
    rot += np.array([x_holder, paddle_y, paddle_z])       # recenter + seat
    return [t for t in rot]


def main():
    m, p = mechanism(), paddle()
    write_stl(os.path.join(OUT, "mechanism.stl"), m)
    write_stl(os.path.join(OUT, "paddle.stl"), p)
    write_stl(os.path.join(OUT, "assembly.stl"), m + p)
    print("wrote mechanism.stl (%d tris), paddle.stl (%d tris), assembly.stl (%d tris)"
          % (len(m), len(p), len(m) + len(p)))
    print("blade face center (contact) at x=%.3f m, +Z normal" % (x_holder + 0.165 + 0.0025))


if __name__ == "__main__":
    main()

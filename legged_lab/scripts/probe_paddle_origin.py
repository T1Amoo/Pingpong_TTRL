"""Probe paddle mesh points to find blade center offset from body origin."""
import sys
import os

_pxr_base = (
    "/home/woan/.conda/envs/pingpong/lib/python3.10/site-packages/"
    "isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310"
)
sys.path.insert(0, _pxr_base)

USD_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "assets", "a1", "X1_URDF_V1_1", "X1_URDF_V1_1.usd",
)

from pxr import Usd, UsdGeom, Gf

stage = Usd.Stage.Open(USD_PATH)
PADDLE_PATH = "/X1_URDF_V1_1/Link_r_paddle"

# visual_0 has scale 0.001 (STL in mm -> m)
scale = 0.001

print("=" * 70)
print("Paddle mesh (visual) points bbox:")
mesh_prim = stage.GetPrimAtPath(f"{PADDLE_PATH}/visuals/visual_0/child_0")
pts_attr = mesh_prim.GetAttribute("points")
if pts_attr:
    pts = pts_attr.Get()
    if pts:
        xs = [p[0] * scale for p in pts]
        ys = [p[1] * scale for p in pts]
        zs = [p[2] * scale for p in pts]
        print(f"  num points: {len(pts)}")
        print(f"  x range: {min(xs):.4f} .. {max(xs):.4f}  span={max(xs)-min(xs):.4f}")
        print(f"  y range: {min(ys):.4f} .. {max(ys):.4f}  span={max(ys)-min(ys):.4f}")
        print(f"  z range: {min(zs):.4f} .. {max(zs):.4f}  span={max(zs)-min(zs):.4f}")
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        cz = (min(zs) + max(zs)) / 2
        print(f"  BLADE BBOX CENTER (local to paddle body) = ({cx:.4f}, {cy:.4f}, {cz:.4f})")
    else:
        print("  no points data")
else:
    print("  no points attribute")

# Also check collision mesh
print("\nPaddle collision mesh points bbox:")
col_prim = stage.GetPrimAtPath(f"{PADDLE_PATH}/collisions/collision_0/child_0")
pts_attr2 = col_prim.GetAttribute("points")
if pts_attr2:
    pts2 = pts_attr2.Get()
    if pts2:
        xs2 = [p[0] * scale for p in pts2]
        ys2 = [p[1] * scale for p in pts2]
        zs2 = [p[2] * scale for p in pts2]
        print(f"  num points: {len(pts2)}")
        print(f"  x range: {min(xs2):.4f} .. {max(xs2):.4f}  span={max(xs2)-min(xs2):.4f}")
        print(f"  y range: {min(ys2):.4f} .. {max(ys2):.4f}  span={max(ys2)-min(ys2):.4f}")
        print(f"  z range: {min(zs2):.4f} .. {max(zs2):.4f}  span={max(zs2)-min(zs2):.4f}")
        cx2 = (min(xs2) + max(xs2)) / 2
        cy2 = (min(ys2) + max(ys2)) / 2
        cz2 = (min(zs2) + max(zs2)) / 2
        print(f"  COLLISION CENTER (local to paddle body) = ({cx2:.4f}, {cy2:.4f}, {cz2:.4f})")
    else:
        print("  no points data")
else:
    print("  no collision points attribute")

print("\nJoint r_paddle:")
jt_prim = stage.GetPrimAtPath("/X1_URDF_V1_1/joints/r_paddle")
for attr in jt_prim.GetAttributes():
    v = attr.Get()
    if v is not None and "pos" in attr.GetName().lower():
        print(f"  {attr.GetName()} = {v}")

print("\n")
print("SUMMARY:")
print("  Link_r_paddle is fixed to Link_r7 by r_paddle.")
print("  r_paddle: parent(Link_r7) localPos0=(0,0,0.172), rpy=(pi,0,0)")
print("  localPos1=(0,0,0) => paddle BODY ORIGIN = joint child anchor")
print("  Paddle mesh center relative to body origin comes from the bbox above.")

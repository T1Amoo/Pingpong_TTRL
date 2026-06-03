#!/usr/bin/env python3
"""
build_tt_urdf.py  -- generate the table-tennis-paddle G1 23DoF URDFs.

Pure-stdlib (xml.etree only); no xacro / ROS required. Produces, next to the
base g1_23dof.urdf:

  g1_23dof_tt.urdf         base with the right rubber-hand visual/collision
                           removed and its inertia zeroed (wrist_roll DoF kept)
  g1_23dof_tt_paddle.urdf  the above + the fixed paddle end-effector

The paddle block is the DEFAULT expansion of g1_tt_paddle_adapter.xacro
(prefix=right_, parent_link=right_wrist_roll_rubber_hand). Inertias are computed
from the same closed-form formulas as the xacro, so the two stay in sync. To
change geometry, edit PARAMS below (or edit the xacro and re-expand with the
`xacro` tool); then re-run this script.

Run:  python3 build_tt_urdf.py
"""
import os
import xml.etree.ElementTree as ET

# script lives in g1_description/tt_paddle/ ; URDFs live one level up in
# g1_description/ so the base "meshes/..." relative paths resolve unchanged.
HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.dirname(HERE)
BASE = os.path.join(ASSET_DIR, "g1_23dof_rev_1_0.urdf")
OUT_TT = os.path.join(ASSET_DIR, "g1_23dof_tt.urdf")
OUT_PADDLE = os.path.join(ASSET_DIR, "g1_23dof_tt_paddle.urdf")

PREFIX = "right_"
PARENT_LINK = "right_wrist_roll_rubber_hand"

# ----- parameters (mirror g1_tt_paddle_adapter.xacro defaults) -----
P = dict(
    mount_xyz="0 0 0", mount_rpy="0 0 0",
    mount_radius=0.025, mount_length=0.012, mount_mass=0.03,
    extension_length=0.16, extension_radius=0.018, extension_mass=0.08,
    # slot enlarged to fit the real paddle handle (~30.8 x 20 mm) + clearance
    slot_length=0.075, slot_width=0.032, slot_height=0.022,
    wall_thickness=0.004, holder_mass=0.05,
    screw_diameter=0.004, screw_x1=0.025, screw_x2=0.055,
    # ---- real paddle MESH (replaces the old handle+blade primitives) ----
    # source STL is in millimetres; rotate so its long (handle->blade) axis
    # maps to connector +X and its face-normal maps to +Z; recenter the handle
    # cross-section on the slot axis (Y,Z=0). MEASURE-ME if you swap the STL.
    paddle_mesh="meshes/tt_paddle.stl",
    paddle_scale=0.001,
    paddle_rpy="0 0 -1.5707963",   # mesh +Y -> connector +X, +Z stays normal
    paddle_y=-0.00351,             # recenter handle laterally on slot axis
    paddle_z=-0.00149,             # recenter handle thickness on slot axis
    insertion_depth=0.075,         # handle tip seats to slot bottom
    # simplified collision + inertia stand-ins (the mesh itself is visual only)
    handle_col_len=0.075, handle_col_w=0.032, handle_col_h=0.022, handle_mass=0.04,
    blade_radius=0.075, blade_thickness=0.005, blade_mass=0.13,
    blade_center_x=0.165,          # blade face center along +X from slot bottom
)

PI_2 = 1.5707963267948966


# ----- inertia helpers (closed form, principal axes) -----
def cyl_x(m, r, L):
    """solid cylinder, symmetry axis == X -> (ixx, iyy, izz)"""
    return 0.5 * m * r**2, m * (3 * r**2 + L**2) / 12, m * (3 * r**2 + L**2) / 12


def cyl_z(m, r, L):
    """solid cylinder, symmetry axis == Z -> (ixx, iyy, izz)"""
    return m * (3 * r**2 + L**2) / 12, m * (3 * r**2 + L**2) / 12, 0.5 * m * r**2


def box(m, x, y, z):
    return (m * (y**2 + z**2) / 12, m * (x**2 + z**2) / 12, m * (x**2 + y**2) / 12)


def inertial(xyz, m, I):
    ixx, iyy, izz = I
    return (f'<inertial><origin xyz="{xyz}" rpy="0 0 0"/><mass value="{m}"/>'
            f'<inertia ixx="{ixx:.9g}" iyy="{iyy:.9g}" izz="{izz:.9g}" '
            f'ixy="0" ixz="0" iyz="0"/></inertial>')


def frame_link(name):
    return (f'<link name="{name}">'
            '<inertial><mass value="1e-6"/>'
            '<inertia ixx="1e-9" iyy="1e-9" izz="1e-9" ixy="0" ixz="0" iyz="0"/>'
            '</inertial></link>')


def build_paddle_xml():
    p = P
    pf = PREFIX
    holder_outer_y = p["slot_width"] + 2 * p["wall_thickness"]
    holder_outer_z = p["slot_height"] + p["wall_thickness"]
    side_wall_y = p["slot_width"] / 2 + p["wall_thickness"] / 2
    bottom_z = -(p["slot_height"] / 2 + p["wall_thickness"] / 2)
    screw_len = holder_outer_y + 0.002
    handle_in_x = p["slot_length"] - p["insertion_depth"]

    parts = []
    parts.append('<material name="tt_red"><color rgba="0.80 0.10 0.10 1.0"/></material>')
    parts.append('<material name="tt_dark"><color rgba="0.20 0.20 0.20 1.0"/></material>')

    # 1) adapter mount (cylinder, axis +X)
    parts.append(
        f'<joint name="{pf}tt_adapter_mount_joint" type="fixed">'
        f'<parent link="{PARENT_LINK}"/><child link="{pf}tt_adapter_mount_link"/>'
        f'<origin xyz="{p["mount_xyz"]}" rpy="{p["mount_rpy"]}"/></joint>')
    parts.append(
        f'<link name="{pf}tt_adapter_mount_link">'
        + inertial(f'{p["mount_length"]/2} 0 0', p["mount_mass"],
                   cyl_x(p["mount_mass"], p["mount_radius"], p["mount_length"]))
        + f'<visual><origin xyz="{p["mount_length"]/2} 0 0" rpy="0 {PI_2} 0"/>'
          f'<geometry><cylinder radius="{p["mount_radius"]}" length="{p["mount_length"]}"/></geometry>'
          '<material name="tt_dark"/></visual>'
        + f'<collision><origin xyz="{p["mount_length"]/2} 0 0" rpy="0 {PI_2} 0"/>'
          f'<geometry><cylinder radius="{p["mount_radius"]}" length="{p["mount_length"]}"/></geometry></collision>'
        + '</link>')
    parts.append(
        f'<joint name="{pf}adapter_mount_frame_joint" type="fixed">'
        f'<parent link="{pf}tt_adapter_mount_link"/><child link="{pf}adapter_mount_frame"/>'
        '<origin xyz="0 0 0" rpy="0 0 0"/></joint>')
    parts.append(frame_link(f"{pf}adapter_mount_frame"))

    # 2) extension tube (cylinder, axis +X)
    parts.append(
        f'<joint name="{pf}tt_extension_joint" type="fixed">'
        f'<parent link="{pf}tt_adapter_mount_link"/><child link="{pf}tt_extension_link"/>'
        f'<origin xyz="{p["mount_length"]} 0 0" rpy="0 0 0"/></joint>')
    parts.append(
        f'<link name="{pf}tt_extension_link">'
        + inertial(f'{p["extension_length"]/2} 0 0', p["extension_mass"],
                   cyl_x(p["extension_mass"], p["extension_radius"], p["extension_length"]))
        + f'<visual><origin xyz="{p["extension_length"]/2} 0 0" rpy="0 {PI_2} 0"/>'
          f'<geometry><cylinder radius="{p["extension_radius"]}" length="{p["extension_length"]}"/></geometry>'
          '<material name="tt_dark"/></visual>'
        + f'<collision><origin xyz="{p["extension_length"]/2} 0 0" rpy="0 {PI_2} 0"/>'
          f'<geometry><cylinder radius="{p["extension_radius"]}" length="{p["extension_length"]}"/></geometry></collision>'
        + '</link>')

    # 3) paddle holder (U-channel) — rolled +90deg about +X (tube axis) so the
    #    seated paddle's face-normal goes from +Z (up) to +Y (lateral/forward in
    #    the forehand pose), matching T1's mounting. Children (paddle + frames) follow.
    parts.append(
        f'<joint name="{pf}tt_paddle_holder_joint" type="fixed">'
        f'<parent link="{pf}tt_extension_link"/><child link="{pf}tt_paddle_holder_link"/>'
        f'<origin xyz="{p["extension_length"]} 0 0" rpy="{PI_2} 0 0"/></joint>')
    sl, sh = p["slot_length"], p["slot_height"]
    wt = p["wall_thickness"]
    parts.append(
        f'<link name="{pf}tt_paddle_holder_link">'
        + inertial(f'{sl/2} 0 {-wt/2}', p["holder_mass"],
                   box(p["holder_mass"], sl, holder_outer_y, holder_outer_z))
        # bottom wall
        + f'<visual><origin xyz="{sl/2} 0 {bottom_z}" rpy="0 0 0"/>'
          f'<geometry><box size="{sl} {holder_outer_y} {wt}"/></geometry><material name="tt_dark"/></visual>'
        # left wall (+Y)
        + f'<visual><origin xyz="{sl/2} {side_wall_y} 0" rpy="0 0 0"/>'
          f'<geometry><box size="{sl} {wt} {sh}"/></geometry><material name="tt_dark"/></visual>'
        # right wall (-Y)
        + f'<visual><origin xyz="{sl/2} {-side_wall_y} 0" rpy="0 0 0"/>'
          f'<geometry><box size="{sl} {wt} {sh}"/></geometry><material name="tt_dark"/></visual>'
        # screw-hole visuals (cylinders along +-Y)
        + f'<visual><origin xyz="{p["screw_x1"]} 0 0" rpy="{PI_2} 0 0"/>'
          f'<geometry><cylinder radius="{p["screw_diameter"]/2}" length="{screw_len}"/></geometry><material name="tt_red"/></visual>'
        + f'<visual><origin xyz="{p["screw_x2"]} 0 0" rpy="{PI_2} 0 0"/>'
          f'<geometry><cylinder radius="{p["screw_diameter"]/2}" length="{screw_len}"/></geometry><material name="tt_red"/></visual>'
        # simplified solid-box collision
        + f'<collision><origin xyz="{sl/2} 0 {-wt/2}" rpy="0 0 0"/>'
          f'<geometry><box size="{sl} {holder_outer_y} {holder_outer_z}"/></geometry></collision>'
        + '</link>')
    parts.append(
        f'<joint name="{pf}paddle_handle_frame_joint" type="fixed">'
        f'<parent link="{pf}tt_paddle_holder_link"/><child link="{pf}paddle_handle_frame"/>'
        f'<origin xyz="{sl/2} 0 0" rpy="0 0 0"/></joint>')
    parts.append(frame_link(f"{pf}paddle_handle_frame"))

    # 4+5) real paddle MESH (handle + blade as one rigid body), seated in slot.
    #      Visual = the imported STL; collision = simplified handle box + blade
    #      disc (cheap & stable). Mesh is mm -> scale; rotated so handle->blade
    #      runs along +X and the face-normal is +Z.
    hl, hw, hh = p["handle_col_len"], p["handle_col_w"], p["handle_col_h"]
    br, bt, bcx = p["blade_radius"], p["blade_thickness"], p["blade_center_x"]
    # lumped inertia: handle box @ x=hl/2 + blade disc @ x=bcx, parallel-axis to COM
    mh, mb = p["handle_mass"], p["blade_mass"]
    com_x = (mh * hl / 2 + mb * bcx) / (mh + mb)
    hI = box(mh, hl, hw, hh)
    bI = cyl_z(mb, br, bt)
    dh, db = (hl / 2 - com_x), (bcx - com_x)   # x-offsets to COM (along X)
    Ixx = hI[0] + bI[0]
    Iyy = hI[1] + mh * dh**2 + bI[1] + mb * db**2
    Izz = hI[2] + mh * dh**2 + bI[2] + mb * db**2
    parts.append(
        f'<joint name="{pf}tt_paddle_joint" type="fixed">'
        f'<parent link="{pf}tt_paddle_holder_link"/><child link="{pf}tt_paddle_link"/>'
        f'<origin xyz="{handle_in_x} 0 0" rpy="0 0 0"/></joint>')
    parts.append(
        f'<link name="{pf}tt_paddle_link">'
        + inertial(f'{com_x} 0 0', mh + mb, (Ixx, Iyy, Izz))
        # visual: imported paddle STL (mm), rotated/recentered into the slot frame
        + f'<visual><origin xyz="0 {p["paddle_y"]} {p["paddle_z"]}" rpy="{p["paddle_rpy"]}"/>'
          f'<geometry><mesh filename="{p["paddle_mesh"]}" '
          f'scale="{p["paddle_scale"]} {p["paddle_scale"]} {p["paddle_scale"]}"/></geometry>'
          '<material name="tt_red"/></visual>'
        # collision: simplified handle box ...
        + f'<collision><origin xyz="{hl/2} 0 0" rpy="0 0 0"/>'
          f'<geometry><box size="{hl} {hw} {hh}"/></geometry></collision>'
        # ... + blade disc (face normal == +Z); set restitution/friction in sim
        + f'<collision><origin xyz="{bcx} 0 0" rpy="0 0 0"/>'
          f'<geometry><cylinder radius="{br}" length="{bt}"/></geometry></collision>'
        + '</link>')
    # contact frame at the blade face center, +Z = outward hitting normal
    parts.append(
        f'<joint name="{pf}paddle_contact_frame_joint" type="fixed">'
        f'<parent link="{pf}tt_paddle_link"/><child link="{pf}paddle_contact_frame"/>'
        f'<origin xyz="{bcx} 0 {bt/2}" rpy="0 0 0"/></joint>')
    parts.append(frame_link(f"{pf}paddle_contact_frame"))

    return parts


def strip_rubber_hand(root):
    """Remove visual/collision of the right rubber hand and zero its inertia.
    The link itself is kept: it is the wrist_roll child (the 23rd DoF)."""
    for link in root.findall("link"):
        if link.get("name") == PARENT_LINK:
            for tag in ("visual", "collision"):
                for el in link.findall(tag):
                    link.remove(el)
            inert = link.find("inertial")
            if inert is not None:
                link.remove(inert)
            new = ET.fromstring(
                '<inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1e-4"/>'
                '<inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial>')
            link.insert(0, new)
            return True
    return False


def validate(root, label):
    links = {l.get("name") for l in root.findall("link")}
    joints = root.findall("joint")
    active = 0
    children = set()
    for j in joints:
        jt = j.get("type")
        pl = j.find("parent").get("link")
        cl = j.find("child").get("link")
        assert pl in links, f"{label}: joint {j.get('name')} parent '{pl}' undefined"
        assert cl in links, f"{label}: joint {j.get('name')} child '{cl}' undefined"
        children.add(cl)
        if jt in ("revolute", "continuous", "prismatic"):
            active += 1
    roots = links - children
    assert len(roots) == 1, f"{label}: expected 1 root link, got {sorted(roots)}"
    print(f"  [{label}] links={len(links)} joints={len(joints)} "
          f"active_DoF={active} root={next(iter(roots))}")
    return active


def main():
    tree = ET.parse(BASE)
    root = tree.getroot()
    base_dof = validate(root, "base g1_23dof")

    assert strip_rubber_hand(root), f"{PARENT_LINK} not found in base"
    ET.ElementTree(root).write(OUT_TT, encoding="utf-8", xml_declaration=True)
    validate(root, "g1_23dof_tt")

    for el in build_paddle_xml():
        root.append(ET.fromstring(el))
    paddle_dof = validate(root, "g1_23dof_tt_paddle")
    assert paddle_dof == base_dof, \
        f"DoF changed! base={base_dof} paddle={paddle_dof} (paddle must be fixed-only)"

    ET.ElementTree(root).write(OUT_PADDLE, encoding="utf-8", xml_declaration=True)
    print(f"\nwrote:\n  {OUT_TT}\n  {OUT_PADDLE}")
    print(f"DoF preserved at {paddle_dof} (no active joints added).")


if __name__ == "__main__":
    main()

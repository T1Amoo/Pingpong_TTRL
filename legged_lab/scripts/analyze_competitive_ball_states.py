"""Filter Google's real ball-state dataset for the A1 backhand workspace.

The source dataset uses ``x=table width, y=table length, z=height above the
table``.  A1 uses ``x=table length, y=table width, z=world height``.  The
proper (right-handed) frame conversion used here is::

    a1_x = source_y
    a1_y = -source_x
    a1_z = source_z + table_height

Each state is replayed with the MuJoCo model shipped by the source repository,
so spin, fluid forces and table contact participate in the filtering.  This is
deliberately an offline analysis tool; training must consume a reviewed,
versioned prior rather than silently depending on a sibling checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "competitive_robot_table_tennis"


def _load_source_model(dataset_root: Path) -> mujoco.MjModel:
    notebook = json.loads((dataset_root / "ball_states_viz.ipynb").read_text())
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "pp_ball_with_table_xml = " in "".join(cell.get("source", []))
    ]
    if len(sources) != 1:
        raise RuntimeError(f"expected one MuJoCo model cell, found {len(sources)}")
    match = re.search(r'pp_ball_with_table_xml\s*=\s*"""(.*?)"""', sources[0], re.S)
    if match is None:
        raise RuntimeError("could not extract pp_ball_with_table_xml from notebook")
    return mujoco.MjModel.from_xml_string(match.group(1))


def _a1_vector(source_xyz: np.ndarray) -> np.ndarray:
    """Rotate a polar or axial vector from the source frame into A1."""

    return np.asarray((source_xyz[1], -source_xyz[0], source_xyz[2]), dtype=np.float64)


def _quantiles(rows: list[dict], key: str) -> list[float]:
    value = np.asarray([row[key] for row in rows], dtype=np.float64)
    return np.quantile(value, (0.0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.0)).round(4).tolist()


def _summarize(rows: list[dict], label: str) -> None:
    print(f"[DATASET_FILTER] subset={label} count={len(rows)}", flush=True)
    if not rows:
        return
    for key in (
        "hit_y_a1_m",
        "hit_z_a1_m",
        "hit_abs_vx_a1_mps",
        "spin_norm_rad_s",
        "spin_x_a1_rad_s",
        "spin_y_a1_rad_s",
        "spin_z_a1_rad_s",
    ):
        print(f"  {key} q0/q1/q5/q50/q95/q99/q100={_quantiles(rows, key)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source", choices=("all", "serves", "rallies"), default="all")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--duration-s", type=float, default=1.8)
    parser.add_argument("--hit-plane-x", type=float, default=-1.243)
    parser.add_argument("--hit-y-range", type=float, nargs=2, default=(-0.22, 0.26))
    parser.add_argument("--hit-z-range", type=float, nargs=2, default=(0.88, 1.32))
    parser.add_argument("--hit-abs-vx-range", type=float, nargs=2, default=(1.0, 5.8))
    parser.add_argument("--net-center-z-min", type=float, default=0.945)
    parser.add_argument("--table-height", type=float, default=0.76)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    model = _load_source_model(dataset_root)
    data = mujoco.MjData(model)
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "geom_ball")
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "geom_table")
    max_steps = int(round(args.duration_s / model.opt.timestep))

    source_names = ("serves", "rallies") if args.source == "all" else (args.source,)
    source_hashes = {}
    states: list[tuple[str, dict]] = []
    for source in source_names:
        source_path = dataset_root / f"{source}.json"
        source_bytes = source_path.read_bytes()
        source_hashes[source] = hashlib.sha256(source_bytes).hexdigest()
        entries = json.loads(source_bytes)
        states.extend((source, entry) for entry in entries)

    stage_counts = {
        "all": len(states),
        "toward_a1": 0,
        "clears_net": 0,
        "receiver_bounce": 0,
        "crosses_hit_plane": 0,
        "inside_a1_window": 0,
    }
    accepted: list[dict] = []
    start = time.monotonic()

    # Source table top is z=0; convert A1 world-height thresholds once.
    source_net_center_min = args.net_center_z_min - args.table_height
    source_hit_z_range = tuple(value - args.table_height for value in args.hit_z_range)

    for source, state in states:
        source_pos = np.asarray(
            (state["pos_x"], state["pos_y"], state["pos_z"]), dtype=np.float64
        )
        source_vel = np.asarray(
            (state["vel_x"], state["vel_y"], state["vel_z"]), dtype=np.float64
        )
        source_spin = np.asarray(
            (state["w_vel_x"], state["w_vel_y"], state["w_vel_z"]), dtype=np.float64
        )
        a1_pos = _a1_vector(source_pos)
        a1_vel = _a1_vector(source_vel)
        a1_spin = _a1_vector(source_spin)
        a1_pos[2] += args.table_height

        if not (a1_pos[0] > 0.0 and a1_vel[0] < -0.05):
            continue
        stage_counts["toward_a1"] += 1

        mujoco.mj_resetData(model, data)
        data.qpos[:3] = source_pos
        data.qvel[:3] = source_vel
        data.qvel[3:] = source_spin
        mujoco.mj_forward(model, data)

        previous = data.qpos[:3].copy()
        net_z = None
        receiver_bounce = None
        hit_pos = None
        hit_vel = None
        for _ in range(max_steps):
            mujoco.mj_step(model, data)
            current = data.qpos[:3].copy()
            velocity = data.qvel[:3].copy()

            if net_z is None and previous[1] > 0.0 >= current[1] and velocity[1] < 0.0:
                alpha = previous[1] / max(previous[1] - current[1], 1.0e-12)
                net_z = previous[2] + alpha * (current[2] - previous[2])

            if receiver_bounce is None:
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    if {contact.geom1, contact.geom2} == {ball_geom, table_geom} and current[1] < 0.0:
                        receiver_bounce = current.copy()
                        break

            if (
                receiver_bounce is not None
                and previous[1] > args.hit_plane_x >= current[1]
                and velocity[1] < 0.0
            ):
                alpha = (previous[1] - args.hit_plane_x) / max(
                    previous[1] - current[1], 1.0e-12
                )
                hit_pos = previous + alpha * (current - previous)
                hit_vel = velocity.copy()
                break

            if current[2] < -0.2 or current[1] < args.hit_plane_x - 0.4:
                break
            previous = current

        if net_z is None or net_z < source_net_center_min:
            continue
        stage_counts["clears_net"] += 1
        if (
            receiver_bounce is None
            or not (-1.35 <= receiver_bounce[1] <= -0.02)
            or abs(receiver_bounce[0]) > 0.7625
        ):
            continue
        stage_counts["receiver_bounce"] += 1
        if hit_pos is None or hit_vel is None:
            continue
        stage_counts["crosses_hit_plane"] += 1

        hit_pos_a1 = _a1_vector(hit_pos)
        hit_vel_a1 = _a1_vector(hit_vel)
        hit_pos_a1[2] += args.table_height
        inside = (
            args.hit_y_range[0] <= hit_pos_a1[1] <= args.hit_y_range[1]
            and source_hit_z_range[0] <= hit_pos[2] <= source_hit_z_range[1]
            and args.hit_abs_vx_range[0]
            <= abs(hit_vel_a1[0])
            <= args.hit_abs_vx_range[1]
        )
        if not inside:
            continue
        stage_counts["inside_a1_window"] += 1
        accepted.append(
            {
                "source": source,
                "id": int(state["id"]),
                "a1_initial_pos_m": a1_pos.round(8).tolist(),
                "a1_initial_lin_vel_mps": a1_vel.round(8).tolist(),
                "a1_initial_ang_vel_rad_s": a1_spin.round(8).tolist(),
                "net_center_z_a1_m": round(float(net_z + args.table_height), 8),
                "hit_pos_a1_m": hit_pos_a1.round(8).tolist(),
                "hit_lin_vel_a1_mps": hit_vel_a1.round(8).tolist(),
                "hit_y_a1_m": float(hit_pos_a1[1]),
                "hit_z_a1_m": float(hit_pos_a1[2]),
                "hit_abs_vx_a1_mps": float(abs(hit_vel_a1[0])),
                "spin_norm_rad_s": float(np.linalg.norm(a1_spin)),
                "spin_x_a1_rad_s": float(a1_spin[0]),
                "spin_y_a1_rad_s": float(a1_spin[1]),
                "spin_z_a1_rad_s": float(a1_spin[2]),
            }
        )

    print(
        f"[DATASET_FILTER] stages={json.dumps(stage_counts, sort_keys=True)} "
        f"elapsed_s={time.monotonic() - start:.2f}",
        flush=True,
    )
    print(f"[DATASET_FILTER] sha256={json.dumps(source_hashes, sort_keys=True)}", flush=True)
    _summarize([row for row in accepted if row["source"] == "serves"], "serves")
    _summarize([row for row in accepted if row["source"] == "rallies"], "rallies")
    _summarize(accepted, "all")

    if args.output is not None:
        split_counts = {}
        for source in source_names:
            source_rows = sorted(
                (row for row in accepted if row["source"] == source),
                key=lambda row: row["id"],
            )
            train_end = int(0.80 * len(source_rows))
            validation_end = int(0.90 * len(source_rows))
            for index, row in enumerate(source_rows):
                row["split"] = (
                    "train"
                    if index < train_end
                    else "validation"
                    if index < validation_end
                    else "test"
                )
            split_counts[source] = {
                name: sum(row["split"] == name for row in source_rows)
                for name in ("train", "validation", "test")
            }
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": "google-deepmind/competitive_robot_table_tennis",
            "license": "CC-BY-4.0",
            "frame": "A1 table frame, world z; x=source_y, y=-source_x",
            "sha256": source_hashes,
            "split": "contiguous source-id 80/10/10 per source",
            "split_counts": split_counts,
            "filters": {
                "hit_plane_x": args.hit_plane_x,
                "hit_y_range": args.hit_y_range,
                "hit_z_range": args.hit_z_range,
                "hit_abs_vx_range": args.hit_abs_vx_range,
                "net_center_z_min": args.net_center_z_min,
            },
            "stage_counts": stage_counts,
            "states": accepted,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[DATASET_FILTER] wrote={output} states={len(accepted)}", flush=True)


if __name__ == "__main__":
    main()

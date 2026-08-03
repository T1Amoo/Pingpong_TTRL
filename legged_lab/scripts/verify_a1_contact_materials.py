"""Read back the effective A1 table-tennis contact materials from PhysX."""

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="a1_tt_backhand_v2")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.utils import task_registry  # noqa: E402


def _shape_rows(asset, body_pattern: str | None = None):
    material = asset.root_physx_view.get_material_properties()[0].cpu()
    if body_pattern is None:
        return material

    body_ids, body_names = asset.find_bodies(body_pattern)
    if not body_ids:
        raise RuntimeError(f"body pattern {body_pattern!r} did not match")
    shape_counts = []
    for link_path in asset.root_physx_view.link_paths[0]:
        link_view = asset._physics_sim_view.create_rigid_body_view(link_path)
        shape_counts.append(link_view.max_shapes)

    rows = []
    for body_id, body_name in zip(body_ids, body_names):
        start = sum(shape_counts[:body_id])
        end = start + shape_counts[body_id]
        rows.append((body_name, start, end, material[start:end]))
    return rows


def _print_material(label: str, values: torch.Tensor):
    unique = torch.unique(values, dim=0)
    print(f"[CONTACT_MATERIAL] {label}: shape_count={values.shape[0]}", flush=True)
    for row in unique.tolist():
        print(
            f"  static_friction={row[0]:.6f} dynamic_friction={row[1]:.6f} "
            f"restitution={row[2]:.6f}",
            flush=True,
        )


def main():
    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = 1
    env = task_registry.get_task_class(args_cli.task)(env_cfg, True)

    _print_material("ball", _shape_rows(env.ball))
    _print_material("table", _shape_rows(env.table))
    for body_name, start, end, values in _shape_rows(env.robot, "Link_r_paddle"):
        _print_material(f"robot body={body_name} shape_idx=[{start},{end})", values)

    env.close()
    simulation_app.close()
    os._exit(0)


main()

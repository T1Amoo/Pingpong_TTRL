# Record a ~1 min Isaac Lab table-tennis clip of the v8_33000 policy.
# Headless rendering: run with --enable_cameras. Frames -> _recordings/run_*/rgb_*.png,
# then encode to mp4 with imageio (bundled ffmpeg). Usage:
#   OMNI_KIT_ACCEPT_EULA=YES python -m legged_lab.scripts.record_play \
#     --task g1_tt_eval --predictor --headless --enable_cameras --num_envs 1 \
#     --load_run 2026-06-23_12-21-52 --checkpoint model_33000.pt
import argparse
import os
import torch
from isaaclab.app import AppLauncher
from rsl_rl.rsl_rl.runners import OnPolicyRunner
from legged_lab.utils import task_registry
import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="g1_tt_eval")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--predictor", action="store_true")
parser.add_argument("--max_steps", type=int, default=1800)  # ~60s at 30fps
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True  # force rendering for capture

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import get_checkpoint_path
from pxr import Gf
from legged_lab.envs import *  # noqa: F401,F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from legged_lab.utils.data_recorder.frame_recorder import FrameRecorder, ensure_world_camera


def play():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    if args_cli.predictor:
        from rsl_rl.rsl_rl.runners import OnPolicyPredictorRegressionRunner
        runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    print(f"[record] loaded {resume_path}", flush=True)

    # -- VIDEO RECORDING SETUP -- down-the-table broadcast view toward the robot (at x=-2.0)
    cam = ensure_world_camera("/World/RecorderCamera", pos=Gf.Vec3d(3.6, 0.0, 1.9),
                              euler_deg=Gf.Vec3f(-14.0, 0.0, 180.0))
    rec = FrameRecorder(camera_prim_paths=[cam], output_root="_recordings", resolution=(1920, 1080))
    rec.start()
    print(f"[record] capturing to: {rec.output_dir}", flush=True)

    obs, _ = env.get_observations()
    n = 0
    try:
        while simulation_app.is_running() and n < args_cli.max_steps:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                if args_cli.predictor:
                    try:
                        runner._record_ball_positions()
                        runner._maybe_predict_and_update_env()
                    except Exception:
                        pass
            n += 1
            if n % 100 == 0:
                print(f"[record] step {n}/{args_cli.max_steps}", flush=True)
    finally:
        out = rec.stop()
        print(f"[record] DONE frames in: {out}", flush=True)
    os._exit(0)


play()

"""
On-policy runner variant with an auxiliary regression predictor for ball pose.

- Maintains a ping-pong ball trajectory buffer per env from `TTEnv.ball_pos`.
- Trains a small MLP to predict a future ball pose from the last 5 positions.
- Supports either the historical environment future-pose target or a physical
  fixed-hit-plane target backfilled when the incoming Isaac ball crosses it.
- Performs inference every step and, if available, calls `env.update_prediction`.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import torch

from .on_policy_runner import OnPolicyRunner


class _MLPPredictor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, int] = (64, 64), output_dim: int = 3):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Linear(h2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OnPolicyPredictorRegressionRunner(OnPolicyRunner):  # noqa: C901
    """Runner with auxiliary regression predictor for future ball pose."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # Predictor config with sensible defaults
        pred_cfg = self.cfg.get("predictor", {})
        self.pred_history_len: int = int(pred_cfg.get("history_len", 5))
        self.pred_traj_maxlen: int = int(pred_cfg.get("traj_max_len", 256))
        self.pred_hidden: Tuple[int, int] = tuple(pred_cfg.get("hidden_sizes", (64, 64)))  # type: ignore
        self.pred_lr: float = float(pred_cfg.get("lr", 1e-3))
        self.pred_epochs: int = int(pred_cfg.get("epochs_per_update", 2))
        self.pred_batch_size: int = int(pred_cfg.get("batch_size", 1024))
        self.pred_target_mode: str = str(pred_cfg.get("target_mode", "env_future_pose"))
        if self.pred_target_mode not in {"env_future_pose", "actual_hit_plane"}:
            raise ValueError(f"unsupported predictor target_mode: {self.pred_target_mode}")

        # optionally stop training predictor after a number of PPO iterations
        self.pred_train_until_iters: int = int(pred_cfg.get("train_until_iters", 1000))

        # Trajectory ring buffer on CPU: [T, N, 3]
        self._traj_maxlen: int = self.pred_traj_maxlen
        self._traj_buf_cpu: torch.Tensor = torch.zeros(
            self._traj_maxlen, self.env.num_envs, 3, dtype=torch.float32, device="cpu"
        )
        # Ground-truth future pose ring buffer on CPU: [T, N, 3] (env.ball_future_pose at each step)
        self._gt_buf_cpu: torch.Tensor = torch.zeros(
            self._traj_maxlen, self.env.num_envs, 3, dtype=torch.float32, device="cpu"
        )
        self._sample_valid_buf_cpu: torch.Tensor = torch.zeros(
            self._traj_maxlen, self.env.num_envs, dtype=torch.bool, device="cpu"
        )
        self._gt_valid_buf_cpu: torch.Tensor = torch.zeros(
            self._traj_maxlen, self.env.num_envs, dtype=torch.bool, device="cpu"
        )
        self._serve_id_buf_cpu: torch.Tensor = torch.zeros(
            self._traj_maxlen, self.env.num_envs, dtype=torch.long, device="cpu"
        )
        self._traj_write_idx: int = 0
        self._traj_len: int = 0  # how many steps recorded (capped at _traj_maxlen)

        # Predictor MLP and optimizer
        self._pred_input_dim = 3 * self.pred_history_len
        self._predictor = _MLPPredictor(self._pred_input_dim, self.pred_hidden, 3).to(self.device)
        self._pred_optim = torch.optim.Adam(self._predictor.parameters(), lr=self.pred_lr)
        self._pred_trained: bool = False
        self._last_pred_loss: Optional[float] = None
        self._last_pred_valid_fraction: Optional[float] = None
        self.pred_min_valid_samples: int = int(pred_cfg.get("min_valid_samples", 64))
        # Prequential validation: at a physical crossing, compare the prediction
        # produced on the PREVIOUS control tick against the newly observed
        # intersection.  That target was not available when inference ran, so
        # these are causal out-of-sample errors rather than training loss.
        self.pred_validation_window_samples: int = int(pred_cfg.get("validation_window_samples", 4096))
        self.pred_validation_interval_iters: int = int(pred_cfg.get("validation_interval_iters", 50))
        self.pred_validation_min_samples: int = int(pred_cfg.get("validation_min_samples", 256))
        self.pred_validation_warn_start_iter: int = int(pred_cfg.get("validation_warn_start_iter", 100))
        self.pred_validation_warn_patience: int = int(pred_cfg.get("validation_warn_patience", 3))
        self.pred_validation_warn_min_rel_improvement: float = float(
            pred_cfg.get("validation_warn_min_rel_improvement", 0.03)
        )
        self.pred_validation_warn_y_mae_m: float = float(pred_cfg.get("validation_warn_y_mae_m", 0.08))
        self.pred_validation_warn_z_mae_m: float = float(pred_cfg.get("validation_warn_z_mae_m", 0.08))
        self.pred_validation_warn_hard_z_mae_m: float = float(
            pred_cfg.get("validation_warn_hard_z_mae_m", 0.10)
        )
        self.pred_validation_slow_abs_vx_mps: float = float(
            pred_cfg.get("validation_slow_abs_vx_mps", 2.2)
        )
        self.pred_validation_high_z_m: float = float(pred_cfg.get("validation_high_z_m", 1.30))
        self._pred_last_prediction_cpu = torch.zeros(self.env.num_envs, 3, dtype=torch.float32)
        self._pred_last_inference_valid_cpu = torch.zeros(self.env.num_envs, dtype=torch.bool)
        self._pred_last_inference_serve_id_cpu = torch.full(
            (self.env.num_envs,), -1, dtype=torch.long
        )
        self._pred_validation_errors_cpu = torch.empty(0, 2, dtype=torch.float32)
        self._pred_validation_slow_cpu = torch.empty(0, dtype=torch.bool)
        self._pred_validation_high_cpu = torch.empty(0, dtype=torch.bool)
        self._pred_validation_crossings_total: int = 0
        self._pred_validation_best_score: float = math.inf
        self._pred_validation_no_improve_checks: int = 0
        self._pred_validation_report_checks: int = 0
        # debug counters
        self._pred_call_count: int = 0

    # ---------------------------
    # Rollout and learning loop
    # ---------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer (copied from parent)
        if self.log_dir is not None and getattr(self, "writer", None) is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # Initialize TT metrics at iter 0 so curves appear immediately
        if self.log_dir is not None and not self.disable_logs and getattr(self, "writer", None) is not None:
            try:
                if not hasattr(self, "_tt_logged_init"):
                    self.writer.add_scalar("Train/TT_success_rate", 0.0, 0)
                    self.writer.add_scalar("Train/TT_hit_rate", 0.0, 0)
                    self.writer.add_scalar("Train/TT_success_rate_sampled", 0.0, 0)
                    self.writer.add_scalar("Train/TT_hit_rate_sampled", 0.0, 0)
                    self.writer.add_scalar("Train/TT_action_noise_l2_mean", 0.0, 0)
                    self.writer.add_scalar("Train/TT_hit_action_noise_l2_mean", 0.0, 0)
                    self.writer.add_scalar("Train/TT_success_action_noise_l2_mean", 0.0, 0)
                    self._tt_logged_init = True
            except Exception:
                pass

        # teacher check
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()

        # Book keeping (copied from parent)
        from collections import deque as _dq

        ep_infos = []
        rewbuffer = _dq(maxlen=100)
        lenbuffer = _dq(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        if self.alg.rnd:
            erewbuffer = _dq(maxlen=100)
            irewbuffer = _dq(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Table-tennis serve and hit tracking for training logs
        # Aggregate across rollout steps and log every 100 updates
        self._tt_succ_total: int = 0
        self._tt_hit_total: int = 0
        self._tt_serve_total: int = 0
        self._tt_action_noise_l2_sum: float = 0.0
        self._tt_action_noise_count: int = 0
        self._tt_hit_action_noise_l2_sum: float = 0.0
        self._tt_hit_action_noise_count: int = 0
        self._tt_success_action_noise_l2_sum: float = 0.0
        self._tt_success_action_noise_count: int = 0
        try:
            self._tt_serve_success_flag = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
            self._tt_serve_hit_flag = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        except Exception:
            self._tt_serve_success_flag = None
            self._tt_serve_hit_flag = None

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions and step
                    actions = self.alg.act(obs, privileged_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # perform normalization and update privileged obs
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    action_noise_l2 = None
                    try:
                        action_mean = self.alg.transition.action_mean
                        action_noise = (actions - action_mean).detach()
                        action_noise_l2 = torch.linalg.vector_norm(action_noise, dim=-1)
                        self._tt_action_noise_l2_sum += float(action_noise_l2.sum().item())
                        self._tt_action_noise_count += int(action_noise_l2.numel())
                    except Exception:
                        action_noise_l2 = None

                    # Aux: record ball positions and run predictor inference
                    try:
                        self._accumulate_actual_hit_plane_validation()
                        # if self.current_learning_iteration < self.pred_train_until_iters:
                        self._record_ball_positions()   
                        self._maybe_predict_and_update_env()
                    except Exception:
                        if self.pred_target_mode == "actual_hit_plane":
                            raise

                    # Track TT success/hit per-serve signals (similar to play.py)
                    try:
                        if (
                            hasattr(self.env, "has_touch_opponent_table_just_now")
                            and hasattr(self.env, "has_touch_paddle")
                            and self._tt_serve_success_flag is not None
                        ):
                            event_mask = (self.env.has_touch_opponent_table_just_now & self.env.has_touch_paddle)
                            self._tt_serve_success_flag |= event_mask.to(self._tt_serve_success_flag.device)
                            if action_noise_l2 is not None and bool(event_mask.any().item()):
                                success_noise = action_noise_l2.to(event_mask.device)[event_mask]
                                self._tt_success_action_noise_l2_sum += float(success_noise.sum().item())
                                self._tt_success_action_noise_count += int(success_noise.numel())
                        if (
                            hasattr(self.env, "ball_contact_rew")
                            and self._tt_serve_hit_flag is not None
                        ):
                            hit_mask = (self.env.ball_contact_rew > 0.0)
                            self._tt_serve_hit_flag |= hit_mask.to(self._tt_serve_hit_flag.device)
                            if action_noise_l2 is not None and bool(hit_mask.any().item()):
                                hit_noise = action_noise_l2.to(hit_mask.device)[hit_mask]
                                self._tt_hit_action_noise_l2_sum += float(hit_noise.sum().item())
                                self._tt_hit_action_noise_count += int(hit_noise.numel())

                        # On serve boundary, aggregate and reset flags
                        if hasattr(self.env, "ball_reset_ids") and self.env.ball_reset_ids is not None:
                            ids = self.env.ball_reset_ids
                            if (
                                isinstance(ids, torch.Tensor)
                                and ids.numel() > 0
                                and self._tt_serve_success_flag is not None
                                and self._tt_serve_hit_flag is not None
                            ):
                                ids_dev = ids.to(self._tt_serve_success_flag.device)
                                self._tt_serve_total += int(ids_dev.numel())
                                self._tt_succ_total += int(self._tt_serve_success_flag[ids_dev].sum().item())
                                self._tt_hit_total += int(self._tt_serve_hit_flag[ids_dev].sum().item())
                                self._tt_serve_success_flag[ids_dev] = False
                                self._tt_serve_hit_flag[ids_dev] = False
                    except Exception:
                        pass

                    # RL algorithm step processing
                    self.alg.process_env_step(rewards, dones, infos)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # Logging bookkeeping mirrors parent
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns (same as parent)
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # Train auxiliary predictor on offline trajectory data (optional cutoff)
            pred_loss_val = None
            if self.current_learning_iteration < self.pred_train_until_iters:
                pred_loss_val = self._train_predictor_offline()
            if pred_loss_val is not None:
                self._last_pred_loss = float(pred_loss_val)

            # Update policy (PPO)
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Attach predictor training loss and independent physical-crossing
            # validation metrics to the normal console/TensorBoard stream.
            validation_metrics = self._predictor_validation_metrics()
            if pred_loss_val is not None or validation_metrics:
                loss_dict = dict(loss_dict)  # shallow copy for augmentation
            if pred_loss_val is not None:
                loss_dict["predictor_mse"] = float(pred_loss_val)
                if self._last_pred_valid_fraction is not None:
                    loss_dict["predictor_valid_fraction"] = float(self._last_pred_valid_fraction)
            loss_dict.update(validation_metrics)
            self._maybe_report_predictor_validation(it, validation_metrics)

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                # Log TT success and hit rates every 100 updates
                if (it + 1) % 100 == 0:
                    serve_total = self._tt_serve_total
                    succ_rate = (self._tt_succ_total / serve_total) if serve_total > 0 else 0.0
                    hit_rate = (self._tt_hit_total / serve_total) if serve_total > 0 else 0.0
                    action_noise_l2_mean = self._tt_action_noise_l2_sum / max(1, self._tt_action_noise_count)
                    hit_action_noise_l2_mean = (
                        self._tt_hit_action_noise_l2_sum / max(1, self._tt_hit_action_noise_count)
                    )
                    success_action_noise_l2_mean = (
                        self._tt_success_action_noise_l2_sum / max(1, self._tt_success_action_noise_count)
                    )
                    try:
                        self.writer.add_scalar("Train/TT_success_rate", succ_rate, it)
                        self.writer.add_scalar("Train/TT_hit_rate", hit_rate, it)
                        self.writer.add_scalar("Train/TT_success_rate_sampled", succ_rate, it)
                        self.writer.add_scalar("Train/TT_hit_rate_sampled", hit_rate, it)
                        self.writer.add_scalar("Train/TT_action_noise_l2_mean", action_noise_l2_mean, it)
                        self.writer.add_scalar("Train/TT_hit_action_noise_l2_mean", hit_action_noise_l2_mean, it)
                        self.writer.add_scalar(
                            "Train/TT_success_action_noise_l2_mean",
                            success_action_noise_l2_mean,
                            it,
                        )
                    except Exception:
                        pass
                    # reset counters for next window
                    self._tt_succ_total = 0
                    self._tt_hit_total = 0
                    self._tt_serve_total = 0
                    self._tt_action_noise_l2_sum = 0.0
                    self._tt_action_noise_count = 0
                    self._tt_hit_action_noise_l2_sum = 0.0
                    self._tt_hit_action_noise_count = 0
                    self._tt_success_action_noise_l2_sum = 0.0
                    self._tt_success_action_noise_count = 0
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                from rsl_rl.utils import store_code_state

                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ---------------------------
    # Auxiliary predictor helpers
    # ---------------------------
    def _record_ball_positions(self):
        """Append current ball positions to the trajectory ring buffer (batched)."""
        if not hasattr(self.env, "ball_pos"):
            return
        with torch.no_grad():
            if hasattr(self.env, "get_predictor_ball_positions"):
                ball_pos_device, sample_valid_device = self.env.get_predictor_ball_positions()
                ball_pos = ball_pos_device.detach().to("cpu")
                sample_valid = sample_valid_device.detach().to("cpu").bool()
            else:
                ball_pos = self.env.ball_pos.detach().to("cpu")  # [N, 3]
                sample_valid = torch.ones(self.env.num_envs, dtype=torch.bool)
            self._traj_buf_cpu[self._traj_write_idx].copy_(ball_pos)
            sample_valid &= (
                torch.isfinite(ball_pos).all(dim=-1)
                & (ball_pos[:, 2] > 0.1)
                & (ball_pos[:, 2] < 3.0)
            )
            try:
                if hasattr(self.env, "ball_reset_ids"):
                    reset_ids = self.env.ball_reset_ids.detach().to("cpu").long()
                    if reset_ids.numel() > 0:
                        sample_valid[reset_ids] = False
            except Exception:
                pass
            self._sample_valid_buf_cpu[self._traj_write_idx].copy_(sample_valid)
            try:
                if hasattr(self.env, "get_predictor_serve_ids"):
                    serve_ids = self.env.get_predictor_serve_ids().detach().to("cpu").long()
                elif hasattr(self.env, "ball_reset_counter"):
                    serve_ids = self.env.ball_reset_counter.detach().to("cpu").long()
                else:
                    serve_ids = torch.zeros(self.env.num_envs, dtype=torch.long)
                self._serve_id_buf_cpu[self._traj_write_idx].copy_(serve_ids)
            except Exception:
                self._serve_id_buf_cpu[self._traj_write_idx].zero_()
            # Each ring slot is reused, so stale target validity must be cleared
            # before either target mode fills it.
            self._gt_valid_buf_cpu[self._traj_write_idx].zero_()
            # Historical mode records the environment's online analytic target.
            # Backhand-v2 instead waits for a physical plane crossing and then
            # retroactively labels every valid pre-crossing history in that serve.
            try:
                if self.current_learning_iteration < self.pred_train_until_iters:
                    if self.pred_target_mode == "env_future_pose" and hasattr(self.env, "ball_future_pose"):
                        gt_pose = self.env.ball_future_pose.detach().to("cpu")  # [N,3]
                        self._gt_buf_cpu[self._traj_write_idx].copy_(gt_pose)
                        gt_valid = torch.isfinite(gt_pose).all(dim=-1) & (gt_pose[:, 2] > 0.70) & (gt_pose[:, 2] < 1.60)
                        if hasattr(self.env, "mask_invalid"):
                            gt_valid &= ~self.env.mask_invalid.detach().to("cpu").bool()
                        if hasattr(self.env, "ball_future_t"):
                            gt_valid &= self.env.ball_future_t.detach().to("cpu").squeeze(-1) > 0.0
                        self._gt_valid_buf_cpu[self._traj_write_idx].copy_(gt_valid)
                    elif self.pred_target_mode == "actual_hit_plane":
                        self._backfill_actual_hit_plane_targets()
            except Exception:
                if self.pred_target_mode == "actual_hit_plane":
                    raise
            self._traj_write_idx = (self._traj_write_idx + 1) % self._traj_maxlen
            self._traj_len = min(self._traj_len + 1, self._traj_maxlen)

    def _backfill_actual_hit_plane_targets(self) -> None:
        """Label recent histories with Isaac's measured crossing for each serve.

        The environment emits a one-step event containing the y/z point
        interpolated from physical states around ``x=hit_plane_x``.  Once that
        event arrives, all still-buffered, valid observations from the same
        serve and before the plane receive exactly that target.  No analytic
        apex or fitted flight parameter enters predictor supervision.
        """

        if not hasattr(self.env, "get_predictor_actual_hit_plane_crossings"):
            raise AttributeError("actual_hit_plane target mode requires environment crossing events")
        target_device, crossed_device = self.env.get_predictor_actual_hit_plane_crossings()
        crossed = crossed_device.detach().to("cpu").bool().reshape(-1)
        ids = torch.nonzero(crossed, as_tuple=False).flatten()
        if ids.numel() == 0:
            return

        targets = target_device.detach().to("cpu")[ids]
        current_serve_ids = self._serve_id_buf_cpu[self._traj_write_idx, ids]
        same_serve = self._serve_id_buf_cpu[:, ids] == current_serve_ids.unsqueeze(0)
        # The predictor consumes camera/coasted positions; only histories whose
        # observed ball is still in front of the target plane are causal inputs.
        before_plane = self._traj_buf_cpu[:, ids, 0] > targets[:, 0].unsqueeze(0)
        backfill = same_serve & before_plane & self._sample_valid_buf_cpu[:, ids]
        existing = self._gt_buf_cpu[:, ids, :]
        expanded_targets = targets.unsqueeze(0).expand(self._traj_maxlen, -1, -1)
        self._gt_buf_cpu[:, ids, :] = torch.where(
            backfill.unsqueeze(-1), expanded_targets, existing
        )
        self._gt_valid_buf_cpu[:, ids] |= backfill

    def _accumulate_actual_hit_plane_validation(self) -> None:
        """Accumulate causal predictor errors at newly observed crossings.

        ``_pred_last_prediction_cpu`` was produced after the previous control
        step.  The environment has only now exposed the physical crossing, so
        this comparison cannot leak the target into its own prediction.
        """

        if self.pred_target_mode != "actual_hit_plane" or not self._pred_trained:
            return
        if not hasattr(self.env, "get_predictor_actual_hit_plane_crossings"):
            raise AttributeError("actual_hit_plane validation requires environment crossing events")
        target_device, crossed_device = self.env.get_predictor_actual_hit_plane_crossings()
        crossed = crossed_device.detach().to("cpu").bool().reshape(-1)
        if not bool(crossed.any()):
            return
        if hasattr(self.env, "get_predictor_serve_ids"):
            current_serve_ids = self.env.get_predictor_serve_ids().detach().to("cpu").long()
        else:
            current_serve_ids = self._pred_last_inference_serve_id_cpu
        targets = target_device.detach().to("cpu")
        predictions = self._pred_last_prediction_cpu
        valid = (
            crossed
            & self._pred_last_inference_valid_cpu
            & (self._pred_last_inference_serve_id_cpu == current_serve_ids)
            & torch.isfinite(targets).all(dim=-1)
            & torch.isfinite(predictions).all(dim=-1)
        )
        if not bool(valid.any()):
            return

        errors = predictions[valid, 1:3] - targets[valid, 1:3]
        if hasattr(self.env, "ball_linvel"):
            abs_vx = torch.abs(self.env.ball_linvel.detach().to("cpu")[valid, 0])
            slow = abs_vx <= self.pred_validation_slow_abs_vx_mps
        else:
            slow = torch.zeros(len(errors), dtype=torch.bool)
        high = targets[valid, 2] >= self.pred_validation_high_z_m
        self._pred_validation_errors_cpu = torch.cat(
            (self._pred_validation_errors_cpu, errors.to(torch.float32)), dim=0
        )
        self._pred_validation_slow_cpu = torch.cat((self._pred_validation_slow_cpu, slow), dim=0)
        self._pred_validation_high_cpu = torch.cat((self._pred_validation_high_cpu, high), dim=0)
        self._pred_validation_crossings_total += len(errors)

        keep = max(1, self.pred_validation_window_samples)
        if len(self._pred_validation_errors_cpu) > keep:
            self._pred_validation_errors_cpu = self._pred_validation_errors_cpu[-keep:]
            self._pred_validation_slow_cpu = self._pred_validation_slow_cpu[-keep:]
            self._pred_validation_high_cpu = self._pred_validation_high_cpu[-keep:]

    def _predictor_validation_metrics(self) -> dict[str, float]:
        """Return rolling physical-crossing metrics for logging and alerting."""

        if self.pred_target_mode != "actual_hit_plane":
            return {}
        errors = self._pred_validation_errors_cpu
        count = len(errors)
        if count == 0:
            return {
                "predictor_actual_crossing_window_n": 0.0,
                "predictor_actual_crossing_total_n": float(self._pred_validation_crossings_total),
            }
        absolute = torch.abs(errors)
        metrics = {
            "predictor_actual_y_mae_m": float(absolute[:, 0].mean()),
            "predictor_actual_z_mae_m": float(absolute[:, 1].mean()),
            "predictor_actual_y_p95_m": float(torch.quantile(absolute[:, 0], 0.95)),
            "predictor_actual_z_p95_m": float(torch.quantile(absolute[:, 1], 0.95)),
            "predictor_actual_crossing_window_n": float(count),
            "predictor_actual_crossing_total_n": float(self._pred_validation_crossings_total),
        }
        hard = self._pred_validation_slow_cpu | self._pred_validation_high_cpu
        metrics["predictor_actual_hard_window_n"] = float(hard.sum())
        if bool(hard.any()):
            metrics["predictor_actual_hard_z_mae_m"] = float(absolute[hard, 1].mean())
            metrics["predictor_actual_hard_z_p95_m"] = float(
                torch.quantile(absolute[hard, 1], 0.95)
            )
        return metrics

    def _maybe_report_predictor_validation(self, iteration: int, metrics: dict[str, float]) -> None:
        """Print fixed-format validation status and warn on a bad plateau."""

        interval = max(1, self.pred_validation_interval_iters)
        if (iteration + 1) % interval != 0:
            return
        self._pred_validation_report_checks += 1
        count = int(metrics.get("predictor_actual_crossing_window_n", 0.0))
        hard_count = int(metrics.get("predictor_actual_hard_window_n", 0.0))
        if count < self.pred_validation_min_samples:
            print(
                "[PREDICTOR_VALIDATE] "
                f"iter={iteration + 1} n={count}/{self.pred_validation_min_samples} "
                "waiting_for_physical_crossings",
                flush=True,
            )
            if (
                iteration + 1 >= self.pred_validation_warn_start_iter
                and self._pred_validation_report_checks >= self.pred_validation_warn_patience
            ):
                print(
                    "[PREDICTOR_ALERT] insufficient causal crossing samples; "
                    "check camera validity, serve resets, and early paddle contacts",
                    flush=True,
                )
            return

        y_mae = metrics["predictor_actual_y_mae_m"]
        z_mae = metrics["predictor_actual_z_mae_m"]
        hard_z_mae = metrics.get("predictor_actual_hard_z_mae_m", float("nan"))
        print(
            "[PREDICTOR_VALIDATE] "
            f"iter={iteration + 1} n={count} y_mae={y_mae:.4f}m z_mae={z_mae:.4f}m "
            f"y_p95={metrics['predictor_actual_y_p95_m']:.4f}m "
            f"z_p95={metrics['predictor_actual_z_p95_m']:.4f}m "
            f"hard_n={hard_count} hard_z_mae={hard_z_mae:.4f}m",
            flush=True,
        )
        if iteration + 1 < self.pred_validation_warn_start_iter:
            return

        ratios = [
            y_mae / max(self.pred_validation_warn_y_mae_m, 1.0e-8),
            z_mae / max(self.pred_validation_warn_z_mae_m, 1.0e-8),
        ]
        if math.isfinite(hard_z_mae) and hard_count >= max(16, self.pred_validation_min_samples // 4):
            ratios.append(hard_z_mae / max(self.pred_validation_warn_hard_z_mae_m, 1.0e-8))
        score = max(ratios)
        required_gain = max(0.0, self.pred_validation_warn_min_rel_improvement)
        if score < self._pred_validation_best_score * (1.0 - required_gain):
            self._pred_validation_best_score = score
            self._pred_validation_no_improve_checks = 0
        else:
            self._pred_validation_no_improve_checks += 1
        if score > 1.0 and self._pred_validation_no_improve_checks >= self.pred_validation_warn_patience:
            print(
                "[PREDICTOR_ALERT] physical hit-plane error is not shrinking enough: "
                f"score={score:.3f} best={self._pred_validation_best_score:.3f} "
                f"stale_checks={self._pred_validation_no_improve_checks}; "
                "inspect y/z/hard curves before continuing the run",
                flush=True,
            )

    def _maybe_predict_and_update_env(self):
        """Run predictor inference given last 5 positions and update env with predictions."""
        if not self._pred_trained:
            if not hasattr(self, "_warn_no_pred"):
                print("[Predictor] Not trained or not loaded; skipping predictions.")
                self._warn_no_pred = True
            return
        # Build batched inputs for all envs with enough history
        if self._traj_len < self.pred_history_len:
            if not hasattr(self, "_warn_short_hist"):
                print(f"[Predictor] Not enough history yet (have {self._traj_len}, need {self.pred_history_len}).")
                self._warn_short_hist = True
            return
        # gather last H time indices in order
        H = self.pred_history_len
        idxs = (torch.arange(-H, 0) + self._traj_write_idx) % self._traj_maxlen  # [H]
        hist = self._traj_buf_cpu[idxs]  # [H, N, 3]
        X = hist.permute(1, 0, 2).reshape(self.env.num_envs, -1).to(self.device)  # [N, H*3]
        history_valid = self._sample_valid_buf_cpu[idxs].all(dim=0)
        same_serve = (
            self._serve_id_buf_cpu[idxs]
            == self._serve_id_buf_cpu[idxs[-1]].unsqueeze(0)
        ).all(dim=0)
        predictor_valid = history_valid & same_serve
        with torch.no_grad():
            preds = self._predictor(X)  # [N, 3]
        self._pred_last_prediction_cpu.copy_(preds.detach().to("cpu"))
        self._pred_last_inference_valid_cpu.copy_(predictor_valid)
        self._pred_last_inference_serve_id_cpu.copy_(self._serve_id_buf_cpu[idxs[-1]])
        self._pred_call_count += 1
        if self._pred_call_count == 1:
            try:
                p0 = preds[0].detach().cpu().numpy()
                print(f"[Predictor] First prediction sample: {p0}")
            except Exception:
                pass
        # If update hook exists in env, call it. Otherwise, skip safely.
        try:
            if hasattr(self.env, "update_prediction"):
                try:
                    self.env.update_prediction(preds, predictor_valid.to(self.env.device))
                except TypeError:
                    # Backward compatibility for non-TT environments with the older hook.
                    self.env.update_prediction(preds)
        except Exception:
            # Do not break PPO rollout if env-side hook is not implemented yet
            pass

    def _train_predictor_offline(self) -> Optional[float]:
        """Create supervised dataset from trajectory buffers and train the predictor.

        Targets in ``_gt_buf_cpu`` come either from the historical environment
        future pose or from a physically measured fixed-plane crossing,
        according to ``pred_target_mode``. For each time t with sufficient
        history H, create X = seq[t-H:t], Y = gt_seq[t-1]. History windows that
        cross a ball reset are skipped to avoid mixing serves.
        """
        H = self.pred_history_len
        L = self._traj_len
        if L < H + 1:
            return None
        # unwrap the last L steps into time order
        idxs = (torch.arange(-L, 0) + self._traj_write_idx) % self._traj_maxlen
        seq = self._traj_buf_cpu[idxs]  # [L, N, 3] on CPU
        gt_seq = self._gt_buf_cpu[idxs]  # [L, N, 3] on CPU
        sample_valid_seq = self._sample_valid_buf_cpu[idxs]  # [L, N] on CPU
        gt_valid_seq = self._gt_valid_buf_cpu[idxs]  # [L, N] on CPU
        serve_id_seq = self._serve_id_buf_cpu[idxs]  # [L, N] on CPU

        X_parts: List[torch.Tensor] = []
        Y_parts: List[torch.Tensor] = []
        selected = 0
        considered = 0
        # Iterate over time (batched over envs). L is capped by traj_max_len.
        for t in range(H, L):
            hist = seq[t - H : t]  # [H, N, 3]
            X_t_full = hist.permute(1, 0, 2).reshape(self.env.num_envs, -1)  # [N, H*3]
            # Use ground-truth at time t-1 (aligned with history window end)
            Y_t_all = gt_seq[t - 1]  # [N, 3]
            hist_valid = sample_valid_seq[t - H : t].all(dim=0)
            target_valid = gt_valid_seq[t - 1]
            same_serve = (serve_id_seq[t - H : t] == serve_id_seq[t - 1].unsqueeze(0)).all(dim=0)
            finite = torch.isfinite(X_t_full).all(dim=-1) & torch.isfinite(Y_t_all).all(dim=-1)
            mask = hist_valid & target_valid & same_serve & finite
            considered += int(mask.numel())
            selected += int(mask.sum().item())
            if mask.any():
                X_parts.append(X_t_full[mask])
                Y_parts.append(Y_t_all[mask])

        if not X_parts:
            self._last_pred_valid_fraction = 0.0
            return None

        X = torch.cat(X_parts, dim=0).to(self.device)
        Y = torch.cat(Y_parts, dim=0).to(self.device)
        self._last_pred_valid_fraction = selected / max(1, considered)
        if X.shape[0] < self.pred_min_valid_samples:
            return None
        # Optional input augmentation: add Gaussian noise if env uses noise
        try:
            if getattr(self.env, "add_noise", False):
                X = X + 0.02 * torch.randn_like(X)
        except Exception:
            pass

        # Simple supervised regression training
        criterion = torch.nn.MSELoss()
        self._predictor.train()
        total_loss = 0.0
        n_batches = 0
        # no shuffling needed: samples already come from varying times and envs

        bs = self.pred_batch_size
        for _ in range(max(1, self.pred_epochs)):
            for s in range(0, X.shape[0], bs):
                e = min(s + bs, X.shape[0])
                xb = X[s:e]
                yb = Y[s:e]
                pred = self._predictor(xb)
                loss = criterion(pred, yb)
                self._pred_optim.zero_grad()
                loss.backward()
                self._pred_optim.step()
                total_loss += float(loss.detach().item())
                n_batches += 1

        self._pred_trained = True
        mean_loss = total_loss / max(1, n_batches)
        return mean_loss

    # ---------------------------
    # Checkpointing (save/load)
    # ---------------------------
    def save(self, path: str, infos=None):
        """Save PPO policy plus auxiliary predictor weights and optimizers."""
        # base dict mirrors parent save()
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # RND (if used by PPO)
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # Normalizers (if used)
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        # Auxiliary predictor
        if hasattr(self, "_predictor") and self._predictor is not None:
            saved_dict["pred_state_dict"] = self._predictor.state_dict()
            saved_dict["pred_optimizer_state_dict"] = self._pred_optim.state_dict()
            saved_dict["pred_cfg"] = {
                "history_len": self.pred_history_len,
                "traj_max_len": self.pred_traj_maxlen,
                "hidden_sizes": list(self.pred_hidden),
                "lr": self.pred_lr,
                "epochs_per_update": self.pred_epochs,
                "batch_size": self.pred_batch_size,
                "target_mode": self.pred_target_mode,
                "validation_window_samples": self.pred_validation_window_samples,
                "validation_interval_iters": self.pred_validation_interval_iters,
                "validation_min_samples": self.pred_validation_min_samples,
                "validation_warn_y_mae_m": self.pred_validation_warn_y_mae_m,
                "validation_warn_z_mae_m": self.pred_validation_warn_z_mae_m,
                "validation_warn_hard_z_mae_m": self.pred_validation_warn_hard_z_mae_m,
                "validation_slow_abs_vx_mps": self.pred_validation_slow_abs_vx_mps,
                "validation_high_z_m": self.pred_validation_high_z_m,
            }

        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        """Load PPO policy and auxiliary predictor if present in checkpoint."""
        loaded_dict = torch.load(path, weights_only=False)
        # -- PPO model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- RND
        if self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Normalizers
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- Optimizers
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- Auxiliary predictor
        if "pred_state_dict" in loaded_dict:
            saved_target_mode = loaded_dict.get("pred_cfg", {}).get("target_mode", "env_future_pose")
            if saved_target_mode != self.pred_target_mode and load_optimizer:
                raise ValueError(
                    "cannot resume predictor training across target semantics: "
                    f"checkpoint={saved_target_mode} current={self.pred_target_mode}. "
                    "Start a fresh run so analytic-apex and actual-hit-plane labels are never mixed."
                )
            try:
                if saved_target_mode != self.pred_target_mode:
                    print(
                        "[Predictor] WARNING target_mode mismatch: "
                        f"checkpoint={saved_target_mode} current={self.pred_target_mode}; "
                        "weights are loaded for visualization only."
                    )
                self._predictor.load_state_dict(loaded_dict["pred_state_dict"])
                if load_optimizer and "pred_optimizer_state_dict" in loaded_dict:
                    self._pred_optim.load_state_dict(loaded_dict["pred_optimizer_state_dict"])
                self._pred_trained = True
                print("[Predictor] Loaded predictor weights from checkpoint; predictions enabled.")
            except Exception:
                # if shape mismatch due to config change, keep running without predictor weights
                self._pred_trained = False
        else:
            try:
                print("[Predictor] No predictor weights found in checkpoint. Keys:", list(loaded_dict.keys()))
            except Exception:
                pass
        # -- current iter
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

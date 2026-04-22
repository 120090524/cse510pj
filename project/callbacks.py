from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from wrappers import infer_fetch_success


class EvalCSVCallback(BaseCallback):
    """Periodic evaluation callback that writes metrics to CSV."""

    def __init__(
        self,
        eval_env_fn: Callable[[], gym.Env],
        csv_path: str | Path,
        best_model_path: str | Path,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.csv_path = Path(csv_path)
        self.best_model_path = Path(best_model_path)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = deterministic
        self._best_score = -np.inf
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized_file = False

    def _on_training_start(self) -> None:
        self._init_csv()

    def _init_csv(self) -> None:
        if self._initialized_file:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timesteps",
                    "mean_reward",
                    "std_reward",
                    "success_rate",
                    "mean_ep_length",
                    "mean_steps_to_success_or_timeout",
                    "mean_steps_to_success_success_only",
                ]
            )
        self._initialized_file = True

    def _evaluate(self) -> tuple[float, float, float, float, float, float]:
        env = self.eval_env_fn()
        rewards: list[float] = []
        successes: list[float] = []
        lengths: list[int] = []
        steps_to_success_or_timeout: list[int] = []
        steps_to_success_success_only: list[int] = []

        for _ in range(self.n_eval_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            ep_success = 0.0
            first_success_step: int | None = None

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)

                if isinstance(obs, dict):
                    current_success = infer_fetch_success(obs, info)
                else:
                    current_success = float(terminated and not truncated)

                ep_success = max(ep_success, current_success)
                if current_success > 0.5 and first_success_step is None:
                    first_success_step = ep_len

            rewards.append(ep_reward)
            successes.append(ep_success)
            lengths.append(ep_len)

            if first_success_step is None:
                steps_to_success_or_timeout.append(ep_len)
            else:
                steps_to_success_or_timeout.append(first_success_step)
                steps_to_success_success_only.append(first_success_step)

        env.close()

        if steps_to_success_success_only:
            mean_success_only = float(np.mean(steps_to_success_success_only))
        else:
            mean_success_only = float("nan")

        return (
            float(np.mean(rewards)),
            float(np.std(rewards)),
            float(np.mean(successes)),
            float(np.mean(lengths)),
            float(np.mean(steps_to_success_or_timeout)),
            mean_success_only,
        )

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            (
                mean_reward,
                std_reward,
                success_rate,
                mean_ep_length,
                mean_steps_to_success_or_timeout,
                mean_steps_to_success_success_only,
            ) = self._evaluate()

            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.num_timesteps,
                        mean_reward,
                        std_reward,
                        success_rate,
                        mean_ep_length,
                        mean_steps_to_success_or_timeout,
                        mean_steps_to_success_success_only,
                    ]
                )

            # For min-time tasks, reward closer to zero also means faster success.
            score = success_rate + 1e-6 * mean_reward
            if score > self._best_score:
                self._best_score = score
                self.model.save(self.best_model_path.as_posix())

            if self.verbose > 0:
                print(
                    f"[Eval] steps={self.num_timesteps} "
                    f"reward={mean_reward:.3f}±{std_reward:.3f} "
                    f"success={success_rate:.3f} len={mean_ep_length:.1f} "
                    f"first_success_or_timeout={mean_steps_to_success_or_timeout:.1f}"
                )
        return True

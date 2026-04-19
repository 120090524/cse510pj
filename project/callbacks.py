from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

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
            writer.writerow(["timesteps", "mean_reward", "std_reward", "success_rate", "mean_ep_length"])
        self._initialized_file = True

    def _evaluate(self) -> tuple[float, float, float, float]:
        env = self.eval_env_fn()
        rewards: list[float] = []
        successes: list[float] = []
        lengths: list[int] = []

        for _ in range(self.n_eval_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            ep_success = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)

                if isinstance(obs, dict):
                    ep_success = max(ep_success, infer_fetch_success(obs, info))
                else:
                    ep_success = max(ep_success, float(terminated and not truncated))

            rewards.append(ep_reward)
            successes.append(ep_success)
            lengths.append(ep_len)

        env.close()
        return (
            float(np.mean(rewards)),
            float(np.std(rewards)),
            float(np.mean(successes)),
            float(np.mean(lengths)),
        )

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            mean_reward, std_reward, success_rate, mean_ep_length = self._evaluate()
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    mean_reward,
                    std_reward,
                    success_rate,
                    mean_ep_length,
                ])

            score = success_rate + 1e-6 * mean_reward
            if score > self._best_score:
                self._best_score = score
                self.model.save(self.best_model_path.as_posix())

            if self.verbose > 0:
                print(
                    f"[Eval] steps={self.num_timesteps} "
                    f"reward={mean_reward:.3f}±{std_reward:.3f} "
                    f"success={success_rate:.3f} len={mean_ep_length:.1f}"
                )
        return True

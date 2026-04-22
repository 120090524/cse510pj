from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

try:
    import gymnasium_robotics  # type: ignore
except Exception:  # pragma: no cover - optional until Fetch experiments
    gymnasium_robotics = None


FETCH_SUCCESS_DISTANCE = 0.05


def resolve_fetch_env_id(env_id: str) -> str:
    """Prefer v4 Fetch envs when a v3 id is provided."""
    if env_id.endswith("-v3"):
        return env_id[:-3] + "-v4"
    return env_id


def register_robotics_envs() -> None:
    """Register Gymnasium-Robotics envs once."""
    if gymnasium_robotics is not None:
        gym.register_envs(gymnasium_robotics)


class MountainCarSparseMinTimeWrapper(gym.Wrapper):
    """
    Sparse minimum-time approximation for MountainCarContinuous.

    Reward = step_penalty until success, success_reward when goal is reached.
    By default this is -1 per step and 0 on the success transition.
    """

    def __init__(
        self,
        env: gym.Env,
        step_penalty: float = -1.0,
        success_reward: float = 0.0,
    ):
        super().__init__(env)
        self.step_penalty = float(step_penalty)
        self.success_reward = float(success_reward)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        success = bool(terminated and not truncated)
        reward = self.success_reward if success else self.step_penalty
        info = dict(info)
        info["is_success"] = float(success)
        return obs, float(reward), terminated, truncated, info


class FetchMinimumTimeWrapper(gym.Wrapper):
    """
    HER-compatible minimum-time wrapper for Fetch goal-reaching tasks.

    Reward:
        - step_penalty until the achieved goal is close enough to the desired goal
        - success_reward on successful transitions

    Termination:
        - if terminate_on_success=True, the episode ends immediately when success is reached
        - if terminate_on_success=False, the reward is minimum-time style but the horizon is unchanged

    HER compatibility:
        Stable-Baselines3 HER needs env.compute_reward(achieved_goal, desired_goal, info)
        so this wrapper implements the same reward rule in a vectorized way.
    """

    def __init__(
        self,
        env: gym.Env,
        step_penalty: float = -1.0,
        success_reward: float = 0.0,
        terminate_on_success: bool = True,
        distance_threshold: float | None = None,
    ):
        super().__init__(env)
        self.step_penalty = float(step_penalty)
        self.success_reward = float(success_reward)
        self.terminate_on_success = bool(terminate_on_success)

        # FetchReach usually uses 0.05. If the wrapped env exposes a threshold, use it.
        if distance_threshold is None:
            distance_threshold = getattr(env.unwrapped, "distance_threshold", FETCH_SUCCESS_DISTANCE)
        self.distance_threshold = float(distance_threshold)

    def _goal_distance(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        return np.linalg.norm(achieved_goal - desired_goal, axis=-1)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return self._goal_distance(achieved_goal, desired_goal) < self.distance_threshold

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Vectorized reward function required by Stable-Baselines3 HerReplayBuffer.

        achieved_goal and desired_goal may be shape (goal_dim,) for one transition
        or shape (batch_size, goal_dim) for HER relabeling.
        """
        success = self._is_success(achieved_goal, desired_goal)
        reward = np.where(success, self.success_reward, self.step_penalty)

        # Keep scalar calls scalar, and batched calls ndarray.
        if np.isscalar(reward) or reward.shape == ():
            return float(reward)
        return reward.astype(np.float32)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)

        achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
        desired = np.asarray(obs["desired_goal"], dtype=np.float32)
        distance = float(self._goal_distance(achieved, desired))
        success = bool(distance < self.distance_threshold)

        reward = self.success_reward if success else self.step_penalty

        # Preserve any termination from the base env; add success termination if requested.
        terminated = bool(terminated) or (self.terminate_on_success and success)

        info = dict(info)
        info["is_success"] = float(success)
        info["distance_to_goal"] = distance
        return obs, float(reward), terminated, truncated, info


@dataclass
class EnvSpec:
    env_id: str
    policy: str
    reward_mode: str


def make_mountaincar_env(reward_mode: str, seed: int | None = None) -> gym.Env:
    env = gym.make("MountainCarContinuous-v0")
    if reward_mode == "sparse":
        env = MountainCarSparseMinTimeWrapper(env)
    elif reward_mode != "dense":
        raise ValueError(f"Unknown reward_mode={reward_mode!r}; expected 'dense' or 'sparse'.")

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_fetch_env(
    env_id: str,
    seed: int | None = None,
    minimum_time: bool = False,
    terminate_on_success: bool = True,
) -> gym.Env:
    register_robotics_envs()
    env_id = resolve_fetch_env_id(env_id)
    env = gym.make(env_id)

    if minimum_time:
        env = FetchMinimumTimeWrapper(
            env,
            terminate_on_success=terminate_on_success,
        )

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def infer_fetch_success(
    obs: dict[str, Any],
    info: dict[str, Any],
    threshold: float = FETCH_SUCCESS_DISTANCE,
) -> float:
    if "is_success" in info:
        try:
            return float(info["is_success"])
        except Exception:
            pass

    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    return float(np.linalg.norm(achieved - desired) < threshold)

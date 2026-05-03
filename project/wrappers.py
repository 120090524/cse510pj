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


# ---------------------------------------------------------------------------
# Registration and env-id helpers
# ---------------------------------------------------------------------------

def resolve_fetch_env_id(env_id: str) -> str:
    """Prefer v4 Fetch envs when a v3 id is provided."""
    if env_id.endswith("-v3"):
        return env_id[:-3] + "-v4"
    return env_id



def to_fetch_dense_env_id(env_id: str) -> str:
    """Convert FetchReach-v4 -> FetchReachDense-v4 if needed."""
    env_id = resolve_fetch_env_id(env_id)
    if "Dense-" in env_id:
        return env_id
    if "Fetch" not in env_id:
        return env_id
    name, version = env_id.rsplit("-", 1)
    return f"{name}Dense-{version}"



def to_fetch_sparse_env_id(env_id: str) -> str:
    """Convert FetchReachDense-v4 -> FetchReach-v4 if needed."""
    env_id = resolve_fetch_env_id(env_id)
    return env_id.replace("Dense-", "-")



def register_robotics_envs() -> None:
    """Register Gymnasium-Robotics envs once."""
    if gymnasium_robotics is not None:
        gym.register_envs(gymnasium_robotics)


# ---------------------------------------------------------------------------
# Shared Fetch helpers
# ---------------------------------------------------------------------------

def fetch_goal_distance(achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
    achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
    desired_goal = np.asarray(desired_goal, dtype=np.float32)
    return np.linalg.norm(achieved_goal - desired_goal, axis=-1)



def fetch_is_success(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    threshold: float = FETCH_SUCCESS_DISTANCE,
) -> np.ndarray:
    return fetch_goal_distance(achieved_goal, desired_goal) < threshold


# ---------------------------------------------------------------------------
# MountainCar wrapper (unchanged behavior)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Fetch wrappers
# ---------------------------------------------------------------------------
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
        if distance_threshold is None:
            distance_threshold = getattr(env.unwrapped, "distance_threshold", FETCH_SUCCESS_DISTANCE)
        self.distance_threshold = float(distance_threshold)

    def _goal_distance(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return fetch_goal_distance(achieved_goal, desired_goal)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return fetch_is_success(achieved_goal, desired_goal, self.distance_threshold)

    def compute_reward(self, achieved_goal, desired_goal, info):
        success = self._is_success(achieved_goal, desired_goal)
        reward = np.where(success, self.success_reward, self.step_penalty)
        if np.isscalar(reward) or np.asarray(reward).shape == ():
            return float(reward)
        return np.asarray(reward, dtype=np.float32)

    def step(self, action: np.ndarray):
        obs, raw_env_reward, terminated, truncated, info = self.env.step(action)
        achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
        desired = np.asarray(obs["desired_goal"], dtype=np.float32)
        distance = float(self._goal_distance(achieved, desired))
        success = bool(distance < self.distance_threshold)

        reward = self.success_reward if success else self.step_penalty
        terminated = bool(terminated) or (self.terminate_on_success and success)

        info = dict(info)
        info["is_success"] = float(success)
        info["distance_to_goal"] = distance
        info["raw_env_reward"] = float(raw_env_reward)
        info["base_reward"] = float(reward)
        info["shaping_bonus"] = 0.0
        return obs, float(reward), terminated, truncated, info


class FetchPBRSMinTimeWrapper(gym.Wrapper):
    """
    Minimum-time reward plus policy-preserving potential-based shaping (PBRS).

    Base reward:
        0 on success, -1 otherwise.

    Shaped reward:
        r'(s, a, s') = r_base(s, a, s') + gamma * Phi(s') - Phi(s)

    Potential:
        Phi(s) = -potential_scale * ||achieved_goal - desired_goal||_2

    This wrapper is intended for plain SAC training, not HER. HER needs a reward that
    can be recomputed from only (achieved_goal, desired_goal, info). PBRS depends on the
    transition (previous state and next state), so we fail loudly if HER tries to use it.
    """

    def __init__(
        self,
        env: gym.Env,
        shaping_gamma: float = 0.99,
        potential_scale: float = 1.0,
        step_penalty: float = -1.0,
        success_reward: float = 0.0,
        terminate_on_success: bool = True,
        distance_threshold: float | None = None,
    ):
        super().__init__(env)
        self.shaping_gamma = float(shaping_gamma)
        self.potential_scale = float(potential_scale)
        self.step_penalty = float(step_penalty)
        self.success_reward = float(success_reward)
        self.terminate_on_success = bool(terminate_on_success)
        if distance_threshold is None:
            distance_threshold = getattr(env.unwrapped, "distance_threshold", FETCH_SUCCESS_DISTANCE)
        self.distance_threshold = float(distance_threshold)
        self._prev_achieved_goal: np.ndarray | None = None
        self._prev_desired_goal: np.ndarray | None = None

    def _goal_distance(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return fetch_goal_distance(achieved_goal, desired_goal)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return fetch_is_success(achieved_goal, desired_goal, self.distance_threshold)

    def _phi(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return -self.potential_scale * self._goal_distance(achieved_goal, desired_goal)

    def compute_reward(self, achieved_goal, desired_goal, info):  # pragma: no cover - safety guard
        raise NotImplementedError(
            "FetchPBRSMinTimeWrapper is transition-based and not HER-compatible. "
            "Use plain SAC, not HER, for reward_mode='pbrs_min_time'."
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32).copy()
        self._prev_desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32).copy()
        return obs, info

    def step(self, action: np.ndarray):
        if self._prev_achieved_goal is None or self._prev_desired_goal is None:
            raise RuntimeError("Call reset() before step() in FetchPBRSMinTimeWrapper.")

        prev_ag = self._prev_achieved_goal.copy()
        prev_dg = self._prev_desired_goal.copy()

        obs, raw_env_reward, terminated, truncated, info = self.env.step(action)
        ag = np.asarray(obs["achieved_goal"], dtype=np.float32)
        dg = np.asarray(obs["desired_goal"], dtype=np.float32)
        success = bool(self._is_success(ag, dg))

        base_reward = self.success_reward if success else self.step_penalty
        shaping_bonus = float(self.shaping_gamma * self._phi(ag, dg) - self._phi(prev_ag, prev_dg))
        reward = float(base_reward + shaping_bonus)

        terminated = bool(terminated) or (self.terminate_on_success and success)

        info = dict(info)
        info["is_success"] = float(success)
        info["distance_to_goal"] = float(self._goal_distance(ag, dg))
        info["raw_env_reward"] = float(raw_env_reward)
        info["base_reward"] = float(base_reward)
        info["shaping_bonus"] = float(shaping_bonus)

        self._prev_achieved_goal = ag.copy()
        self._prev_desired_goal = dg.copy()
        return obs, reward, terminated, truncated, info


class FetchAdHocDistanceWrapper(gym.Wrapper):
    """
    Minimum-time reward plus a configurable non-PBRS dense bonus.

    Core control reward:
        r'(s, a, s') = r_base(s, a, s') - distance_scale * D_shape(s')

    True task semantics remain unchanged:
      - success is always measured against the REAL desired_goal and REAL env threshold
      - base reward is always 0 on success, -1 otherwise
      - terminate_on_success, if enabled, also uses the REAL desired_goal

    Misspecification knobs affect only the shaping term:
      1) goal_offset: use desired_goal + offset inside the shaping distance
      2) action_penalty_scale: add -lambda ||a||^2 to the shaping term
      3) shaping_threshold: use max(distance_to_shaping_goal - shaping_threshold, 0)
         instead of raw distance. If shaping_threshold is wrong, the dense guidance is wrong.

    This wrapper is useful as a control because it is dense, but it is NOT potential-based.
    So it does not come with the policy-invariance guarantee from Ng et al. (1999).
    Like PBRS, this wrapper is intended for plain SAC training, not HER.
    """

    def __init__(
        self,
        env: gym.Env,
        distance_scale: float = 1.0,
        step_penalty: float = -1.0,
        success_reward: float = 0.0,
        terminate_on_success: bool = True,
        distance_threshold: float | None = None,
        goal_offset: np.ndarray | None = None,
        action_penalty_scale: float = 0.0,
        shaping_threshold: float | None = None,
    ):
        super().__init__(env)
        self.distance_scale = float(distance_scale)
        self.step_penalty = float(step_penalty)
        self.success_reward = float(success_reward)
        self.terminate_on_success = bool(terminate_on_success)
        if distance_threshold is None:
            distance_threshold = getattr(env.unwrapped, "distance_threshold", FETCH_SUCCESS_DISTANCE)
        self.distance_threshold = float(distance_threshold)
        if goal_offset is None:
            goal_offset = np.zeros(3, dtype=np.float32)
        self.goal_offset = np.asarray(goal_offset, dtype=np.float32).reshape(-1)
        self.action_penalty_scale = float(action_penalty_scale)
        self.shaping_threshold = None if shaping_threshold is None else float(shaping_threshold)

    def _goal_distance(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return fetch_goal_distance(achieved_goal, desired_goal)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return fetch_is_success(achieved_goal, desired_goal, self.distance_threshold)

    def _shaping_goal(self, desired_goal: np.ndarray) -> np.ndarray:
        return np.asarray(desired_goal, dtype=np.float32) + self.goal_offset

    def _shaping_distance_raw(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return fetch_goal_distance(achieved_goal, self._shaping_goal(desired_goal))

    def _shaping_distance_effective(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        raw = self._shaping_distance_raw(achieved_goal, desired_goal)
        if self.shaping_threshold is None:
            return raw
        return np.maximum(raw - self.shaping_threshold, 0.0)

    def compute_reward(self, achieved_goal, desired_goal, info):  # pragma: no cover - safety guard
        raise NotImplementedError(
            "FetchAdHocDistanceWrapper is not HER-compatible. "
            "Use plain SAC, not HER, for reward_mode='adhoc_distance'."
        )

    def step(self, action: np.ndarray):
        obs, raw_env_reward, terminated, truncated, info = self.env.step(action)
        ag = np.asarray(obs["achieved_goal"], dtype=np.float32)
        dg = np.asarray(obs["desired_goal"], dtype=np.float32)

        true_distance = float(self._goal_distance(ag, dg))
        success = bool(self._is_success(ag, dg))

        shaping_distance_raw = float(self._shaping_distance_raw(ag, dg))
        shaping_distance_effective = float(self._shaping_distance_effective(ag, dg))
        action_penalty = float(self.action_penalty_scale * np.sum(np.square(np.asarray(action, dtype=np.float32))))

        base_reward = self.success_reward if success else self.step_penalty
        shaping_bonus = -self.distance_scale * shaping_distance_effective - action_penalty
        reward = float(base_reward + shaping_bonus)

        terminated = bool(terminated) or (self.terminate_on_success and success)

        info = dict(info)
        info["is_success"] = float(success)
        info["distance_to_goal"] = true_distance
        info["raw_env_reward"] = float(raw_env_reward)
        info["base_reward"] = float(base_reward)
        info["shaping_bonus"] = float(shaping_bonus)
        info["shaping_distance_raw"] = shaping_distance_raw
        info["shaping_distance_effective"] = shaping_distance_effective
        info["action_penalty_term"] = float(action_penalty)
        info["shaping_threshold"] = self.shaping_threshold
        info["goal_offset_x"] = float(self.goal_offset[0]) if self.goal_offset.size > 0 else 0.0
        info["goal_offset_y"] = float(self.goal_offset[1]) if self.goal_offset.size > 1 else 0.0
        info["goal_offset_z"] = float(self.goal_offset[2]) if self.goal_offset.size > 2 else 0.0
        return obs, reward, terminated, truncated, info


@dataclass
class EnvSpec:
    env_id: str
    policy: str
    reward_mode: str


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

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
    reward_mode: str | None = None,
    shaping_gamma: float = 0.99,
    potential_scale: float = 1.0,
    distance_scale: float = 1.0,
    goal_offset: np.ndarray | None = None,
    action_penalty_scale: float = 0.0,
    shaping_threshold: float | None = None,
) -> gym.Env:
    """
    Create a Fetch env with backward compatibility.

    Old code path:
        make_fetch_env(env_id, minimum_time=True/False, terminate_on_success=...)

    New code path:
        make_fetch_env(env_id, reward_mode='pbrs_min_time', ...)
    """
    register_robotics_envs()

    if reward_mode is None or reward_mode == "auto":
        if minimum_time:
            reward_mode = "min_time"
        elif "Dense" in env_id:
            reward_mode = "official_dense"
        else:
            reward_mode = "official_sparse"

    if reward_mode == "official_dense":
        resolved_env_id = to_fetch_dense_env_id(env_id)
        env = gym.make(resolved_env_id)
    else:
        resolved_env_id = to_fetch_sparse_env_id(env_id)
        env = gym.make(resolved_env_id)

        if reward_mode == "official_sparse":
            pass
        elif reward_mode == "min_time":
            env = FetchMinimumTimeWrapper(
                env,
                terminate_on_success=terminate_on_success,
            )
        elif reward_mode == "pbrs_min_time":
            env = FetchPBRSMinTimeWrapper(
                env,
                shaping_gamma=shaping_gamma,
                potential_scale=potential_scale,
                terminate_on_success=terminate_on_success,
            )
        elif reward_mode == "adhoc_distance":
            env = FetchAdHocDistanceWrapper(
                env,
                distance_scale=distance_scale,
                terminate_on_success=terminate_on_success,
                goal_offset=goal_offset,
                action_penalty_scale=action_penalty_scale,
                shaping_threshold=shaping_threshold,
            )
        else:
            raise ValueError(
                f"Unknown reward_mode={reward_mode!r}. "
                "Expected one of: official_dense, official_sparse, min_time, pbrs_min_time, adhoc_distance."
            )

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Inference helpers used by callbacks/evaluation
# ---------------------------------------------------------------------------

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



def infer_fetch_distance(
    obs: dict[str, Any],
    info: dict[str, Any],
) -> float:
    if "distance_to_goal" in info:
        try:
            return float(info["distance_to_goal"])
        except Exception:
            pass
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    return float(np.linalg.norm(achieved - desired))

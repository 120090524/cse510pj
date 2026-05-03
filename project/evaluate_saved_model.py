from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from wrappers import (
    infer_fetch_distance,
    infer_fetch_success,
    make_fetch_env,
    make_mountaincar_env,
    register_robotics_envs,
)


FETCH_REWARD_CHOICES = [
    "auto",
    "official_dense",
    "official_sparse",
    "min_time",
    "pbrs_min_time",
    "adhoc_distance",
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved SAC model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", choices=["mountaincar", "fetch"], required=True)
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="auto",
        help=(
            "For fetch: auto / official_dense / official_sparse / min_time / pbrs_min_time / adhoc_distance. "
            "For mountaincar: dense or sparse."
        ),
    )
    parser.add_argument("--env_id", type=str, default="FetchReach-v4")
    parser.add_argument("--minimum_time", action="store_true")
    parser.add_argument("--terminate_on_success", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--potential_scale", type=float, default=1.0)
    parser.add_argument("--distance_scale", type=float, default=1.0)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()



def resolve_fetch_reward_mode(args: argparse.Namespace) -> str:
    if args.reward_mode != "auto":
        return args.reward_mode
    if args.minimum_time:
        return "min_time"
    return "official_dense" if "Dense" in args.env_id else "official_sparse"



def main() -> None:
    args = parse_args()
    register_robotics_envs()

    if args.task == "mountaincar":
        reward_mode = args.reward_mode
        if reward_mode == "auto":
            reward_mode = "dense"
        env = make_mountaincar_env(reward_mode, seed=args.seed)
        policy_hint = "MlpPolicy"
    else:
        reward_mode = resolve_fetch_reward_mode(args)
        env = make_fetch_env(
            args.env_id,
            seed=args.seed,
            minimum_time=args.minimum_time,
            terminate_on_success=args.terminate_on_success,
            reward_mode=reward_mode,
            shaping_gamma=args.gamma,
            potential_scale=args.potential_scale,
            distance_scale=args.distance_scale,
        )
        policy_hint = "MultiInputPolicy"

    model = SAC.load(args.model_path, env=env, device=args.device)

    rewards: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []
    steps_to_success_or_timeout: list[int] = []
    steps_to_success_success_only: list[int] = []
    final_distances: list[float] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_success = 0.0
        first_success_step: int | None = None
        final_distance = float("nan")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)

            if args.task == "mountaincar":
                current_success = float(terminated and not truncated)
                final_distance = float("nan")
            else:
                current_success = infer_fetch_success(obs, info)
                final_distance = infer_fetch_distance(obs, info)

            ep_success = max(ep_success, current_success)
            if current_success > 0.5 and first_success_step is None:
                first_success_step = ep_len

        rewards.append(ep_reward)
        successes.append(ep_success)
        lengths.append(ep_len)
        final_distances.append(final_distance)
        if first_success_step is None:
            steps_to_success_or_timeout.append(ep_len)
        else:
            steps_to_success_or_timeout.append(first_success_step)
            steps_to_success_success_only.append(first_success_step)

    summary = {
        "task": args.task,
        "policy_hint": policy_hint,
        "reward_mode": reward_mode,
        "mean_reward": float(np.mean(rewards)),
        "mean_success": float(np.mean(successes)),
        "mean_ep_length": float(np.mean(lengths)),
        "mean_steps_to_success_or_timeout": float(np.mean(steps_to_success_or_timeout)),
        "mean_steps_to_success_success_only": (
            float(np.mean(steps_to_success_success_only)) if steps_to_success_success_only else None
        ),
        "mean_final_distance_to_goal": (
            float(np.nanmean(final_distances)) if args.task == "fetch" else None
        ),
        "episodes": args.episodes,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    env.close()


if __name__ == "__main__":
    main()

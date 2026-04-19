from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import SAC

from wrappers import make_fetch_env, make_mountaincar_env, register_robotics_envs, infer_fetch_success



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved SAC model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", choices=["mountaincar", "fetch"], required=True)
    parser.add_argument("--reward_mode", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--env_id", type=str, default="FetchReach-v4")
    parser.add_argument("--minimum_time", action="store_true")
    parser.add_argument("--terminate_on_success", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    register_robotics_envs()

    if args.task == "mountaincar":
        env = make_mountaincar_env(args.reward_mode, seed=args.seed)
        policy_hint = "MlpPolicy"
    else:
        env = make_fetch_env(
            args.env_id,
            seed=args.seed,
            minimum_time=args.minimum_time,
            terminate_on_success=args.terminate_on_success,
        )
        policy_hint = "MultiInputPolicy"

    model = SAC.load(args.model_path, env=env, device=args.device)

    rewards = []
    successes = []
    lengths = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_success = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)
            if args.task == "mountaincar":
                ep_success = max(ep_success, float(terminated and not truncated))
            else:
                ep_success = max(ep_success, infer_fetch_success(obs, info))
        rewards.append(ep_reward)
        successes.append(ep_success)
        lengths.append(ep_len)

    summary = {
        "task": args.task,
        "policy_hint": policy_hint,
        "mean_reward": sum(rewards) / len(rewards),
        "mean_success": sum(successes) / len(successes),
        "mean_ep_length": sum(lengths) / len(lengths),
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

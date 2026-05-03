from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from callbacks import EvalCSVCallback
from train_common import save_metadata, set_global_seed, wrap_monitor
from wrappers import make_fetch_env, register_robotics_envs


REWARD_MODE_CHOICES = [
    "auto",
    "official_dense",
    "official_sparse",
    "min_time",
    "pbrs_min_time",
    "adhoc_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train SAC on Fetch with dense, sparse, minimum-time, PBRS minimum-time, "
            "or ad-hoc dense shaping rewards. Also supports ad-hoc misspecification stress tests."
        )
    )
    parser.add_argument("--env_id", type=str, default="FetchReach-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=250_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--learning_starts", type=int, default=5_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=REWARD_MODE_CHOICES,
        default="auto",
        help=(
            "New interface. 'auto' keeps backward compatibility with the old script: "
            "--minimum_time wins first; otherwise Dense in env_id -> official_dense; else official_sparse."
        ),
    )
    parser.add_argument(
        "--minimum_time",
        action="store_true",
        help="Backward-compatible flag. Equivalent to --reward_mode min_time when reward_mode=auto.",
    )
    parser.add_argument(
        "--terminate_on_success",
        action="store_true",
        help="When using min_time / PBRS / ad-hoc shaping, terminate the episode on success.",
    )
    parser.add_argument(
        "--potential_scale",
        type=float,
        default=1.0,
        help="Potential coefficient alpha in Phi(s) = -alpha * distance for PBRS.",
    )
    parser.add_argument(
        "--distance_scale",
        type=float,
        default=1.0,
        help="Coefficient beta for the non-PBRS ad-hoc dense shaping control.",
    )
    # Misspecification stress-test knobs for adhoc_distance
    parser.add_argument("--goal_offset_x", type=float, default=0.0)
    parser.add_argument("--goal_offset_y", type=float, default=0.0)
    parser.add_argument("--goal_offset_z", type=float, default=0.0)
    parser.add_argument(
        "--action_penalty_scale",
        type=float,
        default=0.0,
        help="Extra action penalty lambda for misspecification stress tests: -lambda ||a||^2.",
    )
    parser.add_argument(
        "--shaping_threshold",
        type=float,
        default=None,
        help=(
            "Optional shaping-only threshold tau. If set, adhoc shaping uses max(distance_to_shaping_goal - tau, 0). "
            "The true success threshold remains unchanged."
        ),
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help="Optional folder name override. Useful for misspecification variants so multiple adhoc runs do not overwrite each other.",
    )
    parser.add_argument("--outdir", type=str, default="./project_outputs/fetch")
    return parser.parse_args()



def resolve_reward_mode(args: argparse.Namespace) -> str:
    if args.reward_mode != "auto":
        return args.reward_mode
    if args.minimum_time:
        return "min_time"
    return "official_dense" if "Dense" in args.env_id else "official_sparse"



def experiment_name_from_reward_mode(reward_mode: str) -> str:
    mapping = {
        "official_dense": "fetch_dense_sac",
        "official_sparse": "fetch_sparse_sac",
        "min_time": "fetch_min_time_sac",
        "pbrs_min_time": "fetch_pbrs_sac",
        "adhoc_distance": "fetch_adhoc_shaping_sac",
    }
    if reward_mode not in mapping:
        raise ValueError(f"Unsupported reward_mode={reward_mode!r}")
    return mapping[reward_mode]



def main() -> None:
    args = parse_args()
    reward_mode = resolve_reward_mode(args)

    set_global_seed(args.seed)
    register_robotics_envs()

    exp_name = args.experiment_name.strip() or experiment_name_from_reward_mode(reward_mode)
    run_dir = Path(args.outdir) / exp_name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    goal_offset = [args.goal_offset_x, args.goal_offset_y, args.goal_offset_z]

    env = wrap_monitor(
        make_fetch_env(
            args.env_id,
            seed=args.seed,
            minimum_time=args.minimum_time,
            terminate_on_success=args.terminate_on_success,
            reward_mode=reward_mode,
            shaping_gamma=args.gamma,
            potential_scale=args.potential_scale,
            distance_scale=args.distance_scale,
            goal_offset=goal_offset,
            action_penalty_scale=args.action_penalty_scale,
            shaping_threshold=args.shaping_threshold,
        ),
        run_dir,
    )

    def eval_env_fn():
        return make_fetch_env(
            args.env_id,
            seed=args.seed + 10_000,
            minimum_time=args.minimum_time,
            terminate_on_success=args.terminate_on_success,
            reward_mode=reward_mode,
            shaping_gamma=args.gamma,
            potential_scale=args.potential_scale,
            distance_scale=args.distance_scale,
            goal_offset=goal_offset,
            action_penalty_scale=args.action_penalty_scale,
            shaping_threshold=args.shaping_threshold,
        )

    callback = EvalCSVCallback(
        eval_env_fn=eval_env_fn,
        csv_path=run_dir / "eval_history.csv",
        best_model_path=run_dir / "best_model",
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        verbose=1,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        seed=args.seed,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        device=args.device,
    )

    save_metadata(
        run_dir,
        {
            "experiment": exp_name,
            "env_id": args.env_id,
            "reward_mode": reward_mode,
            "minimum_time_flag": args.minimum_time,
            "terminate_on_success": args.terminate_on_success,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "algo": "SAC",
            "policy": "MultiInputPolicy",
            "gamma": args.gamma,
            "potential_scale": args.potential_scale,
            "distance_scale": args.distance_scale,
            "goal_offset": goal_offset,
            "action_penalty_scale": args.action_penalty_scale,
            "shaping_threshold": args.shaping_threshold,
        },
    )

    print("=" * 80)
    print("Running Fetch SAC experiment")
    print(f"  env_id              : {args.env_id}")
    print(f"  reward_mode         : {reward_mode}")
    print(f"  experiment_name     : {exp_name}")
    print(f"  terminate_on_success: {args.terminate_on_success}")
    print(f"  gamma               : {args.gamma}")
    print(f"  potential_scale     : {args.potential_scale}")
    print(f"  distance_scale      : {args.distance_scale}")
    print(f"  goal_offset         : {goal_offset}")
    print(f"  action_penalty_scale: {args.action_penalty_scale}")
    print(f"  shaping_threshold   : {args.shaping_threshold}")
    print(f"  seed                : {args.seed}")
    print(f"  output              : {run_dir}")
    print("=" * 80)

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)
    model.save(str(run_dir / "final_model"))
    env.close()


if __name__ == "__main__":
    main()

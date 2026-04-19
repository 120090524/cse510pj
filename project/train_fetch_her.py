from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer

from callbacks import EvalCSVCallback
from train_common import save_metadata, set_global_seed, wrap_monitor
from wrappers import make_fetch_env, register_robotics_envs



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC+HER on FetchReach sparse rewards.")
    parser.add_argument("--env_id", type=str, default="FetchReach-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=250_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--learning_starts", type=int, default=5_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_sampled_goal", type=int, default=4)
    parser.add_argument("--goal_selection_strategy", type=str, default="future")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="./project_outputs/fetch")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    if "Dense" in args.env_id:
        raise ValueError("Use the sparse FetchReach env for HER, e.g. FetchReach-v4.")

    set_global_seed(args.seed)
    register_robotics_envs()

    exp_name = "fetch_sparse_sac_her"
    run_dir = Path(args.outdir) / exp_name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = wrap_monitor(make_fetch_env(args.env_id, seed=args.seed, minimum_time=False), run_dir)

    def eval_env_fn():
        return make_fetch_env(args.env_id, seed=args.seed + 10_000, minimum_time=False)

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
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": args.goal_selection_strategy,
        },
        seed=args.seed,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        device=args.device,
    )

    save_metadata(
        run_dir,
        {
            "experiment": exp_name,
            "env_id": args.env_id,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "algo": "SAC+HER",
            "policy": "MultiInputPolicy",
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": args.goal_selection_strategy,
        },
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)
    model.save(str(run_dir / "final_model"))
    env.close()


if __name__ == "__main__":
    main()

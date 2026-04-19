from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from callbacks import EvalCSVCallback
from train_common import save_metadata, set_global_seed, wrap_monitor
from wrappers import make_fetch_env, register_robotics_envs



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on FetchReach dense/sparse.")
    parser.add_argument("--env_id", type=str, default="FetchReach-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=250_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--learning_starts", type=int, default=5_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--minimum_time", action="store_true", help="Wrap sparse FetchReach into a paper-like min-time task.")
    parser.add_argument("--terminate_on_success", action="store_true", help="When using --minimum_time, terminate the episode on success.")
    parser.add_argument("--outdir", type=str, default="./project_outputs/fetch")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    register_robotics_envs()

    reward_tag = "min_time" if args.minimum_time else ("dense" if "Dense" in args.env_id else "sparse")
    exp_name = f"fetch_{reward_tag}_sac"
    run_dir = Path(args.outdir) / exp_name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = wrap_monitor(
        make_fetch_env(
            args.env_id,
            seed=args.seed,
            minimum_time=args.minimum_time,
            terminate_on_success=args.terminate_on_success,
        ),
        run_dir,
    )

    def eval_env_fn():
        return make_fetch_env(
            args.env_id,
            seed=args.seed + 10_000,
            minimum_time=args.minimum_time,
            terminate_on_success=args.terminate_on_success,
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
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        device=args.device,
    )

    save_metadata(
        run_dir,
        {
            "experiment": exp_name,
            "env_id": args.env_id,
            "minimum_time": args.minimum_time,
            "terminate_on_success": args.terminate_on_success,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "algo": "SAC",
            "policy": "MultiInputPolicy",
        },
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)
    model.save(str(run_dir / "final_model"))
    env.close()


if __name__ == "__main__":
    main()

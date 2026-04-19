from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from callbacks import EvalCSVCallback
from train_common import save_metadata, set_global_seed, wrap_monitor
from wrappers import make_mountaincar_env



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on MountainCarContinuous dense vs sparse.")
    parser.add_argument("--reward_mode", choices=["dense", "sparse"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=150_000)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--learning_starts", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=200_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--outdir", type=str, default="./project_outputs/mountaincar")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    exp_name = f"mountaincar_{args.reward_mode}_sac"
    run_dir = Path(args.outdir) / exp_name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = wrap_monitor(make_mountaincar_env(args.reward_mode, seed=args.seed), run_dir)

    def eval_env_fn():
        return make_mountaincar_env(args.reward_mode, seed=args.seed + 10_000)

    callback = EvalCSVCallback(
        eval_env_fn=eval_env_fn,
        csv_path=run_dir / "eval_history.csv",
        best_model_path=run_dir / "best_model",
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        verbose=1,
    )

    model = SAC(
        policy="MlpPolicy",
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
            "env_id": "MountainCarContinuous-v0",
            "reward_mode": args.reward_mode,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "algo": "SAC",
            "policy": "MlpPolicy",
        },
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=True)
    model.save(str(run_dir / "final_model"))
    env.close()


if __name__ == "__main__":
    main()

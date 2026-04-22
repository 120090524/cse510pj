from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import SAC

from callbacks import EvalCSVCallback
from train_common import save_metadata, set_global_seed, wrap_monitor
from wrappers import make_fetch_env, register_robotics_envs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 training for HER-assisted minimum-time learning. "
            "Load a policy pretrained with SAC+HER, then fine-tune it with plain SAC "
            "on a minimum-time Fetch environment."
        )
    )
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="FetchReach-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--finetune_timesteps", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--learning_starts", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="./project_outputs/fetch")

    # Default is True because stage 2 is meant to optimize fast goal completion.
    parser.add_argument(
        "--terminate_on_success",
        dest="terminate_on_success",
        action="store_true",
        default=True,
        help="Terminate evaluation/training episodes on success. Default: True.",
    )
    parser.add_argument(
        "--no_terminate_on_success",
        dest="terminate_on_success",
        action="store_false",
        help="Keep fixed-horizon episodes even with minimum-time reward.",
    )
    return parser.parse_args()


def copy_policy_weights(pretrained: SAC, target: SAC) -> None:
    """Copy actor/critic weights from a pretrained SAC model into a fresh SAC model."""
    target.policy.load_state_dict(pretrained.policy.state_dict())

    # Optional: copy SAC entropy temperature if both models use automatic entropy tuning.
    if hasattr(pretrained, "log_ent_coef") and hasattr(target, "log_ent_coef"):
        with torch.no_grad():
            target.log_ent_coef.data.copy_(pretrained.log_ent_coef.data)


def main() -> None:
    args = parse_args()
    if "Dense" in args.env_id:
        raise ValueError("Use sparse Fetch env for this experiment, e.g. FetchReach-v4.")

    set_global_seed(args.seed)
    register_robotics_envs()

    exp_name = "fetch_her_to_min_time_sac"
    run_dir = Path(args.outdir) / exp_name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load the HER-pretrained model with its original sparse-style environment.
    # This avoids SB3 HER loading issues because HER needs env.compute_reward().
    pretrain_env = make_fetch_env(args.env_id, seed=args.seed + 20_000, minimum_time=False)
    pretrained = SAC.load(args.pretrained_model_path, env=pretrain_env, device=args.device)

    # Create a fresh plain-SAC model on the minimum-time environment.
    env = wrap_monitor(
        make_fetch_env(
            args.env_id,
            seed=args.seed,
            minimum_time=True,
            terminate_on_success=args.terminate_on_success,
        ),
        run_dir,
    )

    def eval_env_fn():
        return make_fetch_env(
            args.env_id,
            seed=args.seed + 10_000,
            minimum_time=True,
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
    copy_policy_weights(pretrained, model)

    save_metadata(
        run_dir,
        {
            "experiment": exp_name,
            "env_id": args.env_id,
            "seed": args.seed,
            "finetune_timesteps": args.finetune_timesteps,
            "algo": "SAC fine-tuning from SAC+HER policy",
            "policy": "MultiInputPolicy",
            "minimum_time": True,
            "terminate_on_success": args.terminate_on_success,
            "pretrained_model_path": args.pretrained_model_path,
            "learning_starts": args.learning_starts,
        },
    )

    model.learn(total_timesteps=args.finetune_timesteps, callback=callback, progress_bar=True)
    model.save(str(run_dir / "final_model"))

    env.close()
    pretrain_env.close()


if __name__ == "__main__":
    main()

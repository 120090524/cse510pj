from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate eval_history.csv files and make comparison plots.")
    parser.add_argument("--root_dir", type=str, default="./project_outputs")
    parser.add_argument("--task", choices=["mountaincar", "fetch"], required=True)
    parser.add_argument("--output_dir", type=str, default="./project_plots")
    return parser.parse_args()



def collect_runs(root_dir: Path, prefix: str) -> pd.DataFrame:
    records = []
    for csv_path in root_dir.glob(f"{prefix}*/seed_*/eval_history.csv"):
        experiment = csv_path.parent.parent.name
        seed = csv_path.parent.name.replace("seed_", "")
        df = pd.read_csv(csv_path)
        df["experiment"] = experiment
        df["seed"] = seed
        records.append(df)
    if not records:
        raise FileNotFoundError(f"No eval_history.csv files found under {root_dir} for prefix={prefix!r}")
    return pd.concat(records, ignore_index=True)



def plot_metric(df: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for experiment, g in df.groupby("experiment"):
        agg = g.groupby("timesteps")[metric].agg(["mean", "std"]).reset_index()
        plt.plot(agg["timesteps"], agg["mean"], label=experiment)
        plt.fill_between(agg["timesteps"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()



def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)

    prefix = "mountaincar_" if args.task == "mountaincar" else "fetch_"
    df = collect_runs(root_dir, prefix)

    plot_metric(df, "success_rate", f"{args.task}: success rate", output_dir / f"{args.task}_success_rate.png")
    plot_metric(df, "mean_reward", f"{args.task}: mean reward", output_dir / f"{args.task}_mean_reward.png")
    plot_metric(df, "mean_ep_length", f"{args.task}: episode length", output_dir / f"{args.task}_episode_length.png")

    summary = (
        df.sort_values("timesteps")
        .groupby(["experiment", "seed"], as_index=False)
        .tail(1)
        .groupby("experiment")[["mean_reward", "success_rate", "mean_ep_length"]]
        .agg(["mean", "std"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{args.task}_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "success_rate",
    "mean_reward",
    "mean_ep_length",
    "mean_steps_to_success_or_timeout",
    "mean_steps_to_success_success_only",
    "mean_final_distance_to_goal",
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate eval_history.csv files and make comparison plots."
    )
    parser.add_argument("--root_dir", type=str, default="./project_outputs")
    parser.add_argument("--task", choices=["mountaincar", "fetch"], required=True)
    parser.add_argument("--output_dir", type=str, default="./project_plots")
    return parser.parse_args()



def auto_root_dir(root_dir: Path, task: str) -> Path:
    """
    User-friendly behavior:
    - if root_dir already points to the task folder, use it;
    - if root_dir points to project_outputs, automatically descend into /fetch or /mountaincar.
    """
    task_dir = root_dir / task
    if task_dir.exists() and task_dir.is_dir():
        return task_dir
    return root_dir



def collect_runs(root_dir: Path, prefix: str) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for csv_path in root_dir.glob(f"{prefix}*/seed_*/eval_history.csv"):
        experiment = csv_path.parent.parent.name
        seed = csv_path.parent.name.replace("seed_", "")
        df = pd.read_csv(csv_path)
        df["experiment"] = experiment
        df["seed"] = seed
        records.append(df)
    if not records:
        raise FileNotFoundError(
            f"No eval_history.csv files found under {root_dir} for prefix={prefix!r}"
        )
    return pd.concat(records, ignore_index=True)



def _clip_bounds(metric: str, lower: pd.Series, upper: pd.Series) -> tuple[pd.Series, pd.Series]:
    if metric == "success_rate":
        return lower.clip(lower=0.0, upper=1.0), upper.clip(lower=0.0, upper=1.0)
    if metric in {
        "mean_ep_length",
        "mean_steps_to_success_or_timeout",
        "mean_steps_to_success_success_only",
        "mean_final_distance_to_goal",
    }:
        return lower.clip(lower=0.0), upper
    return lower, upper



def plot_metric(df: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plotted_any = False

    for experiment, g in df.groupby("experiment"):
        series = g[["timesteps", metric]].copy()
        series[metric] = pd.to_numeric(series[metric], errors="coerce")
        series = series.dropna(subset=[metric])
        if series.empty:
            continue

        agg = series.groupby("timesteps")[metric].agg(["mean", "std"]).reset_index()
        plt.plot(agg["timesteps"], agg["mean"], label=experiment)
        lower = agg["mean"] - agg["std"].fillna(0.0)
        upper = agg["mean"] + agg["std"].fillna(0.0)
        lower, upper = _clip_bounds(metric, lower, upper)
        plt.fill_between(agg["timesteps"], lower, upper, alpha=0.2)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    plt.xlabel("Timesteps")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()



def build_flat_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    last_rows = (
        df.sort_values("timesteps")
        .groupby(["experiment", "seed"], as_index=False)
        .tail(1)
        .copy()
    )
    # Ensure numeric columns for aggregation.
    for col in metrics:
        if col in last_rows.columns:
            last_rows[col] = pd.to_numeric(last_rows[col], errors="coerce")

    rows = []
    for experiment, g in last_rows.groupby("experiment"):
        row: dict[str, object] = {"experiment": experiment, "num_seeds": int(len(g))}
        for metric in metrics:
            if metric not in g.columns:
                continue
            row[f"{metric}_mean"] = float(np.nanmean(g[metric])) if not g[metric].isna().all() else np.nan
            row[f"{metric}_std"] = float(np.nanstd(g[metric], ddof=1)) if len(g[metric].dropna()) >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("experiment")



def main() -> None:
    args = parse_args()
    root_dir = auto_root_dir(Path(args.root_dir), args.task)
    output_dir = Path(args.output_dir)

    prefix = "mountaincar_" if args.task == "mountaincar" else "fetch_"
    df = collect_runs(root_dir, prefix)

    metrics = [metric for metric in METRICS if metric in df.columns]
    for metric in metrics:
        plot_metric(
            df,
            metric,
            f"{args.task}: {metric}",
            output_dir / f"{args.task}_{metric}.png",
        )

    summary = build_flat_summary(df, metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{args.task}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

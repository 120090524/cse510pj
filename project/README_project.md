# Sparse vs Dense Reward Project Code

This folder contains the course-project code for **Part 2 (toy problem)** and **Part 3 (main FetchReach experiment)**.

## What this code does

### Part 2: Toy problem
We use **MountainCarContinuous-v0** so that the **same base algorithm (SAC)** can be used in both the toy task and the robotics task.

Experiments:
1. `dense`: the official MountainCarContinuous reward
2. `sparse`: a simple minimum-time approximation (`-1` per step, `0` on success)

Main script:
- `train_mountaincar_sac.py`

### Part 3: Main experiment
We use **FetchReach** as the main goal-reaching benchmark.

Experiments:
1. `fetch_dense_sac`: `FetchReachDense-v4` + SAC
2. `fetch_sparse_sac`: `FetchReach-v4` + SAC
3. `fetch_sparse_sac_her`: `FetchReach-v4` + SAC + HER
4. optional: `fetch_min_time_sac`: `FetchReach-v4` + minimum-time wrapper + SAC

Main scripts:
- `train_fetch_sac.py`
- `train_fetch_her.py`

## Install

Create a clean environment and install:

```bash
conda create -n sparse-fetch python=3.10 -y
conda activate sparse-fetch
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Quick test:

```bash
python -c "import gymnasium as gym, gymnasium_robotics; gym.register_envs(gymnasium_robotics); env=gym.make('FetchReach-v4'); obs,_=env.reset(); print(obs.keys())"
```

Expected output is a dict with keys `observation`, `achieved_goal`, `desired_goal`.

## Suggested directory layout inside your repo

Put this folder under the repo root as:

```text
rl_suite/
  project/
    README_project.md
    requirements.txt
    wrappers.py
    callbacks.py
    train_common.py
    train_mountaincar_sac.py
    train_fetch_sac.py
    train_fetch_her.py
    evaluate_saved_model.py
    plot_results.py
```

## Recommended experiment schedule

### Smoke tests first

```bash
python project/train_mountaincar_sac.py --reward_mode dense --seed 0 --total_timesteps 20000
python project/train_mountaincar_sac.py --reward_mode sparse --seed 0 --total_timesteps 20000
python project/train_fetch_sac.py --env_id FetchReachDense-v4 --seed 0 --total_timesteps 30000
python project/train_fetch_sac.py --env_id FetchReach-v4 --seed 0 --total_timesteps 30000
python project/train_fetch_her.py --env_id FetchReach-v4 --seed 0 --total_timesteps 30000
```

### Final runs (3 seeds)

#### Part 2: MountainCarContinuous

```bash
python project/train_mountaincar_sac.py --reward_mode dense --seed 0 --total_timesteps 150000
python project/train_mountaincar_sac.py --reward_mode dense --seed 1 --total_timesteps 150000
python project/train_mountaincar_sac.py --reward_mode dense --seed 2 --total_timesteps 150000

python project/train_mountaincar_sac.py --reward_mode sparse --seed 0 --total_timesteps 150000
python project/train_mountaincar_sac.py --reward_mode sparse --seed 1 --total_timesteps 150000
python project/train_mountaincar_sac.py --reward_mode sparse --seed 2 --total_timesteps 150000
```

#### Part 3: FetchReach

```bash
python project/train_fetch_sac.py --env_id FetchReachDense-v4 --seed 0 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReachDense-v4 --seed 1 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReachDense-v4 --seed 2 --total_timesteps 250000

python project/train_fetch_sac.py --env_id FetchReach-v4 --seed 0 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --seed 1 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --seed 2 --total_timesteps 250000

python project/train_fetch_her.py --env_id FetchReach-v4 --seed 0 --total_timesteps 250000
python project/train_fetch_her.py --env_id FetchReach-v4 --seed 1 --total_timesteps 250000
python project/train_fetch_her.py --env_id FetchReach-v4 --seed 2 --total_timesteps 250000
```

### Optional paper-faithful extension

```bash
python project/train_fetch_sac.py --env_id FetchReach-v4 --minimum_time --terminate_on_success --seed 0 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --minimum_time --terminate_on_success --seed 1 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --minimum_time --terminate_on_success --seed 2 --total_timesteps 250000
```

## Evaluate a saved model

Example:

```bash
python project/evaluate_saved_model.py \
  --task fetch \
  --env_id FetchReach-v4 \
  --model_path ./project_outputs/fetch/fetch_sparse_sac_her/seed_0/best_model.zip \
  --episodes 20
```

## Make plots

```bash
python project/plot_results.py --task mountaincar --root_dir ./project_outputs --output_dir ./project_plots
python project/plot_results.py --task fetch --root_dir ./project_outputs --output_dir ./project_plots
```

## Output files

Each run writes:
- `metadata.json`
- `monitor.csv`
- `eval_history.csv`
- `best_model.zip`
- `final_model.zip`
- tensorboard logs under `tb/`

## Notes

- Use **MountainCarContinuous**, not the discrete MountainCar, if you want to keep the base algorithm fixed as SAC.
- Use **SAC + HER** only on the official sparse FetchReach env. Do **not** combine HER with the custom minimum-time wrapper unless you also implement a matching `compute_reward` method.
- If `FetchReach-v4` is unavailable in your installed version, try `FetchReach-v4` and update the commands consistently.

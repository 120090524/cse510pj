#!/usr/bin/env bash
set -euo pipefail

DEVICE=${1:-cuda}
TIMESTEPS=${2:-100000}
EVAL_FREQ=${3:-5000}
N_EVAL=${4:-20}
OUTDIR=${5:-./project_outputs/misspec_full}

mkdir -p "$OUTDIR"

for seed in 0 1 2

do
  # Reference PBRS and reference adhoc
  python project/train_fetch_sac.py \
    --env_id FetchReach-v4 \
    --reward_mode pbrs_min_time \
    --terminate_on_success \
    --seed "$seed" \
    --total_timesteps "$TIMESTEPS" \
    --eval_freq "$EVAL_FREQ" \
    --n_eval_episodes "$N_EVAL" \
    --gamma 0.99 \
    --potential_scale 0.5 \
    --experiment_name fetch_pbrs_ref \
    --outdir "$OUTDIR" \
    --device "$DEVICE"

  python project/train_fetch_sac.py \
    --env_id FetchReach-v4 \
    --reward_mode adhoc_distance \
    --terminate_on_success \
    --seed "$seed" \
    --total_timesteps "$TIMESTEPS" \
    --eval_freq "$EVAL_FREQ" \
    --n_eval_episodes "$N_EVAL" \
    --gamma 0.99 \
    --distance_scale 1.0 \
    --experiment_name fetch_adhoc_ref \
    --outdir "$OUTDIR" \
    --device "$DEVICE"

  # Recommended misspec variants: one offset, one action penalty, one threshold error
  python project/train_fetch_sac.py \
    --env_id FetchReach-v4 \
    --reward_mode adhoc_distance \
    --terminate_on_success \
    --seed "$seed" \
    --total_timesteps "$TIMESTEPS" \
    --eval_freq "$EVAL_FREQ" \
    --n_eval_episodes "$N_EVAL" \
    --gamma 0.99 \
    --distance_scale 1.0 \
    --goal_offset_x 0.01 \
    --experiment_name fetch_adhoc_offset_x01 \
    --outdir "$OUTDIR" \
    --device "$DEVICE"

  python project/train_fetch_sac.py \
    --env_id FetchReach-v4 \
    --reward_mode adhoc_distance \
    --terminate_on_success \
    --seed "$seed" \
    --total_timesteps "$TIMESTEPS" \
    --eval_freq "$EVAL_FREQ" \
    --n_eval_episodes "$N_EVAL" \
    --gamma 0.99 \
    --distance_scale 1.0 \
    --action_penalty_scale 0.01 \
    --experiment_name fetch_adhoc_actionpen_001 \
    --outdir "$OUTDIR" \
    --device "$DEVICE"

  python project/train_fetch_sac.py \
    --env_id FetchReach-v4 \
    --reward_mode adhoc_distance \
    --terminate_on_success \
    --seed "$seed" \
    --total_timesteps "$TIMESTEPS" \
    --eval_freq "$EVAL_FREQ" \
    --n_eval_episodes "$N_EVAL" \
    --gamma 0.99 \
    --distance_scale 1.0 \
    --shaping_threshold 0.03 \
    --experiment_name fetch_adhoc_threshold_003 \
    --outdir "$OUTDIR" \
    --device "$DEVICE"

done

python project/plot_results.py \
  --task fetch \
  --root_dir "$OUTDIR" \
  --output_dir "$OUTDIR/plots"

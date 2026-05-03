# Revisiting Sparse Rewards and Reward Shaping for Goal-Reaching Reinforcement Learning

> **Final integrated README for the course project**  
> This README supersedes the older generic root README and the earlier `project/README_project.md` for the purpose of understanding the final project, experiments, outputs, and current findings.

## 1. Project overview

This repository contains our full course project on **reward design for goal-reaching reinforcement learning**.

The project started from the question:

- Is a simple **sparse minimum-time reward** already sufficient for learning goal-reaching behaviors?
- Or do we still need **dense hand-crafted rewards** to obtain good performance?

After the initial baseline study, the project evolved into a more theory-driven direction:

- **PBRS (Potential-Based Reward Shaping)** as a theory-grounded shaping method
- **Ad-hoc dense shaping** as a non-PBRS control
- **Misspecification stress tests** to study whether ad-hoc shaping is brittle when the reward is slightly wrong

So the final project is not just “dense vs sparse,” but a staged study of:

1. **Sparse vs. dense reward design**
2. **Stronger sparse baselines with HER**
3. **Minimum-time task specification**
4. **Theory-grounded shaping (PBRS) vs. ad-hoc dense shaping**
5. **Robustness of shaping under reward misspecification**

---

## 2. Main research questions

Our work ultimately addresses the following questions:

1. **Baseline question**  
   How do dense reward, sparse reward, and sparse reward + HER compare on FetchReach?

2. **Minimum-time question**  
   If the task is written directly as a minimum-time problem (reward = \(-1\) per step until success), can SAC learn fast goal-reaching behavior?

3. **Reward-shaping theory question**  
   Can **potential-based reward shaping (PBRS)** provide dense learning signals while remaining aligned with the original minimum-time objective?

4. **Robustness question**  
   If we slightly misspecify an ad-hoc dense reward, does it become more brittle than PBRS?

---

## 3. Repository structure

```text
cse510pj/
├── project/                  # Course-project code (main scripts)
├── project_outputs/          # Saved models, CSV logs, plots, summaries
├── results/                  # Author-code reproduction outputs (dm_reacher_easy)
├── rl_suite/                 # Original/general RL testbed code
├── scripts/                  # Helper shell scripts
├── README.md                 # This final integrated README
└── setup.py
```

### Important code files under `project/`

- `train_fetch_sac.py`  
  Main SAC training script for Fetch tasks. Supports:
  - official dense
  - official sparse
  - minimum-time
  - PBRS minimum-time
  - ad-hoc dense shaping

- `train_fetch_her.py`  
  SAC+HER training for sparse Fetch, and an exploratory HER-compatible minimum-time branch.

- `train_fetch_her_min_time_finetune.py`  
  HER-assisted minimum-time fine-tuning (two-stage training).

- `wrappers.py`  
  Environment wrappers for:
  - Fetch minimum-time reward
  - PBRS minimum-time shaping
  - ad-hoc dense shaping
  - additional misspecified ad-hoc shaping variants

- `callbacks.py`  
  Periodic evaluation callback that records:
  - `success_rate`
  - `mean_reward`
  - `mean_ep_length`
  - `mean_steps_to_success_or_timeout`
  - `mean_steps_to_success_success_only`
  - `mean_final_distance_to_goal`

- `evaluate_saved_model.py`  
  Unified evaluation of saved models.

- `plot_results.py`  
  Aggregates `eval_history.csv` files into plots and summaries.

---

## 4. Output structure and where to find results

### 4.1 Main Fetch experiment outputs

All main Fetch experiment outputs are under:

```text
project_outputs/fetch/
```

Current committed experiment folders include:

- `fetch_dense_sac`
- `fetch_sparse_sac`
- `fetch_sparse_sac_her`
- `fetch_min_time_sac`
- `fetch_min_time_sac_her` *(exploratory; currently incomplete / not a full comparison)*
- `fetch_her_to_min_time_sac`
- `fetch_pbrs_sac`
- `fetch_adhoc_shaping_sac`
- `plots`

Each run folder typically contains:

```text
seed_k/
├── best_model.zip
├── final_model.zip
├── eval_history.csv
├── metadata.json
├── monitor.csv
└── tb/                      # TensorBoard logs
```

### 4.2 Reach scale sweep

Stored under:

```text
project_outputs/reach_scale_sweep/
```

Current sweep folders include:

- `adhoc_0p5/fetch_adhoc_shaping_sac/seed_0`
- `adhoc_1p0/fetch_adhoc_shaping_sac/seed_0`
- `adhoc_2p0/fetch_adhoc_shaping_sac/seed_0`
- `pbrs_0p5/fetch_pbrs_sac/seed_0`
- `pbrs_1p0/fetch_pbrs_sac/seed_0`
- `pbrs_2p0/fetch_pbrs_sac/seed_0`

These are the pilot runs used to choose shaping scales.

### 4.3 Termination ablation

Stored under:

```text
project_outputs/termination_ablation/
├── no_term/
└── term/
```

This compares learning with:

- rollout **continuing after success** (`no_term`)
- rollout **terminating immediately at success** (`term`)

### 4.4 Push pilot

Stored under:

```text
project_outputs/push_pilot/
```

Current pilot experiment folders include:

- `fetch_min_time_sac/seed_0`
- `fetch_pbrs_sac/seed_0`
- `fetch_adhoc_shaping_sac/seed_0`
- `plots/`

### 4.5 Misspecification pilot

Stored under:

```text
project_outputs/misspec_pilot/
```

Current pilot experiment folders include:

- `fetch_adhoc_ref/seed_0`
- `fetch_pbrs_ref/seed_0`
- `fetch_adhoc_offset_x01/seed_0`
- `fetch_adhoc_offset_y01/seed_0`
- `fetch_adhoc_offset_z01/seed_0`
- `fetch_adhoc_actionpen_001/seed_0`
- `fetch_adhoc_actionpen_005/seed_0`
- `fetch_adhoc_threshold_003/seed_0`
- `fetch_adhoc_threshold_008/seed_0`
- `plots/`

### 4.6 Author-code reproduction outputs

The dm_reacher_easy reproduction runs are stored separately under:

```text
results/
├── dm_reacher_easy_timeout25
├── dm_reacher_easy_timeout50
├── dm_reacher_easy_timeout100
└── ...
```

---

## 5. What we actually did

The full project was completed in **four stages**.

---

### Stage A. Original baseline study

#### A1. `dm_reacher_easy` author-code reproduction

We reproduced the timeout effect from the original paper using the author-code environment:

- environment: `dm_reacher_easy`
- algorithm: SAC
- timeout settings: `25`, `50`, `100`

**Key result:**
- `timeout = 50` produced the best late-stage mean return / mean episode length
- `timeout = 25` remained clearly worse
- `timeout = 100` was close to `50`, but slightly worse

This confirmed that **minimum-time learning is sensitive to timeout**.

#### A2. Original FetchReach baselines

We ran the standard baseline comparison on FetchReach:

- `fetch_dense_sac`
- `fetch_sparse_sac`
- `fetch_sparse_sac_her`

These baseline experiments established the initial pattern:

- dense reward is better than vanilla sparse SAC in **early sample efficiency**
- adding HER makes the sparse baseline much stronger

#### A3. MountainCar toy study

We also ran:

- `mountaincar_dense_sac`
- `mountaincar_sparse_sac`

This did **not** become a useful main benchmark, but it was still informative because it highlighted how strongly reward structure and exploration difficulty can dominate behavior.

---

### Stage B. Minimum-time extension

After the original baselines, we extended the Fetch study toward a more paper-faithful minimum-time formulation:

- `fetch_min_time_sac`
- `fetch_min_time_sac_her` *(exploratory)*
- `fetch_her_to_min_time_sac`

The purpose of this stage was to move from fixed-horizon success-only comparison to a setting where the agent is explicitly rewarded for reaching the goal **quickly**.

---

### Stage C. Reward-shaping theory extension

We then reformulated the project as a comparison between:

- **PBRS**: theory-grounded, policy-preserving shaping
- **Ad-hoc dense shaping**: dense shaping without a policy-invariance guarantee

This led to the experiments:

- `fetch_pbrs_sac`
- `fetch_adhoc_shaping_sac`

We also ran a **scale sweep** to choose shaping strength for both families.

---

### Stage D. Robustness and extension studies

Finally, we added three extra experiment groups:

1. **Reach scale sweep**  
   `project_outputs/reach_scale_sweep/`

2. **Termination ablation**  
   `project_outputs/termination_ablation/`

3. **Harder-task pilot on FetchPush**  
   `project_outputs/push_pilot/`

4. **Misspecification stress test**  
   `project_outputs/misspec_pilot/`

These last experiments are what turned the project from a simple “dense vs sparse” study into a more theory-driven reward-shaping analysis.

---

## 6. Main results (integrated summary)

This section summarizes the final state of the project based on the committed outputs.

---

### 6.1 Historical baseline finding from the original FetchReach study

In the original baseline comparison:

- `fetch_dense_sac` was faster than vanilla sparse SAC early in training
- `fetch_sparse_sac_her` substantially improved over vanilla sparse SAC

This established the initial conclusion that **baseline strength matters**: the apparent dense-vs-sparse gap becomes much smaller once the sparse branch is strengthened.

---

### 6.2 Current committed summary for the main Fetch experiment family

From `project_outputs/fetch/plots/fetch_summary.csv`, the committed final summaries are approximately:

| Experiment | Success | Mean steps to success/timeout | Mean final distance |
|---|---:|---:|---:|
| `fetch_dense_sac` | 1.00 | fixed horizon | — |
| `fetch_min_time_sac` | 1.00 | 2.72 | — |
| `fetch_her_to_min_time_sac` | 1.00 | 2.72 | — |
| `fetch_pbrs_sac` | 1.00 | 2.73 | 0.0306 |
| `fetch_adhoc_shaping_sac` | 1.00 | 2.70 | 0.0308 |

Interpretation:

- All well-trained minimum-time / shaping branches eventually learn **very fast goal-reaching behavior**.
- On final performance, `fetch_min_time_sac`, `fetch_her_to_min_time_sac`, `fetch_pbrs_sac`, and `fetch_adhoc_shaping_sac` are all very close.
- The important differences are therefore mainly in **early sample efficiency**, not in final asymptotic behavior.

---

### 6.3 Reach scale-sweep result

The scale sweep on `FetchReach` showed that the two shaping families prefer different magnitudes.

#### PBRS sweep

- `pbrs_0p5` reached 95% success earlier than `pbrs_1p0` and `pbrs_2p0`
- `pbrs_0p5` also produced the best final combination of speed and final distance among the tested PBRS scales

**Chosen PBRS scale:** `0.5`

#### Ad-hoc sweep

- `adhoc_1p0` and `adhoc_2p0` were both much faster than `adhoc_0p5`
- `adhoc_2p0` had the best raw pilot numbers, but `adhoc_1p0` was kept as the main practical reference because it achieved the same early speed while being slightly less aggressive

**Chosen ad-hoc scale for the main comparison:** `1.0`  
**Aggressive appendix variant:** `2.0`

Interpretation:

- PBRS prefers a **weaker shaping scale**
- Ad-hoc shaping benefits from a **stronger dense signal** on FetchReach
- Even after tuning, PBRS remains slower than ad-hoc shaping in early training on the simple Reach task

---

### 6.4 Termination ablation

The termination ablation compares training/evaluation with and without immediate episode termination at success.

The committed summaries show:

#### With termination (`term/`)

- `fetch_min_time_sac`: success 1.0, mean steps ≈ 2.4
- `fetch_pbrs_sac`: success 1.0, mean steps ≈ 2.35
- `fetch_adhoc_shaping_sac`: success 1.0, mean steps ≈ 2.4

#### Without termination (`no_term/`)

- `fetch_min_time_sac`: success only 0.95 and much worse mean steps-to-success/timeout
- `fetch_pbrs_sac`: success 1.0 with good steps-to-success
- `fetch_adhoc_shaping_sac`: success 1.0 with good steps-to-success

Interpretation:

- Termination semantics matter for minimum-time tasks.
- In particular, the plain minimum-time baseline is more affected by no-termination evaluation than the shaped variants.
- This supports the claim that **task specification** (especially whether the rollout ends at success) changes what the reported metrics actually mean.

---

### 6.5 Push pilot

The `FetchPush` pilot is the first harder-task extension.

Current committed pilot summary:

| Experiment | Success | Mean steps / timeout | Final distance |
|---|---:|---:|---:|
| `fetch_min_time_sac` | 0.10 | 45.1 | 0.1823 |
| `fetch_pbrs_sac` | 0.10 | 45.1 | 0.1734 |
| `fetch_adhoc_shaping_sac` | 0.10 | 45.1 | 0.2245 |

Interpretation:

- `FetchPush` is clearly much harder than `FetchReach`
- none of the pilot runs is solved yet
- however, all three methods show *some* progress signal
- among them, **PBRS has the best final distance in the current pilot**

This is important because it suggests that the simple Reach task may be too easy to expose the real difference between shaping families.

---

### 6.6 Misspecification pilot

The misspecification pilot asks whether ad-hoc dense shaping is brittle when the shaping assumptions are slightly wrong.

The following pilot variants were tested (all seed 0):

- `fetch_adhoc_ref`
- `fetch_adhoc_offset_x01`
- `fetch_adhoc_offset_y01`
- `fetch_adhoc_offset_z01`
- `fetch_adhoc_actionpen_001`
- `fetch_adhoc_actionpen_005`
- `fetch_adhoc_threshold_003`
- `fetch_adhoc_threshold_008`
- `fetch_pbrs_ref`

Current committed pilot summary:

- all variants reach **1.0 final success** on `FetchReach`
- all variants end at approximately **2.4 steps to success**
- final goal distances are all tightly clustered around **0.028–0.031**

Interpretation:

- On the simple `FetchReach` task, mild reward misspecification does **not** catastrophically break ad-hoc shaping.
- The main effect of misspecification appears in **early learning speed**, not in final asymptotic behavior.
- In the pilot setting, PBRS does **not** yet show a clear robustness advantage over mildly misspecified ad-hoc shaping.
- This makes `FetchReach` a useful pilot benchmark, but likely too easy to fully expose the theoretical robustness advantage of PBRS.

---

## 7. Key takeaways

Across the whole project, the most important takeaways are:

1. **Dense reward helps mainly in early sample efficiency**, not necessarily in final asymptotic success.
2. **Baseline strength matters**: once the sparse branch is strengthened (for example with HER), the dense-vs-sparse gap becomes much smaller.
3. **Minimum-time task specification is viable** on FetchReach and can learn very fast goal-reaching behavior.
4. **PBRS is effective**, but on simple Reach tasks it is slower early in training than tuned ad-hoc dense shaping.
5. **Termination semantics matter** for minimum-time comparisons.
6. **Misspecification matters, but not dramatically on Reach**: mild ad-hoc reward errors do not immediately destroy learning on the simple benchmark.
7. **Harder tasks such as FetchPush are more promising for future work**, because they are more likely to reveal genuine differences between reward families.

---

## 8. How to reproduce the experiments

## 8.1 Installation

We recommend a clean environment:

```bash
conda create -n sparse-fetch python=3.10 -y
conda activate sparse-fetch
pip install --upgrade pip setuptools wheel
pip install -r project/requirements.txt
```

Quick robotics environment test:

```bash
python -c "import gymnasium as gym, gymnasium_robotics; gym.register_envs(gymnasium_robotics); env=gym.make('FetchReach-v4'); obs,_=env.reset(); print(obs.keys())"
```

Expected keys:

```text
observation, achieved_goal, desired_goal
```

---

## 8.2 Baseline experiments

### FetchReach dense baseline

```bash
python project/train_fetch_sac.py --env_id FetchReachDense-v4 --seed 0 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReachDense-v4 --seed 1 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReachDense-v4 --seed 2 --total_timesteps 250000
```

### FetchReach sparse baseline

```bash
python project/train_fetch_sac.py --env_id FetchReach-v4 --seed 0 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --seed 1 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --seed 2 --total_timesteps 250000
```

### FetchReach sparse + HER

```bash
python project/train_fetch_her.py --env_id FetchReach-v4 --seed 0 --total_timesteps 250000
python project/train_fetch_her.py --env_id FetchReach-v4 --seed 1 --total_timesteps 250000
python project/train_fetch_her.py --env_id FetchReach-v4 --seed 2 --total_timesteps 250000
```

---

## 8.3 Minimum-time experiments

### Minimum-time SAC

```bash
python project/train_fetch_sac.py --env_id FetchReach-v4 --reward_mode min_time --terminate_on_success --seed 0 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --reward_mode min_time --terminate_on_success --seed 1 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --reward_mode min_time --terminate_on_success --seed 2 --total_timesteps 250000
```

### HER-assisted minimum-time fine-tuning

```bash
python project/train_fetch_her_min_time_finetune.py \
  --pretrained_model_path ./project_outputs/fetch/fetch_sparse_sac_her/seed_0/best_model.zip \
  --env_id FetchReach-v4 \
  --seed 0 \
  --finetune_timesteps 100000
```

(Repeat for seeds 1 and 2.)

---

## 8.4 Reward-shaping theory experiments

### PBRS main runs

```bash
python project/train_fetch_sac.py --env_id FetchReach-v4 --reward_mode pbrs_min_time --terminate_on_success --gamma 0.99 --potential_scale 0.5 --seed 0 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --reward_mode pbrs_min_time --terminate_on_success --gamma 0.99 --potential_scale 0.5 --seed 1 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --reward_mode pbrs_min_time --terminate_on_success --gamma 0.99 --potential_scale 0.5 --seed 2 --total_timesteps 250000
```

### Ad-hoc shaping main runs

```bash
python project/train_fetch_sac.py --env_id FetchReach-v4 --reward_mode adhoc_distance --terminate_on_success --gamma 0.99 --distance_scale 1.0 --seed 0 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --reward_mode adhoc_distance --terminate_on_success --gamma 0.99 --distance_scale 1.0 --seed 1 --total_timesteps 250000
python project/train_fetch_sac.py --env_id FetchReach-v4 --reward_mode adhoc_distance --terminate_on_success --gamma 0.99 --distance_scale 1.0 --seed 2 --total_timesteps 250000
```

---

## 8.5 Plotting

### Main Fetch plots

```bash
python project/plot_results.py --task fetch --root_dir ./project_outputs/fetch --output_dir ./project_outputs/fetch/plots
```

### Reach scale sweep plots

Example (PBRS 0.5):

```bash
python project/plot_results.py --task fetch --root_dir ./project_outputs/reach_scale_sweep/pbrs_0p5 --output_dir ./project_outputs/reach_scale_sweep/pbrs_0p5/plots
```

### Termination ablation plots

```bash
python project/plot_results.py --task fetch --root_dir ./project_outputs/termination_ablation/term --output_dir ./project_outputs/termination_ablation/term/plots
python project/plot_results.py --task fetch --root_dir ./project_outputs/termination_ablation/no_term --output_dir ./project_outputs/termination_ablation/no_term/plots
```

### Push pilot plots

```bash
python project/plot_results.py --task fetch --root_dir ./project_outputs/push_pilot --output_dir ./project_outputs/push_pilot/plots
```

### Misspecification pilot plots

```bash
python project/plot_results.py --task fetch --root_dir ./project_outputs/misspec_pilot --output_dir ./project_outputs/misspec_pilot/plots
```

---

## 9. Recommended result-reading order

If you are new to the repository, read the outputs in this order:

1. `project_outputs/fetch/plots/`  
   Main baseline + minimum-time + shaping results

2. `project_outputs/reach_scale_sweep/`  
   Why the chosen PBRS and ad-hoc scales were selected

3. `project_outputs/termination_ablation/`  
   Why termination semantics matter for minimum-time tasks

4. `project_outputs/push_pilot/plots/`  
   First evidence on a harder benchmark

5. `project_outputs/misspec_pilot/plots/`  
   First robustness evidence under reward misspecification

---

## 10. Notes and caveats

- `fetch_min_time_sac_her` is currently exploratory and should **not** be treated as a completed full comparison.
- `FetchReach` is a relatively easy benchmark, so many methods eventually converge to very similar final behavior.
- The most informative comparisons on Reach are therefore often about **early sample efficiency** and **task alignment**, not just final success.
- `FetchPush` is the more informative harder-task pilot in the current repository.
- The misspecification study is currently a **pilot** (seed 0 only), and should be described as preliminary evidence rather than a final statistical conclusion.

---

## 11. References

1. Gautham Vasan, Yan Wang, Fahim Shahriar, James Bergstra, Martin Jagersand, A. Rupam Mahmood.  
   **Revisiting Sparse Rewards for Goal-Reaching Reinforcement Learning.** 2024.

2. Takuya Hiraoka.  
   **Efficient Sparse-Reward Goal-Conditioned Reinforcement Learning with a High Replay Ratio and Regularization.** 2023.

3. Andrew Y. Ng, Daishi Harada, Stuart Russell.  
   **Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.** 1999.

---

## 12. Final one-paragraph project summary

This project began as a reproduction-oriented study of dense versus sparse rewards in goal-reaching RL, then gradually evolved into a more theory-driven analysis of reward shaping. The final repository now contains the original dense/sparse/HER baselines, minimum-time task formulations, HER-assisted fine-tuning, theory-grounded PBRS shaping, ad-hoc dense shaping, a scale sweep, a termination ablation, a harder-task pilot on FetchPush, and a misspecification stress-test pilot. Taken together, the experiments show that baseline strength and task specification strongly affect the apparent dense-vs-sparse conclusion; that minimum-time learning is viable on FetchReach; that PBRS is effective but not obviously superior to tuned ad-hoc shaping on the simple Reach task; and that harder benchmarks are the most promising next step for exposing the real behavioral and robustness differences between shaping families.

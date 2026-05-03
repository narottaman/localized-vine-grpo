# Localized GRPO — MS Thesis Implementation

**Localized GRPO: Critic-Free Step-Level Credit Assignment for Structured Reasoning in Small Language Models**

*Narutto Gangadhara | Arizona State University | Advisor: Prof. Chitta Baral*

---

## Overview

This repository contains the full implementation of all RL post-training methods
evaluated in the thesis, including our proposed **Localized GRPO** algorithm.

---

## Key Results (Puzzle Baron Logic Grid Puzzles — Qwen2.5-3B-Instruct)

| Method | Perfect Solve Rate | Cell Accuracy |
|---|---|---|
| SFT | 0.00% | 14.2% |
| GRPO ORM | 1.49% | 29.1% |
| R++ ORM | 1.78% | 30.1% |
| R++ Hybrid | 2.21% | 47.9% |
| GRPO Hybrid | 2.90% | 49.1% |
| PPO ORM | 2.89% | 50.1% |
| GDPO | 3.10% | 50.2% |
| PPO Hybrid | 4.24% | 52.1% |
| VinePPO | 11.10% | 56.1% |
| **Localized GRPO (Ours)** | **15.20%** | **60.8%** |

---

## Repository Structure

```
verl/trainer/ppo/
  core_algos.py          ← ALL advantage estimators:
                           - GAE (PPO)
                           - GRPO
                           - REINFORCE++
                           - REINFORCE++ w/ Baseline
                           - REINFORCE++ GDPO
                           - VinePPO
                           - Localized GRPO (VINEPPO_GRPO) ← our method
                           - Hybrid GRPO
  ray_trainer.py         ← Training loop with MC rollout generation
                           for VinePPO and Localized GRPO

verl/trainer/config/algorithm/
  vineppo_grpo.yaml      ← Localized GRPO config (our method)
  vine_ppo.yaml          ← VinePPO config
  hybrid_grpo.yaml       ← Hybrid GRPO config
  reinforce_plusplus.yaml
  reinforce_plusplus_gdpo.yaml

verl/utils/reward_score/
  puzzle_baron_rewarder_hybrid.py    ← Hybrid reward (correctness + format)
  puzzle_baron_rewarder_pure_orm.py  ← ORM-only reward
  gsm8k_rewarder.py                  ← GSM8K cross-domain reward

slurm_scripts/           ← SLURM job scripts for all experiments
eval_scripts/            ← Evaluation for Puzzle Baron, ZebraLogic, GSM8K
```

---

## Core Algorithm (Localized GRPO)

```
For each intermediate reasoning step s_t in main trajectory:
  1. Run K MC rollouts from s_t → get raw scores v_t1, v_t2, ..., v_tK
  2. KL subtract: A_tk = v_tk - β * KL_tk

Collect ALL A_tk across ALL steps × ALL rollouts → global pool

Global normalization:
  mu    = mean(pool)
  sigma = std(pool)
  Â_tk  = (A_tk - mu) / sigma

Per-step average and broadcast:
  state_adv_t = mean(Â_t1, Â_t2, ..., Â_tK)  ← K rollouts of step t only
  advantages[step_t_tokens] = state_adv_t       ← broadcast to all tokens of step t
```

See `verl/trainer/ppo/core_algos.py` → `compute_vineppo_grpo_advantage()`

---

## Training

```bash
# Set token (classic PAT with repo scope)
export GITHUB_TOKEN="..."

# Run Localized GRPO (our method)
sbatch slurm_scripts/run_localized_grpo_fixed.sh

# Run VinePPO baseline
sbatch slurm_scripts/run_vineppo_Qwen2-5-3B-Instruct.sh

# Run REINFORCE++ w/ Baseline
sbatch slurm_scripts/run_reinforce_plusplus_baseline_fixed_final.sh
```

---

## Base Model & Infrastructure

- **Base model**: Qwen2.5-3B-Instruct
- **SFT checkpoint**: `global_step_4000`
- **Training framework**: VERL + vLLM + Ray
- **Cluster**: ASU HPC (A100 80GB GPUs)
- **Datasets**: Puzzle Baron (train/val), ZebraLogic (transfer), GSM8K (cross-domain)

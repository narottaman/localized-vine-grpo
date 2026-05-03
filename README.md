# Localized GRPO — Master's Thesis Implementation

**Localized GRPO: State-Wise Group-Relative Optimization for Multi-Step LLM Long Reasoning Chains**

Narottaman Gangadaran  
Arizona State University  
Advisor: Prof. Chitta Baral  

---

## Overview

This repository contains the full implementation of reinforcement learning post-training methods for improving **long-chain reasoning in Large Language Models (LLMs)**.

We introduce **Localized GRPO**, a critic-free method that performs **step-wise credit assignment** using Monte Carlo rollouts from intermediate reasoning states.

---

## Problem Statement: Structured Reasoning with Puzzle Baron

We evaluate models on **Puzzle Baron logic grid puzzles**, which require:

- Multi-step logical deduction
- Constraint propagation across entities
- Consistent reasoning across multiple steps

### Example Puzzle Structure

Each puzzle consists of:
- Entities (people, objects, attributes)
- Clues (constraints)
- A grid that must be filled consistently

### Why This Is Hard

- Requires **long reasoning chains**
- Early mistakes propagate forward
- Final reward alone does not explain *where reasoning failed*

---

### Example Puzzle Visualization

![Puzzle Baron Example](assets/figures/puzzle_example.png)

---

## Why Credit Assignment Matters

In long reasoning tasks:

- The **final answer is sparse feedback**
- Many intermediate steps influence correctness
- Standard RL signals cannot identify:
  - which step helped
  - which step hurt

This leads to:
- high variance learning
- unstable training
- poor reasoning generalization

---

## Architecture Comparison

### Training Paradigms Overview

![Architecture Comparison](assets/figures/architecture_comparison.png)

---

### Method Comparison

| Method | Credit Assignment | Strength | Limitation |
|---|---|---|---|
| **PPO** | Token-level via critic | Stable optimization | Critic becomes inaccurate for long chains |
| **GRPO** | Sequence-level relative reward | No critic needed | No step-level feedback |
| **GDPO** | Structured normalization | Better stability | Still coarse credit assignment |
| **REINFORCE++** | Sampled returns + normalization | Simple | High variance |
| **VinePPO** | Rollouts from intermediate states | Partial step awareness | Limited normalization across steps |
| **Localized GRPO (Ours)** | Step-wise + global normalization | Precise credit assignment | Higher compute cost |

---

## Core Idea: Localized GRPO

![Localized GRPO Pipeline](assets/figures/localized_grpo_pipeline.png)

---

### Algorithm Intuition

For each reasoning step:

1. Sample **K rollouts from intermediate state**
2. Compute reward for each rollout
3. Apply KL penalty:
```

A_tk = v_tk - β * KL_tk

````
4. Pool all rollout advantages across all steps
5. Perform **global normalization**
6. Compute **per-step average advantage**
7. Assign that value to all tokens in that step

---

### Key Insight

Instead of asking:

> “Was the final answer correct?”

We ask:

> “Which reasoning step improved the outcome?”

---

## Key Results

**Puzzle Baron Logic Grid Puzzles — Qwen2.5-3B-Instruct**

| Method | Perfect Solve Rate | Cell Accuracy |
|---|---:|---:|
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

```text
verl/trainer/ppo/
core_algos.py          # All RL algorithms
ray_trainer.py         # Training loop with rollout generation

verl/trainer/config/algorithm/
vineppo_grpo.yaml      # Localized GRPO
vine_ppo.yaml
hybrid_grpo.yaml
reinforce_plusplus.yaml

verl/utils/reward_score/
puzzle_baron_rewarder_hybrid.py
puzzle_baron_rewarder_pure_orm.py
gsm8k_rewarder.py
puzzle_verifier.py

verl/workers/reward_manager/
prm_score_reward_manager.py
prm_verbal_reward_manager.py
verifier_reward_manager.py

slurm_scripts/
eval_scripts/
analysis/
data_prep/
````

---

## Training

```bash
export GITHUB_TOKEN="..."

# Localized GRPO (our method)
sbatch slurm_scripts/run_vineppo_grpo_Qwen2-5-3B-Instruct.sh

# VinePPO baseline
sbatch slurm_scripts/run_vineppo_Qwen2-5-3B-Instruct.sh

# REINFORCE++
sbatch slurm_scripts/run_rein++_Qwen2-5-3B-Instruct.sh
```

---

## Base Model & Infrastructure

* **Model:** Qwen2.5-3B-Instruct
* **Framework:** VERL + vLLM + Ray
* **Cluster:** ASU HPC (A100 GPUs)
* **Datasets:**

  * Puzzle Baron (primary)
  * ZebraLogic (transfer)
  * GSM8K (cross-domain)

---

## Figures Directory

Place images here:

```text
assets/figures/
  puzzle_example.png
  architecture_comparison.png
  localized_grpo_pipeline.png
  results_plot.png
```

---

## Summary

Localized GRPO introduces **step-wise credit assignment** for long reasoning chains by combining:

* intermediate state rollouts
* global normalization
* token-level propagation

This leads to significantly improved performance on structured reasoning tasks.

---

```

---

## 🔥 What you should do next (important)

Create these 3 images (this will **massively boost your repo quality**):

1. **architecture_comparison.png**
   - PPO vs GRPO vs VinePPO vs Localized GRPO flow

2. **localized_grpo_pipeline.png**
   - Step → rollouts → normalization → broadcast

3. **puzzle_example.png**
   - Simple logic grid diagram

---

If you want, I can:
- design those diagrams for you (clean research-paper style)
- or convert your LaTeX figures directly into GitHub-ready PNGs
```

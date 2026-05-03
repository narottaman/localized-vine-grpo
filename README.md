
# Localized GRPO — Master's Thesis Implementation

## Localized GRPO: State-Wise Group-Relative Optimization for Multi-Step LLM Reasoning

**Narottaman Gangadaran**  
Arizona State University  
Advisor: Prof. Chitta Baral  

---

##  Abstract

Large Language Models (LLMs) struggle with **long-chain structured reasoning** due to poor credit assignment. Traditional reinforcement learning approaches provide **sparse or sequence-level feedback**, making it difficult for models to identify which reasoning steps contribute to success.

We propose **Localized GRPO**, a critic-free RL algorithm that performs:

- **State-wise rollout evaluation**
- **Global advantage normalization**
- **Step-level credit propagation**

This leads to significantly improved reasoning performance on structured tasks such as **Puzzle Baron logic grids**, **ZebraLogic**, and **GSM8K**.

---

##  Problem: Long-Chain Reasoning

Logic grid puzzles require:

- Multi-step deduction
- Constraint propagation
- Global consistency

### Example Puzzle

![Puzzle Example](assets/puzzle_background_clues.png)

---

### Step-by-Step Reasoning Chain

![Reasoning](assets/puzzlebaron_chain.png)

---

### Final Answer

![Solution](assets/puzzle_solution.png)

---

##  Core Challenge: Credit Assignment

### Problem

- Reward only given at the **end**
- Early mistakes propagate
- Model cannot identify:
  - useful steps
  - harmful steps

---

### ORM Limitation

![ORM](assets/orm.png)

 ORM evaluates only the **final table**, ignoring reasoning steps.

---

##  Dataset Analysis

### Puzzle Baron Distribution

![Puzzle Distribution](assets/distribution_pb.png)

---

### ZebraLogic Distribution

![Zebra Distribution](assets/distribution_zl.png)

---

##  Training Architectures

---

###  Supervised Fine-Tuning (SFT)

![SFT](assets/sft_pipeline.png)

- Token-level supervision
- No reasoning evaluation

---

###  PPO (Actor-Critic)

![PPO](assets/ppo_arch.png)

- Uses critic for value estimation
- Fails in long reasoning due to unstable value learning

---

###  GRPO (Group Relative Policy Optimization)

![GRPO](assets/GRPO.png)

- Removes critic
- Sequence-level normalization
- No step-level signal

---

###  GDPO

![GDPO](assets/GDPO.png)

- Structured normalization
- Still coarse credit assignment

---

###  REINFORCE++

![R++](assets/R++.png)

- High variance
- Weak temporal credit assignment

---

###  VinePPO

![VinePPO](assets/vineppo.png)

- Rollouts from intermediate states
- Partial step-awareness

---

##  Localized GRPO (Our Contribution)

---

###  Full Architecture

![Localized GRPO](assets/local_grpo_archi.png)

---

###  Advantage Computation

![Advantage Flow](assets/Local_GRPO.png)

---

###  Normalization

![Normalization](assets/adv_distribution.png)

---

###  Key Idea

For each state \( s_t \):

1. Sample **K rollouts**
2. Compute reward for each rollout
3. Normalize across **ALL steps**
4. Compute step-level advantage
5. Broadcast to tokens

---

###  Key Properties

- Correct steps are **never penalized**
- High disagreement → **strong learning signal**
- Identical rollouts → **zero gradient noise**

---

##  Reward Function

---

### Cell-Level Reward

![Cell Reward](assets/orm.png)

---

### Hybrid Reward

![Hybrid Reward](assets/hybrid_reward.png)

\[
R = w_c R_c + w_s R_s + w_{st} R_{st} + w_p R_p
\]

---

##  Results

---

### Perfect Solve Rate

![Perfect Solve](assets/perfect_acc_plot.png)

---

### Cell Accuracy

![Cell Accuracy](assets/cell_acc_plot.png)

---

### Combined Metrics

![Combined](assets/acc_plot.png)

---

### Training Curve

![Training](assets/acc_trainstep.png)

---

### ZebraLogic Transfer

![Zebra](assets/zl_acc.png)

---

### GSM8K Performance

![GSM8K](assets/gsm8k_result.png)

---

##  Key Observations

- Localized GRPO improves:
  - **Perfect solve rate (15.2%)**
  - **Cell accuracy (60.8%)**
- Outperforms:
  - PPO
  - GRPO
  - GDPO
  - REINFORCE++
  - VinePPO

---

##  Why Localized GRPO Works

| Method | Credit Assignment | Limitation |
|------|------------------|-----------|
| PPO | Critic-based | Unstable |
| GRPO | Sequence-level | No step info |
| GDPO | Structured | Still coarse |
| R++ | High variance | Weak signal |
| VinePPO | Partial step | Not global |
| **Localized GRPO** | Step-level + global | ✅ Best |

---

##  Experimental Setup

- Model: **Qwen2.5-3B-Instruct**
- Framework:
  - VERL
  - vLLM
  - Ray
- Hardware: A100 GPUs
- Tasks:
  - Puzzle Baron
  - ZebraLogic
  - GSM8K

---

##  Repository Structure

```

verl/
trainer/ppo/
trainer/config/
utils/reward_score/
workers/reward_manager/

slurm_scripts/
eval_scripts/
analysis/
data_prep/

````

---

##  Training

```bash
export GITHUB_TOKEN="..."

# Localized GRPO
sbatch slurm_scripts/run_vineppo_grpo_Qwen2-5-3B-Instruct.sh

# Baselines
sbatch slurm_scripts/run_vineppo_Qwen2-5-3B-Instruct.sh
sbatch slurm_scripts/run_rein++_Qwen2-5-3B-Instruct.sh
````

---

##  Summary

Localized GRPO introduces:

* Step-wise rollout evaluation
* Global normalization
* Token-level advantage propagation

### Result:

> Stronger learning signal → Better reasoning → Higher accuracy

---

##  Citation

(To be added after thesis submission)

````




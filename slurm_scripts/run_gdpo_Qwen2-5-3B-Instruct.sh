#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-4:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -o output/verl_gdpo_puzzlebaron3b.%j.out
#SBATCH -e output/verl_gdpo_puzzlebaron3b.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =====================================================================
# GDPO: REINFORCE++ with Decoupled Group Reward Normalization
#
# How it works:
#   1. Hybrid rewarder returns THREE components per response:
#        reward_correctness = cell_accuracy          (how correct the table is)
#        reward_structure   = structure_bonus        (format quality, zeroed if acc < 0.1)
#        reward_penalty     = sum of penalties       (duplicates, incomplete table etc.)
#
#   2. Each component is normalized INDEPENDENTLY within its prompt group
#        adv_c = (correctness - mean_group) / std_group
#        adv_s = (structure   - mean_group) / std_group
#        adv_p = (penalty     - mean_group) / std_group
#
#   3. Combined with weights:
#        combined = 0.70*adv_c + 0.20*adv_s + 0.10*adv_p
#
#   4. REINFORCE++ discounted returns scaled by combined signal
#
# Why n=3 (rollout.n=3):
#   GDPO normalizes within prompt groups. With n=1, every group has
#   one sample → std=0 → normalized advantage=0 → no learning.
#   n=3 gives 3 responses per prompt to compare against each other.
#
# Why hybrid reward:
#   Pure ORM only returns cell_accuracy, so reward_structure and
#   reward_penalty are always 0 — GDPO would collapse to REINFORCE++.
#   The hybrid rewarder exposes all three components.
# =====================================================================

source $HOME/.venv/bin/activate
export PYTHONPATH=/scratch/ngangada/thesis/thesis/verl:$PYTHONPATH
export VLLM_USE_V1=1
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES

python -m verl.trainer.main_ppo \
  data.train_files=/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_train_gsm8k_like.parquet \
  data.val_files=/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_val_gsm8k_like.parquet \
  data.train_batch_size=16 \
  data.max_prompt_length=1204 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=/scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_sft_eot/puzzle_sft_eot/global_step_4000/huggingface \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  +actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.do_sample=True \
  +actor_rollout_ref.rollout.repetition_penalty=1.0 \
  +actor_rollout_ref.rollout.stop_sequences=['END_OF_OUTPUT'] \
  actor_rollout_ref.rollout.response_length=2048 \
  actor_rollout_ref.rollout.n=3 \
  trainer.total_epochs=1 \
  +algorithm=reinforce_plusplus_gdpo \
  algorithm.adv_estimator=reinforce_plus_plus_gdpo \
  algorithm.gamma=1.0 \
  algorithm.kl_ctrl.kl_coef=0.01 \
  algorithm.normalize_advantages=true \
  algorithm.kl_ctrl.type=adaptive \
  algorithm.kl_ctrl.target_kl=0.02 \
  algorithm.kl_ctrl.horizon=10000 \
  algorithm.gdpo_w_correctness=0.70 \
  algorithm.gdpo_w_structure=0.20 \
  algorithm.gdpo_w_penalty=0.10 \
  custom_reward_function.path=/scratch/ngangada/thesis/thesis/verl/verl/utils/reward_score/puzzle_baron_rewarder_hybrid.py \
  custom_reward_function.name=compute_puzzle_baron_score \
  reward_model.reward_manager=naive \
  trainer.logger=console \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.project_name=puzzle_3b_gdpo \
  trainer.experiment_name=gdpo_hybrid_n3 \
  trainer.resume_mode=auto \
  2>&1 | tee $SCRATCH/verl_logs/gdpo_hybrid_n3_puzzlebaron3b.log
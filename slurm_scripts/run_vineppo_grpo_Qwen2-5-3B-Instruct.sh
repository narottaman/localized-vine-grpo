#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-04:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -o output/verl_vineppo_grpo_puzzlebarron3b.%j.out
#SBATCH -e output/verl_vineppo_grpo_puzzlebarron3b.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# === Environment setup ===
source $HOME/.venv/bin/activate
export PYTHONPATH=/scratch/ngangada/thesis/thesis/verl:$PYTHONPATH
export VLLM_USE_V1=1
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES

# ── What changed vs original script ──────────────────────────────────────────
#
#  FIXED:
#    1. -t 0-01:00:00 → 0-06:00:00   (1hr was way too short for full training)
#    2. -p public → general           (general partition has more GPU availability)
#    3. do_sample=False → True        (need stochastic sampling for exploration)
#    4. max_response_length=1536 / response_length=2408 mismatch → unified to 2048
#    5. pure_orm rewarder → hybrid    (hybrid gives richer reward signal)
#    6. trainer.save_freq=1 → 619     (saving every step creates huge checkpoint overhead)
#    7. trainer.test_freq=1 → 20      (validation every step slows training significantly)
#    8. val_before_train=False → True (always validate before training to get baseline)
#    9. +algorithm=vineppo_grpo       (must copy vineppo_grpo.yaml to config/algorithm/ first)
#   10. algorithm.use_kl_in_reward removed (handled inside advantage function via vine_grpo_kl)
#
#  KEPT IDENTICAL (same as VinePPO baseline for fair comparison):
#    data.train_batch_size=16
#    ppo_mini_batch_size=8
#    ppo_micro_batch_size_per_gpu=1
#    lr=5e-6, temperature=0.8, top_p=0.95
#    kl settings, gamma, n_gpus, nnodes
#    mc_k=3  (K=3 rollouts per intermediate state)
#
#  PREREQUISITE — copy yaml first:
#    cp vineppo_grpo.yaml \
#       /scratch/ngangada/thesis/thesis/verl/verl/trainer/config/algorithm/vineppo_grpo.yaml
# ─────────────────────────────────────────────────────────────────────────────

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
  trainer.total_epochs=1 \
  +algorithm=vineppo_grpo \
  algorithm.adv_estimator=vineppo_grpo \
  algorithm.gamma=1.0 \
  algorithm.kl_ctrl.kl_coef=0.01 \
  algorithm.normalize_advantages=true \
  algorithm.kl_ctrl.type=adaptive \
  algorithm.kl_ctrl.target_kl=0.02 \
  algorithm.kl_ctrl.horizon=10000 \
  custom_reward_function.path=/scratch/ngangada/thesis/thesis/verl/verl/utils/reward_score/puzzle_baron_rewarder_hybrid.py \
  custom_reward_function.name=compute_puzzle_baron_score \
  reward_model.reward_manager=naive \
  trainer.logger=console \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=619 \
  trainer.test_freq=20 \
  +trainer.mc_k=3 \
  trainer.project_name=puzzle_3b_localized_grpo \
  trainer.experiment_name=localized_grpo_fixed_global_norm \
  2>&1 | tee $SCRATCH/verl_logs/localized_grpo_puzzlebaron3b_fixed.log
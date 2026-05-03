#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-10:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -o output/verl_vineppo_puzzlebarron.%j.out
#SBATCH -e output/verl_vineppo_puzzlebarron.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# === Environment setup ===


# activate your uv venv
source $HOME/.venv/bin/activate
export PYTHONPATH=/scratch/ngangada/thesis/thesis/verl:$PYTHONPATH
# === Training flags ===
export VLLM_USE_V1=1
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES

# === Training flags ===
python -m verl.trainer.main_ppo \
  data.train_files=/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_train_gsm8k_like.parquet \
  data.val_files=/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_val_gsm8k_like.parquet \
  data.train_batch_size=64 \
  data.max_prompt_length=3080 \
  data.max_response_length=4104 \
  actor_rollout_ref.model.path=/scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_sft_05/puzzle_sft_05_run1/global_step_9000/huggingface \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  +algorithm=vine_ppo \
  algorithm.adv_estimator=vine_ppo \
  algorithm.gamma=1.0 \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.normalize_advantages=true \
  custom_reward_function.path=/scratch/ngangada/thesis/thesis/verl/verl/utils/reward_score/puzzle_baron_rewarder_hybrid.py \
  custom_reward_function.name=compute_puzzle_baron_score \
  reward_model.reward_manager=naive \
  trainer.logger=console \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=1 \
  +trainer.mc_k=10 \
  2>&1 | tee $SCRATCH/verl_logs/vine_ppo_puzzlebaron_new_rewarder_after_sft__1.log
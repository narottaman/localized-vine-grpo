#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 0-4:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -o output/verl_sft_gsm8k_raw.%j.out
#SBATCH -e output/verl_sft_gsm8k_raw.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# =====================================================================
# SFT on GSM8K reasoning dataset — raw Qwen2.5-3B-Instruct base
#
# Purpose: teach the model the Step #N. reasoning format on math
# problems BEFORE puzzle baron RL, testing whether diverse step-by-step
# SFT reduces RL sample complexity on logic puzzle tasks.
#
# Chain: Qwen2.5-3B-Instruct → SFT GSM8K → RL (VineGRPO / GDPO)
# Compare against: Qwen2.5-3B-Instruct → SFT PB → RL (baseline)
# =====================================================================

source $HOME/.venv/bin/activate
export PYTHONPATH=/scratch/ngangada/thesis/thesis/verl:$PYTHONPATH
export VLLM_USE_V1=1
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES

torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.fsdp_sft_trainer \
  data.train_files=/scratch/ngangada/thesis/thesis/verl/data/gsm8k/gsm8k_reasoning_sft_train.parquet \
  data.val_files=/scratch/ngangada/thesis/thesis/verl/data/gsm8k/gsm8k_reasoning_sft_val.parquet \
  data.prompt_key=input \
  data.response_key=output \
  data.train_batch_size=8 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=4096 \
  +data.max_prompt_length=1024 \
  +data.max_response_length=3072 \
  model.strategy=fsdp \
  model.partial_pretrain=/scratch/ngangada/models/Qwen2.5-3B-Instruct \
  model.lora_rank=0 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  optim.lr=1e-4 \
  trainer.project_name=puzzle_sft_gsm8k_raw \
  trainer.experiment_name=sft_gsm8k_on_raw_qwen \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=200 \
  trainer.total_epochs=2 \
  trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
  trainer.test_freq=200 \
  trainer.max_ckpt_to_keep=3 \
  trainer.resume_mode=auto \
  2>&1 | tee /scratch/ngangada/verl_logs/sft_gsm8k_raw_qwen.log
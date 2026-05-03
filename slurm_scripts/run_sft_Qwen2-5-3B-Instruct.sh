#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 0-2:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -o output/verl_sft_3b_puzzlebarron.%j.out
#SBATCH -e output/verl_sft_3b_puzzlebarron.%j.err
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


  
  
torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.fsdp_sft_trainer \
  data.train_files=/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_train_sft_clean.v2.parquet \
  data.val_files=/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_val_sft_clean.v2.parquet \
  data.prompt_key=input \
  data.response_key=output \
  data.train_batch_size=8 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=8192 \
  +data.max_prompt_length=4096 \
  +data.max_response_length=4096 \
  model.strategy=fsdp \
  model.partial_pretrain=/scratch/ngangada/models/Qwen2.5-3B-Instruct \
  model.lora_rank=16 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  optim.lr=1e-4 \
  trainer.project_name=puzzle_sf_eot \
  trainer.experiment_name=puzzle_sft_eot \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.total_epochs=5 \
  trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
  trainer.test_freq=100 \
  trainer.max_ckpt_to_keep=100 \
  trainer.resume_mode=auto \
  2>&1 | tee /scratch/ngangada/verl_logs/sft_training_with_eval_3b_4.log

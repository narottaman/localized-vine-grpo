#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-01:00:00
#SBATCH -p htc
#SBATCH -q public
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -o output/eval_zl_naive.%j.out
#SBATCH -e output/eval_zl_naive.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

source $HOME/.venv/bin/activate
export PYTHONPATH=/scratch/ngangada/thesis/thesis/verl:$PYTHONPATH
export VLLM_USE_V1=1
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES

python /scratch/ngangada/thesis/thesis/verl/eval_zebralogic_size.py \
    --checkpoint_dir /scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_3b_vinegrpo_pb_zl/vinegrpo_pb_zl_sft102/global_step_10/actor \
    --output_dir /scratch/ngangada/thesis/thesis/verl/eval_results/vinegrpo_zl_step10_naive \
    --max_samples 1000 \
    --device cuda \
    --save_freq 50
#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-5:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH -o output/vineppo_619_eval.%j.out
#SBATCH -e output/vineppo_619_eval.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# ========================================================================
# Evaluate VinePPO @ step 619 (Final Checkpoint)
# ========================================================================

echo "========================================"
echo "🧪 VINEPPO STEP 619 EVALUATION"
echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# --------- Environment Setup ----------
source $HOME/.venv/bin/activate
export PYTHONPATH=/scratch/ngangada/thesis/thesis/verl:$PYTHONPATH

export VLLM_USE_V1=1
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES

mkdir -p output

# --------- Configuration ----------
DATA_PATH="/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_val_gsm8k_like.parquet"

VINEPPO_619_CHECKPOINT="/scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_3b_vineppo_v2_stepwise/vineppo_fixed_stepwise_ter/global_step_619/actor"

OUTPUT_BASE="/scratch/ngangada/thesis/thesis/eval_results"
VINEPPO_619_OUTPUT="$OUTPUT_BASE/vineppo_step619"

MAX_NEW_TOKENS=4096
DEVICE="cuda"
SAVE_EVERY=50

# Check if checkpoint exists
if [ ! -d "$VINEPPO_619_CHECKPOINT" ]; then
    echo "❌ ERROR: VinePPO step 619 checkpoint not found at: $VINEPPO_619_CHECKPOINT"
    echo "Available checkpoints:"
    ls -la /scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_3b_vineppo_v2_stepwise/vineppo_fixed_stepwise_ter/
    exit 1
fi

mkdir -p "$VINEPPO_619_OUTPUT"

echo ""
echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  VinePPO Checkpoint: $VINEPPO_619_CHECKPOINT"
echo "  Output: $VINEPPO_619_OUTPUT"
echo "  Save every: $SAVE_EVERY samples"
echo ""

# Use the evaluation script
EVAL_SCRIPT="/scratch/ngangada/thesis/thesis/verl/eval_with_resume_fixed.py"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "❌ ERROR: Evaluation script not found at: $EVAL_SCRIPT"
    exit 1
fi

# ========================================================================
# EVALUATION
# ========================================================================
echo ""
echo "========================================"
echo "📊 Evaluating VinePPO @ step 619"
echo "========================================"

python "$EVAL_SCRIPT" \
    --checkpoint_dir "$VINEPPO_619_CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --output_dir "$VINEPPO_619_OUTPUT" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --save_every $SAVE_EVERY \
    --device $DEVICE

echo ""
echo "========================================"
echo "📋 EVALUATION COMPLETE"
echo "========================================"
echo "Job finished at: $(date)"
echo ""
echo "Results saved to: $VINEPPO_619_OUTPUT"
echo ""
echo "Next step: Run comparison script"
echo "  bash /scratch/ngangada/thesis/thesis/verl/run_final_comparison_all.sh"
echo "========================================"

exit 0
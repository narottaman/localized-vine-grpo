#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-8:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH -o output/raw_qwen_eval.%j.out
#SBATCH -e output/raw_qwen_eval.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# ========================================================================
# Evaluate Raw Qwen2.5-3B on Puzzle Baron (Zero-Shot Baseline)
# ========================================================================

echo "========================================"
echo "🧪 RAW QWEN2.5-3B EVALUATION"
echo "========================================"
echo "Testing pretrained model with NO puzzle training"
echo "This provides the true zero-shot baseline"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "========================================"

# Environment
source $HOME/.venv/bin/activate
export PYTHONPATH=/scratch/ngangada/thesis/thesis/verl:$PYTHONPATH

export VLLM_USE_V1=1
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES

mkdir -p output

# Configuration
DATA_PATH="/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_val_gsm8k_like.parquet"

# Raw pretrained model (no puzzle training)
RAW_MODEL="/scratch/ngangada/models/Qwen2.5-3B-Instruct"

OUTPUT_BASE="/scratch/ngangada/thesis/thesis/eval_results"
RAW_OUTPUT="$OUTPUT_BASE/raw_model"

MAX_NEW_TOKENS=4096
DEVICE="cuda"
SAVE_EVERY=50

# Check if model exists
if [ ! -d "$RAW_MODEL" ]; then
    echo "❌ ERROR: Raw Qwen2.5-3B model not found at: $RAW_MODEL"
    echo ""
    echo "Please download the model first:"
    echo "  huggingface-cli download Qwen/Qwen2.5-3B-Instruct"
    echo ""
    echo "Or update RAW_MODEL path to point to your Qwen2.5-3B-Instruct"
    exit 1
fi

mkdir -p "$RAW_OUTPUT"

echo ""
echo "Configuration:"
echo "  Model: Qwen2.5-3B-Instruct (pretrained, NO puzzle training)"
echo "  Data: $DATA_PATH"
echo "  Output: $RAW_OUTPUT"
echo "  Purpose: Zero-shot baseline for complete pipeline analysis"
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
echo "📊 Evaluating Raw Qwen2.5-3B (Zero-Shot)"
echo "========================================"
echo ""
echo "⚠️  IMPORTANT:"
echo "This model has NEVER seen Puzzle Baron puzzles"
echo "Results show pure reasoning capability without training"
echo ""

python "$EVAL_SCRIPT" \
    --checkpoint_dir "$RAW_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$RAW_OUTPUT" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --save_every $SAVE_EVERY \
    --device $DEVICE

echo ""
echo "========================================"
echo "📋 EVALUATION COMPLETE"
echo "========================================"
echo "Job finished at: $(date)"
echo ""
echo "Results saved to: $RAW_OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Compare with SFT baseline (trained on puzzles)"
echo "  2. Compare with RL models (puzzle + RL)"
echo "  3. Run: bash run_final_comparison_complete.sh"
echo "========================================"

exit 0
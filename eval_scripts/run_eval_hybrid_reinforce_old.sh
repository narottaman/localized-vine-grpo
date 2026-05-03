#!/bin/bash
#SBATCH -A grp_cbaral
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-10:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH -o output/eval_reinforce_hybrid_old.%j.out
#SBATCH -e output/eval_reinforce_hybrid_old.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngangada@asu.edu
#SBATCH --export=NONE

# ========================================================================
# Evaluate REINFORCE++ Hybrid Model (FIXED SCORING)
# ========================================================================

echo "========================================"
echo "🧪 REINFORCE++ HYBRID EVALUATION"
echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

FRESH_START=false
if [ "$1" == "--fresh" ]; then
    FRESH_START=true
    echo "🔄 Fresh start requested"
fi

# --------- Environment Setup ----------
source $HOME/.venv/bin/activate
export PYTHONPATH=/scratch/ngangada/thesis/thesis/verl:$PYTHONPATH

export VLLM_USE_V1=1
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES

mkdir -p output

# --------- Configuration ----------
DATA_PATH="/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_val_gsm8k_like.parquet"

# REINFORCE++ Hybrid checkpoint  
CHECKPOINT="/scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_3b_reinforce_plusplus_test/reinforce_plusplus_baseline_test/global_step_400/actor"

OUTPUT_DIR="/scratch/ngangada/thesis/thesis/verl/eval_results/reinforce_plusplus_hybrid_old_400"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ ERROR: Checkpoint not found at: $CHECKPOINT"
    echo "Available checkpoints:"
    ls -la /scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_3b_reinforce_plusplus/reinforce_plusplus_baseline/
    exit 1
fi

if [ "$FRESH_START" = true ]; then
    echo "🗑️  Deleting old results..."
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT_DIR"
echo "  Fresh start: $FRESH_START"
echo ""

# ========================================================================
# EVALUATION (using FIXED scoring)
# ========================================================================

python /scratch/ngangada/thesis/thesis/verl/eval_puzzle_baron_with_resume.py \
    --checkpoint_dir "$CHECKPOINT" \
    --val_file "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --save_freq 50

EVAL_STATUS=$?

echo ""
echo "========================================"
if [ $EVAL_STATUS -eq 0 ]; then
    echo "✅ EVALUATION COMPLETE"
    echo ""
    echo "📊 Results:"
    cat "$OUTPUT_DIR/metrics.json"
else
    echo "❌ EVALUATION FAILED (exit code: $EVAL_STATUS)"
fi
echo "========================================"
echo "Job finished at: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/metrics.json"
echo "  head -20 $OUTPUT_DIR/results_latest.csv"
echo ""
echo "To resume if incomplete:"
echo "  sbatch eval_reinforce_hybrid.sh"
echo ""
echo "To start fresh:"
echo "  sbatch eval_reinforce_hybrid.sh --fresh"
echo "========================================"

exit $EVAL_STATUS
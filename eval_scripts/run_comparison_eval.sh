#!/bin/bash

# Enhanced Puzzle Baron Evaluation - Compare Base vs VinePPO
# For paper-level results

set -e  # Exit on error

# Configuration
BASE_MODEL="/scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_sft_eot/puzzle_sft_eot/global_step_4000/huggingface"  # Your SFT baseline
VINEPPO_MODEL="/scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_3b_vineppo_v2_stepwise/vineppo_fixed_stepwise_ter/global_step_230/actor/huggingface"

DATA_PATH="/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_val_gsm8k_like.parquet"  
OUTPUT_BASE_DIR="/scratch/ngangada/thesis/thesis/eval_results"

# Number of samples (start small for testing, then scale up)
MAX_SAMPLES=50  # Use 50 for quick test, remove for full evaluation

echo "========================================"
echo "🧪 PUZZLE BARON MODEL COMPARISON"
echo "========================================"
echo "Base Model: $BASE_MODEL"
echo "VinePPO Model: $VINEPPO_MODEL"
echo "Max Samples: ${MAX_SAMPLES:-ALL}"
echo "========================================"

# 1. Evaluate Base Model (SFT)
echo ""
echo "📊 [1/2] Evaluating Base Model (SFT)..."
python eval_puzzle_baron_enhanced.py \
    --model_path "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_BASE_DIR/base_model" \
    --max_samples ${MAX_SAMPLES} \
    --split test

echo "✅ Base model evaluation complete!"

# 2. Evaluate VinePPO Model
echo ""
echo "📊 [2/2] Evaluating VinePPO Model..."
python eval_puzzle_baron_enhanced.py \
    --model_path "$VINEPPO_MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_BASE_DIR/vineppo_model" \
    --max_samples ${MAX_SAMPLES} \
    --split test

echo "✅ VinePPO model evaluation complete!"

# 3. Generate comparison report
echo ""
echo "📈 Generating comparison report..."
python compare_results.py \
    --base_results "$OUTPUT_BASE_DIR/base_model/metrics.json" \
    --vineppo_results "$OUTPUT_BASE_DIR/vineppo_model/metrics.json" \
    --output_dir "$OUTPUT_BASE_DIR/comparison"

echo ""
echo "========================================"
echo "✅ EVALUATION COMPLETE!"
echo "========================================"
echo "Results saved to: $OUTPUT_BASE_DIR"
echo ""
echo "Key files:"
echo "  - base_model/metrics.json"
echo "  - vineppo_model/metrics.json"
echo "  - comparison/comparison_report.txt"
echo "  - comparison/comparison_plots.png"
echo "========================================"
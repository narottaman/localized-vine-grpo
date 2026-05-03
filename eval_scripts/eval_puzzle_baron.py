#!/usr/bin/env python3
"""
Puzzle Baron Evaluation for VERL/FSDP Checkpoints
FIXED: Properly extracts prompt content from numpy arrays

Usage:
    python eval_puzzle_baron_fixed.py \
        --checkpoint_dir /path/to/checkpoint/actor \
        --data_path /path/to/data.parquet \
        --output_dir ./eval_results \
        --max_samples 100
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def load_fsdp_checkpoint(checkpoint_dir: str, device: str = "cuda"):
    """Load FSDP checkpoint from VERL format"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Try huggingface subdir first
    hf_dir = checkpoint_dir / "huggingface"
    if not hf_dir.exists():
        hf_dir = checkpoint_dir
    
    print(f"📂 Loading config from {hf_dir}")
    config = AutoConfig.from_pretrained(hf_dir)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    
    print(f"🏗️  Initializing model from config...")
    model = AutoModelForCausalLM.from_config(config)
    
    # Find and load checkpoint
    checkpoint_file = checkpoint_dir / "model_world_size_1_rank_0.pt"
    if not checkpoint_file.exists():
        checkpoint_file = checkpoint_dir / "pytorch_model.bin"
    
    if checkpoint_file.exists():
        print(f"📦 Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove FSDP wrapper prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_fsdp_wrapped_module.', '')
            new_key = new_key.replace('module.', '')
            new_key = new_key.replace('_forward_module.', '')
            new_state_dict[new_key] = value
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"⚠️  Missing keys: {len(missing)}")
        if unexpected:
            print(f"⚠️  Unexpected keys: {len(unexpected)}")
    
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully!")
    
    return model, tokenizer


def extract_prompt_content(prompt_field):
    """
    FIXED: Properly extract prompt content from various formats
    
    The prompt can be:
    1. A string (direct text)
    2. A list of dicts [{"role": "user", "content": "..."}]
    3. A numpy array containing either of the above
    """
    # Handle numpy array
    if isinstance(prompt_field, np.ndarray):
        prompt_field = prompt_field.tolist()
    
    # Handle list (chat messages format)
    if isinstance(prompt_field, list) and len(prompt_field) > 0:
        first_item = prompt_field[0]
        
        # If it's a dict with 'content' key (chat format)
        if isinstance(first_item, dict) and 'content' in first_item:
            return first_item['content']
        
        # If the list item is itself a list (nested)
        if isinstance(first_item, list):
            return extract_prompt_content(first_item)
        
        # Otherwise convert to string
        return str(first_item)
    
    # Handle dict directly
    if isinstance(prompt_field, dict) and 'content' in prompt_field:
        return prompt_field['content']
    
    # Handle string
    if isinstance(prompt_field, str):
        return prompt_field
    
    # Fallback
    return str(prompt_field) if prompt_field is not None else ""


def extract_ground_truth(row):
    """Extract ground truth from row"""
    gt_raw = None
    
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        gt_raw = reward_model.get("ground_truth")
    elif isinstance(reward_model, str):
        gt_raw = reward_model
    
    if gt_raw is None:
        gt_raw = row.get("ground_truth") or row.get("label") or row.get("answer")
    
    if isinstance(gt_raw, dict):
        gt_raw = gt_raw.get("ground_truth", str(gt_raw))
    elif gt_raw is not None and not isinstance(gt_raw, str):
        gt_raw = str(gt_raw)
    
    return gt_raw or ""


def parse_table_from_response(response: str) -> List[List[str]]:
    """Extract table rows from model response"""
    import re
    
    lines = response.strip().split('\n')
    rows = []
    
    for line in lines:
        # Skip lines without enough pipes
        if '|' not in line or line.count('|') < 2:
            continue
        
        # Skip separator lines (just dashes and pipes)
        if re.match(r'^[\s\-:|]+$', line):
            continue
        
        # Split by pipe and clean
        cells = [cell.strip() for cell in line.split('|')]
        cells = [c for c in cells if c]  # Remove empty strings
        
        # Need at least 2 cells for a valid row
        if len(cells) >= 2:
            rows.append(cells)
    
    return rows


def compute_table_accuracy(predicted: List[List[str]], ground_truth: List[List[str]]) -> Tuple[float, int, int]:
    """Compute cell-by-cell accuracy between predicted and ground truth tables"""
    if not predicted or not ground_truth:
        return 0.0, 0, len(ground_truth) * 3 if ground_truth else 0
    
    # Normalize for comparison
    def normalize(s):
        return s.strip().lower()
    
    correct = 0
    total = 0
    
    # Compare row by row
    max_rows = max(len(predicted), len(ground_truth))
    max_cols = max(
        max((len(row) for row in predicted), default=0),
        max((len(row) for row in ground_truth), default=0)
    )
    
    for i in range(max_rows):
        pred_row = predicted[i] if i < len(predicted) else []
        gt_row = ground_truth[i] if i < len(ground_truth) else []
        
        for j in range(max_cols):
            pred_cell = normalize(pred_row[j]) if j < len(pred_row) else ""
            gt_cell = normalize(gt_row[j]) if j < len(gt_row) else ""
            
            total += 1
            if pred_cell == gt_cell:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def evaluate_model(
    checkpoint_dir: str,
    data_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    max_new_tokens: int = 4096
) -> Dict:
    """Evaluate VERL checkpoint on Puzzle Baron dataset"""
    
    # Load model
    model, tokenizer = load_fsdp_checkpoint(checkpoint_dir, device)
    
    # Load data
    print(f"📊 Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    print(f"   Total samples: {len(df)}")
    
    if max_samples:
        df = df.head(max_samples)
        print(f"   Using first {max_samples} samples")
    
    print(f"\n🔍 Evaluating {len(df)} puzzles...")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        # Extract prompt (FIXED)
        prompt = extract_prompt_content(row.get("prompt"))
        ground_truth = extract_ground_truth(row)
        
        if not prompt:
            print(f"⚠️  Skipping row {idx}: empty prompt")
            continue
        
        # Parse ground truth table
        gt_table = parse_table_from_response(ground_truth)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only generated part
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse predicted table
        pred_table = parse_table_from_response(response)
        
        # Compute accuracy
        accuracy, correct_cells, total_cells = compute_table_accuracy(pred_table, gt_table)
        
        is_perfect = accuracy == 1.0 and len(pred_table) == len(gt_table)
        
        result = {
            'puzzle_id': idx,
            'accuracy': accuracy,
            'correct_cells': correct_cells,
            'total_cells': total_cells,
            'is_perfect': is_perfect,
            'num_pred_rows': len(pred_table),
            'num_gt_rows': len(gt_table),
            'response_length': len(response),
        }
        
        results.append(result)
        
        # Show progress every 50 samples
        if (idx + 1) % 50 == 0:
            current_acc = np.mean([r['accuracy'] for r in results])
            current_perfect = np.mean([r['is_perfect'] for r in results])
            print(f"\n   Progress: {idx+1}/{len(df)} | Acc: {current_acc:.3f} | Perfect: {current_perfect:.1%}")
    
    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(results)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    
    return metrics


def compute_aggregate_metrics(results: List[Dict]) -> Dict:
    """Compute aggregate metrics"""
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    metrics = {
        'total_puzzles': len(df),
        'mean_cell_accuracy': float(df['accuracy'].mean()),
        'std_cell_accuracy': float(df['accuracy'].std()),
        'median_cell_accuracy': float(df['accuracy'].median()),
        'perfect_solve_rate': float(df['is_perfect'].mean()),
        'num_perfect_solves': int(df['is_perfect'].sum()),
        'total_cells': int(df['total_cells'].sum()),
        'total_correct_cells': int(df['correct_cells'].sum()),
        'overall_cell_accuracy': float(df['correct_cells'].sum() / df['total_cells'].sum()) if df['total_cells'].sum() > 0 else 0,
        'mean_response_length': float(df['response_length'].mean()),
        'accuracy_quartiles': {
            'q25': float(df['accuracy'].quantile(0.25)),
            'q50': float(df['accuracy'].quantile(0.50)),
            'q75': float(df['accuracy'].quantile(0.75)),
        }
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate VERL/FSDP Checkpoint on Puzzle Baron')
    parser.add_argument('--checkpoint_dir', type=str, required=True, 
                        help='Path to checkpoint directory (e.g., .../global_step_230/actor/)')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to parquet data')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Max samples to evaluate (default: all)')
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                        help='Max new tokens to generate')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 PUZZLE BARON EVALUATION - VERL CHECKPOINT")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print("=" * 80)
    
    metrics = evaluate_model(
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )
    
    print("\n" + "=" * 80)
    print("📈 FINAL METRICS")
    print("=" * 80)
    print(f"Total Puzzles: {metrics['total_puzzles']}")
    print(f"Perfect Solve Rate: {metrics['perfect_solve_rate']:.1%} ({metrics['num_perfect_solves']}/{metrics['total_puzzles']})")
    print(f"Mean Cell Accuracy: {metrics['mean_cell_accuracy']:.3f} ± {metrics['std_cell_accuracy']:.3f}")
    print(f"Overall Cell Accuracy: {metrics['overall_cell_accuracy']:.3f}")
    print(f"Accuracy Quartiles: Q25={metrics['accuracy_quartiles']['q25']:.3f}, Q50={metrics['accuracy_quartiles']['q50']:.3f}, Q75={metrics['accuracy_quartiles']['q75']:.3f}")
    print("=" * 80)
    print(f"\n✅ Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
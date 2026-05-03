
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def load_checkpoint(checkpoint_dir: str, device: str = "cuda"):
    """Load checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    
    hf_dir = checkpoint_dir / "huggingface"
    if not hf_dir.exists():
        hf_dir = checkpoint_dir
    
    print(f"📂 Loading config from {hf_dir}")
    config = AutoConfig.from_pretrained(hf_dir)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    
    fsdp_file = checkpoint_dir / "model_world_size_1_rank_0.pt"
    
    if fsdp_file.exists():
        print(f"🏗️  Loading FSDP checkpoint...")
        model = AutoModelForCausalLM.from_config(config)
        
        checkpoint = torch.load(fsdp_file, map_location="cpu")
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_fsdp_wrapped_module.', '')
            new_key = new_key.replace('module.', '')
            new_key = new_key.replace('_forward_module.', '')
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"🏗️  Loading HuggingFace checkpoint...")
        model = AutoModelForCausalLM.from_pretrained(hf_dir)
    
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully!")
    
    return model, tokenizer


def extract_prompt_content(prompt_field):
    """Properly extract prompt content"""
    if isinstance(prompt_field, np.ndarray):
        prompt_field = prompt_field.tolist()
    
    if isinstance(prompt_field, list) and len(prompt_field) > 0:
        first_item = prompt_field[0]
        if isinstance(first_item, dict) and 'content' in first_item:
            return first_item['content']
        if isinstance(first_item, list):
            return extract_prompt_content(first_item)
        return str(first_item)
    
    if isinstance(prompt_field, dict) and 'content' in prompt_field:
        return prompt_field['content']
    
    if isinstance(prompt_field, str):
        return prompt_field
    
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
        if '|' not in line or line.count('|') < 2:
            continue
        if re.match(r'^[\s\-:|]+$', line):
            continue
        
        cells = [cell.strip() for cell in line.split('|')]
        cells = [c for c in cells if c]
        
        if len(cells) >= 2:
            rows.append(cells)
    
    return rows


def compute_table_accuracy(predicted: List[List[str]], ground_truth: List[List[str]]) -> Tuple[float, int, int, bool]:
    """Compute cell-by-cell accuracy"""
    if not predicted or not ground_truth:
        total = len(ground_truth) * 3 if ground_truth else 0
        return 0.0, 0, total, False
    
    def normalize(s):
        return s.strip().lower()
    
    correct = 0
    total = 0
    
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
    is_perfect = accuracy == 1.0 and len(predicted) == len(ground_truth)
    
    return accuracy, correct, total, is_perfect


def load_existing_results(output_dir: str) -> Tuple[List[Dict], int]:
    """
    Load existing results to enable resume.
    Returns (results_list, start_index)
    """
    results = []
    start_idx = 0
    
    # Find the best file to load from
    results_file = os.path.join(output_dir, 'detailed_results.csv')
    latest_file = os.path.join(output_dir, 'detailed_results_latest.csv')
    
    # Check for checkpoint files
    checkpoint_files = sorted(Path(output_dir).glob('detailed_results_checkpoint_*.csv'))
    
    # Priority: final > latest > checkpoint
    if os.path.exists(results_file):
        best_file = results_file
    elif os.path.exists(latest_file):
        best_file = latest_file
    elif checkpoint_files:
        best_file = str(checkpoint_files[-1])
    else:
        return [], 0
    
    try:
        df = pd.read_csv(best_file)
        results = df.to_dict('records')
        
        # Find the maximum puzzle_id that was evaluated
        # This is the index we should continue FROM (exclusive)
        if len(results) > 0:
            max_puzzle_id = max(r['puzzle_id'] for r in results)
            start_idx = max_puzzle_id + 1
        
        print(f"📂 Loaded {len(results)} existing results from {best_file}")
        print(f"   Last evaluated puzzle_id: {start_idx - 1}")
        print(f"   Will resume from puzzle_id: {start_idx}")
        
    except Exception as e:
        print(f"⚠️  Could not load existing results: {e}")
        return [], 0
    
    return results, start_idx


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
        'accuracy_distribution': {
            'zero': float((df['accuracy'] == 0).mean()),
            'low_0_to_25': float(((df['accuracy'] > 0) & (df['accuracy'] <= 0.25)).mean()),
            'mid_25_to_50': float(((df['accuracy'] > 0.25) & (df['accuracy'] <= 0.5)).mean()),
            'high_50_to_75': float(((df['accuracy'] > 0.5) & (df['accuracy'] <= 0.75)).mean()),
            'very_high_75_to_99': float(((df['accuracy'] > 0.75) & (df['accuracy'] < 1.0)).mean()),
            'perfect': float((df['accuracy'] == 1.0).mean()),
        },
        'accuracy_quartiles': {
            'q25': float(df['accuracy'].quantile(0.25)),
            'q50': float(df['accuracy'].quantile(0.50)),
            'q75': float(df['accuracy'].quantile(0.75)),
        }
    }
    
    return metrics


def evaluate_model(
    checkpoint_dir: str,
    data_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    max_new_tokens: int = 4096,
    print_every: int = 25,
    save_every: int = 50,
    resume: bool = True
) -> Dict:
    """Evaluate model with FIXED resume capability"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing results if resuming
    results = []
    start_idx = 0
    sample_outputs = []
    
    if resume:
        results, start_idx = load_existing_results(output_dir)
        
        # Load sample outputs if they exist
        sample_file = os.path.join(output_dir, 'sample_outputs_latest.json')
        if os.path.exists(sample_file):
            try:
                with open(sample_file, 'r') as f:
                    sample_outputs = json.load(f)
            except:
                pass
    
    # Load data
    print(f"📊 Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    total_samples = len(df)
    print(f"   Total samples in dataset: {total_samples}")
    
    if max_samples:
        df = df.head(max_samples)
        total_samples = len(df)
        print(f"   Limiting to first {max_samples} samples")
    
    # Calculate remaining work
    remaining = total_samples - start_idx
    
    print(f"\n🔍 Evaluation status:")
    print(f"   Total samples: {total_samples}")
    print(f"   Already done: {start_idx}")
    print(f"   Remaining: {remaining}")
    print(f"   Estimated time: {remaining * 125 / 3600:.1f} hours")
    
    if remaining <= 0:
        print("\n✅ All samples already evaluated!")
        metrics = compute_aggregate_metrics(results)
        return metrics
    
    # Load model (only if there's work to do)
    model, tokenizer = load_checkpoint(checkpoint_dir, device)
    
    # Create progress bar for ONLY the remaining samples
    print(f"\n🚀 Starting evaluation from sample {start_idx}...")
    
    newly_evaluated = 0
    
    # FIXED: Only iterate over remaining samples
    remaining_df = df.iloc[start_idx:]
    
    for idx, row in tqdm(remaining_df.iterrows(), 
                         total=len(remaining_df), 
                         desc=f"Evaluating ({start_idx}-{total_samples})",
                         initial=0):
        
        prompt = extract_prompt_content(row.get("prompt"))
        ground_truth = extract_ground_truth(row)
        
        if not prompt:
            # Still record it as skipped with 0 accuracy
            result = {
                'puzzle_id': idx,
                'accuracy': 0.0,
                'correct_cells': 0,
                'total_cells': 0,
                'is_perfect': False,
                'num_pred_rows': 0,
                'num_gt_rows': 0,
                'response_length': 0,
                'skipped': True
            }
            results.append(result)
            newly_evaluated += 1
            continue
        
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
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        pred_table = parse_table_from_response(response)
        accuracy, correct_cells, total_cells, is_perfect = compute_table_accuracy(pred_table, gt_table)
        
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
        newly_evaluated += 1
        
        # Store some examples
        if len(sample_outputs) < 10 or is_perfect or (len(sample_outputs) < 20 and accuracy > 0.5):
            sample_outputs.append({
                'puzzle_id': idx,
                'prompt': prompt[:500],
                'response': response[:2000],
                'ground_truth': ground_truth,
                'accuracy': accuracy,
                'is_perfect': is_perfect
            })
        
        # Progress update
        if newly_evaluated % print_every == 0:
            current_acc = np.mean([r['accuracy'] for r in results])
            current_perfect = np.mean([r['is_perfect'] for r in results])
            print(f"\n   [Total: {len(results)}/{total_samples}] Acc: {current_acc:.3f} | Perfect: {current_perfect:.1%}")
        
        # Checkpoint saving
        if newly_evaluated % save_every == 0:
            print(f"\n   💾 Saving checkpoint ({len(results)} total results)...")
            
            # Sort results by puzzle_id before saving
            results_sorted = sorted(results, key=lambda x: x['puzzle_id'])
            
            temp_df = pd.DataFrame(results_sorted)
            temp_df.to_csv(os.path.join(output_dir, f'detailed_results_checkpoint_{len(results)}.csv'), index=False)
            temp_df.to_csv(os.path.join(output_dir, 'detailed_results_latest.csv'), index=False)
            
            with open(os.path.join(output_dir, 'sample_outputs_latest.json'), 'w') as f:
                json.dump(sample_outputs, f, indent=2, default=str)
            
            partial_metrics = compute_aggregate_metrics(results_sorted)
            partial_metrics['evaluated_samples'] = len(results)
            partial_metrics['total_samples'] = total_samples
            partial_metrics['remaining'] = total_samples - len(results)
            partial_metrics['last_puzzle_id'] = idx
            
            with open(os.path.join(output_dir, 'metrics_latest.json'), 'w') as f:
                json.dump(partial_metrics, f, indent=2, default=float)
            
            print(f"   ✅ Checkpoint saved! Last puzzle_id: {idx}")
    
    # Final save
    print(f"\n💾 Saving final results...")
    
    # Sort results by puzzle_id
    results_sorted = sorted(results, key=lambda x: x['puzzle_id'])
    
    metrics = compute_aggregate_metrics(results_sorted)
    metrics['evaluated_samples'] = len(results)
    metrics['total_samples'] = total_samples
    metrics['is_complete'] = len(results) >= total_samples
    
    results_df = pd.DataFrame(results_sorted)
    results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    results_df.to_csv(os.path.join(output_dir, 'detailed_results_latest.csv'), index=False)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=float)
    
    with open(os.path.join(output_dir, 'sample_outputs.json'), 'w') as f:
        json.dump(sample_outputs, f, indent=2, default=str)
    
    print(f"   ✅ Results saved! Total: {len(results)}/{total_samples}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_resume', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 PUZZLE BARON EVALUATION (FIXED RESUME)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Resume: {not args.no_resume}")
    print("=" * 80)
    
    metrics = evaluate_model(
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        save_every=args.save_every,
        resume=not args.no_resume
    )
    
    print("\n" + "=" * 80)
    print("📈 RESULTS")
    print("=" * 80)
    print(f"Evaluated: {metrics.get('evaluated_samples', 'N/A')}/{metrics.get('total_samples', 'N/A')}")
    print(f"Complete: {metrics.get('is_complete', False)}")
    print(f"Perfect Solve Rate: {metrics['perfect_solve_rate']:.1%}")
    print(f"Mean Cell Accuracy: {metrics['mean_cell_accuracy']:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
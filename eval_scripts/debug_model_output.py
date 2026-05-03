#!/usr/bin/env python3
"""
Debug script - Check what model is actually generating
FIXED: Uses same sampling params as training validation to ensure consistency
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path
import re


def load_fsdp_checkpoint(checkpoint_dir: str, device: str = "cuda"):
    """Load FSDP checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    hf_dir = checkpoint_dir / "huggingface"
    
    config = AutoConfig.from_pretrained(hf_dir)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForCausalLM.from_config(config)
    
    checkpoint_file = checkpoint_dir / "model_world_size_1_rank_0.pt"
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
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_fsdp_wrapped_module.', '')
        new_key = new_key.replace('module.', '')
        new_key = new_key.replace('_forward_module.', '')
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def extract_prompt(row):
    """Extract prompt from row"""
    prompt_field = row.get("prompt")
    if prompt_field is None:
        return ""
    
    if isinstance(prompt_field, list) and len(prompt_field) > 0:
        if isinstance(prompt_field[0], dict) and "content" in prompt_field[0]:
            return prompt_field[0]["content"]
        return str(prompt_field[0])
    elif isinstance(prompt_field, str):
        return prompt_field
    else:
        return str(prompt_field)


def extract_ground_truth(row):
    """Extract ground truth from row"""
    gt_raw = None
    
    if isinstance(row.get("reward_model"), dict):
        gt_raw = row["reward_model"].get("ground_truth")
    elif isinstance(row.get("reward_model"), str):
        gt_raw = row["reward_model"]
    
    if gt_raw is None:
        gt_raw = row.get("ground_truth") or row.get("label") or row.get("answer")
    
    if isinstance(gt_raw, dict):
        gt_raw = gt_raw.get("ground_truth", str(gt_raw))
    elif not isinstance(gt_raw, str):
        gt_raw = str(gt_raw)
    
    return gt_raw


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, 
                        default="/scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_3b_vineppo_v2_stepwise/vineppo_fixed_stepwise_ter/global_step_230/actor")
    parser.add_argument('--data_path', type=str,
                        default="/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_val_gsm8k_like.parquet")
    parser.add_argument('--mode', type=str, default='greedy', choices=['greedy', 'sampling'],
                        help='Generation mode: greedy (deterministic) or sampling (with anti-repetition)')
    args = parser.parse_args()
    
    print("=" * 100)
    print("🔍 DEBUG - CHECK MODEL OUTPUT (FIXED SAMPLING)")
    print("=" * 100)
    
    print(f"\n📋 Generation mode: {args.mode.upper()}")
    
    print("\n1️⃣ Loading model...")
    model, tokenizer = load_fsdp_checkpoint(args.checkpoint_dir)
    print("✅ Model loaded")
    
    print("\n2️⃣ Loading data...")
    df = pd.read_parquet(args.data_path)
    print(f"✅ Data loaded ({len(df)} rows)")
    
    # Get first sample
    print("\n3️⃣ Processing first sample...")
    row = df.iloc[0]
    
    prompt = extract_prompt(row)
    ground_truth = extract_ground_truth(row)
    
    print("\n" + "=" * 100)
    print("PROMPT (first 500 chars):")
    print("=" * 100)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    print("\n" + "=" * 100)
    print("GROUND TRUTH:")
    print("=" * 100)
    print(ground_truth[:500])
    
    # Generate
    print("\n4️⃣ Generating response...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        if args.mode == 'greedy':
            # GREEDY MODE - Same as training validation
            # This is what produces good results during training
            print("   Using GREEDY generation (same as training validation)")
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,           # Deterministic
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # SAMPLING MODE - With anti-repetition measures
            # Use this to test exploration capability
            print("   Using SAMPLING with repetition_penalty=1.2")
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.3,           # Low for coherence
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.2,    # CRITICAL: Prevent repetition loops
                pad_token_id=tokenizer.eos_token_id
            )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    print("\n" + "=" * 100)
    print("MODEL RESPONSE:")
    print("=" * 100)
    print(response)
    
    print("\n" + "=" * 100)
    print("RESPONSE ANALYSIS:")
    print("=" * 100)
    print(f"Total length: {len(response)} chars")
    print(f"Contains '|': {('|' in response)}")
    print(f"Number of '|' chars: {response.count('|')}")
    print(f"Number of newlines: {response.count(chr(10))}")
    
    # Check for foreign/garbage characters
    non_ascii = [c for c in response if ord(c) > 127]
    if non_ascii:
        print(f"⚠️  Non-ASCII characters found: {len(non_ascii)}")
        print(f"   Sample: {non_ascii[:20]}")
    else:
        print("✅ No foreign characters detected")
    
    # Count lines with tables
    lines_with_pipes = [line for line in response.split('\n') if '|' in line]
    print(f"Lines with '|': {len(lines_with_pipes)}")
    
    if lines_with_pipes:
        print("\nFirst 10 lines with pipes:")
        for i, line in enumerate(lines_with_pipes[:10]):
            print(f"  {i+1}: {line[:80]}")
    
    # Try to parse table
    print("\n" + "=" * 100)
    print("TABLE PARSING TEST:")
    print("=" * 100)
    
    lines = response.strip().split('\n')
    rows = []
    
    for i, line in enumerate(lines):
        if '|' not in line or line.count('|') < 2:
            continue
        
        if re.match(r'^\|[\s\-:]+\|', line):
            print(f"  Skipping separator line {i}: {line[:60]}")
            continue
        
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if cells and any(cell for cell in cells):
            rows.append(cells)
            print(f"  ✅ Row {len(rows)}: {cells}")
    
    print(f"\nTotal rows parsed: {len(rows)}")
    
    if not rows:
        print("\n❌ NO TABLE FOUND IN RESPONSE!")
        print("\nPossible reasons:")
        print("1. Model didn't generate a table")
        print("2. Table format is different than expected")
        print("3. Response is truncated or corrupted")
    else:
        print("\n✅ TABLE FOUND!")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
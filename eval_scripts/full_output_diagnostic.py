#!/usr/bin/env python3
"""
FULL OUTPUT DIAGNOSTIC - No truncation, shows complete generation
"""

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path
import argparse
import re


def load_fsdp_checkpoint(checkpoint_dir: str, device: str = "cuda"):
    """Load FSDP checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    hf_dir = checkpoint_dir / "huggingface"
    
    if not hf_dir.exists():
        hf_dir = checkpoint_dir
    
    config = AutoConfig.from_pretrained(hf_dir)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForCausalLM.from_config(config)
    
    checkpoint_file = checkpoint_dir / "model_world_size_1_rank_0.pt"
    if not checkpoint_file.exists():
        checkpoint_file = checkpoint_dir / "pytorch_model.bin"
    if not checkpoint_file.exists():
        checkpoint_file = checkpoint_dir / "model.safetensors"
    
    if checkpoint_file.exists():
        print(f"   Loading weights from: {checkpoint_file}")
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


def extract_prompt_content_fixed(prompt_field):
    """Properly extract prompt content from various formats"""
    if isinstance(prompt_field, np.ndarray):
        prompt_field = prompt_field.tolist()
    
    if isinstance(prompt_field, list) and len(prompt_field) > 0:
        first_item = prompt_field[0]
        
        if isinstance(first_item, dict) and 'content' in first_item:
            return first_item['content'], prompt_field
        
        if isinstance(first_item, list):
            return extract_prompt_content_fixed(first_item)
        
        return str(first_item), None
    
    if isinstance(prompt_field, dict) and 'content' in prompt_field:
        return prompt_field['content'], [prompt_field]
    
    if isinstance(prompt_field, str):
        return prompt_field, None
    
    return str(prompt_field), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, 
                        default="/scratch/ngangada/thesis/thesis/verl/checkpoints/puzzle_3b_vineppo_v2_stepwise/vineppo_fixed_stepwise_ter/global_step_230/actor")
    parser.add_argument('--data_path', type=str,
                        default="/scratch/ngangada/thesis/thesis/verl/data/puzzle_baron_hints/puzzle_baron_val_gsm8k_like.parquet")
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                        help='Maximum new tokens to generate')
    args = parser.parse_args()
    
    print("=" * 100)
    print("🔬 FULL OUTPUT DIAGNOSTIC (NO TRUNCATION)")
    print("=" * 100)
    
    # Load model and tokenizer
    print("\n1️⃣ Loading model and tokenizer...")
    model, tokenizer = load_fsdp_checkpoint(args.checkpoint_dir)
    print(f"   ✅ Loaded")
    
    # Load data
    print("\n2️⃣ Loading data...")
    df = pd.read_parquet(args.data_path)
    row = df.iloc[args.sample_idx]
    
    prompt_field = row.get("prompt")
    raw_content, chat_messages = extract_prompt_content_fixed(prompt_field)
    
    print(f"   Prompt length: {len(raw_content)} chars")
    
    # Tokenize
    tokens = tokenizer(raw_content, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens["input_ids"].to(model.device)
    attention_mask = tokens["attention_mask"].to(model.device)
    
    print(f"   Input tokens: {input_ids.shape[1]}")
    
    # Generate
    print(f"\n3️⃣ Generating (max {args.max_new_tokens} new tokens, greedy)...")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract generated part only
    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"   Generated {len(generated_ids)} tokens ({len(generated_text)} chars)")
    
    # Show FULL output
    print("\n" + "=" * 100)
    print("FULL GENERATED OUTPUT:")
    print("=" * 100)
    print(generated_text)
    print("=" * 100)
    
    # Ground truth
    gt = row.get("reward_model", {})
    if isinstance(gt, dict):
        gt_text = gt.get("ground_truth", "")
    else:
        gt_text = str(gt) if gt else ""
    
    print("\nGROUND TRUTH:")
    print("=" * 100)
    print(gt_text)
    print("=" * 100)
    
    # Quick analysis
    print("\n📊 ANALYSIS:")
    
    # Count reasoning steps
    steps = re.findall(r'Step #\d+', generated_text)
    print(f"   Reasoning steps: {len(steps)}")
    
    # Check for final answer
    has_final = "Final Answer" in generated_text
    print(f"   Has Final Answer block: {has_final}")
    
    # Parse table
    lines = generated_text.split('\n')
    table_lines = []
    in_table = False
    
    for line in lines:
        # Look for table start
        if '|' in line and line.count('|') >= 2:
            # Skip separator lines
            if not re.match(r'^[\s\-:|]+$', line):
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if len(cells) >= 2:
                    table_lines.append(cells)
    
    print(f"   Table rows found: {len(table_lines)}")
    if table_lines:
        print("\n   Parsed table:")
        for i, row_cells in enumerate(table_lines[:10]):  # Show up to 10 rows
            print(f"      Row {i+1}: {row_cells}")
    
    # Check for gibberish
    cjk_chars = [c for c in generated_text if ord(c) > 0x4E00 and ord(c) < 0x9FFF]
    if cjk_chars:
        print(f"\n   ⚠️  CJK characters found: {len(cjk_chars)}")
        print(f"      Sample: {cjk_chars[:20]}")
    else:
        print(f"\n   ✅ No CJK/gibberish characters")
    
    # Check for excessive repetition
    repeated_phrases = re.findall(r'(In short,[^.]+\.)\s*\1', generated_text)
    if repeated_phrases:
        print(f"   ⚠️  Repetition detected")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
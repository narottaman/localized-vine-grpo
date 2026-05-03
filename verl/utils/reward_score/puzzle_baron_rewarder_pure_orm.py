"""
Pure ORM Reward Function for Puzzle Baron
Only checks final answer table - no process supervision!
"""

import re
from typing import Dict, Any, List, Optional
from collections import Counter


def _normalize_cell(cell: str) -> str:
    """Normalize cell for comparison."""
    if not cell:
        return ""
    normalized = ' '.join(cell.split())
    normalized = normalized.lower()
    normalized = normalized.strip('"\'')
    normalized = normalized.rstrip('.,;:')
    return normalized.strip()


def _parse_raw_ground_truth(ground_truth: Any) -> List[List[str]]:
    """Parse ground truth table."""
    if not ground_truth:
        return []
    
    if isinstance(ground_truth, (list, tuple)):
        rows = []
        for r in ground_truth:
            if isinstance(r, str) and '|' in r:
                cells = [c.strip() for c in r.split('|') if c.strip()]
                if cells:
                    rows.append(cells)
            elif isinstance(r, (list, tuple)):
                cells = [str(c).strip() for c in r if str(c).strip()]
                if cells:
                    rows.append(cells)
        return rows
    
    if isinstance(ground_truth, str):
        rows = []
        for line in ground_truth.strip().splitlines():
            line = line.strip()
            if not line or re.match(r'^\|?[\s\-:]+\|', line):
                continue
            if '|' in line:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    rows.append(cells)
        return rows
    
    return []


def _extract_table_from_final_block(solution_str: str,
                                     extra_info: Optional[Dict] = None,
                                     ground_truth: Optional[str] = None) -> List[List[str]]:
    """Extract table from ### Final Answer Block."""
    m = re.search(r'###\s*Final Answer Block\s*\n(.*)', solution_str, re.DOTALL | re.IGNORECASE)
    if not m:
        return []

    block = m.group(1)
    
    expected_rows = None
    expected_cols = None

    if ground_truth:
        gt_rows = _parse_raw_ground_truth(ground_truth)
        if len(gt_rows) > 0:
            expected_rows = len(gt_rows)
            expected_cols = max(len(r) for r in gt_rows)

    if expected_rows is None and extra_info and isinstance(extra_info, dict):
        vars_dict = extra_info.get("variables") or extra_info.get("variable") or {}
        if isinstance(vars_dict, dict) and len(vars_dict) > 0:
            non_null = [v for v in vars_dict.values() if v is not None]
            expected_cols = len(non_null)
            for v in vars_dict.values():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    expected_rows = len(v)
                    break

    raw_lines = [ln.rstrip() for ln in block.splitlines()]
    parsed_rows = []
    started = False
    
    for ln in raw_lines:
        if '|' not in ln:
            if started:
                break
            else:
                continue
        if re.match(r'^\s*\|?\s*-{2,}\s*(\|\s*-{2,}\s*)+\|?\s*$', ln):
            continue
        
        parts = [p.strip() for p in ln.split('|')]
        parts = [p for p in parts if p != ""]
        
        if len(parts) < 1:
            continue
        
        if expected_cols is not None and len(parts) > expected_cols:
            parts = parts[:expected_cols]
        
        parsed_rows.append(parts)
        started = True

    if expected_rows is not None and len(parsed_rows) > expected_rows:
        parsed_rows = parsed_rows[:expected_rows]

    clean_rows = []
    for row in parsed_rows:
        clean_row = [_normalize_cell(c) for c in row]
        clean_rows.append(clean_row)

    return clean_rows


def compute_puzzle_baron_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict
) -> Dict[str, Any]:
    """
    PURE ORM: Only final table correctness.
    
    Score = cell_accuracy (that's it!)
    
    No structure bonuses, no format penalties, no reasoning checks.
    """
    
    out = {
        "score": 0.0,
        "cell_accuracy": 0.0,
        "correct_cells": 0,
        "total_cells": 0,
        "perfect_rows": 0,
        "num_pred_rows": 0,
        "num_gt_rows": 0,
        "reason": "ok"
    }

    try:
        pred_rows = _extract_table_from_final_block(
            solution_str,
            extra_info=extra_info,
            ground_truth=ground_truth
        )
        out["num_pred_rows"] = len(pred_rows)

        gt_table = _parse_raw_ground_truth(ground_truth)
        out["num_gt_rows"] = len(gt_table)
        
        if len(gt_table) == 0:
            out["reason"] = "no_ground_truth"
            return out
        
        gt_table = [[_normalize_cell(c) for c in row] for row in gt_table]
        
        exp_rows = len(gt_table)
        exp_cols = max(len(r) for r in gt_table) if gt_table else 0
        
        pred_table = pred_rows
        if exp_cols > 0:
            pred_table = [
                (row[:exp_cols] + ([""] * max(0, exp_cols - len(row))))
                for row in pred_table
            ]
        
        if len(pred_table) < exp_rows:
            for _ in range(exp_rows - len(pred_table)):
                pred_table.append([""] * exp_cols)
        
        if len(pred_table) > exp_rows:
            pred_table = pred_table[:exp_rows]
        
        total_cells = exp_rows * exp_cols
        out["total_cells"] = total_cells
        
        # FIXED: Cell-by-cell matching (not column-wise!)
        correct_cells = 0
        
        for row_idx in range(exp_rows):
            pred_row = pred_table[row_idx] if row_idx < len(pred_table) else []
            gt_row = gt_table[row_idx] if row_idx < len(gt_table) else []
            
            for col_idx in range(exp_cols):
                pred_cell = pred_row[col_idx] if col_idx < len(pred_row) else ""
                gt_cell = gt_row[col_idx] if col_idx < len(gt_row) else ""
                
                # Normalize and compare
                pred_cell = _normalize_cell(pred_cell)
                gt_cell = _normalize_cell(gt_cell)
                
                if pred_cell == gt_cell and pred_cell != "":
                    correct_cells += 1
        
        out["correct_cells"] = correct_cells
        
        if total_cells > 0:
            out["cell_accuracy"] = correct_cells / total_cells
        
        # Count perfect rows (diagnostic only)
        perfect_rows = 0
        for i, pred_row in enumerate(pred_table):
            if i < len(gt_table):
                gt_row = gt_table[i]
                pred_padded = list(pred_row) + [""] * max(0, len(gt_row) - len(pred_row))
                gt_padded = list(gt_row) + [""] * max(0, len(pred_row) - len(gt_row))
                if pred_padded == gt_padded:
                    perfect_rows += 1
        out["perfect_rows"] = perfect_rows
        
        # PURE ORM: Score = Cell Accuracy
        out["score"] = out["cell_accuracy"]
        
    except Exception as e:
        out["reason"] = f"error: {str(e)[:100]}"
        out["score"] = 0.0
    
    return out
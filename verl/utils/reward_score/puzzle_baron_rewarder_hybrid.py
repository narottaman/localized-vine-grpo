import re
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter


def _parse_raw_ground_truth(ground_truth: Any) -> List[List[str]]:
    """
    Parse ground_truth which is raw pipe-separated table text (NO markdown headers).
    
    Example input:
        "250 | Irycia | football\n500 | Garroda | rustic village\n..."
    
    Returns:
        List of rows, each row is a list of cell strings.
    """
    if not ground_truth:
        return []
    
    # Handle list/tuple input (list of row strings or list of cell lists)
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
    
    # Handle string input (newline-separated rows)
    if isinstance(ground_truth, str):
        rows = []
        for line in ground_truth.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip markdown separator lines like |---|---|
            if re.match(r'^\|?[\s\-:]+\|', line):
                continue
            if '|' in line:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    rows.append(cells)
        return rows
    
    return []


def _extract_table_rows_from_final_block(solution_str: str,
                                         extra_info: Optional[Dict] = None,
                                         ground_truth: Optional[str] = None) -> List[List[str]]:
    """
    Return a list of rows, where each row is a list of cell strings.
    - Uses ground_truth (if present) or extra_info['variables'] to infer expected shape:
        expected_rows = number of values per variable (N)
        expected_cols = number of non-null variables (M)
    - Normalizes spacing, removes markdown separators, truncates columns > expected_cols
      and truncates number of rows > expected_rows.
    - If expected shape unknown, returns all parsed rows (as lists), normalized.
    """
    # Find final block
    m = re.search(r'###\s*Final Answer Block\s*\n(.*)', solution_str, re.DOTALL | re.IGNORECASE)
    if not m:
        return []

    block = m.group(1)

    # Infer expected shape
    expected_rows = None
    expected_cols = None

    # 1) Try ground_truth first (most reliable) - use the new parser
    if ground_truth:
        gt_rows = _parse_raw_ground_truth(ground_truth)
        if len(gt_rows) > 0:
            expected_rows = len(gt_rows)
            expected_cols = max(len(r) for r in gt_rows)

    # 2) If not available, try extra_info['variables']
    if expected_rows is None and extra_info and isinstance(extra_info, dict):
        vars_dict = extra_info.get("variables") or extra_info.get("variable") or {}
        if isinstance(vars_dict, dict) and len(vars_dict) > 0:
            # non-null keys (values lists)
            non_null = [v for v in vars_dict.values() if v is not None and (not isinstance(v, (list, tuple)) or len(v) > 0 or isinstance(v, str) and v.strip() != "")]
            expected_cols = len(non_null)
            # number of values per variable - try to infer from first non-null value that is a list
            for v in vars_dict.values():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    expected_rows = len(v)
                    break

    # fallback defaults
    if expected_rows is None:
        expected_rows = None  # leave None to not truncate rows if unknown
    if expected_cols is None:
        expected_cols = None  # leave None to not truncate cols if unknown

    # Parse candidate lines in block
    raw_lines = [ln.rstrip() for ln in block.splitlines()]

    parsed_rows = []
    started = False
    for ln in raw_lines:
        if '|' not in ln:
            # if we already started collecting table and see non-table line -> stop collecting
            if started:
                break
            else:
                continue
        # skip markdown separators
        if re.match(r'^\s*\|?\s*-{2,}\s*(\|\s*-{2,}\s*)+\|?\s*$', ln):
            continue
        # normalize parts
        parts = [p.strip() for p in ln.split('|')]
        # remove empty segments from start/end but keep empty cells within row
        # turn ['','250','','Irycia','football',''] -> ['250','Irycia','football']
        parts = [p for p in parts if p != ""]
        if len(parts) < 1:
            continue
        # Truncate columns if known
        if expected_cols is not None and len(parts) > expected_cols:
            parts = parts[:expected_cols]
        parsed_rows.append(parts)
        started = True

    # Truncate rows if expected shape known
    if expected_rows is not None and len(parsed_rows) > expected_rows:
        parsed_rows = parsed_rows[:expected_rows]

    # Normalize cells
    clean_rows = []
    for row in parsed_rows:
        clean_row = [_normalize_cell(c) for c in row]
        clean_rows.append(clean_row)

    return clean_rows


def _normalize_cell(cell: str) -> str:
    """Normalize cell for comparison - lowercase, strip, remove quotes/punctuation."""
    if not cell:
        return ""
    normalized = ' '.join(cell.split())  # normalize whitespace
    normalized = normalized.lower()
    normalized = normalized.strip('"\'')
    normalized = normalized.rstrip('.,;:')
    return normalized.strip()


def _parse_row_cells(row: str) -> List[str]:
    return [c.strip() for c in row.split("|")]


def _compute_reasoning_correctness(solution_str: str, extra_info: dict) -> float:
    """
    Compare reasoning chain in solution with ground truth reasoning.
    Returns score 0.0-1.0 based on how many reasoning steps match.
    """
    if not extra_info or 'answer' not in extra_info:
        return 0.0
    
    gt_reasoning = extra_info.get('answer', '')
    if not gt_reasoning or not isinstance(gt_reasoning, str):
        return 0.0
    
    # Extract reasoning chain from solution
    solution_reasoning_match = re.search(
        r'###\s*Step-by-Step Reasoning Chain\s*\n(.*?)(?=###|$)',
        solution_str,
        re.DOTALL | re.IGNORECASE
    )
    
    if not solution_reasoning_match:
        return 0.0
    
    solution_reasoning = solution_reasoning_match.group(1).strip()
    
    # Extract individual reasoning steps
    def extract_reasoning_facts(text: str) -> set:
        """Extract 'X equals Y' or 'X cannot equal Y' facts from reasoning."""
        facts = set()
        
        # Pattern: "X equals Y" or "X cannot equal Y"
        # Examples: "250 equals football", "rustic village cannot equal Irycia"
        patterns = [
            r'(\w+(?:\s+\w+)?)\s+equals\s+(\w+(?:\s+\w+)?)',
            r'(\w+(?:\s+\w+)?)\s+cannot\s+equal\s+(\w+(?:\s+\w+)?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Normalize to lowercase and sort alphabetically
                fact = tuple(sorted([match[0].lower().strip(), match[1].lower().strip()]))
                relation = 'equals' if 'equals' in pattern else 'cannot_equal'
                facts.add((relation, fact))
        
        return facts
    
    # Extract facts from both
    gt_facts = extract_reasoning_facts(gt_reasoning)
    solution_facts = extract_reasoning_facts(solution_reasoning)
    
    if len(gt_facts) == 0:
        return 0.0
    
    # Compute overlap
    matching_facts = gt_facts & solution_facts
    reasoning_score = len(matching_facts) / len(gt_facts)
    
    return float(reasoning_score)


def compute_puzzle_baron_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict) -> Dict[str, Any]:
    """
    Smooth, robust scoring for PuzzleBaron-style table tasks.

    Returns a dict:
      {
        "score": float,
        "cell_accuracy": float,
        "correct_cells": int,
        "total_cells": int,
        "perfect_rows": int,
        "num_pred_rows": int,
        "num_gt_rows": int,
        "num_reasoning_sections": int,
        "num_table_sections": int,
        "structure_bonus": float,
        "clean_ending_bonus": float,
        "incomplete_table_penalty": float,
        "missing_values_penalty": float,
        "reason": str
      }
    """

    
    # defaults
    out = {
        "score": 0.0,
        "cell_accuracy": 0.0,
        "correct_cells": 0,
        "total_cells": 0,
        "perfect_rows": 0,
        "num_pred_rows": 0,
        "num_gt_rows": 0,
        "num_reasoning_sections": 0,
        "num_table_sections": 0,
        "structure_bonus": 0.0,
        "clean_ending_bonus": 0.0,
        "incomplete_table_penalty": 0.0,
        "missing_values_penalty": 0.0,
        "multiple_sections_penalty": 0.0,
        "post_table_penalty": 0.0,
        # new defensive diagnostics (always present)
        "duplicate_count_in_columns": 0,
        "duplicate_penalty_count": 0,
        "reasoning_correctness": 0.0,
        "reason": "",
    }


    try:
        # Count sections
        out["num_reasoning_sections"] = solution_str.count("### Step-by-Step Reasoning Chain")
        out["num_table_sections"] = solution_str.count("### Final Answer Block")

        # Structure bonus if exactly one reasoning and one final table section
        if out["num_reasoning_sections"] == 1 and out["num_table_sections"] == 1:
            out["structure_bonus"] = 0.10
        elif out["num_table_sections"] == 1 and out["num_reasoning_sections"] == 0:
            # small structure bonus for only table but no reasoning
            out["structure_bonus"] = 0.05
        else:
            out["multiple_sections_penalty"] = -0.05 * max(0, out["num_table_sections"] - 1)

        # Extract predicted rows from the response
        pred_rows = _extract_table_rows_from_final_block(solution_str, extra_info=extra_info, ground_truth=ground_truth)
        out["num_pred_rows"] = len(pred_rows)

        # Compute post-table content penalty (content after final block)
        final_match = re.search(r'###\s*Final Answer Block\s*\n(.*)', solution_str, re.DOTALL | re.IGNORECASE)
        if final_match:
            # everything after the final block lines â€” crude but effective
            body = final_match.group(1)
            # if there's significant natural language after the last table line -> penalty
            if re.search(r'\n(?!\s*$)[^\n]*[a-zA-Z].*', body) and len(pred_rows) > 0:
                # detect content lines that are not table rows (no '|')
                leftover_lines = [ln for ln in body.splitlines() if '|' not in ln and ln.strip()]
                if len(leftover_lines) > 0:
                    out["post_table_penalty"] = -0.03 * min(3, len(leftover_lines))

        # ============================================================
        # FIX: Parse ground_truth using the dedicated parser (not _extract_table_rows_from_final_block)
        # ============================================================
        gt_table = _parse_raw_ground_truth(ground_truth)
        out["num_gt_rows"] = len(gt_table)
        
        # Normalize gt_table cells
        gt_table = [[_normalize_cell(c) for c in row] for row in gt_table]

        # Convert pred_rows to normalized format (already normalized in _extract_table_rows_from_final_block)
        pred_table = pred_rows
        
        # Determine expected shape from ground truth
        exp_rows = len(gt_table) if len(gt_table) > 0 else None
        exp_cols = max((len(r) for r in gt_table), default=None) if len(gt_table) > 0 else None
        
        # Fallback to extra_info if no ground truth
        if exp_rows is None and extra_info:
            exp_rows = extra_info.get("expected_rows")
        if exp_cols is None and extra_info:
            vars_dict = extra_info.get("variables") or {}
            if isinstance(vars_dict, dict) and len(vars_dict) > 0:
                non_null = [v for v in vars_dict.values() if v is not None and (isinstance(v, (list, tuple)) and len(v) > 0 or isinstance(v, str) and v.strip())]
                exp_cols = len(non_null) if non_null else None
        
        # If still None, infer from pred_table if available
        if exp_cols is None and len(pred_table) > 0:
            exp_cols = max(len(r) for r in pred_table)
        if exp_rows is None and len(pred_table) > 0:
            exp_rows = len(pred_table)
        
        # Safe guard: ensure exp_rows/exp_cols are integers
        exp_rows = int(exp_rows) if exp_rows is not None else 0
        exp_cols = int(exp_cols) if exp_cols is not None else 0

        
        # Truncate/pad pred_table to expected shape
        if exp_cols > 0:
            pred_table = [(row[:exp_cols] + ([""] * max(0, exp_cols - len(row)))) for row in pred_table]
        if exp_rows > 0 and len(pred_table) > exp_rows:
            pred_table = pred_table[:exp_rows]
        # Pad with empty rows if needed
        if exp_rows > 0 and len(pred_table) < exp_rows:
            for _ in range(exp_rows - len(pred_table)):
                pred_table.append([""] * (exp_cols or 1))
        
        # Compute correct_cells by column-wise multiset intersection
        correct_cells = 0
        if exp_rows > 0 and exp_cols > 0:
            total_cells = exp_rows * exp_cols
        else:
            if pred_table and len(pred_table) > 0:
                total_cells = len(pred_table) * len(pred_table[0])
                exp_rows = exp_rows or len(pred_table)
                exp_cols = exp_cols or len(pred_table[0])
            else:
                total_cells = 0
                exp_rows = exp_rows or 0
                exp_cols = exp_cols or 0
        
        out["total_cells"] = total_cells
        
        # FIXED: Cell-by-cell comparison (not column-wise multiset!)
        correct_cells = 0
        dup_count_all = 0
        dup_penalty_count = 0
        
        # Cell-by-cell matching
        for row_idx in range(exp_rows):
            pred_row = pred_table[row_idx] if row_idx < len(pred_table) else []
            gt_row = gt_table[row_idx] if row_idx < len(gt_table) else []
            
            for col_idx in range(exp_cols):
                pred_cell = pred_row[col_idx] if col_idx < len(pred_row) else ""
                gt_cell = gt_row[col_idx] if col_idx < len(gt_row) else ""
                
                # Normalize
                pred_cell = _normalize_cell(pred_cell)
                gt_cell = _normalize_cell(gt_cell)
                
                # Match check
                if pred_cell == gt_cell and pred_cell != "":
                    correct_cells += 1
        
        # Check for duplicate values in each column (still useful for penalty)
        for col_idx in range(exp_cols):
            pred_col = [
                _normalize_cell(pred_table[r][col_idx]) if col_idx < len(pred_table[r]) else ""
                for r in range(len(pred_table))
            ]
            pred_counts = Counter([v for v in pred_col if v])
            
            for val, cnt in pred_counts.items():
                if cnt > 1:
                    dup_count_all += cnt - 1
                    dup_penalty_count += cnt - 1
        
        out["correct_cells"] = correct_cells
        out["duplicate_count_in_columns"] = dup_count_all
        out["duplicate_penalty_count"] = dup_penalty_count
        
        # Cell accuracy
        if total_cells > 0:
            out["cell_accuracy"] = correct_cells / total_cells
        
        # Perfect rows: count rows where all cells match exactly
        perfect_rows = 0
        for i, pred_row in enumerate(pred_table):
            if i < len(gt_table):
                gt_row = gt_table[i]
                # Pad to same length for comparison
                pred_padded = list(pred_row) + [""] * max(0, len(gt_row) - len(pred_row))
                gt_padded = list(gt_row) + [""] * max(0, len(pred_row) - len(gt_row))
                if pred_padded == gt_padded:
                    perfect_rows += 1
        out["perfect_rows"] = perfect_rows
        
        # Incomplete table penalty
        if out["num_pred_rows"] < out["num_gt_rows"]:
            missing = out["num_gt_rows"] - out["num_pred_rows"]
            out["incomplete_table_penalty"] = -0.15 * missing
        
        # Clean ending bonus (no extra content after table)
        if out["post_table_penalty"] == 0.0 and out["num_pred_rows"] >= out["num_gt_rows"]:
            out["clean_ending_bonus"] = 0.05
        
        # Duplicate penalty
        dup_penalty = -0.02 * dup_penalty_count
        
        # Final score computation
        base_score = 0.50 * out["cell_accuracy"] + 0.30 * (perfect_rows / max(1, exp_rows))
        bonuses = out["structure_bonus"] + out["clean_ending_bonus"]
        penalties = out["post_table_penalty"] + out["incomplete_table_penalty"] + out["multiple_sections_penalty"] + dup_penalty
        
        out["score"] = max(0.0, min(1.0, base_score + bonuses + penalties))
        out["reason"] = "ok"
        
    except Exception as e:
        out["reason"] = f"error: {str(e)[:100]}"
        out["score"] = 0.0
        
    out["reasoning_correctness"] = _compute_reasoning_correctness(solution_str, extra_info)

    # If reasoning is good but table is wrong, give partial credit
    if out["reasoning_correctness"] > 0.5 and out["cell_accuracy"] < 0.5:
        out["score"] = max(out["score"], 0.3 * out["reasoning_correctness"])
        
        
    # ── GDPO: expose individual reward components ──────────────
    # These are used by reinforce_plus_plus_gdpo for decoupled normalization.
    # Conditioned structure reward: only give structure credit when puzzle
    # is at least partially solved, preventing format-gaming without solving.
    out["reward_correctness"] = float(out["cell_accuracy"])
    out["reward_structure"]   = float(out["structure_bonus"] + out["clean_ending_bonus"]) \
                                if out["cell_accuracy"] >= 0.1 else 0.0
    out["reward_penalty"]     = float(
        out["post_table_penalty"]
        + out["incomplete_table_penalty"]
        + out["multiple_sections_penalty"]
        - 0.02 * out["duplicate_penalty_count"]
    )
    
    return out
"""
gsm8k_rewarder.py — GSM8K reward function for VineGRPO intermediate state scoring.

Two entry points:
1. compute_puzzle_baron_score(data_source, solution_str, ground_truth, extra_info)
   → VERL entry point, same signature as puzzle rewarder
   → Returns full reward dict with CANONICAL_KEYS

2. compute_mc_partial_score_gsm8k(solution_str, ground_truth, extra_info)
   → Used by ray_trainer.py compute_mc_partial_score for intermediate states
   → Returns float score for partial GSM8K trajectories

Copy to:
/scratch/ngangada/thesis/thesis/verl/verl/utils/reward_score/gsm8k_rewarder.py
"""

import re
from typing import Any, Dict, Optional

# ── Canonical keys (must match multisource_rewarder.py) ──────
CANONICAL_KEYS = {
    "score":            0.0,
    "cell_accuracy":    0.0,
    "correct_cells":    0,
    "total_cells":      0,
    "perfect_rows":     0,
    "num_pred_rows":    0,
    "num_gt_rows":      0,
    "reason":           "ok",
    "predicted":        "",
    "hacking_detected": False,
    "hacking_reason":   "ok",
}


def _fill_canonical(result: dict) -> dict:
    for key, default in CANONICAL_KEYS.items():
        result.setdefault(key, default)
    return result


# ── Final answer extractor ────────────────────────────────────

def _extract_final_answer(solution_str: str) -> Optional[str]:
    """
    Extract number from either:
      ### Final Answer\n48           (our SFT format)
      #### 48                        (standard GSM8K format)
    """
    if not solution_str:
        return None

    # Our trained format: ### Final Answer\n<number>
    m = re.search(
        r'###\s*Final Answer\s*\n\s*(-?[0-9][0-9,. ]*)',
        solution_str, re.IGNORECASE
    )
    if m:
        raw = m.group(1).strip().replace(",", "").replace(" ", "")
        return raw[:20]

    # Standard GSM8K format: #### <number>
    m2 = re.search(r'####\s*(-?[0-9][0-9,. ]*)', solution_str)
    if m2:
        raw = m2.group(1).strip().replace(",", "").replace(" ", "")
        return raw[:20]

    # Fallback: last number in response
    numbers = re.findall(r'-?[0-9]+(?:\.[0-9]+)?', solution_str)
    if numbers:
        return numbers[-1][:20]

    return None


# ── Full reward function ──────────────────────────────────────

def compute_puzzle_baron_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
) -> Dict[str, Any]:
    """
    VERL entry point for GSM8K-only RL training.
    Same signature as puzzle_baron_rewarder_pure_orm.compute_puzzle_baron_score.
    """
    out = dict(CANONICAL_KEYS)
    out["reason"] = "no_match"

    predicted = _extract_final_answer(solution_str)
    if predicted is None:
        out["reason"] = "no_final_answer_found"
        return out

    out["predicted"] = predicted

    # Check for repetition hacking (model outputs 480000000...)
    if len(predicted) >= 15:
        out["reason"]           = "absurd_value"
        out["score"]            = -0.5  # negative: actively discourage repetition
        out["hacking_detected"] = True
        out["hacking_reason"]   = "repeated_digits"
        return out

    try:
        pred_float = float(predicted)
        gt_float   = float(str(ground_truth).strip().replace(",", ""))

        if not (-1e12 < pred_float < 1e12):
            out["reason"]           = "absurd_value"
            out["score"]            = -0.5
            out["hacking_detected"] = True
            out["hacking_reason"]   = "out_of_range"
            return out

        if abs(pred_float - gt_float) < 1e-6:
            out["score"]         = 1.0
            out["cell_accuracy"] = 1.0
            out["reason"]        = "correct"
        else:
            out["reason"] = f"wrong: got {pred_float}, expected {gt_float}"

    except (ValueError, TypeError, OverflowError) as e:
        out["reason"] = f"parse_error: {e}"

    return _fill_canonical(out)


# ── MC partial scorer for VineGRPO intermediate states ───────

def compute_mc_partial_score_gsm8k(
    solution_str: str,
    ground_truth: str = "",
    extra_info: Optional[Dict] = None,
) -> float:
    """
    Score a PARTIAL GSM8K trajectory for VineGRPO MC value estimation.

    Unlike the full reward, this rewards:
    1. Correct reasoning structure (Step #N. format)
    2. Presence of arithmetic calculations
    3. Proximity to correct final answer (if present)

    Penalizes:
    - Repetition hacking (repeated digits)
    - Contradictory reasoning
    """
    if not solution_str or not solution_str.strip():
        return 0.0

    score = 0.0

    # ── Penalize repetition hacking immediately ───────────────
    # If response contains long runs of the same digit → severe penalty
    if re.search(r'(\d)\1{10,}', solution_str):
        return -1.0

    # ── 1. Reasoning structure (0.1 pts) ─────────────────────
    has_reasoning = bool(re.search(
        r'###\s*Step-by-Step Reasoning Chain',
        solution_str, re.IGNORECASE
    ))
    if has_reasoning:
        score += 0.1

    # ── 2. Valid numbered steps (up to 0.3 pts) ──────────────
    steps = re.findall(r'Step\s*#?\d+\.', solution_str, re.IGNORECASE)
    if steps:
        score += min(0.3, 0.06 * min(len(steps), 5))

    # ── 3. Arithmetic calculations present (0.1 pts) ─────────
    # Look for patterns like "3 * 4 = 12" or "36 / 3 = 12"
    calc_pattern = r'\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+'
    calcs = re.findall(calc_pattern, solution_str)
    if calcs:
        score += 0.1

    # ── 4. Final answer present and valid (up to 0.5 pts) ────
    predicted = _extract_final_answer(solution_str)
    if predicted is not None:
        # Check it's not absurd
        if len(predicted) < 15:
            try:
                pred_float = float(predicted)

                if -1e12 < pred_float < 1e12:
                    score += 0.2  # valid non-absurd answer present

                    # Check if matches ground truth
                    if ground_truth:
                        try:
                            gt_float = float(str(ground_truth).strip().replace(",", ""))
                            if abs(pred_float - gt_float) < 1e-6:
                                score += 0.3  # correct answer bonus
                        except (ValueError, TypeError):
                            pass
                else:
                    score -= 0.3  # absurd answer penalty
            except (ValueError, TypeError):
                pass
        else:
            score -= 0.5  # repeated digits penalty

    return max(-1.0, min(1.0, score))


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":

    # Test 1: correct answer
    resp_correct = """### Step-by-Step Reasoning Chain
Step #1. Jake eats 3 per week, brother 5, father 4. Total = 12 per week.
Step #2. For 4 weeks: 12 * 4 = 48.

### Final Answer
48"""
    r = compute_puzzle_baron_score("gsm8k", resp_correct, "48", {})
    assert r["score"] == 1.0, f"Expected 1.0 got {r['score']}"
    print(f"Correct answer: score={r['score']} reason={r['reason']}")

    # Test 2: wrong answer
    r2 = compute_puzzle_baron_score("gsm8k", resp_correct, "50", {})
    assert r2["score"] == 0.0
    print(f"Wrong answer:   score={r2['score']} reason={r2['reason']}")

    # Test 3: repetition hacking
    hack = "### Final Answer\n" + "4" + "8" * 50
    r3 = compute_puzzle_baron_score("gsm8k", hack, "48", {})
    assert r3["score"] == -0.5
    assert r3["hacking_detected"] == True
    print(f"Hacking:        score={r3['score']} reason={r3['reason']}")

    # Test 4: partial MC scorer — correct intermediate
    partial = """### Step-by-Step Reasoning Chain
Step #1. Total per week = 3 + 5 + 4 = 12.
Step #2. For 4 weeks: 12 * 4 = 48."""
    s = compute_mc_partial_score_gsm8k(partial, "48")
    print(f"MC partial (good): score={s:.3f}")
    assert s > 0.3

    # Test 5: partial MC scorer — repetition hacking
    hack_partial = "Step #1. " + "9" * 30
    s2 = compute_mc_partial_score_gsm8k(hack_partial, "48")
    assert s2 == -1.0
    print(f"MC partial (hack): score={s2}")

    print("\nAll tests passed!")
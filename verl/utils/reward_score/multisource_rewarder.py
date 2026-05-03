"""
Multi-source reward function for combined Puzzle Baron + ZebraLogic + GSM8K RL training.

Routes to the correct scorer based on data_source:
  - "puzzle_baron" or "zebralogic" → puzzle_baron_rewarder_pure_orm
  - "gsm8k"                        → gsm8k number extraction scorer

CRITICAL: Every scorer must return the EXACT SAME set of keys.
VERL's validation loop asserts len(lst) == len(sample_scores) for every key
that appears in ANY reward dict across the entire batch. If puzzle scorer
returns keys A,B,C and gsm8k scorer returns A,B,C,D then D causes an
AssertionError. All keys must be present in every return dict.
"""

import re
from typing import Any, Dict, Optional


# ── Import puzzle scorer ──────────────────────────────────────
from verl.utils.reward_score.puzzle_baron_rewarder_pure_orm import (
    compute_puzzle_baron_score as _puzzle_score
)


# ── Canonical key set — every scorer must return ALL of these ─
# Add new keys here whenever either scorer adds a new output field.
CANONICAL_KEYS = {
    # Core metric (both scorers)
    "score":            0.0,
    "cell_accuracy":    0.0,
    # Puzzle-specific (zero/empty for GSM8K)
    "correct_cells":    0,
    "total_cells":      0,
    "perfect_rows":     0,
    "num_pred_rows":    0,
    "num_gt_rows":      0,
    "reason":           "ok",
    # GSM8K-specific (empty/neutral for puzzles)
    "predicted":        "",
    "hacking_detected": False,
    "hacking_reason":   "ok",
}


def _fill_canonical(result: dict) -> dict:
    """
    Ensure result has every key in CANONICAL_KEYS.
    Missing keys are filled with their default value.
    This guarantees VERL's length assertion never fires.
    """
    for key, default in CANONICAL_KEYS.items():
        result.setdefault(key, default)
    return result


# ── GSM8K scorer ─────────────────────────────────────────────

def _extract_final_answer(solution_str: str) -> Optional[str]:
    """Extract the number from ### Final Answer block."""
    if not solution_str:
        return None

    # Primary: look for ### Final Answer block
    match = re.search(
        r'###\s*Final Answer\s*\n\s*(-?[0-9][0-9,. ]*)',
        solution_str, re.IGNORECASE
    )
    if match:
        raw = match.group(1).strip().replace(",", "").replace(" ", "")
        return raw[:20]  # truncate to prevent absurd-length strings

    # Fallback: last number in response
    numbers = re.findall(r'-?[0-9]+(?:\.[0-9]+)?', solution_str)
    if numbers:
        return numbers[-1][:20]

    return None


def compute_gsm8k_score(data_source, solution_str, ground_truth, extra_info) -> Dict[str, Any]:
    """Score a GSM8K response by extracting and comparing the final number."""
    out = dict(CANONICAL_KEYS)  # start from canonical defaults
    out["hacking_detected"] = False
    out["hacking_reason"]   = "ok"
    out["reason"]           = "no_match"

    predicted = _extract_final_answer(solution_str)
    if predicted is None:
        out["reason"] = "no_final_answer_found"
        return out

    out["predicted"] = predicted

    try:
        pred_float = float(predicted)
        gt_float   = float(str(ground_truth).strip().replace(",", ""))

        # Guard against repetition-hacking (model outputs 480000000000...)
        if not (-1e15 < pred_float < 1e15):
            out["reason"] = "absurd_value"
            out["score"] = -0.5  # negative reward to actively discourage repetition
            out["cell_accuracy"] = 0.0
            return out

        if abs(pred_float - gt_float) < 1e-6:
            out["score"]         = 1.0
            out["cell_accuracy"] = 1.0
            out["reason"]        = "correct"
        else:
            out["reason"] = f"wrong: got {pred_float}, expected {gt_float}"

    except (ValueError, TypeError, OverflowError) as e:
        out["reason"] = f"parse_error: {e}"

    return out


# ── Multi-source router ───────────────────────────────────────

PUZZLE_SOURCES = {"puzzle_baron", "zebralogic", "puzzle baron"}


def compute_multisource_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
) -> Dict[str, Any]:
    """
    Route to the correct reward function based on data_source,
    then fill any missing canonical keys so VERL's length assertion
    always passes across mixed batches.
    """
    src = str(data_source).lower().strip()

    if src in PUZZLE_SOURCES:
        result = _puzzle_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

    elif src == "gsm8k":
        result = compute_gsm8k_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

    else:
        print(f"WARNING: Unknown data_source: '{data_source}' — returning score=0.0")
        result = {
            "score":         0.0,
            "cell_accuracy": 0.0,
            "reason":        f"unknown_data_source:{data_source}",
        }

    # Fill every missing canonical key — guarantees consistent key set
    return _fill_canonical(result)


# ── VERL entry point ──────────────────────────────────────────
# VERL calls: compute_puzzle_baron_score(data_source, solution_str, ground_truth, extra_info)
# Name kept as-is so existing SLURM scripts need no changes.

def compute_puzzle_baron_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
) -> Dict[str, Any]:
    """VERL entry point — routes to correct scorer by data_source."""
    return compute_multisource_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. GSM8K correct answer
    gsm8k_response = """### Step-by-Step Reasoning Chain
Step #1. Fabric section: 36 * 1/3 = 12.
Step #2. Jewelry section: 36 * 1/4 = 9.
Step #3. Stationery: 36 - 12 - 9 = 15.

### Final Answer
15"""
    r = compute_puzzle_baron_score("gsm8k", gsm8k_response, "15", {})
    assert r["score"] == 1.0, f"Expected 1.0, got {r['score']}"
    assert set(r.keys()) == set(CANONICAL_KEYS.keys()), \
        f"Key mismatch: {set(r.keys()) ^ set(CANONICAL_KEYS.keys())}"
    print(f"GSM8K correct:  score={r['score']}, reason={r['reason']}")

    # 2. GSM8K wrong answer
    r2 = compute_puzzle_baron_score("gsm8k", gsm8k_response, "20", {})
    assert r2["score"] == 0.0
    print(f"GSM8K wrong:    score={r2['score']}, reason={r2['reason']}")

    # 3. GSM8K absurd hacking response
    hack_response = "### Final Answer\n" + "4" + "8" * 60
    r3 = compute_puzzle_baron_score("gsm8k", hack_response, "48", {})
    assert r3["score"] == 0.0
    print(f"GSM8K hacking:  score={r3['score']}, reason={r3['reason']}")

    # 4. Puzzle Baron routing
    pb_response = """### Step-by-Step Reasoning Chain
Step #1. Test. In short, A equals B

### Final Answer Block
250 | Irycia | football
500 | Garroda | rustic village"""
    r4 = compute_puzzle_baron_score(
        "puzzle_baron", pb_response,
        "250 | Irycia | football\n500 | Garroda | rustic village", {}
    )
    assert set(r4.keys()) == set(CANONICAL_KEYS.keys()), \
        f"Key mismatch: {set(r4.keys()) ^ set(CANONICAL_KEYS.keys())}"
    print(f"Puzzle Baron:   score={r4['score']}, reason={r4.get('reason', 'ok')}")

    # 5. ZebraLogic routing
    r5 = compute_puzzle_baron_score(
        "zebralogic", pb_response,
        "250 | irycia | football\n500 | garroda | rustic village", {}
    )
    assert set(r5.keys()) == set(CANONICAL_KEYS.keys())
    print(f"ZebraLogic:     score={r5['score']}, reason={r5.get('reason', 'ok')}")

    # 6. Key consistency — all results must have identical key sets
    all_results = [r, r2, r3, r4, r5]
    key_sets = [set(res.keys()) for res in all_results]
    assert all(ks == key_sets[0] for ks in key_sets), \
        "Key sets differ across scorers!"
    print(f"Key consistency: all {len(all_results)} results have identical keys")

    print("\nAll tests passed!")
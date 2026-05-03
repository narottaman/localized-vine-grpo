# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward function for Puzzle Baron grid-based logic puzzles.
Designed for use with Vine-PPO, RLv, and Hybrid-GRPO algorithms.
"""

import re
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import numpy as np


class PuzzleBaronRewarder:
    """
    Reward function for grid-based logic puzzles with structured reasoning.
    
    Provides rewards for:
    1. Step-by-step reasoning quality
    2. Final answer correctness
    3. Structural compliance
    
    With strong anti-reward-hacking mechanisms.
    """
    
    def __init__(
        self,
        step_reward_weight: float = 0.3,
        final_reward_weight: float = 0.7,
        novelty_bonus: float = 0.1,
        repetition_penalty: float = -0.5,  # Increased from -0.2
        format_reward: float = 0.1,
        max_reasonable_steps: int = 30,     # New: limit on steps
        severe_repetition_threshold: int = 3,  # New: aggressive penalty threshold
        no_progress_penalty: float = -0.3,  # New: penalty for meaningless steps
    ):
        """
        Args:
            step_reward_weight: Weight for step-by-step reasoning rewards
            final_reward_weight: Weight for final answer rewards
            novelty_bonus: Bonus for discovering new facts
            repetition_penalty: Penalty for repeating facts (per repetition)
            format_reward: Reward for proper formatting
            max_reasonable_steps: Maximum expected number of steps (penalize beyond this)
            severe_repetition_threshold: Number of repetitions before severe penalty
            no_progress_penalty: Penalty for steps that don't add conclusions
        """
        self.step_weight = step_reward_weight
        self.final_weight = final_reward_weight
        self.novelty_bonus = novelty_bonus
        self.repetition_penalty = repetition_penalty
        self.format_reward = format_reward
        self.max_reasonable_steps = max_reasonable_steps
        self.severe_repetition_threshold = severe_repetition_threshold
        self.no_progress_penalty = no_progress_penalty
        
    def compute_score(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: Dict,
    ) -> float:
        """
        Main scoring function compatible with VERL framework.
        
        Args:
            data_source: Source identifier (e.g., "puzzle_baron")
            solution_str: Model's generated response
            ground_truth: Correct answer string
            extra_info: Dictionary containing 'answer', 'variables', etc.
            
        Returns:
            Float score between 0.0 and 1.0
        """
        # Parse the solution into reasoning and answer sections
        reasoning_section, answer_section = self._parse_solution(solution_str)
        
        # Extract reference reasoning steps from extra_info
        reference_steps = self._extract_reference_steps(extra_info.get("answer", ""))
        
        # Extract variables for validation
        variables = extra_info.get("variables", {})
        
        # Score reasoning steps with anti-hacking measures
        step_score, step_metrics = self._score_reasoning_steps(
            reasoning_section, reference_steps, variables
        )
        
        # Score final answer
        final_score, final_metrics = self._score_final_answer(
            answer_section, ground_truth, variables
        )
        
        # Apply severe penalties for reward hacking patterns
        hacking_penalty = self._detect_reward_hacking(step_metrics, final_metrics)
        
        # Compute total score
        total_score = (
            self.step_weight * step_score +
            self.final_weight * final_score +
            hacking_penalty
        )
        
        # Clip to valid range
        total_score = np.clip(total_score, 0.0, 1.0)
        
        return float(total_score)
    
    def _parse_solution(self, solution_str: str) -> Tuple[str, str]:
        """
        Parse solution into reasoning and answer sections.
        
        Returns:
            (reasoning_section, answer_section) tuple
        """
        # Look for reasoning section
        reasoning_match = re.search(
            r'###\s*Step-by-Step Reasoning Chain\s*\n(.*?)(?=###|$)',
            solution_str,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Look for answer section
        answer_match = re.search(
            r'###\s*Final Answer Block\s*\n(.*)',
            solution_str,
            re.IGNORECASE | re.DOTALL
        )
        answer = answer_match.group(1).strip() if answer_match else ""
        
        return reasoning, answer
    
    def _extract_reference_steps(self, answer_text: str) -> List[Dict]:
        """
        Extract reference reasoning steps from the 'answer' field in extra_info.
        
        Each step is parsed to extract:
        - Step number
        - Values mentioned
        - Relationship (equals/cannot equal)
        - Conclusion
        """
        steps = []
        if not answer_text:
            return steps
        
        # Split by "Answer #" markers
        step_blocks = re.split(r'Answer #\d+:', answer_text)
        
        for block in step_blocks:
            if not block.strip():
                continue
            
            # Extract conclusion using "In short," pattern
            conclusion_match = re.search(
                r'In short,\s*([^.]+)\s*(equals|cannot equal)\s*([^.]+)',
                block,
                re.IGNORECASE
            )
            
            if conclusion_match:
                value1 = conclusion_match.group(1).strip()
                relation = conclusion_match.group(2).strip().lower()
                value2 = conclusion_match.group(3).strip()
                
                steps.append({
                    "value1": value1,
                    "value2": value2,
                    "relation": "equal" if "equal" in relation and "cannot" not in relation else "not_equal",
                    "text": block.strip()
                })
        
        return steps
    
    def _score_reasoning_steps(
        self,
        reasoning_text: str,
        reference_steps: List[Dict],
        variables: Dict
    ) -> Tuple[float, Dict]:
        """
        Score the step-by-step reasoning section with strong anti-hacking measures.
        
        Returns:
            (score, metrics) tuple
        """
        if not reasoning_text:
            return 0.0, {"error": "no_reasoning_found", "num_steps": 0}
        
        # Extract model's reasoning steps
        model_steps = self._extract_model_steps(reasoning_text)
        
        if not model_steps:
            return 0.0, {"error": "no_valid_steps", "num_steps": 0}
        
        # Track discovered facts with repetition counting
        fact_counts = defaultdict(int)
        unique_facts = []
        repeated_facts = []
        no_conclusion_steps = 0
        
        for step in model_steps:
            # Check if step has a valid conclusion
            if not step.get("value1") or not step.get("value2"):
                no_conclusion_steps += 1
                continue
                
            fact_signature = self._create_fact_signature(step)
            fact_counts[fact_signature] += 1
            
            if fact_counts[fact_signature] == 1:
                unique_facts.append(step)
            else:
                repeated_facts.append((step, fact_counts[fact_signature]))
        
        # Compare with reference steps
        correct_facts = 0
        incorrect_facts = 0
        novel_facts = 0
        
        # Create reference fact signatures for comparison
        ref_signatures = {
            self._create_fact_signature(ref): ref 
            for ref in reference_steps
        }
        
        for step in unique_facts:
            sig = self._create_fact_signature(step)
            
            # Check if fact matches reference
            if sig in ref_signatures:
                correct_facts += 1
            else:
                # Check if it's a valid novel deduction
                if self._is_valid_fact(step, variables):
                    novel_facts += 1
                else:
                    incorrect_facts += 1
        
        # Calculate score components
        num_unique = len(unique_facts)
        num_repeated = len(repeated_facts)
        num_reference = len(reference_steps)
        total_steps = len(model_steps)
        
        # Base score: correctness
        correctness = correct_facts / max(num_reference, 1) if num_reference > 0 else 0.0
        
        # Novelty bonus for valid new facts (capped)
        novelty_score = min(novel_facts * self.novelty_bonus, 0.3)
        
        # ANTI-HACKING PENALTIES
        
        # 1. Basic repetition penalty (linear)
        repetition_score = num_repeated * self.repetition_penalty
        
        # 2. Severe repetition penalty (for 3+ repetitions of same fact)
        severe_repetitions = sum(1 for _, count in repeated_facts if count >= self.severe_repetition_threshold)
        severe_penalty = severe_repetitions * -0.5
        
        # 3. Excessive steps penalty (beyond reasonable limit)
        if total_steps > self.max_reasonable_steps:
            excess = total_steps - self.max_reasonable_steps
            excess_penalty = -min(excess * 0.02, 0.5)  # Cap at -0.5
        else:
            excess_penalty = 0.0
        
        # 4. No-progress penalty (steps without conclusions like "This is consistent")
        no_progress_score = no_conclusion_steps * self.no_progress_penalty
        
        # 5. Repetition ratio penalty (if >50% are repetitions, heavily penalize)
        repetition_ratio = num_repeated / max(total_steps, 1)
        if repetition_ratio > 0.5:
            ratio_penalty = -0.5 * repetition_ratio
        else:
            ratio_penalty = 0.0
        
        # Format bonus: check if steps are properly numbered
        format_score = self.format_reward if self._has_proper_step_format(reasoning_text) else 0.0
        
        # Combine all components
        total = correctness + novelty_score + repetition_score + severe_penalty + \
                excess_penalty + no_progress_score + ratio_penalty + format_score
        
        # Ensure non-negative (but can be 0)
        total = max(0.0, min(total, 1.0))
        
        metrics = {
            "num_steps": total_steps,
            "unique_steps": num_unique,
            "repeated_steps": num_repeated,
            "correct_facts": correct_facts,
            "novel_facts": novel_facts,
            "incorrect_facts": incorrect_facts,
            "no_conclusion_steps": no_conclusion_steps,
            "correctness": float(correctness),
            "has_proper_format": bool(format_score > 0),
            "severe_repetitions": severe_repetitions,
            "repetition_ratio": float(repetition_ratio),
            "excess_steps": max(0, total_steps - self.max_reasonable_steps),
        }
        
        return float(total), metrics
    
    def _extract_model_steps(self, reasoning_text: str) -> List[Dict]:
        """
        Extract reasoning steps from model output.
        Returns steps even without proper conclusions (for penalty tracking).
        """
        steps = []
        
        # Find all step blocks (Step #N: ...)
        step_blocks = re.findall(
            r'Step\s*#?\s*\d+\s*:\s*(.+?)(?=Step\s*#?\s*\d+|$)',
            reasoning_text,
            re.IGNORECASE | re.DOTALL
        )
        
        for block in step_blocks:
            # Extract conclusion
            conclusion_match = re.search(
                r'In short,\s*([^.]+?)\s*(equals|cannot equal|does not equal)\s*([^.]+)',
                block,
                re.IGNORECASE
            )
            
            if conclusion_match:
                value1 = self._normalize_value(conclusion_match.group(1))
                relation_text = conclusion_match.group(2).strip().lower()
                value2 = self._normalize_value(conclusion_match.group(3))
                
                relation = "equal" if relation_text == "equals" else "not_equal"
                
                steps.append({
                    "value1": value1,
                    "value2": value2,
                    "relation": relation,
                    "text": block.strip()
                })
            else:
                # Step exists but has no valid conclusion (for penalty tracking)
                steps.append({
                    "value1": None,
                    "value2": None,
                    "relation": None,
                    "text": block.strip()
                })
        
        return steps
    
    def _create_fact_signature(self, step: Dict) -> str:
        """
        Create a canonical signature for a fact to detect duplicates.
        """
        v1 = step.get("value1")
        v2 = step.get("value2")
        relation = step.get("relation")
        
        if v1 is None or v2 is None:
            # No valid conclusion - use text hash for deduplication
            return f"no_conclusion:{hash(step.get('text', ''))}"
        
        # Sort values for canonical form
        v1_norm, v2_norm = sorted([v1, v2])
        return f"{v1_norm}|{relation}|{v2_norm}"
    
    def _normalize_value(self, value: str) -> str:
        """
        Normalize value strings for comparison.
        """
        # Remove extra whitespace, convert to lowercase
        normalized = re.sub(r'\s+', ' ', value.strip().lower())
        # Remove quotes
        normalized = normalized.strip('"\'')
        return normalized
    
    def _is_valid_fact(self, step: Dict, variables: Dict) -> bool:
        """
        Check if a fact is valid given the puzzle variables.
        """
        v1 = step.get("value1")
        v2 = step.get("value2")
        
        if not v1 or not v2:
            return False
        
        # Check if both values exist in the variables
        found_v1 = False
        found_v2 = False
        
        for var_name, var_values in variables.items():
            if var_values is None:
                continue
            
            normalized_values = [self._normalize_value(str(v)) for v in var_values]
            
            if v1 in normalized_values:
                found_v1 = True
            if v2 in normalized_values:
                found_v2 = True
        
        return found_v1 and found_v2
    
    def _has_proper_step_format(self, reasoning_text: str) -> bool:
        """
        Check if reasoning follows proper "Step #N:" format.
        """
        step_pattern = re.compile(r'Step\s*#?\s*\d+\s*:', re.IGNORECASE)
        matches = step_pattern.findall(reasoning_text)
        return len(matches) >= 2  # At least 2 steps
    
    def _score_final_answer(
        self,
        answer_text: str,
        ground_truth: str,
        variables: Dict
    ) -> Tuple[float, Dict]:
        """
        Score the final answer table against ground truth.
        
        Returns:
            (score, metrics) tuple
        """
        if not answer_text:
            return 0.0, {"error": "no_answer_found"}
        
        # Parse ground truth
        gt_mappings = self._parse_ground_truth(ground_truth)
        
        # Extract table from answer
        table_mappings = self._extract_table_mappings(answer_text, variables)
        
        if not table_mappings:
            return 0.0, {"error": "no_valid_table"}
        
        # Check structural constraints
        structure_score, structure_metrics = self._check_table_structure(
            table_mappings, variables
        )
        
        # Check correctness against ground truth
        correctness_score, correctness_metrics = self._check_mappings_correctness(
            table_mappings, gt_mappings
        )
        
        # Combined score
        total = 0.5 * structure_score + 0.5 * correctness_score
        
        metrics = {
            **structure_metrics,
            **correctness_metrics,
        }
        
        return float(total), metrics
    
    def _parse_ground_truth(self, ground_truth: str) -> List[Dict]:
        """
        Parse ground truth string into list of mappings.
        
        Expected format: "val1 | val2 | val3\nval4 | val5 | val6\n..."
        """
        mappings = []
        lines = ground_truth.strip().split('\n')
        
        for line in lines:
            values = [v.strip() for v in line.split('|')]
            if values:
                mappings.append(values)
        
        return mappings
    
    def _extract_table_mappings(
        self,
        answer_text: str,
        variables: Dict
    ) -> List[Dict]:
        """
        Extract table mappings from markdown table in answer.
        """
        # Find markdown table
        table_match = re.search(
            r'\|(.+?)\|.*?\n\s*\|[-:\s|]+\|\s*\n((?:\s*\|.+?\|\s*\n)+)',
            answer_text,
            re.MULTILINE | re.DOTALL
        )
        
        if not table_match:
            return []
        
        # Extract header
        header_line = table_match.group(1)
        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        
        # Extract rows
        rows_text = table_match.group(2)
        rows = []
        
        for row_line in rows_text.strip().split('\n'):
            if not row_line.strip():
                continue
            
            cells = [c.strip() for c in row_line.split('|') if c.strip()]
            
            if len(cells) == len(headers):
                row_dict = {headers[i]: cells[i] for i in range(len(headers))}
                rows.append(row_dict)
        
        return rows
    
    def _check_table_structure(
        self,
        table_mappings: List[Dict],
        variables: Dict
    ) -> Tuple[float, Dict]:
        """
        Check if table satisfies structural constraints.
        """
        if not table_mappings:
            return 0.0, {"valid_structure": False}
        
        # Get active variables (non-None)
        active_vars = {k: v for k, v in variables.items() if v is not None}
        
        if not active_vars:
            return 0.0, {"valid_structure": False, "error": "no_active_variables"}
        
        num_vars = len(active_vars)
        expected_rows = len(next(iter(active_vars.values())))
        
        # Check number of rows
        num_rows = len(table_mappings)
        row_score = min(num_rows / expected_rows, 1.0) if expected_rows > 0 else 0.0
        
        # Check number of columns
        num_cols = len(table_mappings[0]) if table_mappings else 0
        col_score = 1.0 if num_cols == num_vars else 0.5
        
        # Check uniqueness within columns
        uniqueness_violations = 0
        for col in (table_mappings[0].keys() if table_mappings else []):
            values = [row.get(col, "") for row in table_mappings]
            if len(values) != len(set(values)):
                uniqueness_violations += 1
        
        uniqueness_score = 1.0 - (uniqueness_violations / max(num_cols, 1))
        
        total = (row_score + col_score + uniqueness_score) / 3.0
        
        metrics = {
            "valid_structure": total > 0.5,
            "num_rows": num_rows,
            "expected_rows": expected_rows,
            "num_columns": num_cols,
            "expected_columns": num_vars,
            "uniqueness_violations": uniqueness_violations,
        }
        
        return float(total), metrics
    
    def _check_mappings_correctness(
        self,
        table_mappings: List[Dict],
        gt_mappings: List[List[str]]
    ) -> Tuple[float, Dict]:
        """
        Check correctness of mappings against ground truth.
        """
        if not table_mappings or not gt_mappings:
            return 0.0, {"correct_mappings": 0, "total_mappings": 0}
        
        # Normalize ground truth to set of tuples
        gt_set = set()
        for gt_row in gt_mappings:
            normalized = tuple(self._normalize_value(v) for v in gt_row)
            gt_set.add(normalized)
        
        # Check each table row
        correct_rows = 0
        partial_correct = 0
        
        for row in table_mappings:
            row_values = tuple(self._normalize_value(v) for v in row.values())
            
            if row_values in gt_set:
                correct_rows += 1
            else:
                # Check partial correctness
                for gt_row in gt_mappings:
                    gt_normalized = tuple(self._normalize_value(v) for v in gt_row)
                    matches = sum(1 for i in range(len(row_values)) 
                                if i < len(gt_normalized) and row_values[i] == gt_normalized[i])
                    partial_correct = max(partial_correct, matches)
        
        num_rows = len(table_mappings)
        row_accuracy = correct_rows / num_rows if num_rows > 0 else 0.0
        
        # Cell-level accuracy
        total_cells = num_rows * len(table_mappings[0]) if table_mappings else 0
        correct_cells = correct_rows * len(table_mappings[0]) if table_mappings else 0
        cell_accuracy = correct_cells / total_cells if total_cells > 0 else 0.0
        
        total = 0.7 * row_accuracy + 0.3 * cell_accuracy
        
        metrics = {
            "correct_rows": correct_rows,
            "total_rows": num_rows,
            "row_accuracy": float(row_accuracy),
            "cell_accuracy": float(cell_accuracy),
        }
        
        return float(total), metrics
    
    def _detect_reward_hacking(self, step_metrics: Dict, final_metrics: Dict) -> float:
        """
        Detect and penalize reward hacking patterns.
        
        Returns:
            Additional penalty (negative value) if hacking detected
        """
        penalty = 0.0
        
        # Pattern 1: Excessive repetition (>50% repeated)
        repetition_ratio = step_metrics.get("repetition_ratio", 0.0)
        if repetition_ratio > 0.5:
            penalty -= 0.3
        
        # Pattern 2: Too many steps with no conclusions
        no_conclusion = step_metrics.get("no_conclusion_steps", 0)
        if no_conclusion > 10:
            penalty -= 0.2
        
        # Pattern 3: Excessive total steps (likely padding)
        excess_steps = step_metrics.get("excess_steps", 0)
        if excess_steps > 20:
            penalty -= 0.3
        
        # Pattern 4: Severe repetitions (same fact 3+ times)
        severe_reps = step_metrics.get("severe_repetitions", 0)
        if severe_reps > 0:
            penalty -= 0.4
        
        # Pattern 5: Multiple uniqueness violations in table
        uniqueness_violations = final_metrics.get("uniqueness_violations", 0)
        if uniqueness_violations > 1:
            penalty -= 0.2
        
        return penalty


def compute_puzzle_baron_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict,
) -> float:
    """
    Main entry point compatible with VERL's reward scoring interface.
    
    Returns a single float score (not a dict) for compatibility.
    """
    rewarder = PuzzleBaronRewarder(
        step_reward_weight=0.3,
        final_reward_weight=0.7,
        novelty_bonus=0.1,
        repetition_penalty=-0.5,      # Increased penalty
        format_reward=0.1,
        max_reasonable_steps=30,       # Beyond this, penalize
        severe_repetition_threshold=3, # 3+ repetitions = severe penalty
        no_progress_penalty=-0.3,      # Penalize "This is consistent" steps
    )
    
    return rewarder.compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
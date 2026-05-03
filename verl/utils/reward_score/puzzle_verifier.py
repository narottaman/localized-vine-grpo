"""
Puzzle Baron Verifier - VAL-like verification for logic puzzles.

Inspired by PDDL-INSTRUCT's VAL approach:
- Checks logical consistency of reasoning steps
- Provides detailed feedback on errors
- Suggests corrections (like VAL's "Plan Repair Advice")
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class PuzzleVerifier:
    """
    Verifier for logic puzzles (similar to VAL for PDDL).
    
    Provides:
    1. Step-by-step verification
    2. Detailed feedback on errors
    3. Correction suggestions
    """
    
    def __init__(self):
        self.current_state = {}  # Track deduced facts
        self.constraints = []    # Store puzzle constraints
        
    def verify_response(
        self,
        response_str: str,
        ground_truth: str,
        extra_info: Dict,
    ) -> Dict[str, Any]:
        """
        Verify a complete response.
        
        Returns dict with:
        - is_valid: bool
        - step_results: List of step verification results
        - final_table_check: Dict with table accuracy
        - detailed_feedback: List of feedback strings
        - repair_advice: List of suggested corrections
        """
        # Initialize state
        self.current_state = {}
        self.constraints = extra_info.get("clues", []) if extra_info else []
        variables = extra_info.get("variables", {}) if extra_info else {}
        
        result = {
            "is_valid": False,
            "step_results": [],
            "final_table_check": {},
            "detailed_feedback": [],
            "repair_advice": [],
        }
        
        # Parse ground truth
        gt_table = self._parse_ground_truth(ground_truth)
        
        # Extract and verify reasoning steps
        steps = self._extract_steps(response_str)
        
        for step_idx, step in enumerate(steps):
            step_result = self._verify_step(step, gt_table, variables)
            result["step_results"].append(step_result)
            
            if not step_result["is_valid"]:
                result["detailed_feedback"].append(
                    f"Step {step_idx + 1}: {step_result['error_type']} - {step_result['error_detail']}"
                )
                if step_result.get("repair_advice"):
                    result["repair_advice"].append(
                        f"Step {step_idx + 1}: {step_result['repair_advice']}"
                    )
        
        # Verify final answer table
        table_result = self._verify_final_table(response_str, gt_table, variables)
        result["final_table_check"] = table_result
        
        # Overall validity
        all_steps_valid = all(s["is_valid"] for s in result["step_results"]) if result["step_results"] else True
        table_valid = table_result.get("cell_accuracy", 0) >= 0.9
        result["is_valid"] = all_steps_valid and table_valid
        
        return result
    
    def _extract_steps(self, response_str: str) -> List[str]:
        """Extract reasoning steps from response."""
        steps = []
        
        # Find reasoning section
        reasoning_match = re.search(
            r'###\s*Step-by-Step Reasoning Chain\s*\n(.*?)(?=###|$)',
            response_str,
            re.DOTALL | re.IGNORECASE
        )
        
        if reasoning_match:
            reasoning_text = reasoning_match.group(1)
            
            # Split by numbered steps
            step_splits = re.split(r'\n\s*\d+\.\s*', reasoning_text)
            for split in step_splits:
                if split.strip() and len(split.strip()) > 10:
                    steps.append(split.strip())
        
        return steps
    
    def _verify_step(
        self,
        step: str,
        gt_table: List[List[str]],
        variables: Dict,
    ) -> Dict[str, Any]:
        """
        Verify a single reasoning step.
        
        Checks:
        1. Logical validity of deductions
        2. Consistency with previous deductions
        3. Correctness against ground truth
        """
        result = {
            "is_valid": True,
            "score": 1.0,
            "error_type": None,
            "error_detail": None,
            "repair_advice": None,
        }
        
        # Extract deductions from step
        deductions = self._extract_deductions(step)
        
        for entity, relation, value in deductions:
            # Check 1: Consistency with current state
            if entity in self.current_state:
                existing = self.current_state[entity]
                if relation == "is" and existing.get("is") and existing["is"] != value:
                    result["is_valid"] = False
                    result["score"] = 0.0
                    result["error_type"] = "CONTRADICTION"
                    result["error_detail"] = f"'{entity}' was already determined to be '{existing['is']}', cannot be '{value}'"
                    result["repair_advice"] = f"Review earlier deduction about '{entity}'"
                    return result
            
            # Check 2: Correctness against ground truth
            gt_facts = self._get_gt_facts(gt_table)
            if relation == "is":
                fact = (entity.lower(), value.lower())
                reverse_fact = (value.lower(), entity.lower())
                if fact not in gt_facts and reverse_fact not in gt_facts:
                    # Check if it's a valid elimination instead
                    if relation != "is_not":
                        result["score"] = 0.5  # Uncertain, not verified
                else:
                    result["score"] = 1.0
                    # Update current state
                    if entity not in self.current_state:
                        self.current_state[entity] = {}
                    self.current_state[entity]["is"] = value
        
        return result
    
    def _extract_deductions(self, step: str) -> List[Tuple[str, str, str]]:
        """Extract (entity, relation, value) triples from step text."""
        deductions = []
        
        # Pattern: "X is Y" or "X equals Y"
        is_pattern = r'(\w+(?:\s+\w+)?)\s+(?:is|equals?|=)\s+(\w+(?:\s+\w+)?)'
        for match in re.finditer(is_pattern, step, re.IGNORECASE):
            entity, value = match.groups()
            deductions.append((entity.strip(), "is", value.strip()))
        
        # Pattern: "X cannot be Y" or "X is not Y"
        not_pattern = r'(\w+(?:\s+\w+)?)\s+(?:cannot|can\'t|is\s+not|doesn\'t)\s+(?:be|have|equal)?\s*(\w+(?:\s+\w+)?)'
        for match in re.finditer(not_pattern, step, re.IGNORECASE):
            entity, value = match.groups()
            deductions.append((entity.strip(), "is_not", value.strip()))
        
        return deductions
    
    def _verify_final_table(
        self,
        response_str: str,
        gt_table: List[List[str]],
        variables: Dict,
    ) -> Dict[str, Any]:
        """Verify the final answer table against ground truth."""
        result = {
            "cell_accuracy": 0.0,
            "correct_cells": 0,
            "total_cells": 0,
            "errors": [],
        }
        
        # Extract predicted table
        pred_table = self._extract_final_table(response_str)
        
        if not pred_table or not gt_table:
            return result
        
        # Compare tables
        total_cells = 0
        correct_cells = 0
        
        # Normalize both tables
        gt_normalized = [[c.lower().strip() for c in row] for row in gt_table]
        pred_normalized = [[c.lower().strip() for c in row] for row in pred_table]
        
        for row_idx, (pred_row, gt_row) in enumerate(zip(pred_normalized, gt_normalized)):
            for col_idx, (pred_cell, gt_cell) in enumerate(zip(pred_row, gt_row)):
                total_cells += 1
                if pred_cell == gt_cell:
                    correct_cells += 1
                else:
                    result["errors"].append({
                        "row": row_idx,
                        "col": col_idx,
                        "predicted": pred_cell,
                        "expected": gt_cell,
                        "feedback": f"Cell ({row_idx},{col_idx}): expected '{gt_cell}', got '{pred_cell}'"
                    })
        
        result["correct_cells"] = correct_cells
        result["total_cells"] = total_cells
        result["cell_accuracy"] = correct_cells / total_cells if total_cells > 0 else 0.0
        
        return result
    
    def _extract_final_table(self, response_str: str) -> List[List[str]]:
        """Extract final answer table from response."""
        rows = []
        
        match = re.search(
            r'###\s*Final Answer Block\s*\n(.*)',
            response_str,
            re.DOTALL | re.IGNORECASE
        )
        
        if match:
            table_text = match.group(1)
            for line in table_text.split('\n'):
                if '|' in line and not re.match(r'^\s*\|?[\s\-:]+\|', line):
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if cells:
                        rows.append(cells)
        
        return rows
    
    def _parse_ground_truth(self, ground_truth: str) -> List[List[str]]:
        """Parse ground truth into table format."""
        rows = []
        
        if not ground_truth:
            return rows
        
        for line in ground_truth.strip().split('\n'):
            if '|' in line:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    rows.append(cells)
        
        return rows
    
    def _get_gt_facts(self, gt_table: List[List[str]]) -> set:
        """Convert ground truth table to set of facts."""
        facts = set()
        for row in gt_table:
            for i, cell1 in enumerate(row):
                for j, cell2 in enumerate(row):
                    if i != j:
                        facts.add((cell1.lower(), cell2.lower()))
        return facts


def verify_puzzle_response(
    response_str: str,
    ground_truth: str,
    extra_info: Dict = None,
) -> Dict[str, Any]:
    """
    Convenience function to verify a puzzle response.
    
    Usage:
        result = verify_puzzle_response(response, gt, extra_info)
        print(result["detailed_feedback"])  # VAL-like feedback
        print(result["repair_advice"])      # Correction suggestions
    """
    verifier = PuzzleVerifier()
    return verifier.verify_response(response_str, ground_truth, extra_info or {})
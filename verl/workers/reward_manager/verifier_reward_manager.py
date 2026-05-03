# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Extended for Verifier-based Reward with Detailed Feedback

"""
Verifier Reward Manager - Inspired by PDDL-INSTRUCT's VAL approach.

Provides:
1. Step-by-step verification of reasoning chains
2. Detailed feedback on WHY steps are incorrect
3. Token-level rewards at step boundaries
4. Final outcome reward

This is similar to VAL in PDDL-INSTRUCT but for logic puzzles.
"""

import re
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("verifier")
class VerifierRewardManager(AbstractRewardManager):
    """
    Verifier-based Reward Manager with Detailed Feedback.
    
    Inspired by PDDL-INSTRUCT paper:
    - Provides step-level verification (like VAL)
    - Generates detailed feedback explaining errors
    - Assigns intermediate rewards at step boundaries
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        # Verifier-specific parameters
        use_detailed_feedback: bool = True,
        intermediate_reward_scale: float = 0.1,
        use_external_prm: bool = False,
        prm_model_name: str = None,
    ) -> None:
        """
        Initialize the Verifier Reward Manager.
        
        Args:
            tokenizer: Tokenizer for decoding responses
            num_examine: Number of examples to print for debugging
            compute_score: Function to compute final outcome score
            reward_fn_key: Key for data source
            use_detailed_feedback: If True, generate detailed error explanations
            intermediate_reward_scale: Scale for step-level rewards (0.0-1.0)
            use_external_prm: If True, use external Process Reward Model
            prm_model_name: HuggingFace model name for PRM (if use_external_prm=True)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.use_detailed_feedback = use_detailed_feedback
        self.intermediate_reward_scale = intermediate_reward_scale
        self.use_external_prm = use_external_prm
        
        # Initialize external PRM if requested
        self.prm_model = None
        if use_external_prm and prm_model_name:
            self._load_prm_model(prm_model_name)
    
    def _load_prm_model(self, model_name: str):
        """Load external Process Reward Model from HuggingFace."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            print(f"Loading PRM model: {model_name}")
            self.prm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.prm_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.prm_model.eval()
            print(f"✓ PRM model loaded successfully")
        except Exception as e:
            print(f"⚠ Failed to load PRM model: {e}")
            print("  Falling back to rule-based verification")
            self.prm_model = None
    
    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Compute rewards with step-level verification and detailed feedback.
        
        This mimics VAL's approach from PDDL-INSTRUCT:
        1. Parse response into reasoning steps
        2. Verify each step for logical consistency
        3. Generate detailed feedback for errors
        4. Assign rewards at step boundaries
        """
        # If RM scores already computed, return them
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": {}}
            return data.batch["rm_scores"]
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Decode prompt and response
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Get ground truth
            gt_dict = data_item.non_tensor_batch.get("reward_model", {}) or {}
            if isinstance(gt_dict, dict):
                ground_truth = gt_dict.get("ground_truth", "")
            else:
                ground_truth = ""
            
            extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "puzzle_baron")
            
            # ================================================================
            # STEP 1: Compute final outcome score (like VAL's plan validation)
            # ================================================================
            if self.compute_score:
                score_result = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                if isinstance(score_result, dict):
                    final_reward = float(score_result.get("score", 0.0))
                    score_details = score_result
                else:
                    final_reward = float(score_result)
                    score_details = {"score": final_reward}
            else:
                final_reward = 0.0
                score_details = {"score": 0.0}
            
            # ================================================================
            # STEP 2: Parse reasoning steps and verify each one
            # ================================================================
            steps, step_positions = self._extract_reasoning_steps(
                response_str=response_str,
                response_ids=response_ids,
                valid_length=valid_response_length,
            )
            
            # ================================================================
            # STEP 3: Verify each step and generate detailed feedback
            # ================================================================
            if self.use_external_prm and self.prm_model is not None:
                # Use external PRM for step verification
                step_scores, step_feedback = self._verify_steps_with_prm(
                    prompt_str=prompt_str,
                    steps=steps,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            else:
                # Use rule-based verification (like VAL)
                step_scores, step_feedback = self._verify_steps_rule_based(
                    response_str=response_str,
                    steps=steps,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            
            # ================================================================
            # STEP 4: Assign rewards at step boundaries
            # ================================================================
            for step_idx, (pos, score) in enumerate(zip(step_positions, step_scores)):
                if pos < valid_response_length:
                    # Scale intermediate reward
                    step_reward = score * self.intermediate_reward_scale
                    reward_tensor[i, pos] = step_reward
            
            # Final outcome reward at last token
            reward_tensor[i, valid_response_length - 1] = final_reward
            
            # Store detailed feedback in extra_info
            reward_extra_info["step_scores"].append(step_scores)
            reward_extra_info["step_feedback"].append(step_feedback)
            reward_extra_info["final_score"].append(final_reward)
            
            for key, value in score_details.items():
                reward_extra_info[key].append(value)
            
            # Logging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n{'='*60}")
                print(f"[VERIFIER REWARD MANAGER - Example {i}]")
                print(f"[prompt] {prompt_str[:200]}...")
                print(f"[response] {response_str[:500]}...")
                print(f"[ground_truth] {ground_truth[:200]}...")
                print(f"\n[STEP VERIFICATION] (like VAL feedback)")
                for idx, (step, score, feedback) in enumerate(zip(steps, step_scores, step_feedback)):
                    status = "✓ VALID" if score > 0.5 else "✗ INVALID"
                    print(f"  Step {idx+1}: {status} (score={score:.2f})")
                    print(f"    Content: {step[:100]}...")
                    if self.use_detailed_feedback and feedback:
                        print(f"    Feedback: {feedback}")
                print(f"\n[FINAL OUTCOME] score={final_reward:.3f}")
                print(f"{'='*60}\n")
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor
    
    def _extract_reasoning_steps(
        self,
        response_str: str,
        response_ids: torch.Tensor,
        valid_length: int,
    ) -> Tuple[List[str], List[int]]:
        """
        Extract reasoning steps from response.
        
        For Puzzle Baron format:
        - Steps in "### Step-by-Step Reasoning Chain" section
        - Each numbered step or bullet point
        
        Returns:
            steps: List of step text
            positions: List of token positions where each step ends
        """
        steps = []
        positions = []
        
        # Pattern 1: Numbered steps (1., 2., 3., etc.)
        step_pattern = r'(?:^|\n)\s*(\d+\.?\s*[^\n]+(?:\n(?!\d+\.).*)*)'
        
        # Pattern 2: Bullet points
        bullet_pattern = r'(?:^|\n)\s*[-•]\s*([^\n]+)'
        
        # Pattern 3: "Step X:" format
        step_x_pattern = r'Step\s*\d+[:\.]?\s*([^\n]+(?:\n(?!Step\s*\d).*)*)'
        
        # Try to find reasoning section
        reasoning_match = re.search(
            r'###\s*Step-by-Step Reasoning Chain\s*\n(.*?)(?=###|$)',
            response_str,
            re.DOTALL | re.IGNORECASE
        )
        
        if reasoning_match:
            reasoning_section = reasoning_match.group(1)
            
            # Extract numbered steps
            for match in re.finditer(step_pattern, reasoning_section):
                step_text = match.group(1).strip()
                if len(step_text) > 10:  # Skip very short matches
                    steps.append(step_text)
            
            # If no numbered steps, try bullets
            if not steps:
                for match in re.finditer(bullet_pattern, reasoning_section):
                    step_text = match.group(1).strip()
                    if len(step_text) > 10:
                        steps.append(step_text)
        
        # If still no steps, try Step X: pattern on full response
        if not steps:
            for match in re.finditer(step_x_pattern, response_str, re.IGNORECASE):
                step_text = match.group(1).strip()
                if len(step_text) > 10:
                    steps.append(step_text)
        
        # Calculate token positions for each step
        if steps:
            # Approximate: divide response evenly among steps
            step_length = valid_length // (len(steps) + 1)
            for idx in range(len(steps)):
                pos = min((idx + 1) * step_length, valid_length - 1)
                positions.append(int(pos))
        
        return steps, positions
    
    def _verify_steps_rule_based(
        self,
        response_str: str,
        steps: List[str],
        ground_truth: str,
        extra_info: Dict,
    ) -> Tuple[List[float], List[str]]:
        """
        Rule-based verification of reasoning steps (like VAL for PDDL).
        
        Checks:
        1. Logical consistency of deductions
        2. Whether deductions follow from given clues
        3. Whether conclusions match ground truth
        
        Returns:
            scores: List of scores (0.0-1.0) for each step
            feedback: List of detailed feedback strings
        """
        scores = []
        feedback = []
        
        # Get puzzle constraints from extra_info
        variables = extra_info.get("variables", {}) if extra_info else {}
        clues = extra_info.get("clues", []) if extra_info else []
        
        # Parse ground truth table
        gt_facts = self._parse_ground_truth_facts(ground_truth)
        
        for step_idx, step in enumerate(steps):
            step_score = 0.5  # Default: neutral
            step_feedback = ""
            
            # Check 1: Does step contain a valid deduction pattern?
            deduction_patterns = [
                r'(\w+)\s+(?:is|has|equals?|=)\s+(\w+)',
                r'(\w+)\s+(?:cannot|can\'t|doesn\'t)\s+(?:be|have|equal)\s+(\w+)',
                r'(?:therefore|so|thus|hence)\s+(\w+)\s+(?:is|has|must be)\s+(\w+)',
            ]
            
            has_deduction = False
            deduction_correct = False
            
            for pattern in deduction_patterns:
                matches = re.findall(pattern, step, re.IGNORECASE)
                if matches:
                    has_deduction = True
                    # Check if deduction matches ground truth
                    for match in matches:
                        entity, value = match[0].lower(), match[1].lower()
                        if (entity, value) in gt_facts or (value, entity) in gt_facts:
                            deduction_correct = True
                            break
            
            if has_deduction:
                if deduction_correct:
                    step_score = 1.0
                    step_feedback = "✓ Deduction is correct and matches solution"
                else:
                    step_score = 0.3
                    step_feedback = "✗ Deduction made but doesn't match solution"
            else:
                # Check if step references clues (setup step)
                if any(clue_word in step.lower() for clue_word in ['clue', 'given', 'since', 'because']):
                    step_score = 0.7
                    step_feedback = "○ Setup step referencing clues"
                else:
                    step_score = 0.5
                    step_feedback = "? Step doesn't contain clear deduction"
            
            # Check 2: Logical consistency (no contradictions)
            contradiction_patterns = [
                (r'(\w+)\s+is\s+(\w+)', r'\1\s+is\s+not\s+\2'),
                (r'(\w+)\s+has\s+(\w+)', r'\1\s+doesn\'t\s+have\s+\2'),
            ]
            
            for pos_pattern, neg_pattern in contradiction_patterns:
                pos_matches = re.findall(pos_pattern, step, re.IGNORECASE)
                neg_matches = re.findall(neg_pattern, step, re.IGNORECASE)
                if pos_matches and neg_matches:
                    for pm in pos_matches:
                        for nm in neg_matches:
                            if pm[0].lower() == nm[0].lower() and pm[1].lower() == nm[1].lower():
                                step_score = 0.0
                                step_feedback = f"✗ CONTRADICTION: '{pm[0]} is {pm[1]}' contradicts '{nm[0]} is not {nm[1]}'"
            
            scores.append(step_score)
            feedback.append(step_feedback)
        
        return scores, feedback
    
    def _verify_steps_with_prm(
        self,
        prompt_str: str,
        steps: List[str],
        ground_truth: str,
        extra_info: Dict,
    ) -> Tuple[List[float], List[str]]:
        """
        Verify steps using external Process Reward Model.
        
        Returns:
            scores: List of scores (0.0-1.0) for each step
            feedback: List of feedback strings (based on score interpretation)
        """
        scores = []
        feedback = []
        
        if self.prm_model is None:
            # Fallback to rule-based
            return self._verify_steps_rule_based("", steps, ground_truth, extra_info)
        
        # Build cumulative context for each step
        context = prompt_str
        
        for step_idx, step in enumerate(steps):
            # Add step to context
            context = context + "\n" + step
            
            # Score with PRM
            try:
                inputs = self.prm_tokenizer(
                    context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.prm_model.device)
                
                with torch.no_grad():
                    outputs = self.prm_model(**inputs)
                    # Most PRMs output logits for [negative, positive]
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    score = probs[0, 1].item()  # Probability of "correct"
                
                scores.append(score)
                
                # Generate feedback based on score
                if score >= 0.8:
                    feedback.append("✓ High confidence: Step is correct")
                elif score >= 0.5:
                    feedback.append("○ Medium confidence: Step may be correct")
                elif score >= 0.3:
                    feedback.append("? Low confidence: Step may have issues")
                else:
                    feedback.append("✗ Very low confidence: Step likely incorrect")
                    
            except Exception as e:
                print(f"PRM scoring error: {e}")
                scores.append(0.5)
                feedback.append(f"? Error during PRM scoring: {str(e)[:50]}")
        
        return scores, feedback
    
    def _parse_ground_truth_facts(self, ground_truth: str) -> set:
        """Parse ground truth into a set of (entity, value) facts."""
        facts = set()
        
        if not ground_truth:
            return facts
        
        # Parse pipe-separated table format
        for line in ground_truth.strip().split('\n'):
            if '|' in line:
                cells = [c.strip().lower() for c in line.split('|') if c.strip()]
                # Add all pairs from the row as facts
                for i, cell1 in enumerate(cells):
                    for j, cell2 in enumerate(cells):
                        if i != j and cell1 and cell2:
                            facts.add((cell1, cell2))
        
        return facts


# ================================================================
# Convenience registration for different verifier configurations
# ================================================================

@register("verifier_detailed")
class DetailedVerifierRewardManager(VerifierRewardManager):
    """Verifier with detailed feedback enabled (recommended)."""
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            use_detailed_feedback=True,
            intermediate_reward_scale=0.1,
            use_external_prm=False,
        )


@register("verifier_prm")
class PRMVerifierRewardManager(VerifierRewardManager):
    """Verifier using external Process Reward Model."""
    
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        prm_model_name: str = "peiyi9979/math-shepherd-mistral-7b-prm",
    ):
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            use_detailed_feedback=True,
            intermediate_reward_scale=0.1,
            use_external_prm=True,
            prm_model_name=prm_model_name,
        )
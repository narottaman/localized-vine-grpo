# File: /scratch/ngangada/thesis/thesis/verl/verl/workers/reward_manager/prm_verbal_reward_manager.py

"""
PRM + Verbal Reasoning Reward Manager.

Approach 2: Uses PRM to score each step AND generates verbal explanations
for why steps are wrong/right.

This provides richer feedback signal, similar to "detailed feedback" 
in PDDL-INSTRUCT paper which improved accuracy by 5-15%.
"""

import re
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("prm_verbal")
class PRMVerbalRewardManager(AbstractRewardManager):
    """
    PRM + Verbal Reasoning Reward Manager.
    
    Scores each step with PRM AND generates verbal feedback explaining
    why the step is correct/incorrect.
    
    Benefits:
    - Richer debugging information
    - Can analyze common error patterns
    - Potentially usable for future self-improvement training
    
    Similar to "detailed feedback" in PDDL-INSTRUCT paper.
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        # PRM configuration
        prm_model_name: str = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
        intermediate_reward_scale: float = 0.1,
        prm_max_length: int = 4096,
        # Verbal feedback configuration
        feedback_threshold: float = 0.5,  # Generate feedback for scores below this
        max_feedback_length: int = 150,
        use_critique_prompt: bool = True,
    ) -> None:
        """
        Initialize PRM + Verbal Reasoning Reward Manager.
        
        Args:
            tokenizer: Tokenizer for the policy model
            num_examine: Number of examples to print for debugging
            compute_score: Function to compute final outcome score
            reward_fn_key: Key for data source
            prm_model_name: HuggingFace model name for PRM
            intermediate_reward_scale: Scale for step-level rewards (0.0-1.0)
            prm_max_length: Maximum sequence length for PRM
            feedback_threshold: Generate verbal feedback for scores below this
            max_feedback_length: Maximum length of verbal feedback
            use_critique_prompt: If True, use structured critique prompt
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.intermediate_reward_scale = intermediate_reward_scale
        self.prm_max_length = prm_max_length
        self.feedback_threshold = feedback_threshold
        self.max_feedback_length = max_feedback_length
        self.use_critique_prompt = use_critique_prompt
        
        # Load PRM
        self.prm_model = None
        self.prm_tokenizer = None
        self._load_prm(prm_model_name)
    
    def _load_prm(self, model_name: str):
        """Load the Process Reward Model."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            print(f"🔄 [PRM-Verbal] Loading: {model_name}")
            
            self.prm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            
            # Ensure pad token exists
            if self.prm_tokenizer.pad_token is None:
                self.prm_tokenizer.pad_token = self.prm_tokenizer.eos_token
            
            self.prm_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.prm_model.eval()
            
            print(f"✅ [PRM-Verbal] Model loaded: {model_name}")
            
        except Exception as e:
            print(f"❌ [PRM-Verbal] Failed to load: {e}")
            print("   Using fallback scoring (no verbal feedback)")
            self.prm_model = None
    
    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute rewards with PRM scores and verbal feedback."""
        
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": {}}
            return data.batch["rm_scores"]
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print = {}
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Decode
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Get ground truth
            gt_dict = data_item.non_tensor_batch.get("reward_model", {}) or {}
            ground_truth = gt_dict.get("ground_truth", "") if isinstance(gt_dict, dict) else ""
            extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "puzzle_baron")
            
            # ================================================================
            # Compute final outcome score
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
            # Extract steps
            # ================================================================
            steps, step_positions = self._extract_steps(response_str, valid_response_length)
            
            # ================================================================
            # Score steps with PRM + generate verbal feedback
            # ================================================================
            step_scores, step_feedback = self._score_and_explain_steps(
                prompt_str=prompt_str,
                steps=steps,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # ================================================================
            # Assign rewards at step boundaries
            # ================================================================
            for pos, score in zip(step_positions, step_scores):
                if pos < valid_response_length:
                    reward_tensor[i, pos] = score * self.intermediate_reward_scale
            
            # Final reward
            reward_tensor[i, valid_response_length - 1] += final_reward
            
            # Store detailed info
            reward_extra_info["step_scores"].append(step_scores)
            reward_extra_info["step_feedback"].append(step_feedback)
            reward_extra_info["final_score"].append(final_reward)
            reward_extra_info["num_steps"].append(len(steps))
            reward_extra_info["steps_text"].append(steps)
            
            # Add score details
            for key, value in score_details.items():
                reward_extra_info[key].append(value)
            
            # ================================================================
            # Detailed logging
            # ================================================================
            if data_source not in already_print:
                already_print[data_source] = 0
            
            if already_print[data_source] < self.num_examine:
                already_print[data_source] += 1
                self._print_detailed_example(
                    idx=i,
                    prompt_str=prompt_str,
                    response_str=response_str,
                    steps=steps,
                    step_scores=step_scores,
                    step_feedback=step_feedback,
                    final_reward=final_reward,
                )
        
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        return reward_tensor
    
    def _extract_steps(self, response_str: str, valid_length: int) -> Tuple[List[str], List[int]]:
        """Extract reasoning steps and their token positions."""
        steps = []
        positions = []
        
        # Find reasoning section
        match = re.search(
            r'###\s*Step-by-Step Reasoning Chain\s*\n(.*?)(?=###|$)',
            response_str, re.DOTALL | re.IGNORECASE
        )
        text = match.group(1) if match else response_str
        
        # Extract numbered steps
        for m in re.finditer(r'(?:^|\n)\s*(\d+)\.\s*([^\n]+(?:\n(?!\d+\.).*)*)', text):
            step_text = m.group(2).strip()
            if len(step_text) > 10:
                steps.append(step_text)
        
        # Fallback: bullet points
        if not steps:
            for m in re.finditer(r'(?:^|\n)\s*[-•]\s*([^\n]+)', text):
                step_text = m.group(1).strip()
                if len(step_text) > 15:
                    steps.append(step_text)
        
        # Fallback: "Step X:" format
        if not steps:
            for m in re.finditer(r'Step\s*(\d+)[:\.]?\s*([^\n]+)', response_str, re.IGNORECASE):
                step_text = m.group(2).strip()
                if len(step_text) > 10:
                    steps.append(step_text)
        
        # Calculate positions
        if steps:
            step_len = valid_length // (len(steps) + 1)
            for idx in range(len(steps)):
                positions.append(min((idx + 1) * step_len, valid_length - 1))
        
        return steps, positions
    
    def _score_and_explain_steps(
        self,
        prompt_str: str,
        steps: List[str],
        ground_truth: str,
        extra_info: Dict,
    ) -> Tuple[List[float], List[str]]:
        """
        Score each step with PRM and generate verbal feedback.
        
        Returns:
            scores: List of PRM scores (0.0-1.0)
            feedback: List of verbal feedback strings
        """
        scores = []
        feedback = []
        
        if self.prm_model is None:
            # Fallback
            return self._fallback_scoring(steps, ground_truth)
        
        context = prompt_str + "\n"
        
        for step_idx, step in enumerate(steps):
            context += f"\n{step}"
            
            try:
                # ============================================================
                # Step 1: Get PRM score
                # ============================================================
                score = self._get_prm_score(context)
                scores.append(score)
                
                # ============================================================
                # Step 2: Generate verbal feedback if score is low
                # ============================================================
                if score < self.feedback_threshold:
                    # Generate explanation for why this step might be wrong
                    fb = self._generate_verbal_feedback(
                        context=context,
                        current_step=step,
                        score=score,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
                    feedback.append(fb)
                elif score >= 0.8:
                    feedback.append(f"✓ CORRECT (score={score:.2f}): Step reasoning appears sound")
                else:
                    feedback.append(f"○ UNCERTAIN (score={score:.2f}): Step may be correct but confidence is moderate")
                    
            except Exception as e:
                print(f"⚠ [PRM-Verbal] Error at step {step_idx}: {e}")
                scores.append(0.5)
                feedback.append(f"? ERROR: Could not evaluate step ({str(e)[:30]})")
        
        return scores, feedback
    
    def _get_prm_score(self, context: str) -> float:
        """Get PRM score for the current context."""
        inputs = self.prm_tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self.prm_max_length,
        ).to(self.prm_model.device)
        
        with torch.no_grad():
            outputs = self.prm_model(**inputs)
            logits = outputs.logits[:, -1, :]
            
            # Try to get good/bad token probabilities
            try:
                good_id = self.prm_tokenizer.encode("+", add_special_tokens=False)[0]
                bad_id = self.prm_tokenizer.encode("-", add_special_tokens=False)[0]
                probs = torch.softmax(torch.stack([logits[0, bad_id], logits[0, good_id]]), dim=0)
                score = probs[1].item()
            except:
                # Fallback: use average logit
                score = torch.sigmoid(logits.mean()).item()
        
        return score
    
    def _generate_verbal_feedback(
        self,
        context: str,
        current_step: str,
        score: float,
        ground_truth: str,
        extra_info: Dict,
    ) -> str:
        """
        Generate verbal feedback explaining why a step is incorrect.
        
        This is the key differentiator from PRM-Score-Only approach!
        """
        if not self.use_critique_prompt:
            # Simple feedback based on score
            return f"✗ LOW SCORE ({score:.2f}): This reasoning step may contain an error"
        
        # ================================================================
        # Construct critique prompt
        # ================================================================
        critique_prompt = self._build_critique_prompt(
            context=context,
            current_step=current_step,
            score=score,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        
        try:
            inputs = self.prm_tokenizer(
                critique_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.prm_model.device)
            
            with torch.no_grad():
                outputs = self.prm_model.generate(
                    **inputs,
                    max_new_tokens=self.max_feedback_length,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.prm_tokenizer.pad_token_id,
                )
            
            # Decode generated feedback
            generated = outputs[0, inputs.input_ids.shape[1]:]
            feedback_text = self.prm_tokenizer.decode(generated, skip_special_tokens=True)
            
            # Clean up
            feedback_text = feedback_text.strip()
            # Stop at first newline or period
            if '\n' in feedback_text:
                feedback_text = feedback_text.split('\n')[0]
            if len(feedback_text) > self.max_feedback_length:
                feedback_text = feedback_text[:self.max_feedback_length] + "..."
            
            return f"✗ INCORRECT (score={score:.2f}): {feedback_text}"
            
        except Exception as e:
            return f"✗ LOW SCORE ({score:.2f}): Error generating explanation: {str(e)[:50]}"
    
    def _build_critique_prompt(
        self,
        context: str,
        current_step: str,
        score: float,
        ground_truth: str,
        extra_info: Dict,
    ) -> str:
        """Build a prompt for generating verbal feedback."""
        
        # Extract clues if available
        clues_text = ""
        if extra_info and "clues" in extra_info:
            clues = extra_info["clues"]
            if isinstance(clues, list):
                clues_text = "\n".join([f"- {c}" for c in clues[:5]])
        
        prompt = f"""You are a logic puzzle reasoning verifier. Analyze the following reasoning step and explain why it might be incorrect.

PUZZLE CLUES:
{clues_text if clues_text else "(Not provided)"}

REASONING SO FAR:
{context[-800:]}

STEP BEING EVALUATED:
"{current_step}"

This step received a low confidence score of {score:.2f}, indicating it may contain an error.

In ONE brief sentence, explain what might be wrong with this reasoning step:"""
        
        return prompt
    
    def _fallback_scoring(
        self,
        steps: List[str],
        ground_truth: str,
    ) -> Tuple[List[float], List[str]]:
        """Fallback scoring when PRM is not available."""
        scores = []
        feedback = []
        
        # Parse ground truth
        gt_facts = set()
        if ground_truth:
            for line in ground_truth.split('\n'):
                if '|' in line:
                    cells = [c.strip().lower() for c in line.split('|') if c.strip()]
                    for i, c1 in enumerate(cells):
                        for j, c2 in enumerate(cells):
                            if i != j:
                                gt_facts.add((c1, c2))
        
        for step in steps:
            # Check for deduction patterns
            deductions = re.findall(r'(\w+)\s+(?:is|has|equals?)\s+(\w+)', step, re.IGNORECASE)
            
            if deductions:
                correct = any((d[0].lower(), d[1].lower()) in gt_facts for d in deductions)
                if correct:
                    scores.append(0.9)
                    feedback.append("✓ Deduction matches ground truth (fallback)")
                else:
                    scores.append(0.4)
                    feedback.append("✗ Deduction not found in ground truth (fallback)")
            else:
                scores.append(0.5)
                feedback.append("? No clear deduction pattern found (fallback)")
        
        return scores, feedback
    
    def _print_detailed_example(
        self,
        idx: int,
        prompt_str: str,
        response_str: str,
        steps: List[str],
        step_scores: List[float],
        step_feedback: List[str],
        final_reward: float,
    ):
        """Print detailed example with verbal feedback."""
        print(f"\n{'='*70}")
        print(f"🔍 [PRM-Verbal] Example {idx}")
        print(f"{'='*70}")
        print(f"\n[PROMPT] {prompt_str[:250]}...")
        print(f"\n[RESPONSE PREVIEW] {response_str[:400]}...")
        
        print(f"\n{'─'*70}")
        print(f"[STEP-BY-STEP VERIFICATION WITH VERBAL FEEDBACK]")
        print(f"{'─'*70}")
        
        for i, (step, score, fb) in enumerate(zip(steps, step_scores, step_feedback)):
            if score >= 0.7:
                status = "✓"
                color_code = "32"  # Green
            elif score >= 0.5:
                status = "○"
                color_code = "33"  # Yellow
            else:
                status = "✗"
                color_code = "31"  # Red
            
            print(f"\n  Step {i+1}: {status} (PRM score: {score:.3f})")
            print(f"    Text: {step[:80]}...")
            print(f"    Feedback: {fb}")
        
        print(f"\n{'─'*70}")
        avg_score = sum(step_scores) / len(step_scores) if step_scores else 0
        low_score_steps = sum(1 for s in step_scores if s < 0.5)
        print(f"[SUMMARY]")
        print(f"  • Total steps: {len(steps)}")
        print(f"  • Average step score: {avg_score:.3f}")
        print(f"  • Steps with low scores: {low_score_steps}")
        print(f"  • Final outcome reward: {final_reward:.3f}")
        print(f"{'='*70}\n")


# ================================================================
# Convenience registrations
# ================================================================

@register("prm_verbal_skywork")
class SkyworkPRMVerbalRewardManager(PRMVerbalRewardManager):
    """PRM + Verbal using Skywork model."""
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            prm_model_name="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
        )


@register("prm_verbal_mathshepherd")
class MathShepherdPRMVerbalRewardManager(PRMVerbalRewardManager):
    """PRM + Verbal using Math-Shepherd model."""
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            prm_model_name="peiyi9979/math-shepherd-mistral-7b-prm",
        )
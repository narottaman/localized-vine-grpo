# File: /scratch/ngangada/thesis/thesis/verl/verl/workers/reward_manager/prm_score_reward_manager.py

"""
PRM Score-Only Reward Manager.

Approach 1: Uses PRM to score each step, assigns reward based on score.
No verbal feedback - just numerical scores.

This is simpler and faster than PRM + Verbal approach.
"""

import re
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("prm_score")
class PRMScoreRewardManager(AbstractRewardManager):
    """
    PRM Score-Only Reward Manager.
    
    Scores each reasoning step with PRM and assigns intermediate rewards.
    No verbal feedback generation - faster and simpler.
    
    Similar to "binary feedback" in PDDL-INSTRUCT paper.
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
    ) -> None:
        """
        Initialize PRM Score-Only Reward Manager.
        
        Args:
            tokenizer: Tokenizer for the policy model
            num_examine: Number of examples to print for debugging
            compute_score: Function to compute final outcome score
            reward_fn_key: Key for data source
            prm_model_name: HuggingFace model name for PRM
            intermediate_reward_scale: Scale for step-level rewards (0.0-1.0)
            prm_max_length: Maximum sequence length for PRM
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.intermediate_reward_scale = intermediate_reward_scale
        self.prm_max_length = prm_max_length
        
        # Load PRM
        self.prm_model = None
        self.prm_tokenizer = None
        self._load_prm(prm_model_name)
    
    def _load_prm(self, model_name: str):
        """Load the Process Reward Model."""
        try:
            
            from transformers import AutoModel, AutoTokenizer
            print(f"🔄 [PRM-Score] Loading: {model_name}")
            
            self.prm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            
            self.prm_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.prm_model.eval()
            
            print(f"✅ [PRM-Score] Model loaded: {model_name}")
            
        except Exception as e:
            print(f"❌ [PRM-Score] Failed to load: {e}")
            print("   Using fallback heuristic scoring")
            self.prm_model = None
    
    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute rewards with PRM scores at step boundaries."""
        
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
                final_reward = float(score_result.get("score", 0.0)) if isinstance(score_result, dict) else float(score_result)
            else:
                final_reward = 0.0
            
            # ================================================================
            # Extract steps and score with PRM
            # ================================================================
            steps, step_positions = self._extract_steps(response_str, valid_response_length)
            step_scores = self._score_steps(prompt_str, steps)
            
            # ================================================================
            # Assign rewards
            # ================================================================
            for pos, score in zip(step_positions, step_scores):
                if pos < valid_response_length:
                    reward_tensor[i, pos] = score * self.intermediate_reward_scale
            
            # Final reward
            reward_tensor[i, valid_response_length - 1] += final_reward
            
            # Store info
            reward_extra_info["step_scores"].append(step_scores)
            reward_extra_info["final_score"].append(final_reward)
            reward_extra_info["num_steps"].append(len(steps))
            
            # Logging
            if data_source not in already_print:
                already_print[data_source] = 0
            if already_print[data_source] < self.num_examine:
                already_print[data_source] += 1
                print(f"\n{'='*60}")
                print(f"[PRM-Score] Example {i}")
                print(f"Steps: {len(steps)}, Scores: {[f'{s:.2f}' for s in step_scores]}")
                print(f"Final: {final_reward:.3f}, Avg Step: {sum(step_scores)/len(step_scores) if step_scores else 0:.3f}")
                print(f"{'='*60}")
        
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
        for m in re.finditer(r'(?:^|\n)\s*(\d+)\.\s*([^\n]+)', text):
            step_text = m.group(2).strip()
            if len(step_text) > 10:
                steps.append(step_text)
        
        # Fallback: bullet points
        if not steps:
            for m in re.finditer(r'(?:^|\n)\s*[-•]\s*([^\n]+)', text):
                step_text = m.group(1).strip()
                if len(step_text) > 15:
                    steps.append(step_text)
        
        # Calculate positions
        if steps:
            step_len = valid_length // (len(steps) + 1)
            for idx in range(len(steps)):
                positions.append(min((idx + 1) * step_len, valid_length - 1))
        
        return steps, positions
    
    def _score_steps(self, prompt_str: str, steps: List[str]) -> List[float]:
        """Score each step with PRM."""
        if self.prm_model is None:
            return [0.5] * len(steps)  # Fallback
        
        scores = []
        context = prompt_str + "\n"
        
        for step in steps:
            context += f"\n{step}"
            
            try:
                inputs = self.prm_tokenizer(
                    context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.prm_max_length,
                ).to(self.prm_model.device)
                
                with torch.no_grad():
                    outputs = self.prm_model(**inputs)
                    logits = outputs.logits[:, -1, :]
                    
                    # Get score (model-specific logic)
                    try:
                        good_id = self.prm_tokenizer.encode("+", add_special_tokens=False)[0]
                        bad_id = self.prm_tokenizer.encode("-", add_special_tokens=False)[0]
                        probs = torch.softmax(torch.stack([logits[0, bad_id], logits[0, good_id]]), dim=0)
                        score = probs[1].item()
                    except:
                        score = torch.sigmoid(logits.mean()).item()
                
                scores.append(score)
                
            except Exception as e:
                print(f"⚠ PRM error: {e}")
                scores.append(0.5)
        
        return scores
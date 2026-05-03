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
Optimized NaiveRewardManager for Vine-PPO and state-aware algorithms.

Key improvements:
1. Sparse intermediate rewards at section boundaries
2. Final reward at end of sequence
3. Optional: Dense shaping rewards for better credit assignment
"""

from collections import defaultdict
from typing import Any
import re
import ast
import json

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """
    Reward manager optimized for Vine-PPO and state-aware RL algorithms.
    
    Provides:
    - Sparse intermediate rewards at section completions
    - Final reward at end of sequence
    - Optional dense shaping for better credit assignment
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source",
        use_intermediate_rewards=True,
        intermediate_reward_scale=0.3,
        use_dense_shaping=False,
    ) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data.
            use_intermediate_rewards: If True, provide rewards at section boundaries.
            intermediate_reward_scale: Scale factor for intermediate rewards (0.0-1.0).
            use_dense_shaping: If True, spread a small reward across all tokens.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.use_intermediate_rewards = use_intermediate_rewards
        self.intermediate_reward_scale = intermediate_reward_scale
        self.use_dense_shaping = use_dense_shaping

    def _extract_ground_truth(self, data_item) -> str:
        """
        Robustly extract ground_truth text from a DataProto non_tensor_batch item.
        Returns a plain string.
        """
        non_tb = getattr(data_item, "non_tensor_batch", {}) or {}

        # 1) reward_model may be a dict-like
        rm = non_tb.get("reward_model", None)
        if isinstance(rm, dict):
            gt = rm.get("ground_truth") or rm.get("answer") or rm.get("gt") or ""
            if isinstance(gt, str):
                return gt
            if isinstance(gt, (list, tuple)):
                return "\n".join(str(x) for x in gt)
            if gt is not None:
                return str(gt)

        # 2) extra_info may carry ground_truth
        ei = non_tb.get("extra_info", {}) or {}
        if isinstance(ei, dict):
            gt = ei.get("ground_truth") or ei.get("groundtruth") or ei.get("answer")
            if isinstance(gt, str):
                return gt
            if isinstance(gt, (list, tuple)):
                return "\n".join(str(x) for x in gt)
            if gt is not None:
                return str(gt)

        # 3) reward_model might be a string (JSON or python-literal)
        if isinstance(rm, str):
            try:
                parsed = json.loads(rm)
                if isinstance(parsed, dict) and "ground_truth" in parsed:
                    g = parsed["ground_truth"]
                    if isinstance(g, (list, tuple)):
                        return "\n".join(str(x) for x in g)
                    return str(g)
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(rm)
                if isinstance(parsed, dict) and "ground_truth" in parsed:
                    g = parsed["ground_truth"]
                    if isinstance(g, (list, tuple)):
                        return "\n".join(str(x) for x in g)
                    return str(g)
            except Exception:
                pass
            return rm

        # 4) maybe there's a top-level 'ground_truth' key
        if "ground_truth" in non_tb:
            gt = non_tb["ground_truth"]
            if isinstance(gt, str):
                return gt
            if isinstance(gt, (list, tuple)):
                return "\n".join(str(x) for x in gt)
            return str(gt)

        # 5) fallback empty
        return ""

    def _is_reward_hacking(self, response_str: str, variables: dict) -> dict:
        """
        Detect reward-hacking patterns and return a dict:
          {"flag": bool, "reason": str, "meta": {...}}
        """
        result = {"flag": False, "reason": "", "meta": {}}

        # Check for multiple Final Answer Blocks
        if response_str.count('### Final Answer Block') > 1:
            result["flag"] = True
            result["reason"] = "multiple_final_blocks"
            result["meta"]["count"] = response_str.count('### Final Answer Block')
            return result

        # Extract table from Final Answer Block
        answer_match = re.search(r'###\s*Final Answer Block\s*\n(.*)', response_str, re.DOTALL | re.IGNORECASE)
        if not answer_match:
            result["flag"] = False
            result["reason"] = "no_final_block"
            return result

        table_text = answer_match.group(1)
        rows = [l.strip() for l in table_text.split('\n') if '|' in l and not l.startswith('|---')]
        
        # Remove header rows
        rows = [r for r in rows if not any(w in r.lower() for w in ['pieces', 'companies', 'themes', 'variable'])]

        if not rows:
            result["flag"] = False
            result["reason"] = "empty_table"
            return result

        # Check for duplicate rows
        if len(rows) != len(set(rows)):
            result["flag"] = True
            result["reason"] = "duplicate_rows"
            result["meta"]["total_rows"] = len(rows)
            result["meta"]["unique_rows"] = len(set(rows))
            return result

        # Check for too many rows
        expected_rows = 4  # Default
        if variables and isinstance(variables, dict) and len(variables) > 0:
            try:
                first_value = list(variables.values())[0]
                if first_value and hasattr(first_value, '__len__'):
                    expected_rows = len(first_value)
            except (IndexError, TypeError, AttributeError):
                pass

        if len(rows) > expected_rows * 1.5:
            result["flag"] = True
            result["reason"] = "too_many_rows"
            result["meta"]["num_rows"] = len(rows)
            result["meta"]["expected_rows"] = expected_rows
            return result

        # Check for identical rows repeated
        normalized_rows = ["|".join([c.strip() for c in r.split("|")]) for r in rows]
        tokens_per_row = [tuple(r.split("|")) for r in normalized_rows]
        if len(tokens_per_row) > 1 and all(tr == tokens_per_row[0] for tr in tokens_per_row[1:]):
            result["flag"] = True
            result["reason"] = "identical_rows_repeated"
            return result

        result["flag"] = False
        result["reason"] = "ok"
        result["meta"]["num_rows"] = len(rows)
        result["meta"]["expected_rows"] = expected_rows
        return result

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute rewards with optional intermediate signals for Vine-PPO."""

        # If there is rm score, directly return it
        # --- NEW: Support per-call override to disable intermediate rewards ---
        # If the trainer sets data.meta_info["disable_intermediate_rewards"] = True,
        # temporarily treat use_intermediate_rewards as False for this call.
        disable_intermediate_flag = False
        try:
            disable_intermediate_flag = bool(data.meta_info.get("disable_intermediate_rewards", False))
        except Exception:
            disable_intermediate_flag = False
    
        # effective boolean to use in this call
        use_intermediate = (self.use_intermediate_rewards and not disable_intermediate_flag)
 
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Extract ground truth using robust method
            gt_dict = data_item.non_tensor_batch.get("reward_model", {}) or {}
            if isinstance(gt_dict, dict):
                ground_truth = gt_dict.get("ground_truth", "")
            else:
                ground_truth = self._extract_ground_truth(data_item)
            
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {}) or {}
            if not isinstance(extra_info, dict):
                extra_info = {}

            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            # Compute score
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            # Normalize scalar -> dict
            if isinstance(score, dict):
                final_reward = float(score.get("score", 0.0))
            else:
                final_reward = float(score)
                score = {"score": final_reward}
            
            # Check for reward hacking
            hacking_res = self._is_reward_hacking(response_str, extra_info.get("variables", {}))
            
            if hacking_res.get("flag", False):
                final_reward = -0.35
                score["score"] = final_reward
                score["hacking_detected"] = True
                score["hacking_reason"] = hacking_res.get("reason", "unknown")
            else:
                score["hacking_detected"] = False
                score["hacking_reason"] = "ok"

            # Add all score keys to reward_extra_info
            for key, value in score.items():
                reward_extra_info[key].append(value)

            # Initialize section_positions
            section_positions = []

            # Intermediate rewards for Vine-PPO
            if use_intermediate  and valid_response_length > 1:
                num_reasoning = response_str.count("### Step-by-Step Reasoning Chain")
                num_tables = response_str.count("### Final Answer Block")
                
                if num_reasoning == 1 and num_tables == 1:
                    section_positions = self._find_section_boundaries(
                        response_ids=response_ids,
                        valid_length=valid_response_length,
                        tokenizer=self.tokenizer
                    )
                    
                    for pos in section_positions:
                        if pos < valid_response_length:
                            intermediate_reward = final_reward * self.intermediate_reward_scale
                            reward_tensor[i, pos] = intermediate_reward

            # Sparse reward at end (default)
            if not self.use_dense_shaping:
                reward_tensor[i, valid_response_length - 1] = final_reward
            else:
                # Dense shaping (advanced)
                dense_reward_per_token = (final_reward * 0.1) / valid_response_length
                reward_tensor[i, :valid_response_length] = dense_reward_per_token
                reward_tensor[i, valid_response_length - 1] += final_reward * 0.9

            # Logging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                
                if self.use_intermediate_rewards:
                    print(f"[intermediate_rewards_at]", section_positions)
                
                for key, value in score.items():
                    print(f"[{key}]", value)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    
    def _find_section_boundaries(
        self,
        response_ids: torch.Tensor,
        valid_length: int,
        tokenizer
    ) -> list[int]:
        """
        Find token positions where sections complete.
        
        For puzzle_baron, we look for:
        - "### Step-by-Step Reasoning Chain" completion
        - "### Final Answer Block" start
        
        Returns:
            List of token positions (indices) where intermediate rewards should be given.
        """
        positions = []
        
        # Decode the response to find section markers
        full_response = tokenizer.decode(response_ids[:valid_length], skip_special_tokens=False)
        
        # Pattern 1: After "### Step-by-Step Reasoning Chain" section
        reasoning_pattern = r'###\s*Step-by-Step Reasoning Chain.*?(?=###|$)'
        reasoning_match = re.search(reasoning_pattern, full_response, re.IGNORECASE | re.DOTALL)
        
        if reasoning_match:
            reasoning_end_char = reasoning_match.end()
            reasoning_end_token = min(int(reasoning_end_char / 4), valid_length - 1)
            if reasoning_end_token > 10:
                positions.append(reasoning_end_token)
        
        # Pattern 2: After "### Final Answer Block" header
        answer_pattern = r'###\s*Final Answer Block'
        answer_match = re.search(answer_pattern, full_response, re.IGNORECASE)
        
        if answer_match:
            answer_start_char = answer_match.end()
            answer_start_token = min(int(answer_start_char / 4), valid_length - 1)
            reasoning_end_token = positions[0] if positions else 0
            if answer_start_token > reasoning_end_token + 10:
                positions.append(answer_start_token)
        
        return positions


@register("naive_dense")
class NaiveRewardManagerDense(NaiveRewardManager):
    """
    Variant with dense shaping enabled by default.
    Use this ONLY if standard approach doesn't work.
    """
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            use_intermediate_rewards=False,
            intermediate_reward_scale=0.2,
            use_dense_shaping=True,
        )
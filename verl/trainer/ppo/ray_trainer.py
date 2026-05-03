# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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


import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import pickle

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.reward_score.puzzle_baron_rewarder_hybrid import compute_puzzle_baron_score
from verl.trainer.ppo.restem_trainer import RestEMTrainer, RestEMConfig

import re
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import Counter


# ================================================================
# PARTIAL TRAJECTORY SCORER
# ================================================================
def compute_mc_partial_score(
    solution_str: str,
    ground_truth: str = "",
    extra_info: Optional[Dict] = None,
    data_source: str = "puzzle_baron",
) -> float:
    """
    Score a PARTIAL trajectory for VineGRPO MC value estimation.
    Routes to GSM8K scorer or puzzle scorer based on data_source.
    Puzzle logic (puzzle_baron, zebralogic) is UNCHANGED from original.
    """
    if not solution_str or not solution_str.strip():
        return 0.0

    src = str(data_source).lower().strip()
    if src == "gsm8k":
        return _mc_partial_score_gsm8k(solution_str, ground_truth)
    return _mc_partial_score_puzzle(solution_str, ground_truth)


def _mc_partial_score_gsm8k(solution_str: str, ground_truth: str = "") -> float:
    """MC partial scorer for GSM8K arithmetic — rewards Steps + arithmetic + correct number."""
    # Immediately penalize repeated-digit hacking (480000000...)
    if re.search(r'(\d)\1{10,}', solution_str):
        return -1.0

    score = 0.0

    if re.search(r'###\s*Step-by-Step Reasoning Chain', solution_str, re.IGNORECASE):
        score += 0.1

    steps = re.findall(r'Step\s*#?\d+\.', solution_str, re.IGNORECASE)
    if steps:
        score += min(0.3, 0.06 * min(len(steps), 5))

    if re.findall(r'\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+', solution_str):
        score += 0.1

    # Extract final answer (our format OR standard GSM8K #### format)
    predicted = None
    m = re.search(r'###\s*Final Answer\s*\n\s*(-?[0-9][0-9,. ]*)', solution_str, re.IGNORECASE)
    if m:
        predicted = m.group(1).strip().replace(",", "").replace(" ", "")[:20]
    else:
        m2 = re.search(r'####\s*(-?[0-9][0-9,. ]*)', solution_str)
        if m2:
            predicted = m2.group(1).strip().replace(",", "").replace(" ", "")[:20]

    if predicted is not None:
        if len(predicted) >= 15:
            score -= 0.5  # repeated digits in answer
        else:
            try:
                pred_float = float(predicted)
                if -1e12 < pred_float < 1e12:
                    score += 0.2
                    if ground_truth:
                        try:
                            gt_float = float(str(ground_truth).strip().replace(",", ""))
                            if abs(pred_float - gt_float) < 1e-6:
                                score += 0.3
                        except (ValueError, TypeError):
                            pass
                else:
                    score -= 0.3
            except (ValueError, TypeError):
                pass

    return max(-1.0, min(1.0, score))


def _mc_partial_score_puzzle(solution_str: str, ground_truth: str = "") -> float:
    """
    MC partial scorer for logic puzzles — uses FINAL GRID SCORE ONLY.
    
    Scores the completed MC response using actual table accuracy,
    exactly like the main reward function. No format rewards, no
    partial credit for reasoning steps — this prevents reward hacking
    where the model repeats conclusions to accumulate format rewards.
    
    Returns 0.0-1.0 based on cell accuracy of the predicted table.
    Returns -0.5 for responses with clear repetition hacking.
    """
    if not solution_str or not solution_str.strip():
        return 0.0
    
    # ── Penalize repetition hacking — same conclusion repeated 5+ times ──
    conclusions = re.findall(
        r'In short,\s*[^.\n]+',
        solution_str, re.IGNORECASE
    )
    if conclusions:
        from collections import Counter
        counts = Counter(c.strip().lower() for c in conclusions)
        if counts.most_common(1)[0][1] >= 5:
            return -0.5  # hard penalty for looping

    # ── Extract Final Answer Block table ──
    table_match = re.search(
        r'###\s*Final Answer Block\s*\n(.*)',
        solution_str, re.DOTALL | re.IGNORECASE
    )
    if not table_match:
        return 0.0  # no table = no reward, no penalty

    table_content = table_match.group(1)
    pred_rows = []
    for line in table_content.split('\n'):
        if '|' not in line:
            if pred_rows:
                break
            continue
        if re.match(r'^\s*\|?[\s\-:]+\|', line):
            continue
        cells = [_normalize_cell(c) for c in line.split('|') if c.strip()]
        if cells:
            pred_rows.append(cells)

    if not pred_rows:
        return 0.0

    # ── Score against ground truth using cell accuracy ──
    gt_rows = _parse_simple_gt(ground_truth)
    if not gt_rows:
        return 0.0  # No GT = no score (don't give free partial credit)

    # Normalize GT cells consistently
    gt_rows = [[_normalize_cell(c) for c in row] for row in gt_rows]

    exp_rows = len(gt_rows)
    exp_cols = max(len(r) for r in gt_rows)

    # Pad/truncate predicted to match expected shape
    pred = []
    for i in range(exp_rows):
        if i < len(pred_rows):
            row = pred_rows[i][:exp_cols]
            row += [""] * max(0, exp_cols - len(row))
        else:
            row = [""] * exp_cols
        pred.append(row)

    # Positional cell-by-cell comparison (the correct metric)
    total = exp_rows * exp_cols
    correct = 0
    for i in range(exp_rows):
        for j in range(exp_cols):
            if pred[i][j] and gt_rows[i][j] and pred[i][j] == gt_rows[i][j]:
                correct += 1

    return correct / total if total > 0 else 0.0


def _normalize_cell(cell: str) -> str:
    """Normalize cell for comparison — must match the main rewarder."""
    if not cell:
        return ""
    normalized = ' '.join(cell.split())
    normalized = normalized.lower()
    normalized = normalized.strip('"\'')
    normalized = normalized.rstrip('.,;:')
    return normalized.strip()

def _parse_simple_gt(ground_truth: str) -> List[List[str]]:
    """Simple parser for ground truth table."""
    if not ground_truth:
        return []
    
    rows = []
    for line in ground_truth.strip().split('\n'):
        line = line.strip()
        if not line or re.match(r'^[\s\-:|]+$', line):
            continue
        if '|' in line:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells:
                rows.append(cells)
    return rows

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch.get("token_level_rewards", None),
            "response_mask": data.batch.get("response_mask", None),
            "config": config,
        }

        # Optional parameters
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
            
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # ===== CRITICAL FIX: Pass vine_mc_values with correct key name =====
        if "vine_mc_values" in data.batch:
            adv_kwargs["vine_mc_values"] = data.batch["vine_mc_values"]  
        
        if "vine_grpo_scores" in data.meta_info:
            adv_kwargs["vine_grpo_scores"] = data.meta_info["vine_grpo_scores"]
        if "vine_step_ranges" in data.meta_info:
            adv_kwargs["vine_step_ranges"] = data.meta_info["vine_step_ranges"]
        if "vine_grpo_kl" in data.meta_info:
            adv_kwargs["vine_grpo_kl"] = data.meta_info["vine_grpo_kl"]
        if hasattr(config, "kl_ctrl"):
            adv_kwargs["kl_coef"] = float(config.kl_ctrl.get("kl_coef", 0.01))
            
        # Pass gamma and lam if available (needed by some estimators)
        if hasattr(config, "gamma"):
            adv_kwargs["gamma"] = config.gamma
        if hasattr(config, "gae_lambda"):
            adv_kwargs["lam"] = config.gae_lambda
            
        if hasattr(config, "alpha"):
            adv_kwargs["alpha"] = config.alpha
        
        
        # Pass values from critic if available (for hybrid methods)
        if "values" in data.batch:
            adv_kwargs["values"] = data.batch["values"]
            
        # ── GDPO: pass individual reward components if present ──────────
        # These are populated by puzzle_baron_rewarder_hybrid.py and stored
        # in non_tensor_batch by the reward manager during compute_reward().
        if "reward_correctness" in data.non_tensor_batch:
            adv_kwargs["reward_correctness"] = torch.tensor(
                data.non_tensor_batch["reward_correctness"],
                dtype=torch.float32,
                device=data.batch["token_level_rewards"].device,
            )
        if "reward_structure" in data.non_tensor_batch:
            adv_kwargs["reward_structure"] = torch.tensor(
                data.non_tensor_batch["reward_structure"],
                dtype=torch.float32,
                device=data.batch["token_level_rewards"].device,
            )
        if "reward_penalty" in data.non_tensor_batch:
            adv_kwargs["reward_penalty"] = torch.tensor(
                data.non_tensor_batch["reward_penalty"],
                dtype=torch.float32,
                device=data.batch["token_level_rewards"].device,
            )
        # ────────────────────────────────────────────────────────────────


        # Calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        
        # Initialize RestEM if enabled
        if self.config.algorithm.get('use_restem', False):
            restem_config = RestEMConfig(
                n_samples_per_prompt=self.config.algorithm.get('restem_n_samples', 16),
                min_correct_ratio=self.config.algorithm.get('restem_min_correct_ratio', 0.1),
                temperature=self.config.actor_rollout_ref.rollout.temperature,
                top_p=self.config.actor_rollout_ref.rollout.top_p,
                sft_epochs=self.config.algorithm.get('restem_sft_epochs', 1),
            )
            self.restem_trainer = RestEMTrainer(
                actor_rollout_wg=self.actor_rollout_wg,
                reward_fn=self.reward_fn,
                config=restem_config,
                reward_threshold=self.config.algorithm.get('restem_reward_threshold', 0.9),
            )
        
        self.use_grpo_per_state = self.config.trainer.get('use_grpo_per_state', False)
        self.mc_k = int(self.config.trainer.get('mc_k', 1))
        self.use_grpo_per_state = (self.config.algorithm.adv_estimator == "vineppo_grpo")
        print(f"🔧 Advantage computation mode: {'GRPO-per-State' if self.use_grpo_per_state else 'MC Rollout'}")
        print(f"   mc_k (samples per state): {self.mc_k}")
        
    

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch
    
    
    
    def compute_score(self, data_source, solution_str, ground_truth, extra_info):
        """Wrapper to call reward function for MC scoring."""
        # Import your custom reward function
        from verl.utils.reward_score.puzzle_baron_rewarder_hybrid import compute_puzzle_baron_score
        
        return compute_puzzle_baron_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info
        )
    
    def _extract_intermediate_states(self, batch: DataProto) -> tuple[list, dict]:
        """
        Extract intermediate states AND step boundaries from batch.
        
        CRITICAL: Skips responses that are already terminal (contain stop sequences or final answer).
        """
        import re
        from omegaconf import OmegaConf
        
        states = []
        step_boundaries = {}
        
        tokenizer = self.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        
        responses = batch.batch.get("responses", None)
        prompts = batch.batch.get("prompts", None)
        response_mask = batch.batch.get("response_mask", None)
        uids = batch.non_tensor_batch.get("uid", [None] * len(batch))
        
        if responses is None or prompts is None:
            return states, step_boundaries
        
        # ================================================================
        # Get stop sequences
        # ================================================================
        stop_seqs = getattr(self, "_stop_sequences", None)
        if stop_seqs is None:
            try:
                stop_seqs = list(OmegaConf.select(
                    self.config, 
                    "actor_rollout_ref.rollout.stop_sequences", 
                    default=[]
                ) or [])
            except Exception:
                stop_seqs = []
            if not stop_seqs:
                stop_seqs = ["END_OF_OUTPUT", "\n\n"]
            if "\n\n" not in stop_seqs:
                stop_seqs = stop_seqs + ["\n\n"]
            self._stop_sequences = stop_seqs
        
        # Track skipped responses
        skipped_terminal = 0
        skipped_no_structure = 0
        processed = 0
        
        for i in range(len(batch)):
            try:
                uid = uids[i] if i < len(uids) else None
                resp_ids_tensor = responses[i]
                prompt_ids_tensor = prompts[i]
                
                # Get valid response length
                if response_mask is not None:
                    valid_resp_len = int(response_mask[i].sum().item())
                else:
                    valid_resp_len = resp_ids_tensor.size(0)
                
                # Get token IDs as lists
                resp_ids_list = resp_ids_tensor[:valid_resp_len].cpu().tolist()
                prompt_ids_list = prompt_ids_tensor.cpu().tolist()
                
                # Remove padding from prompt
                prompt_len = sum(1 for t in prompt_ids_list if t != pad_token_id)
                prompt_ids_clean = prompt_ids_list[:prompt_len]
                
                # Decode response for analysis
                resp_text = tokenizer.decode(resp_ids_list, skip_special_tokens=True)
                
            except Exception as e:
                print(f"Warning: Failed to process batch item {i}: {e}")
                continue
            
            # ================================================================
            # CHANGED: Don't skip terminal responses entirely!
            # Instead, extract intermediate states but exclude the final step.
            # ================================================================
            lower_resp = resp_text.lower()
            
            # Check for stop sequences
            has_stop_seq = any(s.lower() in lower_resp for s in stop_seqs)
            
            # Check for final answer markers
            has_final_answer = (
                "final answer" in lower_resp or 
                "### final" in lower_resp or
                "answer block" in lower_resp
            )
            
            is_terminal = has_stop_seq or has_final_answer
            
            # ================================================================
            # Find step delimiters (do this for ALL responses, terminal or not)
            # ================================================================
            step_pattern = r'(###\s*(?:Step|Reasoning|Final)[^\n]*|Step\s*#?\d+[:\.])'
            
            
            delimiter_char_positions = []
            for match in re.finditer(step_pattern, resp_text, re.IGNORECASE):
                delimiter_char_positions.append(match.start())
            
            if not delimiter_char_positions:
                # No steps found
                step_boundaries[i] = [(0, valid_resp_len)]
                if is_terminal:
                    skipped_terminal += 1
                else:
                    skipped_no_structure += 1
                continue
            
            # ================================================================
            # Convert character positions to token positions
            # ================================================================
            token_to_char_end = []
            cumulative_chars = 0
            for tok_idx, tok_id in enumerate(resp_ids_list):
                tok_text = tokenizer.decode([tok_id], skip_special_tokens=True)
                cumulative_chars += len(tok_text)
                token_to_char_end.append(cumulative_chars)
            
            def char_pos_to_token_pos(char_pos):
                for tok_idx, char_end in enumerate(token_to_char_end):
                    if char_pos <= char_end:
                        return tok_idx
                return len(token_to_char_end) - 1
            
            # Build step boundaries
            boundaries = []
            prev_token_pos = 0
            
            for char_pos in delimiter_char_positions:
                token_pos = char_pos_to_token_pos(char_pos)
                token_pos = min(token_pos, valid_resp_len)
                
                if token_pos > prev_token_pos:
                    boundaries.append((prev_token_pos, token_pos))
                prev_token_pos = token_pos
            
            # Add final segment
            if prev_token_pos < valid_resp_len:
                boundaries.append((prev_token_pos, valid_resp_len))
            
            if not boundaries:
                step_boundaries[i] = [(0, valid_resp_len)]
                continue
            
            step_boundaries[i] = boundaries
            
            # ================================================================
            # Create states for MC generation
            # For TERMINAL responses: extract all steps EXCEPT the last one
            # For NON-TERMINAL responses: extract all steps
            # ================================================================
            steps_to_process = boundaries[:-1] if is_terminal else boundaries
            
            if is_terminal and len(boundaries) <= 1:
                # Terminal response with only 1 step - nothing to extract
                skipped_terminal += 1
                continue
            MAX_STATES_PER_EXAMPLE = 35 
            for step_idx, (start_tok, end_tok) in enumerate(steps_to_process):
            
                if step_idx >= MAX_STATES_PER_EXAMPLE:
                    break
                # Skip if this step contains final answer text
                step_text = tokenizer.decode(resp_ids_list[start_tok:end_tok], skip_special_tokens=True)
                if "final answer" in step_text.lower() or "### final" in step_text.lower():
                    continue
                
                # Build prefix: prompt + response[:end_tok]
                prefix_resp_ids = resp_ids_list[:end_tok]
                
                # Strip terminal tokens from prefix
                while prefix_resp_ids and prefix_resp_ids[-1] == pad_token_id:
                    prefix_resp_ids.pop()
                if eos_token_id is not None:
                    while prefix_resp_ids and prefix_resp_ids[-1] == eos_token_id:
                        prefix_resp_ids.pop()
                
                full_prefix_ids = prompt_ids_clean + prefix_resp_ids
                
                states.append({
                    "batch_idx": int(i),
                    "episode_id": str(uid) if uid is not None else str(i),
                    "step_idx": int(step_idx),
                    "prefix": full_prefix_ids,
                    "token_position": int(end_tok),
                    "step_start": int(start_tok),
                    "step_end": int(end_tok),
                })
            
            processed += 1

        
        print(f"ðŸ“Š Extracted {len(states)} intermediate states from {len(batch)} batch items")
        print(f"ðŸ“Š Processed: {processed}, Skipped terminal: {skipped_terminal}, Skipped no structure: {skipped_no_structure}")
        print(f"ðŸ“Š Step boundaries: {len(step_boundaries)} items with boundaries")
        
        return states, step_boundaries
    
    

    # OPTIMIZED VERSION OF _generate_mc_rollouts_bucketed
    # Remove ALL debug prints and text decoding
    
    def _generate_mc_rollouts_bucketed(self, prefixes, episode_ids, batch_idxs, step_idxs,
                                        token_positions, step_starts, step_ends,
                                        mc_k=1, max_new_tokens=None, batch=None):
        """
        Generate MC rollouts in buckets.
        OPTIMIZED: Removed all debug prints and unnecessary text decoding.
        """
        from copy import deepcopy
        import numpy as np
        from omegaconf import OmegaConf
        
        # ================================================================
        # Config-derived params
        # ================================================================
        vocab_size = len(self.tokenizer)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        valid_pad = pad_token_id if pad_token_id < vocab_size else 0
        
        desired_new_tokens = min(int(max_new_tokens or 512), 512)
        
        # Compute pad_to
        try:
            chunk_from_cfg = int(OmegaConf.select(self.config, "trainer.n_gpus_per_node", default=0) or 0)
            micro_batch = int(OmegaConf.select(self.config, "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", default=1) or 1)
            pad_to = max(1, chunk_from_cfg * micro_batch)
        except Exception:
            pad_to = 2
        
        try:
            if hasattr(self.actor_rollout_wg, "num_workers"):
                pad_to = max(pad_to, int(self.actor_rollout_wg.num_workers or 1))
        except Exception:
            pass
        
        try:
            ppo_micro = int(OmegaConf.select(self.config, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", default=2) or 2)
            pad_to = max(pad_to, ppo_micro)
        except Exception:
            pass
        
        pad_to = max(2, pad_to)
        
        n = len(prefixes)
        if n == 0:
            return []
        
        # Safe dummy tokens
        try:
            safe_dummy_tokens = self.tokenizer.encode("Hello", add_special_tokens=False)
            if not safe_dummy_tokens or all(t == valid_pad for t in safe_dummy_tokens):
                safe_dummy_tokens = [1]
        except Exception:
            safe_dummy_tokens = [1]
        
        safe_dummy_tokens = [t for t in safe_dummy_tokens if 0 <= t < vocab_size and t != valid_pad]
        if not safe_dummy_tokens:
            safe_dummy_tokens = [1]
        
        # Sort and bucket
        indexed = [(i, prefixes[i]) for i in range(n)]
        indexed.sort(key=lambda x: len(x[1]))
        
        bucket_size = 8
        buckets = [indexed[i:i+bucket_size] for i in range(0, n, bucket_size)]
        
        # Output container
        all_completions = [{
            "episode_id": episode_ids[i],
            "batch_idx": batch_idxs[i],
            "step_idx": step_idxs[i],
            "token_position": token_positions[i],
            "step_start": step_starts[i],
            "step_end": step_ends[i],
            "completions_ids": [],
            "completions_text": []
        } for i in range(n)]
        
        orig_ntb = batch.non_tensor_batch if batch and hasattr(batch, "non_tensor_batch") else {}
        orig_batch_size = len(batch) if batch and hasattr(batch, "__len__") else None
        
        total_non_empty = 0
        total_empty = 0
        
        # Process buckets
        for b_idx, bucket in enumerate(buckets):
            bucket_orig_indices = [item[0] for item in bucket]
            bucket_prefixes = [list(item[1]) for item in bucket]
            
            real_batch_len = len(bucket_prefixes)
            
            for i, pref in enumerate(bucket_prefixes):
                if len(pref) == 0:
                    bucket_prefixes[i] = list(safe_dummy_tokens)
            
            bucket_max_len = max(len(p) for p in bucket_prefixes)
            
            pad_needed = (-len(bucket_prefixes)) % pad_to
            if pad_needed > 0:
                reference_prefix = bucket_prefixes[-1][:min(100, len(bucket_prefixes[-1]))]
                if not reference_prefix:
                    reference_prefix = list(safe_dummy_tokens)
                
                for _ in range(pad_needed):
                    bucket_orig_indices.append(bucket_orig_indices[-1])
                    bucket_prefixes.append(list(reference_prefix))
                
                bucket_max_len = max(len(p) for p in bucket_prefixes)
            
            original_lengths = [len(p) for p in bucket_prefixes]
            
            # ================================================================
            # Build tensors with LEFT padding (required for decoder-only models)
            # ================================================================
            bsize = len(bucket_prefixes)
            prompt_tensor = torch.full((bsize, bucket_max_len), valid_pad, dtype=torch.long)
            attention_mask = torch.zeros((bsize, bucket_max_len), dtype=torch.long)
            
            for i, p in enumerate(bucket_prefixes):
                seq_len = len(p)
                # LEFT padding: content goes at the END
                start_idx = bucket_max_len - seq_len
                prompt_tensor[i, start_idx:] = torch.tensor(p, dtype=torch.long)
                attention_mask[i, start_idx:] = 1
            
            # Compute position_ids using cumsum (robust for left-padding)
            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            position_ids = position_ids.clamp(min=0).long()
            
            gen_batch_dict = {
                "prompts": prompt_tensor,
                "input_ids": prompt_tensor,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            gen_data = DataProto.from_single_dict(gen_batch_dict)
            
            # Build non_tensor_batch
            bucket_ntb = {}
            bucket_batch_idxs = [batch_idxs[idx] for idx in bucket_orig_indices[:real_batch_len]]
            bucket_batch_idxs.extend([bucket_batch_idxs[-1] if bucket_batch_idxs else 0] * pad_needed)
            
            for key, val in orig_ntb.items():
                try:
                    is_per_example = (
                        orig_batch_size is not None
                        and hasattr(val, "__len__")
                        and not isinstance(val, (str, dict))
                        and len(val) == orig_batch_size
                    )
                    if is_per_example:
                        mapped = [val[bidx] if bidx < len(val) else None for bidx in bucket_batch_idxs]
                        bucket_ntb[key] = np.array(mapped, dtype=object)
                    else:
                        bucket_ntb[key] = deepcopy(val)
                except Exception:
                    bucket_ntb[key] = deepcopy(val)
            
            gen_data.non_tensor_batch = bucket_ntb
            
            # Remove cached raw_prompt_ids
            if "raw_prompt_ids" in gen_data.non_tensor_batch:
                del gen_data.non_tensor_batch["raw_prompt_ids"]
            
            # Generate
            # Get stop sequences from config (same as main rollout!)
            stop_seqs = getattr(self, "_stop_sequences", None)
            if stop_seqs is None:
                try:
                    from omegaconf import OmegaConf
                    stop_seqs = list(OmegaConf.select(
                        self.config, 
                        "actor_rollout_ref.rollout.stop_sequences", 
                        default=[]
                    ) or [])
                except Exception:
                    stop_seqs = []
                if not stop_seqs:
                    stop_seqs = ["END_OF_OUTPUT", "\n\n"]
                if "\n\n" not in stop_seqs:
                    stop_seqs = stop_seqs + ["\n\n"]
            
            for k_idx in range(mc_k):
                mc_meta = {
                    "do_sample": True,
                    "validate": False,
                    "temperature": 0.8,              # âœ… Match main
                    "top_p": 0.95,                   # âœ… Match main
                    "top_k": -1,                     # âœ… No filtering
                    "repetition_penalty": 1.5,       # âœ… Match main (NOT 1.05!)
                    "max_new_tokens": min(desired_new_tokens, 512),
                    "min_new_tokens": 1,
                    "eos_token_id": eos_token_id if eos_token_id and eos_token_id < vocab_size else None,
                    "pad_token_id": valid_pad,
                    "stop": stop_seqs,               # âœ… CRITICAL FIX!
                    "stop_sequences": stop_seqs,     # âœ… CRITICAL FIX!
                }
                gen_data.meta_info = mc_meta 
                try:
                    result = self.actor_rollout_wg.generate_sequences(gen_data)
                    
                    # vLLM returns "responses" which contains ONLY newly generated tokens
                    if hasattr(result, "batch"):
                        out_seqs = result.batch.get("responses", result.batch.get("sequences"))
                    else:
                        out_seqs = result
                    
                    # Extract completions
                    for local_idx in range(real_batch_len):
                        orig_index = bucket_orig_indices[local_idx]
                        full_seq = out_seqs[local_idx].cpu().tolist()
                        
                        # Find where actual content ends (last non-pad token)
                        last_nonpad_idx = -1
                        for i in range(len(full_seq) - 1, -1, -1):
                            if full_seq[i] != valid_pad:
                                last_nonpad_idx = i
                                break
                        
                        if last_nonpad_idx >= 0:
                            # Extract all content up to last_nonpad
                            completion_ids = full_seq[:last_nonpad_idx + 1]
                            if completion_ids:
                                total_non_empty += 1
                                completion_text = ""  # Don't decode - we don't use it!
                                
                                # ADD THIS DIAGNOSTIC:
                                if b_idx == 0 and k_idx == 0 and local_idx == 0:
                                    print(f"ðŸ” SAMPLE MC COMPLETION:")
                                    print(f"   Length: {len(completion_ids)} tokens")
                                    print(f"   First 10 tokens: {completion_ids[:10]}")
                                    try:
                                        sample_text = self.tokenizer.decode(completion_ids[:100], skip_special_tokens=True)
                                        print(f"   Text preview: {sample_text[:200]}")
                                    except:
                                        pass
                            else:
                                total_empty += 1
                                completion_text = ""
                            
                            
                            # Clean up eos tokens at the end
                            if eos_token_id is not None:
                                while completion_ids and completion_ids[-1] == eos_token_id:
                                    completion_ids.pop()
                            
                            # Remove any embedded pad tokens (defensive)
                            completion_ids = [t for t in completion_ids if t != valid_pad]
                        else:
                            completion_ids = []
                        
                        # REMOVED: Text decoding here (huge speedup!)
                        # We only need completion_ids, not text
                        if completion_ids:
                            total_non_empty += 1
                            completion_text = ""  # Don't decode unless needed
                        else:
                            total_empty += 1
                            completion_text = ""
                        
                        all_completions[orig_index]["completions_ids"].append(completion_ids)
                        all_completions[orig_index]["completions_text"].append(completion_text)
                    
                    del result, out_seqs
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    # Only print actual errors, no verbose traceback
                    print(f"MC generation error in bucket {b_idx}: {str(e)[:100]}")
                    
                    for local_idx in range(real_batch_len):
                        orig_index = bucket_orig_indices[local_idx]
                        all_completions[orig_index]["completions_ids"].append([])
                        all_completions[orig_index]["completions_text"].append("")
        
        # Only one summary print at the end
        total_attempts = n * mc_k
        print(f"âœ… MC: {total_non_empty}/{total_attempts} completions ({100*total_non_empty/max(1,total_attempts):.1f}%)")
        
        return all_completions
    
       
    def _generate_mc_rollouts_for_intermediate_states(
        self, 
        batch: DataProto, 
        mc_k: int = 1, 
        max_new_tokens: int | None = None
    ):
        """
        Generate MC rollouts from intermediate prefixes using bucketed generation.
        """
        from copy import deepcopy
        import numpy as np
        from omegaconf import OmegaConf
        
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        vocab_size = len(self.tokenizer)
        valid_pad = pad_token_id if pad_token_id < vocab_size else 0
        
        #print(f"DEBUG: vocab_size={vocab_size}, pad_token_id={pad_token_id}, eos_token_id={eos_token_id}")
        
      
        
        # Get stop sequences
        stop_seqs = getattr(self, "_stop_sequences", None)
        if stop_seqs is None:
            try:
                stop_seqs = list(OmegaConf.select(
                    self.config,
                    "actor_rollout_ref.rollout.stop_sequences",
                    default=[]
                ) or [])
            except Exception:
                stop_seqs = []
            if not stop_seqs:
                stop_seqs = ["END_OF_OUTPUT", "\n\n"]
            if "\n\n" not in stop_seqs:
                stop_seqs = stop_seqs + ["\n\n"]
            self._stop_sequences = stop_seqs
        
        #print(f"DEBUG: stop_sequences = {stop_seqs}")
        
        # Check response quality
        #print("DEBUG: Checking response quality in batch...")
        responses = batch.batch.get("responses", None)
        terminal_count = 0
        good_count = 0
        
        if responses is not None:
            for i in range(min(5, len(batch))):
                try:
                    resp_ids = responses[i].cpu().tolist()
                    resp_ids = [t for t in resp_ids if t != pad_token_id and 0 <= t < vocab_size]
                    if len(resp_ids) == 0:
                        continue
                    resp_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                    
                    is_terminal = (
                        any(s.lower() in resp_text.lower() for s in stop_seqs) or
                        "final answer" in resp_text.lower()
                    )
                    
                    if is_terminal:
                        terminal_count += 1
                    else:
                        good_count += 1
                except Exception:
                    pass
        
        #print(f"DEBUG: {good_count} non-terminal, {terminal_count} terminal in sample")
        
        # Extract intermediate states
        states, step_boundaries = self._extract_intermediate_states(batch)
        
        if not states:
            #print("âš ï¸ No valid intermediate states")
            return [], {}
        
        self._current_step_boundaries = step_boundaries
        
        # Get worker context cap
        actor_response_length = 0
        try:
            actor_response_length = int(OmegaConf.select(
                self.config, 
                "actor_rollout_ref.rollout.response_length", 
                default=0
            ) or 0)
        except Exception:
            pass
        
        if actor_response_length > 0:
            worker_context_cap = actor_response_length
        else:
            worker_context_cap = 4096
        
        SAFETY_MARGIN = 100
        worker_context_cap = max(256, worker_context_cap - SAFETY_MARGIN)
        
        print(f"ðŸ“ Context: actor_response_length={actor_response_length}, worker_cap={worker_context_cap}")
        
        # Build prefixes
        MIN_GENERATION_TOKENS = 64
        desired_new_tokens = min(int(max_new_tokens or 512), 512)
        max_allowed_prefix = worker_context_cap - desired_new_tokens
        
        print(f"ðŸ“ Max allowed prefix: {max_allowed_prefix}, desired new tokens: {desired_new_tokens}")
        
        prefixes = []
        episode_ids = []
        batch_idxs = []
        step_idxs = []
        token_positions = []
        step_starts = []
        step_ends = []
        
        skipped_too_long = 0
        trimmed_count = 0
        
        for s in states:
            full_prefix = list(s["prefix"])
            
            # ================================================================
            # CRITICAL FIX: Remove ALL pad tokens, not just trailing ones!
            # ================================================================
            # First, remove all pad tokens from the sequence
            full_prefix = [t for t in full_prefix if t != valid_pad]
            
            # Also remove eos tokens
            if eos_token_id is not None:
                full_prefix = [t for t in full_prefix if t != eos_token_id]
            
            # Sanitize - keep only valid vocab tokens
            full_prefix = [t for t in full_prefix if 0 <= t < vocab_size]
            
            # DEBUG: Check for pad tokens after cleaning
            remaining_pads = sum(1 for t in full_prefix if t == valid_pad)
            if remaining_pads > 0:
                print(f"âš ï¸ WARNING: {remaining_pads} pad tokens remain after cleaning!")
            
            # Skip if too short
            if len(full_prefix) < 64:
                continue
            
            # Trim if too long â€” KEEP MOST RECENT TOKENS (safe for reasoning)
            if len(full_prefix) > max_allowed_prefix:
                if trimmed_count < 3:
                    print(
                        f"âœ‚ï¸ Truncating prefix from the front: "
                        f"{len(full_prefix)} -> {max_allowed_prefix} (keeping most recent context)"
                    )
                full_prefix = full_prefix[-max_allowed_prefix:]
                trimmed_count += 1

            
            if len(full_prefix) < 64:
                continue
            
            # Add continuation hint (space token)
            try:
                space_tok = self.tokenizer.encode(" ", add_special_tokens=False)
                if space_tok and all(0 <= t < vocab_size for t in space_tok):
                    full_prefix = full_prefix + space_tok[:1]
            except Exception:
                pass
            
            # ================================================================
            # FINAL VERIFICATION: No pads should exist
            # ================================================================
            final_pad_count = sum(1 for t in full_prefix if t == valid_pad)
            if final_pad_count > 0:
                print(f"âŒ CRITICAL: Prefix still has {final_pad_count} pads after all cleaning!")
                # Force remove them
                full_prefix = [t for t in full_prefix if t != valid_pad]
            
            prefixes.append(full_prefix)
            episode_ids.append(s.get("episode_id"))
            batch_idxs.append(int(s["batch_idx"]))
            step_idxs.append(int(s["step_idx"]))
            token_positions.append(s.get("token_position", 0))
            step_starts.append(s.get("step_start", 0))
            step_ends.append(s.get("step_end", 0))
        
        print(f"âœ‚ï¸ Trimmed: {trimmed_count}, Skipped too long: {skipped_too_long}")
        
        if not prefixes:
            print("âš ï¸ No valid prefixes after filtering")
            return [], step_boundaries
        
        # ================================================================
        # DEBUG: Verify no prefixes contain pads
        # ================================================================
        for i, p in enumerate(prefixes[:3]):
            pad_count = sum(1 for t in p if t == valid_pad)
            print(f"DEBUG: prefix[{i}] len={len(p)}, pad_count={pad_count}")
            if pad_count > 0:
                print(f"  âŒ Found pads at positions: {[j for j, t in enumerate(p) if t == valid_pad][:10]}...")
        
        print(f"ðŸ“Š Prepared {len(prefixes)} prefixes for bucketed generation")
        prefix_lens = [len(p) for p in prefixes]
        print(f"ðŸ“ Prefix lengths: min={min(prefix_lens)}, max={max(prefix_lens)}, mean={sum(prefix_lens)/len(prefix_lens):.0f}")
        
        # USE BUCKETED GENERATION
        completions_by_state = self._generate_mc_rollouts_bucketed(
            prefixes=prefixes,
            episode_ids=episode_ids,
            batch_idxs=batch_idxs,
            step_idxs=step_idxs,
            token_positions=token_positions,
            step_starts=step_starts,
            step_ends=step_ends,
            mc_k=mc_k,
            max_new_tokens=desired_new_tokens,
            batch=batch
        )
        
        return completions_by_state, step_boundaries


    
    def _score_mc_rollouts(self, mc_rollouts: list[dict]) -> list[dict]:
        """Score MC rollouts using puzzle_baron reward function."""
        if not mc_rollouts or (mc_rollouts and "rewards" in mc_rollouts[0]):
            return mc_rollouts
        
        scored = []
        
        for rollout in mc_rollouts:
            ep_id = str(rollout["episode_id"])
            step_idx = int(rollout["step_idx"])
            
            # Get completions
            completions = rollout.get("completions_text", [])
            if not completions and "completions_ids" in rollout:
                completions = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in rollout["completions_ids"]
                ]
            
            # Score each completion
            rewards = []
            for comp_text in completions:
                try:
                    # Your puzzle_baron scorer expects: 
                    # compute_puzzle_baron_score(data_source, solution_str, ground_truth, extra_info)
                    # We give it the completion and score it
                    reward = self.reward_fn.__self__.compute_score(
                        data_source="puzzle_baron",
                        solution_str=comp_text,
                        ground_truth="",  # Empty since we are just evaluating quality
                        extra_info={},
                    )
                    if isinstance(reward, dict):
                        rewards.append(float(reward.get("score", 0.0)))
                    else:
                        rewards.append(float(reward))
                except Exception as e:
                    print(f"Warning: MC scoring failed: {e}")
                    rewards.append(0.0)
            
            scored.append({
                "episode_id": ep_id,
                "step_idx": step_idx,
                "rewards": rewards
            })
        
        return scored

    def _process_mc_rollouts_to_values(
        self,
        mc_rollouts: list[dict],
        batch: DataProto,  # FIXED: Type hint
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """
        Process MC rollouts into value estimates.
        
        Args:
            mc_rollouts: List of dicts with 'episode_id', 'step_idx', 'completions_ids', 'completions_text'
            batch: Current training batch (DataProto)
            gamma: Discount factor
        
        Returns:
            vine_mc_values: Tensor of shape (batch_size, response_length)
        """
        # FIXED: Access tensors correctly
        responses = batch.batch["responses"]
        batch_size = responses.shape[0]
        response_length = responses.shape[1]
        
        
        
        # Initialize with zeros
        vine_mc_values = torch.zeros(batch_size, response_length, dtype=torch.float32)
        
        # Group rollouts by episode_id and step_idx
        from collections import defaultdict
        rollout_groups = defaultdict(list)
        
        for rollout in mc_rollouts:
            key = (rollout["episode_id"], rollout["step_idx"])
            rollout_groups[key].append(rollout)
        
        # Compute MC values
        for (episode_id, step_idx), group in rollout_groups.items():
            # Extract batch index from episode_id (format: "batch{i}_step{step}")
            try:
                batch_idx = int(episode_id.split("_")[0].replace("batch", ""))
            except (ValueError, IndexError):
                continue
            
            if batch_idx >= batch_size:
                continue
            
            # Compute rewards for each completion
            mc_returns = []
            for rollout in group:
                # Get completions
                completions = rollout.get("completions_text", [""])
                
                # Compute reward for each completion
                for completion_text in completions:
                    # Call your actual reward function
                    reward = self._compute_mc_reward(completion_text, batch, batch_idx)
                    mc_returns.append(reward)
            
            # Average MC returns
            if mc_returns:
                avg_mc_value = sum(mc_returns) / len(mc_returns)
                if step_idx < response_length:
                    vine_mc_values[batch_idx, step_idx] = avg_mc_value
        
        return vine_mc_values
    
    def _compute_mc_value_estimates(self, mc_sequences, data, mc_k):
        """
        Compute Monte Carlo value estimates from MC rollouts.
        Returns tensor of shape (batch_size, response_length)
        """
        import torch
        
        batch_size = mc_sequences.shape[0]
        seq_len = mc_sequences.shape[2]
        
        # Compute rewards for each MC rollout
        # This is a placeholder - implement based on your reward function
        mc_rewards = torch.zeros(batch_size, mc_k, seq_len, device=mc_sequences.device)
        
        for b in range(batch_size):
            for k in range(mc_k):
                # Compute reward for this MC rollout
                # You'll need to use your reward function here
                reward = self._compute_sequence_reward(mc_sequences[b, k], data[b])
                mc_rewards[b, k, -1] = reward  # Sparse reward at end
        
        # Average across MC samples to get value estimate
        mc_values = mc_rewards.mean(dim=1)  # (batch_size, seq_len)
        
        return mc_values
    
    
    
    
    def _compute_mc_reward(self, completion_text: str, batch: DataProto, batch_idx: int) -> float:
        """
        Compute reward for a single MC rollout completion.
        Integrates with puzzle_baron_rewarder_hybrid.
        """
        # FIXED: Access non-tensor data correctly
        try:
            # Get ground truth and extra info for this specific batch item
            data_item = batch[batch_idx]  # Returns DataProtoItem
            
            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth")
            data_source = data_item.non_tensor_batch.get("data_source", "puzzle_baron")
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            
            # Call your reward function (from puzzle_baron_rewarder_hybrid.py)
            score = self.rm_wg.compute_score(
                data_source=data_source,
                solution_str=completion_text,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if isinstance(score, dict):
                return float(score["score"])
            else:
                return float(score)
                
        except Exception as e:
            print(f"Warning: Failed to compute MC reward for batch {batch_idx}: {e}")
            return 0.0
    def _compute_vine_values(self, mc_rewards: list[dict], batch: DataProto, response_length: int, gamma: float) -> torch.Tensor:
        from collections import defaultdict
        if not mc_rewards:
            return torch.zeros((len(batch), response_length), dtype=torch.float32)
    
        grouped = defaultdict(list)
        for entry in mc_rewards:
            key = (str(entry["episode_id"]), int(entry["step_idx"]))
            grouped[key].extend(entry.get("rewards", []))
    
        batch_size = len(batch)
        vine_values = torch.zeros((batch_size, response_length), dtype=torch.float32)
    
        uid_list_raw = batch.non_tensor_batch.get("uid", [None] * batch_size)
        uid_list = [str(u) for u in uid_list_raw]  # normalize to str
        uid_to_idx = {uid: idx for idx, uid in enumerate(uid_list)}
    
        for (ep_id, step_idx), rewards in grouped.items():
            idx = uid_to_idx.get(str(ep_id), None)
            if idx is None:
                continue
            si = int(step_idx)
            if 0 <= si < response_length and len(rewards) > 0:
                vine_values[idx, si] = float(sum(rewards) / len(rewards))
    
        return vine_values


 
    def _compute_grpo_per_state_advantages(
        self,
        mc_rollouts: list,
        batch,
        mc_k: int,
        step_boundaries: dict = None,
    ):
        """
        Localized GRPO: score K completions per intermediate state,
        compute per-rollout KL, return raw dicts for global normalization
        in compute_vineppo_grpo_advantage (core_algos.py).
     
        VinePPO is NOT affected — it uses _compute_mc_values_from_rollouts instead.
     
        Returns:
            per_state_scores : {(bidx, step_idx): [score_k1, ..., score_kK]}
            per_state_range  : {(bidx, step_idx): (token_start, token_end)}
            per_state_kl     : {(bidx, step_idx): [kl_k1,    ..., kl_kK]}
                               KL_k = mean token-level log(π_θ / π_ref) for rollout k
        """
        if not mc_rollouts:
            print("⚠️  No MC rollouts for Localized GRPO computation")
            return {}, {}, {}
     
        # ── Setup ─────────────────────────────────────────────────────────────────
        device           = batch.batch["response_mask"].device
        batch_size       = batch.batch["response_mask"].shape[0]
        pad_token_id     = getattr(self.tokenizer, "pad_token_id", 0)
        eos_token_id     = getattr(self.tokenizer, "eos_token_id", None)
        responses_tensor = batch.batch.get("responses")          # (bs, T)
        orig_ntb         = batch.non_tensor_batch if hasattr(batch, "non_tensor_batch") else {}
     
        if responses_tensor is None:
            return {}, {}, {}
     
        # Reference log-probs for KL fallback (already computed by VERL per main trajectory)
        ref_log_probs = batch.batch.get("ref_log_prob", None)   # (bs, T)
        old_log_probs = batch.batch.get("old_log_probs", None)  # (bs, T)
     
        per_state_scores = {}
        per_state_range  = {}
        per_state_kl     = {}   # ← NEW
        total_completions = 0
        skipped_empty     = 0
     
        # ── Step 1: Score every completion + compute per-rollout KL ───────────────
        for row in mc_rollouts:
            bidx       = int(row.get("batch_idx", -1))
            step_idx   = int(row.get("step_idx", 0))
            step_start = int(row.get("step_start", 0))
            step_end   = int(row.get("step_end", row.get("token_position", 0)))
     
            if bidx < 0 or bidx >= batch_size:
                continue
     
            # Build prefix: response tokens up to this step
            prefix_resp_ids = responses_tensor[bidx, :step_end].cpu().tolist()
            prefix_resp_ids = [t for t in prefix_resp_ids if t != pad_token_id]
            if eos_token_id is not None:
                prefix_resp_ids = [t for t in prefix_resp_ids if t != eos_token_id]
     
            # Ground truth for reward scoring
            gt = ""
            ei = {}
            if "reward_model" in orig_ntb:
                try:
                    rm = orig_ntb["reward_model"]
                    if hasattr(rm, "__getitem__") and bidx < len(rm):
                        rm_item = rm[bidx]
                        if rm_item and isinstance(rm_item, dict):
                            gt = rm_item.get("ground_truth", "")
                            ei = rm_item.get("extra_info", {})
                except Exception:
                    pass
     
            key = (bidx, step_idx)
            per_state_range[key] = (step_start, step_end)
     
            # Get log prob lists for KL computation if stored in rollout row
            comp_log_probs     = row.get("completions_log_probs", [])
            comp_ref_log_probs = row.get("completions_ref_log_probs", [])
     
            for comp_idx, comp_ids in enumerate(row.get("completions_ids", [])):
                if not comp_ids:
                    skipped_empty += 1
                    continue
     
                # ── Score this completion ─────────────────────────────────────
                full_ids = list(prefix_resp_ids) + list(comp_ids)
                try:
                    full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
                except Exception:
                    full_text = ""
     
                score = compute_mc_partial_score(full_text, gt, ei)
                per_state_scores.setdefault(key, []).append(float(score))
                total_completions += 1
     
                # ── Per-rollout KL computation ────────────────────────────────
                # Priority 1: use per-completion log probs if stored in rollout row
                if (comp_idx < len(comp_log_probs) and
                    comp_idx < len(comp_ref_log_probs) and
                    len(comp_log_probs[comp_idx]) > 0):
     
                    lp     = torch.tensor(comp_log_probs[comp_idx],     dtype=torch.float32)
                    lp_ref = torch.tensor(comp_ref_log_probs[comp_idx], dtype=torch.float32)
                    kl_val = (lp - lp_ref).mean().item()
     
                # Priority 2: fallback — use main trajectory log probs at this step range
                elif ref_log_probs is not None and old_log_probs is not None:
                    step_lp     = old_log_probs[bidx, step_start:step_end]
                    step_lp_ref = ref_log_probs[bidx, step_start:step_end]
                    kl_val = (step_lp - step_lp_ref).mean().item()
     
                # Priority 3: no KL info available — use 0.0 (KL term dropped)
                else:
                    kl_val = 0.0
     
                per_state_kl.setdefault(key, []).append(float(kl_val))
     
        print(f"🎯 Localized GRPO: scored {total_completions} completions across "
              f"{len(per_state_scores)} states (skipped {skipped_empty} empty)")
     
        if not per_state_scores:
            return {}, {}, {}
     
        # NO local normalization here — that belongs in compute_vineppo_grpo_advantage
        return per_state_scores, per_state_range, per_state_kl
     
    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}
                

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            
            '''
            print(f"DEBUG: test_gen_batch_padded input_ids shape: {test_gen_batch_padded.batch['input_ids'].shape}")
            print(f"DEBUG: test_gen_batch_padded attention_mask shape: {test_gen_batch_padded.batch['attention_mask'].shape}")
            '''
            
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)


    

    
    
    """
    DIAGNOSTIC VERSION of _compute_mc_values_from_rollouts
    
    This version adds extensive logging to understand why all rewards are 0.
    """
    
    def _compute_mc_values_from_rollouts(
        self, 
        mc_rollouts: list, 
        batch, 
        mc_k: int,
        step_boundaries: dict = None
    ):
        """
        Score MC completions using PARTIAL TRAJECTORY SCORER.
        Propagates values to all tokens within each step.
        """
        from copy import deepcopy
        
        if not mc_rollouts:
            print("DEBUG: mc_rollouts is empty")
            return torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
        
        device = batch.batch["response_mask"].device
        batch_size = batch.batch["response_mask"].shape[0]
        response_length = batch.batch["response_mask"].shape[1]
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        
        if step_boundaries is None:
            step_boundaries = getattr(self, "_current_step_boundaries", {})
        
        responses_tensor = batch.batch.get("responses")
        prompts_tensor = batch.batch.get("prompts", batch.batch.get("input_ids"))
        orig_ntb = batch.non_tensor_batch if hasattr(batch, "non_tensor_batch") else {}
        
        if responses_tensor is None:
            print("ERROR: Batch must contain 'responses'")
            return torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
        
        # ================================================================
        # Build scoring examples
        # ================================================================
        scoring_examples = []
        skipped_empty = 0
        
        for row in mc_rollouts:
            bidx = int(row.get("batch_idx", -1))
            step_idx = int(row.get("step_idx", 0))
            step_start = int(row.get("step_start", 0))
            step_end = int(row.get("step_end", row.get("token_position", 0)))
            
            if bidx < 0 or bidx >= batch_size:
                continue
            
            # Get prefix response (response up to this step)
            prefix_resp_tensor = responses_tensor[bidx, :step_end]
            prefix_resp_ids = prefix_resp_tensor.cpu().tolist()
            prefix_resp_ids = [t for t in prefix_resp_ids if t != pad_token_id]
            if eos_token_id is not None:
                prefix_resp_ids = [t for t in prefix_resp_ids if t != eos_token_id]
            
            for comp_ids in row.get("completions_ids", []):
                if not comp_ids or len(comp_ids) == 0:
                    skipped_empty += 1
                    continue
                
                scoring_examples.append({
                    "batch_idx": bidx,
                    "step_idx": step_idx,
                    "step_start": step_start,
                    "step_end": step_end,
                    "prefix_resp_ids": prefix_resp_ids,
                    "comp_ids": comp_ids
                })
        
        print(f"DEBUG: scoring_examples count={len(scoring_examples)}, skipped_empty={skipped_empty}")
        
        if not scoring_examples:
            return torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
        
        # ================================================================
        # Score using PARTIAL TRAJECTORY SCORER (not full reward function!)
        # ================================================================
        scored_rewards = []
        
        for ex in scoring_examples:
            bidx = ex["batch_idx"]
            prefix_resp_ids = ex["prefix_resp_ids"]
            comp_ids = ex["comp_ids"]
            
            # Build full response tokens
            full_response_ids = list(prefix_resp_ids) + list(comp_ids)
            
            # Decode to text
            try:
                full_response_text = self.tokenizer.decode(full_response_ids, skip_special_tokens=True)
            except:
                full_response_text = ""
            
            # Get ground truth if available
            gt = ""
            ei = {}
            if "reward_model" in orig_ntb:
                try:
                    rm = orig_ntb["reward_model"]
                    if hasattr(rm, "__getitem__") and bidx < len(rm):
                        rm_item = rm[bidx]
                        if rm_item and isinstance(rm_item, dict):
                            gt = rm_item.get("ground_truth", "")
                            ei = rm_item.get("extra_info", {})
                except:
                    pass
            
            # Score using partial trajectory scorer
            score = compute_mc_partial_score(full_response_text, gt, ei)
            scored_rewards.append(score)
            
            
        
        # Show sample scores
        print(f"DEBUG: First 10 MC scores: {scored_rewards[:10]}")
        
        
        
        # ── NEW: Build per-state score lists for GRPO-per-State ──
        vine_grpo_scores = {}   # {(batch_idx, step_idx): [s1, s2, ..., sK]}
        vine_step_ranges = {}   # {(batch_idx, step_idx): (step_start, step_end)}
        
        for ex, r in zip(scoring_examples, scored_rewards):
            key = (ex["batch_idx"], ex["step_idx"])
            vine_grpo_scores.setdefault(key, []).append(float(r))
            vine_step_ranges[key] = (ex["step_start"], ex["step_end"])
        
        # Store on self so fit() can pick them up
        self._vine_grpo_scores = vine_grpo_scores
        self._vine_step_ranges = vine_step_ranges
        
        # ================================================================
        # Aggregate by (batch_idx, step_idx)
        # ================================================================
        reward_accumulator = {}
        step_info = {}
        
        for ex, r in zip(scoring_examples, scored_rewards):
            key = (ex["batch_idx"], ex["step_idx"])
            reward_accumulator.setdefault(key, []).append(float(r))
            step_info[key] = (ex["step_start"], ex["step_end"])
        
        if not reward_accumulator:
            return torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
        
        averaged_values = {k: sum(v) / max(1, len(v)) for k, v in reward_accumulator.items()}
        
        # Stats
        reward_values = list(averaged_values.values())
        print(f"ðŸ” REWARD STATS:")
        print(f"   Min: {min(reward_values) if reward_values else 0:.4f}")
        print(f"   Max: {max(reward_values) if reward_values else 0:.4f}")
        print(f"   Mean: {sum(reward_values)/len(reward_values) if reward_values else 0:.4f}")
        print(f"   Non-zero: {sum(1 for v in reward_values if v > 0.01)}")
        print(f"   Zero: {sum(1 for v in reward_values if v <= 0.01)}")
        
        # ================================================================
        # Propagate to vine_values tensor
        # ================================================================
        vine_values = torch.zeros((batch_size, response_length), dtype=torch.float32, device=device)
        
        positions_filled = 0
        for (bidx, step_idx), val in averaged_values.items():
            if bidx < 0 or bidx >= batch_size:
                continue
            
            step_start, step_end = step_info.get((bidx, step_idx), (0, 0))
            step_start = max(0, min(step_start, response_length - 1))
            step_end = max(step_start + 1, min(step_end, response_length))
            
            vine_values[bidx, step_start:step_end] = float(val)
            positions_filled += (step_end - step_start)
        
        coverage = 100.0 * positions_filled / max(1, batch_size * response_length)
        print(f"âœ“ MC values: {positions_filled} positions ({coverage:.1f}% coverage)")
        
        non_zero = (vine_values > 0.01).sum().item()
        print(f"âœ“ Non-zero vine_values: {non_zero}")
        
        import gc
        del scoring_examples
        del scored_rewards
        del reward_accumulator
        del averaged_values
        del step_info
        gc.collect()
        torch.cuda.empty_cache()
        
        return vine_values
        
        



    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        # ================================================================
                        # DEBUG: check rollout config right before generation
                        # ================================================================
                        #print(f"DEBUG: Checking vLLM/rollout config...")
                        '''
                        try:
                            from omegaconf import OmegaConf
                            max_prompt_len = OmegaConf.select(self.config, "data.max_prompt_length", default=None)
                            max_seq_len = OmegaConf.select(self.config, "actor_rollout_ref.rollout.max_seq_len", default=None)
                            max_model_len = OmegaConf.select(self.config, "actor_rollout_ref.rollout.max_model_len", default=None)
                            response_length = OmegaConf.select(self.config, "actor_rollout_ref.rollout.response_length", default=None)
                            prompt_length = OmegaConf.select(self.config, "actor_rollout_ref.rollout.prompt_length", default=None)
                    
                            print(f"  data.max_prompt_length: {max_prompt_len}")
                            print(f"  rollout.max_seq_len: {max_seq_len}")
                            print(f"  rollout.max_model_len: {max_model_len}")
                            print(f"  rollout.response_length: {response_length}")
                            print(f"  rollout.prompt_length: {prompt_length}")
                    
                            # RayWorkerGroup itself typically won't expose worker config here,
                            # but print what you can.
                            print(f"  actor_rollout_wg type: {type(self.actor_rollout_wg)}")
                            print(f"  actor_rollout_wg world_size: {getattr(self.actor_rollout_wg, 'world_size', None)}")
                        except Exception as e:
                            print(f"  Config check error: {e}")
                        '''
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            '''
                            print("="*60)
                            #print("DEBUG: MAIN POLICY ROLLOUT OUTPUT")
                            print("="*60)
                            responses = gen_batch_output.batch.get("responses")
                            if responses is not None:
                                for i in range(min(3, len(responses))):
                                    resp = responses[i]
                                    # Find actual content length
                                    non_pad = (resp != 151643).sum().item()  # 151643 is your pad_token_id
                                    decoded = self.tokenizer.decode(resp[:min(200, non_pad)], skip_special_tokens=True)
                                    print(f"Response {i} ({non_pad} tokens): {decoded[:300]}...")
                            '''
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    
                    '''
                    print("DEBUG: Sample policy rollout output:")
                    sample_response = batch.batch["responses"][0]
                    decoded = self.tokenizer.decode(sample_response[:200], skip_special_tokens=True)
                    print(f"  {decoded[:300]}...")
                    '''


                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                        
                        
                    # ===== VinePPO MC Value Estimation =====
                    if self.config.algorithm.adv_estimator in ("vine_ppo", "vineppo_grpo"):
                        is_validation = batch.meta_info.get("validate", False)
                    
                        if not is_validation:
                            mc_k = int(self.config.trainer.get("mc_k", 1))
                            #print(f"ðŸ”„ Generating MC rollouts at step {self.global_steps} (mc_k={mc_k})")
                    
                            try:
                                max_new_tokens = int(OmegaConf.select(
                                    self.config,
                                    "actor_rollout_ref.rollout.max_new_tokens",
                                    default=512,
                                ))
                    
                                # Generate MC rollouts (returns rollouts + step boundaries)
                                mc_rollouts, step_boundaries = self._generate_mc_rollouts_for_intermediate_states(
                                    batch=batch,
                                    mc_k=mc_k,
                                    max_new_tokens=max_new_tokens,
                                )
                    
                                if mc_rollouts and len(mc_rollouts) > 0:
                                    # Choose between MC rollout and GRPO-per-State
                                    if self.use_grpo_per_state:
                                        print(f"📊 Localized GRPO: scoring K={mc_k} rollouts per state")
                                        per_state_scores, per_state_range, per_state_kl = \
                                            self._compute_grpo_per_state_advantages(
                                                mc_rollouts=mc_rollouts,
                                                batch=batch,
                                                mc_k=mc_k,
                                                step_boundaries=step_boundaries,
                                            )
                                        # Store raw dicts — global normalization happens in compute_vineppo_grpo_advantage
                                        batch.meta_info["vine_grpo_scores"] = per_state_scores
                                        batch.meta_info["vine_step_ranges"]  = per_state_range
                                        batch.meta_info["vine_grpo_kl"]      = per_state_kl
                                        # Dummy tensor so downstream vine_mc_values checks don't raise KeyError
                                        batch.batch["vine_mc_values"] = torch.zeros_like(
                                            batch.batch["response_mask"], dtype=torch.float32
                                        )
                                    else:
                                        # VinePPO unchanged — MC value estimation (V_next - V_current)
                                        print(f"📊 VinePPO using MC Rollout (K={mc_k})")
                                        mc_values = self._compute_mc_values_from_rollouts(
                                            mc_rollouts=mc_rollouts,
                                            batch=batch,
                                            mc_k=mc_k,
                                            step_boundaries=step_boundaries,
                                        )
                                        assert mc_values.shape == batch.batch["response_mask"].shape, \
                                            "vine_mc_values shape mismatch"
                                        batch.batch["vine_mc_values"] = mc_values.to(batch.batch["response_mask"].device)
                                else:
                                    print("⚠️ MC generation returned empty - using zero vine_mc_values")
                                    batch.batch["vine_mc_values"] = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                    
                            except Exception as e:
                                print(f"âŒ ERROR in MC generation: {e}")
                                import traceback
                                traceback.print_exc()
                                batch.batch["vine_mc_values"] = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                                print("  Using fallback zero MC values")
                    
                        else:
                            print("âš ï¸ Skipping MC generation during validation")
                            batch.batch["vine_mc_values"] = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                    else:
                        batch.batch["vine_mc_values"] = None
                    # ===== VinePPO MC Value Estimation (end) =====

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # --- Ensure reward manager has correct mode for VinePPO ---
                    # If using VinePPO, disable intermediate rewards so token-level rewards
                    # remain sparse (final reward only) and are consistent with MC rollouts.
                    if self.config.algorithm.adv_estimator in ("vine_ppo", "vineppo_grpo"):
                        batch.meta_info["disable_intermediate_rewards"] = True
                    else:
                        # Clear the override for other algorithms to avoid unexpected behavior.
                        batch.meta_info.pop("disable_intermediate_rewards", None)
                    
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                    
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor
                        
                        # existing adv_kwargs
                        adv_kwargs = {
                            "token_level_rewards": batch.batch.get("token_level_rewards", None),
                            "response_mask": batch.batch.get("response_mask", None),
                            "config": self.config.algorithm,   # algorithm config expected by adv estimators
                        }
                        # add vine values if present in the batch
                        if "vine_mc_values" in batch.batch:
                            adv_kwargs["vine_mc_values"] = batch.batch["vine_mc_values"]
                        # If uid exists, forward it
                        if "uid" in batch.non_tensor_batch:
                            adv_kwargs["index"] = batch.non_tensor_batch["uid"]
                        if "reward_baselines" in batch.batch:
                            adv_kwargs["reward_baselines"] = batch.batch["reward_baselines"]
                        
                        # Localized GRPO: pass raw dicts for global normalization in core_algos.py
                        # These are only present when adv_estimator == "vineppo_grpo"
                        #if "vine_grpo_scores" in batch.non_tensor_batch:
                        #    adv_kwargs["vine_grpo_scores"] = batch.non_tensor_batch["vine_grpo_scores"]
                        #if "vine_step_ranges" in batch.non_tensor_batch:
                        #    adv_kwargs["vine_step_ranges"] = batch.non_tensor_batch["vine_step_ranges"]
                        #if "vine_grpo_kl" in batch.non_tensor_batch:
                        #    adv_kwargs["vine_grpo_kl"] = batch.non_tensor_batch["vine_grpo_kl"]
                        # Pass kl_coef so advantage function uses the same coefficient as training
                        #adv_kwargs["kl_coef"] = float(
                        #    self.config.algorithm.kl_ctrl.get("kl_coef", 0.01)
                        #    if hasattr(self.config.algorithm, "kl_ctrl")
                        #    else self.config.algorithm.get("kl_ctrl", {}).get("kl_coef", 0.01)
                        #)                        

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        #print("DEBUG batch keys:", list(batch.batch.keys()))

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        
                    '''   
                    try:
                        # show worker handles stored inside worker group, if accessible
                        first_worker = None
                        try:
                            first_worker = self.actor_rollout_wg.workers[0]
                        except Exception:
                            # some RayWorkerGroup wrappers hide .workers; try get actor names:
                            print("WARN: actor_rollout_wg has no .workers attribute; try listing named actors")
                            first_worker = None
                    
                        #print("DEBUG: actor_rollout_wg type:", type(self.actor_rollout_wg))
                        #print("DEBUG: first_worker:", first_worker)
                        if first_worker is not None:
                            # this prints the remote actor methods available (Ray ActorHandle proxies __dir__ differently)
                            #print("DEBUG: dir(first_worker):", dir(first_worker))
                            # Try a harmless remote ping if actor exposes 'check_health' or 'ping'
                            if hasattr(first_worker, "check_health"):
                                print("DEBUG: check_health ->", ray.get(first_worker.check_health.remote()))
                    except Exception as e:
                        print("DEBUG introspect error:", e)    
                    '''
                    
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
                    
                    
    
    
    def fit_restem(self):
        """Main RestEM training loop."""
        print("="*80)
        print("ðŸŽ¯ Starting RestEM Training")
        print("="*80)
        
        for epoch in range(self.config.trainer.total_epochs):
            print(f"\n{'='*80}")
            print(f"ðŸ“… RestEM Epoch {epoch+1}/{self.config.trainer.total_epochs}")
            print(f"{'='*80}")
            
            epoch_metrics = []
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                prompts = batch['prompts']
                
                print(f"\nðŸ”„ Batch {batch_idx+1}, {len(prompts)} prompts")
                
                # Run RestEM iteration
                metrics = self.restem_trainer.train_iteration(prompts)
                epoch_metrics.append(metrics)
                
                # Log metrics
                avg_success_rate = np.mean([m['success_rate'] for m in epoch_metrics])
                print(f"   ðŸ“Š Epoch avg success rate: {100*avg_success_rate:.1f}%")
            
            # Validation
            if (epoch + 1) % self.config.trainer.val_freq == 0:
                val_metrics = self.validate()
                print(f"\nâœ… Validation: {val_metrics}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.trainer.save_freq == 0:
                self.save_checkpoint(epoch)
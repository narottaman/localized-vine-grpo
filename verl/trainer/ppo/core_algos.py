# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig
from collections import defaultdict


try:
    from verl.utils import as_torch_index, group_mean_std  # use if your utils exports them
except Exception:
    # Fallbacks so this file works even if utils doesn’t expose these helpers
    def as_torch_index(index, device=None):
        if isinstance(index, torch.Tensor):
            return index.to(device=device, dtype=torch.long)
        return torch.as_tensor(index, device=device, dtype=torch.long)

    def group_mean_std(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6):
        """
        Compute per-group mean/std of x given group ids g.
        Returns: mean_g, std_g, count_g with shape [n_groups]
        """
        uniq, inv = torch.unique(g, return_inverse=True)
        n = uniq.numel()
        counts = torch.bincount(inv, minlength=n).clamp_min(1).to(x.dtype)
        sums   = torch.bincount(inv, weights=x, minlength=n)
        means  = sums / counts
        sqs    = torch.bincount(inv, weights=x * x, minlength=n)
        var    = (sqs / counts) - means * means
        stds   = torch.sqrt(torch.clamp(var, min=eps))
        return means, stds, counts


PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | AlgoConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    HYBRID_GRPO = "hybrid_grpo"  # ADD THIS LINE
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    VINEPPO_GRPO = "vineppo_grpo"
    REINFORCE_PLUS_PLUS_GDPO = "reinforce_plus_plus_gdpo"

ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]




class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages



@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT 1: REINFORCE++ (k=1, or k>1 without group baseline)
# Uses global normalization only — no group mean subtraction
# ─────────────────────────────────────────────────────────────────────────────
@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length)
    response_mask: torch.Tensor,          # (bs, response_length)
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    REINFORCE++ basic variant (k=1 or k>1 without group baseline).
 
    Paper Eq. for basic variant:
        R_i        = sum of token rewards for response i  (scalar per response)
        A_norm_i   = (R_i - mean_global(R)) / (std_global(R) + ε)
 
    Key fix vs original:
      - gamma=1.0 (flat reward, NO positional decay)
      - Global whiten operates on per-RESPONSE scalars, not per-token returns
      - Every token in a response gets the same normalized scalar
 
    Args:
        token_level_rewards: (bs, response_length) — sparse reward at EOS token
        response_mask: (bs, response_length)
        config: AlgoConfig — must have config.gamma (set to 1.0 in SLURM script)
    """
    assert config is not None
    epsilon = 1e-6
 
    with torch.no_grad():
        # ── Step 1: collapse token rewards → one scalar per response ──────────
        # With gamma=1.0 this is just the sum (reward sits at EOS position)
        scores = token_level_rewards.sum(dim=-1)           # (bs,)
 
        # ── Step 2: global normalization over the entire batch ─────────────────
        mean_global = scores.mean()
        std_global  = scores.std()
        scores_norm = (scores - mean_global) / (std_global + epsilon)  # (bs,)
 
        # ── Step 3: broadcast normalized scalar to every token in response ─────
        advantages = scores_norm.unsqueeze(-1) * response_mask         # (bs, T)
        returns    = advantages  # returns == advantages for outcome-only reward
 
    return advantages, returns
 
 
# ─────────────────────────────────────────────────────────────────────────────
# VARIANT 2: REINFORCE++ w/ Baseline (k>1)
# Two-stage: group mean subtraction → global std normalization
# ─────────────────────────────────────────────────────────────────────────────
@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE)
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length)
    response_mask: torch.Tensor,          # (bs, response_length)
    index: torch.Tensor,                  # (bs,) — prompt UID for grouping
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    REINFORCE++ w/ Baseline (k>1) — correct two-stage normalization.
 
    Paper Equations 6 & 7:
        Stage 1 (group mean subtraction, Eq. 6):
            A'_i = R_i - mean_{j in group(i)}(R_j)
 
        Stage 2 (global batch normalization, Eq. 7):
            A_norm_i = (A'_i - mean_batch(A')) / (std_batch(A') + ε)
 
    Why two stages?
        - Stage 1 removes prompt-level difficulty bias (easy vs hard prompts)
          and makes rewards scale-invariant (works for 0/1 or -1/+1 rewards)
        - Stage 2 uses global std (NOT local std like GRPO) → never collapses
          to zero even if all rollouts for one prompt get identical rewards
 
    What was WRONG in the original:
        - Used a length-weighted mean instead of simple mean for group baseline
        - Returned raw score after group subtraction with NO global normalization
        - Completely missing Stage 2 (the main contribution of the paper)
 
    Args:
        token_level_rewards: (bs, response_length) — sparse reward at EOS
        response_mask: (bs, response_length)
        index: (bs,) — prompt UID; responses with same UID form one group
        epsilon: numerical stability for std division
        config: AlgoConfig
    """
    assert config is not None
 
    with torch.no_grad():
        # ── Step 1: collapse token rewards → one scalar per response ──────────
        scores = token_level_rewards.sum(dim=-1)    # (bs,)
        bsz = scores.shape[0]
 
        # ── Step 2: Stage 1 — subtract GROUP mean per prompt ──────────────────
        # Build group → list of (global_index, score) mapping
        id2scores   = defaultdict(list)
        id2indices  = defaultdict(list)
 
        for i in range(bsz):
            uid = index[i].item() if hasattr(index[i], 'item') else index[i]
            id2scores[uid].append(scores[i])
            id2indices[uid].append(i)
 
        # Compute group means and subtract
        prime_scores = scores.clone()   # A'_i after Stage 1
        for uid, group_scores in id2scores.items():
            group_tensor = torch.stack(group_scores)         # (k,)
            group_mean   = group_tensor.mean()               # scalar
            for i in id2indices[uid]:
                prime_scores[i] = scores[i] - group_mean    # A'_i = R_i - μ_group
 
        # ── Step 3: Stage 2 — normalize by GLOBAL batch std ───────────────────
        # mean_batch(A') ≈ 0 after group subtraction, but we compute it anyway
        # for correctness (paper Eq. 7 subtracts mean_batch too)
        mean_batch   = prime_scores.mean()
        std_batch    = prime_scores.std()
        scores_norm  = (prime_scores - mean_batch) / (std_batch + epsilon)  # (bs,)
 
        # ── Step 4: broadcast to token level ──────────────────────────────────
        advantages = scores_norm.unsqueeze(-1) * response_mask    # (bs, T)
        returns    = advantages
 
    return advantages, returns
 

@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if config.tis_imp_ratio_cap > 0 and rollout_log_probs is not None:
        # Apply truncated importance sampling -> https://fengyao.notion.site/off-policy-rl
        tis_imp_ratio = torch.exp(old_log_prob - rollout_log_probs)
        tis_imp_ratio = torch.clamp(tis_imp_ratio, max=config.tis_imp_ratio_cap)
        pg_losses = pg_losses * tis_imp_ratio

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    pg_losses = -log_prob * advantages

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return pg_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio
    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
    
# ============================
# Hybrid-GRPO: GRPO + Optional Critic
# ============================
@register_adv_est("hybrid_grpo")
def compute_hybrid_grpo_advantage(
    token_level_rewards: torch.Tensor,      # (bs, T)
    response_mask: torch.Tensor,            # (bs, T)
    index: np.ndarray,                       # (bs,) - group IDs for GRPO
    values: torch.Tensor | None = None,     # (bs, T) - critic values (optional)
    gamma: float = 1.0,
    lam: float = 0.95,                       # GAE lambda for critic part
    alpha: float = 0.3,                      # Weight for critic vs GRPO (0 = pure GRPO)
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hybrid-GRPO: Combines GRPO group baseline with optional critic-based advantages.
    
    Formula: A = α * A_critic + (1 - α) * A_grpo
    
    Args:
        token_level_rewards: Per-token rewards (bs, T)
        response_mask: Valid token mask (bs, T)  
        index: Group IDs for GRPO grouping (bs,)
        values: Critic value predictions (bs, T) - optional
        gamma: Discount factor
        lam: GAE lambda (for critic component)
        alpha: Weight for critic component (0.0 = pure GRPO, 1.0 = pure critic)
        epsilon: Small value for numerical stability
        norm_adv_by_std_in_grpo: Whether to normalize GRPO advantages by std
        config: Algorithm config
    
    Returns:
        advantages: (bs, T) - Combined advantages
        returns: (bs, T) - For value function training
    """
    bs, T = token_level_rewards.shape
    device = token_level_rewards.device
    
    # Get config parameters
    if config is not None:
        alpha = float(getattr(config, "alpha", alpha))
        gamma = float(getattr(config, "gamma", gamma))
        lam = float(getattr(config, "gae_lambda", lam))
    
    with torch.no_grad():
        # ============================================================
        # Part 1: Compute GRPO advantages (group-normalized)
        # ============================================================
        scores = token_level_rewards.sum(dim=-1)  # (bs,) - total reward per sequence
        
        # Group statistics
        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}
        
        for i in range(bs):
            id2score[index[i]].append(scores[i])
        
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=device)
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor).clamp(min=epsilon)
        
        # Compute GRPO advantages
        grpo_scores = scores.clone()
        for i in range(bs):
            if norm_adv_by_std_in_grpo:
                grpo_scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                grpo_scores[i] = scores[i] - id2mean[index[i]]
        
        # Expand to token level
        A_grpo = grpo_scores.unsqueeze(-1) * response_mask  # (bs, T)
        
        # ============================================================
        # Part 2: Compute Critic-based GAE advantages (if values provided)
        # ============================================================
        if values is not None and values.shape == token_level_rewards.shape and alpha > 0:
            # Standard GAE computation
            nextvalues = torch.zeros(bs, device=device)
            lastgaelam = torch.zeros(bs, device=device)
            A_critic = torch.zeros_like(token_level_rewards)
            
            for t in reversed(range(T)):
                delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
                lastgaelam_ = delta + gamma * lam * lastgaelam
                
                # Freeze after EOS
                nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
                lastgaelam = lastgaelam_ * response_mask[:, t]
                
                A_critic[:, t] = lastgaelam
            
            # Normalize critic advantages
            A_critic = verl_F.masked_whiten(A_critic, response_mask) * response_mask
            
            # Combine: A = α * A_critic + (1 - α) * A_grpo
            advantages = alpha * A_critic + (1.0 - alpha) * A_grpo
            
            # Compute returns for critic training
            returns = (advantages + values) * response_mask
        else:
            # Pure GRPO (no critic)
            advantages = A_grpo
            returns = A_grpo  # Use advantages as returns when no critic
        
        # ============================================================
        # Final normalization
        # ============================================================
        advantages = verl_F.masked_whiten(advantages, response_mask) * response_mask
    
    return advantages, returns


# ============================
# Vine-PPO (FIXED - Paper-Compliant)
# ============================
@register_adv_est("vine_ppo")
def compute_vine_ppo_advantage(
    token_level_rewards: torch.Tensor,        # (bs, T) - rewards from POLICY rollouts only
    response_mask: torch.Tensor,              # (bs, T)
    values: torch.Tensor | None = None,       # NOT USED - VinePPO uses MC values
    gamma: float | torch.Tensor | None = None,
    lam: float | torch.Tensor | None = None,  # NOT USED - VinePPO doesn't use GAE
    config: Optional[AlgoConfig] = None,
    # ===== CRITICAL: MC value estimates from auxiliary rollouts =====
    vine_mc_values: torch.Tensor | None = None,  # (bs, T) - V_MC(s_t) for each token/step
    vine_k_rollouts: int = 4,                     # Number of MC rollouts used (for logging)
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    VinePPO advantage estimation (Paper: https://arxiv.org/abs/2410.01679)
    
    Key Formula (Eq. 3 in paper):
        A(s_t, a_t) = r(s_t, a_t) + γ * V_MC(s_{t+1}) - V_MC(s_t)
    
    CRITICAL REQUIREMENTS:
    1. vine_mc_values MUST come from separate MC rollouts (not policy rollouts)
    2. MC rollouts must NOT be in the training batch (only used for value estimation)
    3. Advantages are computed PER-STEP (group tokens by reasoning steps)
    4. Advantages MUST be normalized (zero mean, unit variance)
    
    Args:
        token_level_rewards: Rewards from POLICY rollouts (bs, T)
        response_mask: Valid token mask (bs, T)
        vine_mc_values: Monte-Carlo value estimates V_MC(s_t) from K auxiliary rollouts (bs, T)
                        Shape MUST match token_level_rewards. Each entry is the average of K rollouts.
        vine_k_rollouts: Number of MC rollouts used (for logging/debugging)
        gamma: Discount factor (default from config)
        config: Algorithm config
    
    Returns:
        advantages: (bs, T) - Normalized advantages
        returns: (bs, T) - For value function training (if using critic)
    """
    assert config is not None, "AlgoConfig required for VinePPO"
    
    # Default gamma
    if gamma is None:
        gamma = float(getattr(config, "gamma", 1.0))
    else:
        gamma = float(gamma) if not torch.is_tensor(gamma) else gamma
    
    bs, T = token_level_rewards.shape
    device = token_level_rewards.device
    
    # ============================================================
    # CRITICAL CHECK: vine_mc_values must be provided
    # ============================================================
    if vine_mc_values is None:
        raise ValueError(
            "VinePPO requires vine_mc_values (MC value estimates from auxiliary rollouts). "
            "You must run K auxiliary rollouts per intermediate state and pass the averaged V_MC(s_t). "
            "See paper Section 3.2: 'We estimate V(s) by running K auxiliary rollouts...'"
        )
    
    assert vine_mc_values.shape == token_level_rewards.shape, (
        f"vine_mc_values shape {vine_mc_values.shape} must match token_level_rewards {token_level_rewards.shape}"
    )
    
    # ============================================================
    # PAPER FORMULA: A(s_t, a_t) = r(s_t, a_t) + γ * V_MC(s_{t+1}) - V_MC(s_t)
    # ============================================================
    with torch.no_grad():
        V_next = torch.cat([vine_mc_values[:, 1:], torch.zeros(bs, 1, device=device)], dim=1)
        
        # Compute raw TD residuals (unnormalized)
        deltas_raw = token_level_rewards + gamma * V_next - vine_mc_values
        deltas_raw = deltas_raw * response_mask
        
        # Normalize ONLY for policy loss (advantages)
        advantages = verl_F.masked_whiten(deltas_raw, response_mask) * response_mask
        
        # Returns on original scale (for optional critic training)
        # Option B is correct if you're training a critic
        returns = (deltas_raw + vine_mc_values) * response_mask
    
    # Shape validation (catch bugs early)
    assert advantages.shape == (bs, T), f"advantages must be (bs, T), got {advantages.shape}"
    assert returns.shape == (bs, T), f"returns must be (bs, T), got {returns.shape}"
    
    return advantages, returns

 
@register_adv_est(AdvantageEstimator.VINEPPO_GRPO)
def compute_vineppo_grpo_advantage(
    token_level_rewards: torch.Tensor,          # (bs, T) — main policy rewards
    response_mask: torch.Tensor,                # (bs, T)
    vine_mc_values: torch.Tensor | None = None, # (bs, T) — not used here, kept for compat
    vine_grpo_scores: dict | None = None,
    # {(batch_idx, step_idx): [score_k1, score_k2, ..., score_kK]}
    vine_step_ranges: dict | None = None,
    # {(batch_idx, step_idx): (token_start, token_end)}
    vine_grpo_kl: dict | None = None,
    # {(batch_idx, step_idx): [kl_k1, kl_k2, ..., kl_kK]}
    # Per-rollout KL penalties — subtracted BEFORE global normalization
    # If None, KL subtraction is skipped (degrades to raw score normalization)
    kl_coef: float = 0.01,                      # β for KL penalty scaling
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Localized GRPO (FIXED): Per-state GRPO with global normalization.
 
    Step-by-step matching the diagram:
 
    Step 1-2: For each (batch_idx, step_idx), subtract KL from each rollout:
        A_tk = score_tk - kl_coef * kl_tk
 
    Step 3: Collect ALL A_tk values into one flat global pool.
 
    Step 4: Compute global statistics over the pool:
        μ_allstep = mean of all A_tk
        σ_allstep = std  of all A_tk
 
    Step 5: Normalize each A_tk globally:
        Â_tk = (A_tk - μ_allstep) / (σ_allstep + ε)
 
    Step 6: Average normalized advantages across K rollouts per step:
        Â_t = mean_k(Â_tk)
 
    Step 7: Broadcast Â_t to token range [t_start, t_end] of step t.
    """
    bs, T = response_mask.shape
    device = token_level_rewards.device
 
    advantages = torch.zeros(bs, T, dtype=torch.float32, device=device)
 
    if vine_grpo_scores is None or vine_step_ranges is None:
        print("WARNING: vine_grpo_scores not provided, returning zero advantages")
        return advantages, advantages
 
    with torch.no_grad():
 
        # ── Step 1-2: subtract KL penalty from each rollout score ─────────────
        # Build: adjusted_scores[(bidx, step_idx)] = [A_k1, A_k2, ..., A_kK]
        adjusted_scores = {}
 
        for (bidx, step_idx), scores in vine_grpo_scores.items():
            if bidx >= bs or not scores:
                continue
 
            kl_list = []
            if vine_grpo_kl is not None:
                kl_list = vine_grpo_kl.get((bidx, step_idx), [])
 
            # A_tk = score_tk - β * KL_tk
            # If no KL provided for a rollout, use 0.0
            adjusted = []
            for k_idx, score in enumerate(scores):
                kl_val = kl_list[k_idx] if k_idx < len(kl_list) else 0.0
                adjusted.append(score - kl_coef * kl_val)
 
            adjusted_scores[(bidx, step_idx)] = adjusted
 
        if not adjusted_scores:
            return advantages, advantages
 
        # ── Step 3: collect ALL adjusted scores into global pool ──────────────
        all_values = []
        for vals in adjusted_scores.values():
            all_values.extend(vals)
 
        all_tensor = torch.tensor(all_values, dtype=torch.float32, device=device)
 
        # ── Step 4: global statistics ─────────────────────────────────────────
        mu_allstep    = all_tensor.mean()
        sigma_allstep = all_tensor.std().clamp(min=epsilon) if norm_adv_by_std_in_grpo else torch.tensor(1.0, device=device)
 
        # ── Step 5-6: normalize globally then average per step ────────────────
        # For each step: Â_t = mean_k[(A_tk - μ) / σ]
        state_advantages = {}  # (bidx, step_idx) → scalar Â_t
 
        for (bidx, step_idx), adj_vals in adjusted_scores.items():
            adj_t = torch.tensor(adj_vals, dtype=torch.float32, device=device)
 
            # Normalize each rollout globally (Step 5)
            norm_adv_k = (adj_t - mu_allstep) / sigma_allstep   # (K,)
 
            # Average across K rollouts (Step 6)
            state_adv = norm_adv_k.mean()                        # scalar
 
            state_advantages[(bidx, step_idx)] = state_adv
 
        # ── Step 7: broadcast to token ranges ────────────────────────────────
        for (bidx, step_idx), state_adv in state_advantages.items():
            step_range = vine_step_ranges.get((bidx, step_idx))
            if step_range is None:
                continue
 
            t_start, t_end = step_range
            t_start = max(0, min(t_start, T - 1))
            t_end   = max(t_start + 1, min(t_end, T))
 
            advantages[bidx, t_start:t_end] = state_adv
 
        advantages = advantages * response_mask
 
    returns = advantages
    return advantages, returns

 


# ============================
# Helper: Step-Level Advantage Aggregation (Optional Optimization)
# ============================
def group_advantages_by_step(
    token_advantages: torch.Tensor,      # (bs, T)
    response_mask: torch.Tensor,         # (bs, T)
    step_boundaries: list[list[int]],    # List of [start, end] indices per step, per batch
) -> torch.Tensor:
    """
    OPTIONAL: Group token-level advantages into step-level advantages.
    
    Paper Section 3.3: "We group tokens by reasoning steps to reduce computation."
    
    This is an optimization - you can skip this if token-level is fast enough.
    
    Args:
        token_advantages: Token-level advantages (bs, T)
        response_mask: Valid token mask (bs, T)
        step_boundaries: List of step boundaries per batch
                         Example: [[[0, 10], [10, 25], [25, 50]], ...]  # 3 steps for batch 0
    
    Returns:
        step_advantages: (bs, T) - Same shape, but each step has uniform advantage
    """
    bs, T = token_advantages.shape
    step_advantages = torch.zeros_like(token_advantages)
    
    for b in range(bs):
        for start, end in step_boundaries[b]:
            if start >= end or end > T:
                continue
            
            # Average advantage over tokens in this step
            step_mask = response_mask[b, start:end]
            if step_mask.sum() > 0:
                step_adv = (token_advantages[b, start:end] * step_mask).sum() / step_mask.sum()
                step_advantages[b, start:end] = step_adv * step_mask
    
    return step_advantages


# ============================
# Fallback: REINFORCE-style if MC values not available
# ============================
@register_adv_est("vine_ppo_fallback")
def compute_vine_ppo_fallback(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float | torch.Tensor | None = None,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fallback when MC values are not available.
    Uses standard REINFORCE returns (not recommended for VinePPO).
    """
    print("WARNING: VinePPO MC values not provided. Falling back to REINFORCE returns.")
    
    gamma = float(gamma or getattr(config, "gamma", 1.0))
    
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running = 0
        
        for t in reversed(range(token_level_rewards.shape[1])):
            running = token_level_rewards[:, t] + gamma * running
            returns[:, t] = running
            running = running * response_mask[:, t]
        
        advantages = verl_F.masked_whiten(returns, response_mask) * response_mask
    
    return advantages, returns
    
    
    
    
# ============================
# RL^v (verifier-shaped)
# ============================
from verl.utils import torch_functional as verl_F

@register_adv_est("rlv")
def compute_rlv_advantage(
    token_level_rewards: torch.Tensor,     # (bs, T)
    response_mask: torch.Tensor,           # (bs, T)
    values: torch.Tensor | None = None,    # (bs, T) or None
    verifier_scores: torch.Tensor | None = None,  # (bs, T) or None
    config: Optional[AlgoConfig] = None,
    gamma: float = 1.0,
    lam: float = 0.95,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RLv: GAE advantages + optional verifier shaping.
    Returns (advantages, returns) each shaped (bs, T).
    """
    assert token_level_rewards.dim() == 2, f"rewards shape must be (bs,T), got {token_level_rewards.shape}"
    bs, T = token_level_rewards.shape
    device = token_level_rewards.device
    response_mask = response_mask.to(device)

    # ---- compute GAE advantages if values provided, else zeros
    if values is not None and values.shape == token_level_rewards.shape:
        with torch.no_grad():
            nextvalues = torch.zeros(bs, device=device)
            lastgaelam = torch.zeros(bs, device=device)
            adv = torch.zeros_like(token_level_rewards)

            for t in reversed(range(T)):
                delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
                lastgaelam = delta + gamma * lam * lastgaelam
                # freeze after EOS
                nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
                lastgaelam = lastgaelam * response_mask[:, t] + (1 - response_mask[:, t]) * 0
                adv[:, t] = lastgaelam

            advantages = verl_F.masked_whiten(adv, response_mask) * response_mask
            returns = (advantages + values) * response_mask
    else:
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)

    # ---- verifier shaping (optional)
    if verifier_scores is not None and verifier_scores.shape == advantages.shape:
        vcoef = 0.2
        if config is not None and hasattr(config, "verifier_coef"):
            vcoef = float(config.verifier_coef)
        vshape = verl_F.masked_whiten(verifier_scores, response_mask)
        advantages = verl_F.masked_whiten(advantages + vcoef * vshape, response_mask) * response_mask
        # keep returns consistent
        if values is not None and values.shape == advantages.shape:
            returns = (advantages + values) * response_mask

    # FINAL SHAPE CHECKS (will catch the bug you’re seeing)
    assert advantages.shape == (bs, T), f"advantages must be (bs,T), got {advantages.shape}"
    assert returns.shape == (bs, T), f"returns must be (bs,T), got {returns.shape}"
    return advantages, returns
    
    
@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_GDPO)
def compute_reinforce_plus_plus_gdpo_advantage(
    token_level_rewards: torch.Tensor,   # (bs, T) — carries combined score at EOS
    response_mask: torch.Tensor,          # (bs, T)
    reward_correctness: torch.Tensor | None = None,  # (bs,) — cell_accuracy per response
    reward_structure: torch.Tensor | None = None,    # (bs,) — conditioned structure reward
    reward_penalty: torch.Tensor | None = None,      # (bs,) — penalties
    index: np.ndarray | None = None,                 # (bs,) — prompt uid for grouping
    epsilon: float = 1e-6,
    w_correctness: float = 0.70,   # weight for cell accuracy advantage
    w_structure: float   = 0.20,   # weight for structure advantage
    w_penalty: float     = 0.10,   # weight for penalty advantage
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    REINFORCE++ with GDPO-style decoupled reward normalization.

    Instead of normalizing the combined reward, each reward component
    (correctness, structure, penalty) is normalized independently within
    its prompt group, then combined with weights.

    Conditioned structure reward: reward_structure is already zeroed when
    cell_accuracy < 0.1 (done in puzzle_baron_rewarder_hybrid.py), so the
    model cannot game format without solving the puzzle.

    Falls back to standard REINFORCE++ if individual components not provided.
    """
    assert config is not None
    gamma = config.gamma
    bs, T = token_level_rewards.shape

    # ── Fallback: no individual components → standard REINFORCE++ ──
    if reward_correctness is None or index is None:
        print("REINFORCE++GDPO: no individual rewards provided, falling back to standard REINFORCE++")
        with torch.no_grad():
            returns = torch.zeros_like(token_level_rewards)
            running_return = 0
            for t in reversed(range(T)):
                running_return = token_level_rewards[:, t] + gamma * running_return
                returns[:, t] = running_return
                running_return = running_return * response_mask[:, t]
            advantages = verl_F.masked_whiten(returns, response_mask) * response_mask
        return advantages, returns

    # ── Step 1: Compute discounted returns from combined reward ──
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0
        for t in reversed(range(T)):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            running_return = running_return * response_mask[:, t]

    # ── Step 2: GDPO decoupled normalization per reward component ──
    # Group responses by prompt uid (index), normalize each reward within group

    def gdpo_normalize(reward_vec: torch.Tensor) -> torch.Tensor:
        """
        Normalize reward_vec (bs,) within each prompt group defined by index.
        Returns normalized scalar per sample (bs,).
        """
        id2vals = defaultdict(list)
        for i in range(bs):
            id2vals[index[i]].append(reward_vec[i])

        normalized = torch.zeros(bs, dtype=torch.float32, device=reward_vec.device)
        for i in range(bs):
            uid = index[i]
            group = torch.stack(id2vals[uid])
            if len(id2vals[uid]) == 1:
                normalized[i] = 0.0   # single sample — no relative comparison
            else:
                mean_g = group.mean()
                std_g  = group.std().clamp(min=epsilon)
                normalized[i] = (reward_vec[i] - mean_g) / std_g
        return normalized

    with torch.no_grad():
        r_c = reward_correctness.to(token_level_rewards.device)
        r_s = (reward_structure if reward_structure is not None
               else torch.zeros(bs, device=token_level_rewards.device))
        r_p = (reward_penalty   if reward_penalty   is not None
               else torch.zeros(bs, device=token_level_rewards.device))

        adv_c = gdpo_normalize(r_c)   # normalized cell accuracy advantage
        adv_s = gdpo_normalize(r_s)   # normalized structure advantage
        adv_p = gdpo_normalize(r_p)   # normalized penalty advantage

        # Weighted combination
        combined_scalar = w_correctness * adv_c + w_structure * adv_s + w_penalty * adv_p

        # ── Step 3: Broadcast to token level (sparse at EOS, like REINFORCE++) ──
        # Use returns as base, then scale by the GDPO-normalized signal
        # This preserves the temporal credit assignment of REINFORCE++
        # while using GDPO's multi-reward normalization
        gdpo_returns = returns.clone()

        # Scale each sequence's returns by the combined GDPO scalar
        # (positive scalar amplifies good responses, negative suppresses bad ones)
        for i in range(bs):
            gdpo_returns[i] = returns[i] * combined_scalar[i].sign() + \
                              combined_scalar[i].unsqueeze(0) * response_mask[i]

        advantages = verl_F.masked_whiten(gdpo_returns, response_mask) * response_mask

    return advantages, gdpo_returns

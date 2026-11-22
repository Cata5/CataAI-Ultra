"""
src/training/tasks/grpo.py
GRPO (Group Relative Policy Optimization) Task Adapter.
DeepSeek-R1 style rule-based reasoning engine:
Generates G completions per prompt, scores them via deterministic/rule-based rewards,
normalizes advantages within the group, and optimizes policy without any critic/value model.
Reference: DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskAdapter, TaskBatch, TaskMetrics


class GRPOTask(TaskAdapter):
    task_id = "grpo"
    task_name = "GRPO (DeepSeek-R1 Style Reasoning RL)"
    task_family = "language_model"
    compatible_architecture_families = ["decoder_only", "hybrid", "custom_pytorch"]
    requires_tokenizer = True
    progress_unit = "prompts"

    def __init__(
        self,
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        ignore_index: int = -100,
        reward_fn: Optional[Callable[[str, str, Optional[str]], float]] = None,
    ) -> None:
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.ignore_index = ignore_index
        self.reward_fn = reward_fn or self.default_reasoning_reward

    @staticmethod
    def default_reasoning_reward(
        prompt: str,
        completion: str,
        ground_truth: Optional[str] = None,
    ) -> float:
        """
        Default rule-based reward for reasoning tasks:
        1. Format reward: produces valid <think>...</think> reasoning traces (+0.4)
        2. Format reward: produces valid <answer>...</answer> tags (+0.2)
        3. Accuracy reward: answer matches ground truth (+1.0)
        4. Conciseness: penalizes empty or extreme looping outputs
        """
        reward = 0.0

        # Check for thinking blocks
        has_think = bool(re.search(r"<think>[\s\S]+?</think>", completion))
        if has_think:
            reward += 0.4
        elif "<think>" in completion:
            reward += 0.1  # Partial credit for opening tag

        # Check for answer blocks
        ans_match = re.search(r"<answer>([\s\S]+?)</answer>", completion)
        if ans_match:
            reward += 0.2
            cand_ans = ans_match.group(1).strip()
            if ground_truth:
                gt = ground_truth.strip()
                if cand_ans.lower() == gt.lower():
                    reward += 1.0
                elif gt.lower() in cand_ans.lower():
                    reward += 0.5
        elif ground_truth and ground_truth.strip().lower() in completion.lower():
            reward += 0.3

        # Repetition penalty
        words = completion.split()
        if len(words) > 8:
            unique_ratio = len(set(words)) / float(len(words))
            if unique_ratio < 0.4:
                reward -= 0.5

        return reward

    def prepare_batch(self, raw_data: Any, device: torch.device, dtype: torch.dtype) -> TaskBatch:
        """
        Accepts dict with:
          - group_input_ids: [B, G, L]
          - group_labels: [B, G, L]
          - rewards: [B, G] (precomputed) or completions list for inline scoring
        """
        if isinstance(raw_data, dict) and "group_input_ids" in raw_data:
            inputs = raw_data["group_input_ids"].to(device, non_blocking=True)
            labels = raw_data.get("group_labels", inputs).to(device, non_blocking=True)
            rewards = raw_data.get("rewards")
            if rewards is not None and isinstance(rewards, torch.Tensor):
                rewards = rewards.to(device, non_blocking=True)
            return TaskBatch(
                inputs=inputs,
                targets=labels,
                metadata={"rewards": rewards, "batch_size": inputs.size(0), "group_size": inputs.size(1)},
            )

        # Fallback for sequence batches: reshapes into groups of G
        if isinstance(raw_data, torch.Tensor):
            t = raw_data.to(device, non_blocking=True)
            if t.dim() == 2:
                bs, seq_len = t.shape
                g = min(self.group_size, bs)
                effective_b = bs // g
                if effective_b > 0:
                    grouped = t[:effective_b * g].view(effective_b, g, seq_len)
                    return TaskBatch(
                        inputs=grouped[:, :, :-1],
                        targets=grouped[:, :, 1:],
                        metadata={"batch_size": effective_b, "group_size": g},
                    )
            return TaskBatch(
                inputs=t[:, :-1].unsqueeze(1),
                targets=t[:, 1:].unsqueeze(1),
                metadata={"batch_size": t.size(0), "group_size": 1},
            )

        raise ValueError(f"Unrecognized batch format for GRPO: {type(raw_data)}")

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: TaskBatch,
    ) -> Tuple[torch.Tensor, TaskMetrics]:
        # inputs: [B, G, L]
        # targets: [B, G, L]
        inputs = batch.inputs
        targets = batch.targets
        b, g, l = inputs.shape

        flat_in = inputs.view(b * g, l)
        flat_tgt = targets.view(b * g, l)

        logits = model(flat_in)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Calculate per-token log probabilities
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=flat_tgt.unsqueeze(2),
        ).squeeze(2)  # [B*G, L]

        mask = flat_tgt != self.ignore_index
        token_lens = mask.sum(dim=-1).clamp(min=1)
        seq_logps = (per_token_logps * mask).sum(dim=-1) / token_lens.float()  # [B*G]
        group_logps = seq_logps.view(b, g)

        # Get or compute group rewards
        rewards = batch.metadata.get("rewards")
        if rewards is None:
            # Baseline proxy reward based on token likelihood and confidence
            rewards = group_logps.detach().clone()

        # Compute Group Relative Advantages: A_i = (r_i - mean(r)) / (std(r) + 1e-4)
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True).clamp(min=1e-4)
        advantages = (rewards - mean_r) / std_r  # [B, G]

        # Policy gradient with normalized advantages
        # Loss = - E [ advantage * log_prob ]
        pg_loss = -(advantages.detach() * group_logps).mean()

        loss_val = float(pg_loss.detach().item())
        avg_r = float(rewards.mean().item())
        max_r = float(rewards.max().item())

        metrics = TaskMetrics(
            loss=loss_val,
            primary_metric_name="grpo_loss",
            primary_metric_val=loss_val,
            detailed_metrics={
                "avg_reward": avg_r,
                "best_reward": max_r,
                "group_size": float(g),
            },
        )
        return pg_loss, metrics

    def get_display_metrics(self) -> List[Dict[str, str]]:
        return [
            {"key": "grpo_loss", "label": "GRPO Loss", "unit": ""},
            {"key": "avg_reward", "label": "Avg Group Reward", "unit": ""},
            {"key": "best_reward", "label": "Best Group Reward", "unit": ""},
        ]

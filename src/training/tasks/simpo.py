"""
src/training/tasks/simpo.py
SimPO (Simple Preference Optimization) Task Adapter.
Reference-free, length-normalized preference optimization with target reward margin.
Reference: Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward" (NeurIPS 2024).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskAdapter, TaskBatch, TaskMetrics


class SimPOTask(TaskAdapter):
    task_id = "simpo"
    task_name = "SimPO (Simple Preference Optimization)"
    task_family = "language_model"
    compatible_architecture_families = ["decoder_only", "hybrid", "custom_pytorch"]
    requires_tokenizer = True
    progress_unit = "pairs"

    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.5,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def prepare_batch(self, raw_data: Any, device: torch.device, dtype: torch.dtype) -> TaskBatch:
        """
        Accepts dict with:
          - chosen_ids: [B, L_c]
          - chosen_labels: [B, L_c] (-100 on prompt tokens)
          - rejected_ids: [B, L_r]
          - rejected_labels: [B, L_r] (-100 on prompt tokens)
        Or flat tuple/list of paired tensors.
        """
        if isinstance(raw_data, dict) and "chosen_ids" in raw_data:
            chosen_ids = raw_data["chosen_ids"].to(device, non_blocking=True)
            chosen_labels = raw_data.get("chosen_labels", chosen_ids).to(device, non_blocking=True)
            rejected_ids = raw_data["rejected_ids"].to(device, non_blocking=True)
            rejected_labels = raw_data.get("rejected_labels", rejected_ids).to(device, non_blocking=True)
            return TaskBatch(
                inputs={"chosen": chosen_ids, "rejected": rejected_ids},
                targets={"chosen": chosen_labels, "rejected": rejected_labels},
                metadata={"batch_size": chosen_ids.size(0)},
            )

        # Fallback for standard sequence tokens (splits batch in two: first half chosen, second half rejected)
        if isinstance(raw_data, torch.Tensor):
            t = raw_data.to(device, non_blocking=True)
            bs = t.size(0) // 2
            if bs > 0:
                return TaskBatch(
                    inputs={"chosen": t[:bs, :-1], "rejected": t[bs:2*bs, :-1]},
                    targets={"chosen": t[:bs, 1:], "rejected": t[bs:2*bs, 1:]},
                    metadata={"batch_size": bs},
                )
            return TaskBatch(
                inputs={"chosen": t[:, :-1], "rejected": t[:, :-1]},
                targets={"chosen": t[:, 1:], "rejected": t[:, 1:]},
                metadata={"batch_size": t.size(0)},
            )

        if isinstance(raw_data, (tuple, list)) and len(raw_data) >= 2:
            c = raw_data[0].to(device, non_blocking=True)
            r = raw_data[1].to(device, non_blocking=True)
            return TaskBatch(
                inputs={"chosen": c[:, :-1], "rejected": r[:, :-1]},
                targets={"chosen": c[:, 1:], "rejected": r[:, 1:]},
                metadata={"batch_size": c.size(0)},
            )

        raise ValueError(f"Unrecognized batch format for SimPO: {type(raw_data)}")

    def _get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes sum of token log-probabilities and token counts for each sample."""
        # Shift so tokens predict the next token
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2),
        ).squeeze(2)

        loss_mask = labels != self.ignore_index
        token_counts = loss_mask.sum(dim=-1).clamp(min=1)
        sum_logps = (per_token_logps * loss_mask).sum(dim=-1)
        return sum_logps, token_counts

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: TaskBatch,
    ) -> Tuple[torch.Tensor, TaskMetrics]:
        chosen_in = batch.inputs["chosen"]
        chosen_tgt = batch.targets["chosen"]
        rejected_in = batch.inputs["rejected"]
        rejected_tgt = batch.targets["rejected"]

        # Forward passes
        chosen_logits = model(chosen_in)
        if isinstance(chosen_logits, tuple):
            chosen_logits = chosen_logits[0]

        rejected_logits = model(rejected_in)
        if isinstance(rejected_logits, tuple):
            rejected_logits = rejected_logits[0]

        chosen_logps, chosen_lens = self._get_batch_logps(chosen_logits, chosen_tgt)
        rejected_logps, rejected_lens = self._get_batch_logps(rejected_logits, rejected_tgt)

        # SimPO Length-Normalized Rewards: (beta / |y|) * sum(log pi)
        pi_chosen = chosen_logps / chosen_lens.float()
        pi_rejected = rejected_logps / rejected_lens.float()

        reward_chosen = self.beta * pi_chosen
        reward_rejected = self.beta * pi_rejected
        reward_margin = reward_chosen - reward_rejected

        # SimPO Loss: -log sigmoid( beta*(pi_w - pi_l) - gamma )
        loss = -F.logsigmoid(reward_margin - self.gamma).mean()

        # Metrics
        loss_val = float(loss.detach().item())
        acc = float((reward_margin > 0).float().mean().item())
        avg_chosen_r = float(reward_chosen.detach().mean().item())
        avg_rejected_r = float(reward_rejected.detach().mean().item())
        avg_margin = float(reward_margin.detach().mean().item())

        metrics = TaskMetrics(
            loss=loss_val,
            primary_metric_name="simpo_loss",
            primary_metric_val=loss_val,
            detailed_metrics={
                "margin": avg_margin,
                "accuracy": acc,
                "chosen_reward": avg_chosen_r,
                "rejected_reward": avg_rejected_r,
            },
        )
        return loss, metrics

    def get_display_metrics(self) -> List[Dict[str, str]]:
        return [
            {"key": "simpo_loss", "label": "SimPO Loss", "unit": ""},
            {"key": "margin", "label": "Reward Margin", "unit": ""},
            {"key": "accuracy", "label": "Chosen Win Rate", "unit": "%"},
        ]

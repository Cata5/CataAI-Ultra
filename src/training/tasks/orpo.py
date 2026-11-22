"""
src/training/tasks/orpo.py
ORPO (Odds Ratio Preference Optimization) Task Adapter.
Monolithic preference optimization combining SFT loss on chosen response with an
odds ratio penalty on rejected response without any reference model.
Reference: Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model" (2024).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskAdapter, TaskBatch, TaskMetrics


class ORPOTask(TaskAdapter):
    task_id = "orpo"
    task_name = "ORPO (Odds Ratio Preference Optimization)"
    task_family = "language_model"
    compatible_architecture_families = ["decoder_only", "hybrid", "custom_pytorch"]
    requires_tokenizer = True
    progress_unit = "pairs"

    def __init__(
        self,
        lambda_or: float = 0.1,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        self.lambda_or = lambda_or
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def prepare_batch(self, raw_data: Any, device: torch.device, dtype: torch.dtype) -> TaskBatch:
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

        raise ValueError(f"Unrecognized batch format for ORPO: {type(raw_data)}")

    def _get_logps(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2),
        ).squeeze(2)

        loss_mask = labels != self.ignore_index
        token_counts = loss_mask.sum(dim=-1).clamp(min=1)
        mean_logps = (per_token_logps * loss_mask).sum(dim=-1) / token_counts.float()
        return mean_logps, token_counts

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: TaskBatch,
    ) -> Tuple[torch.Tensor, TaskMetrics]:
        chosen_in = batch.inputs["chosen"]
        chosen_tgt = batch.targets["chosen"]
        rejected_in = batch.inputs["rejected"]
        rejected_tgt = batch.targets["rejected"]

        # Forward chosen
        chosen_logits = model(chosen_in)
        if isinstance(chosen_logits, tuple):
            chosen_logits = chosen_logits[0]

        # 1. Standard SFT loss on chosen
        sft_loss = F.cross_entropy(
            chosen_logits.view(-1, chosen_logits.size(-1)),
            chosen_tgt.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        # Forward rejected
        rejected_logits = model(rejected_in)
        if isinstance(rejected_logits, tuple):
            rejected_logits = rejected_logits[0]

        chosen_mean_logp, _ = self._get_logps(chosen_logits, chosen_tgt)
        rejected_mean_logp, _ = self._get_logps(rejected_logits, rejected_tgt)

        # Odds ratio calculation: log_odds = log(P / (1 - P)) = log_p - log(1 - exp(log_p))
        # Numerically stable formulation:
        log_odds_chosen = chosen_mean_logp - torch.log1p(-torch.exp(chosen_mean_logp).clamp(max=0.9999))
        log_odds_rejected = rejected_mean_logp - torch.log1p(-torch.exp(rejected_mean_logp).clamp(max=0.9999))

        log_odds_ratio = log_odds_chosen - log_odds_rejected
        odds_ratio_loss = -F.logsigmoid(log_odds_ratio).mean()

        total_loss = sft_loss + self.lambda_or * odds_ratio_loss

        loss_val = float(total_loss.detach().item())
        sft_val = float(sft_loss.detach().item())
        or_val = float(odds_ratio_loss.detach().item())
        acc = float((log_odds_ratio > 0).float().mean().item())

        metrics = TaskMetrics(
            loss=loss_val,
            primary_metric_name="orpo_loss",
            primary_metric_val=loss_val,
            detailed_metrics={
                "sft_loss": sft_val,
                "odds_ratio_loss": or_val,
                "accuracy": acc,
                "odds_ratio_margin": float(log_odds_ratio.detach().mean().item()),
            },
        )
        return total_loss, metrics

    def get_display_metrics(self) -> List[Dict[str, str]]:
        return [
            {"key": "orpo_loss", "label": "ORPO Loss", "unit": ""},
            {"key": "sft_loss", "label": "SFT Loss", "unit": ""},
            {"key": "accuracy", "label": "Odds Win Rate", "unit": "%"},
        ]

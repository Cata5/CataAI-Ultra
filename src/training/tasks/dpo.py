"""
src/training/tasks/dpo.py
Direct Preference Optimization (DPO) Task Adapter.
Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskAdapter, TaskBatch, TaskMetrics


class DPOTask(TaskAdapter):
    task_id = "dpo"
    task_name = "DPO (Direct Preference Optimization)"
    task_family = "language_model"
    compatible_architecture_families = ["decoder_only", "hybrid", "custom_pytorch"]
    requires_tokenizer = True
    progress_unit = "pairs"

    def __init__(
        self,
        beta: float = 0.1,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        ref_model: Optional[nn.Module] = None,
    ) -> None:
        self.beta = beta
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.ref_model = ref_model

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

        raise ValueError(f"Unrecognized batch format for DPO: {type(raw_data)}")

    def _get_logps(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2),
        ).squeeze(2)
        loss_mask = labels != self.ignore_index
        return (per_token_logps * loss_mask).sum(dim=-1)

    def forward_and_loss(
        self,
        model: nn.Module,
        batch: TaskBatch,
    ) -> Tuple[torch.Tensor, TaskMetrics]:
        chosen_in = batch.inputs["chosen"]
        chosen_tgt = batch.targets["chosen"]
        rejected_in = batch.inputs["rejected"]
        rejected_tgt = batch.targets["rejected"]

        # Policy forward pass
        chosen_logits = model(chosen_in)
        if isinstance(chosen_logits, tuple): chosen_logits = chosen_logits[0]
        rejected_logits = model(rejected_in)
        if isinstance(rejected_logits, tuple): rejected_logits = rejected_logits[0]

        policy_chosen_logps = self._get_logps(chosen_logits, chosen_tgt)
        policy_rejected_logps = self._get_logps(rejected_logits, rejected_tgt)

        # Reference forward pass (if ref_model available, otherwise self-reference with detached baseline)
        if self.ref_model is not None:
            with torch.no_grad():
                ref_chosen_logits = self.ref_model(chosen_in)
                if isinstance(ref_chosen_logits, tuple): ref_chosen_logits = ref_chosen_logits[0]
                ref_rejected_logits = self.ref_model(rejected_in)
                if isinstance(ref_rejected_logits, tuple): ref_rejected_logits = ref_rejected_logits[0]
                ref_chosen_logps = self._get_logps(ref_chosen_logits, chosen_tgt)
                ref_rejected_logps = self._get_logps(ref_rejected_logits, rejected_tgt)
        else:
            # Reference-free fallback (log ratio = 0)
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits_diff = self.beta * (pi_logratios - ref_logratios)
        losses = -F.logsigmoid(logits_diff)
        loss = losses.mean()

        loss_val = float(loss.detach().item())
        acc = float((logits_diff > 0).float().mean().item())

        metrics = TaskMetrics(
            loss=loss_val,
            primary_metric_name="dpo_loss",
            primary_metric_val=loss_val,
            detailed_metrics={
                "accuracy": acc,
                "reward_margin": float(logits_diff.detach().mean().item()),
            },
        )
        return loss, metrics

    def get_display_metrics(self) -> List[Dict[str, str]]:
        return [
            {"key": "dpo_loss", "label": "DPO Loss", "unit": ""},
            {"key": "reward_margin", "label": "Implicit Margin", "unit": ""},
            {"key": "accuracy", "label": "Chosen Accuracy", "unit": "%"},
        ]

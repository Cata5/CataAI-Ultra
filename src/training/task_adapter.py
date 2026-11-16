"""
src/training/task_adapter.py
Pluggable Task Adapter Architecture for RoCanon Studio v2.
Decouples model execution and training loss from hardcoded causal language modeling.
Supports Causal LM, Masked LM, Seq2Seq, Classification, and Regression.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTaskAdapter(ABC):
    """
    Abstract interface for training tasks.
    Encapsulates input formatting, forward pass, loss calculation, metrics, and evaluation.
    """
    task_name: str = "base_task"
    description: str = "Base task adapter interface"

    @abstractmethod
    def prepare_batch(
        self,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[Any, Any]:
        """Formats and moves inputs and targets to device."""
        pass

    @abstractmethod
    def forward_loss(
        self,
        model: nn.Module,
        inputs: Any,
        targets: Any,
        loss_fn: Optional[nn.Module] = None,
        valid_normalizer: Optional[int] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        """
        Executes forward pass and loss computation.
        Returns:
            Tuple of (loss_tensor, normalizer_count, extra_metrics_dict).
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        val_loader: Any,
        device: torch.device,
        autocast_dtype: Optional[torch.dtype] = None,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        """Evaluates model over validation data, returning metrics including 'loss'."""
        pass


class CausalLMTaskAdapter(BaseTaskAdapter):
    """Causal Autoregressive Language Modeling with chunked next-token cross-entropy."""
    task_name = "causal_lm"
    description = "Autoregressive Next-Token Prediction with causal masking"

    def prepare_batch(
        self,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            inp = batch.get("input_ids", batch.get("inputs"))
            lbl = batch.get("labels", inp)
        elif isinstance(batch, (tuple, list)):
            inp = batch[0]
            lbl = batch[1] if len(batch) > 1 and batch[1] is not None else inp
        else:
            inp = batch
            lbl = batch

        input_ids = inp.to(device=device, dtype=torch.long, non_blocking=True)
        target_ids = lbl.to(device=device, dtype=torch.long, non_blocking=True)
        return input_ids, target_ids

    def forward_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        valid_normalizer: Optional[int] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        device_type = inputs.device.type
        cast_dtype = autocast_dtype or (torch.bfloat16 if device_type == "cuda" else torch.float32)
        enable_autocast = (device_type == "cuda") or (device_type == "cpu" and cast_dtype in (torch.bfloat16, torch.float16))

        with torch.autocast(device_type=device_type, dtype=cast_dtype, enabled=enable_autocast):
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()

            if loss_fn is not None and hasattr(loss_fn, "__call__"):
                try:
                    loss, valid_tokens = loss_fn(
                        shift_logits,
                        shift_targets,
                        valid_normalizer=valid_normalizer,
                    )
                except TypeError:
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
                    valid_tokens = (shift_targets != -100).sum().item()
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )
                valid_tokens = (shift_targets != -100).sum().item()

        loss_f = float(loss.item())
        ppl = math.exp(min(20.0, loss_f)) if loss_f < 30 else float("inf")
        return loss, max(1, valid_tokens), {"loss": loss_f, "perplexity": round(ppl, 2)}

    def evaluate(
        self,
        model: nn.Module,
        val_loader: Any,
        device: torch.device,
        autocast_dtype: Optional[torch.dtype] = None,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        device_type = device.type
        cast_dtype = autocast_dtype or (torch.bfloat16 if device_type == "cuda" else torch.float32)

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if idx >= max_batches:
                    break
                inputs, targets = self.prepare_batch(batch, device=device)
                with torch.autocast(device_type=device_type, dtype=cast_dtype):
                    logits = model(inputs)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_targets = targets[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_targets.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    v_tokens = (shift_targets != -100).sum().item()
                    total_loss += float(loss.item())
                    total_tokens += v_tokens

        mean_loss = (total_loss / max(1, total_tokens)) if total_tokens > 0 else 0.0
        ppl = math.exp(min(20.0, mean_loss)) if mean_loss < 30 else float("inf")
        return {"loss": mean_loss, "perplexity": round(ppl, 2), "val_tokens": total_tokens}


class MaskedLMTaskAdapter(BaseTaskAdapter):
    """Masked Language Modeling (BERT-style) with cross-entropy over masked positions."""
    task_name = "masked_lm"
    description = "Masked Language Modeling with unmasked position loss masking"

    def prepare_batch(
        self,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            inp = batch.get("input_ids", batch.get("inputs"))
            lbl = batch.get("labels", batch.get("targets"))
        elif isinstance(batch, (tuple, list)):
            inp = batch[0]
            lbl = batch[1]
        else:
            inp = batch
            lbl = batch

        return inp.to(device=device, dtype=torch.long), lbl.to(device=device, dtype=torch.long)

    def forward_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        valid_normalizer: Optional[int] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        device_type = inputs.device.type
        cast_dtype = autocast_dtype or (torch.bfloat16 if device_type == "cuda" else torch.float32)
        enable_autocast = (device_type == "cuda") or (device_type == "cpu" and cast_dtype in (torch.bfloat16, torch.float16))

        with torch.autocast(device_type=device_type, dtype=cast_dtype, enabled=enable_autocast):
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            valid_tokens = (targets != -100).sum().item()

        return loss, max(1, valid_tokens), {"loss": float(loss.item())}

    def evaluate(
        self,
        model: nn.Module,
        val_loader: Any,
        device: torch.device,
        autocast_dtype: Optional[torch.dtype] = None,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if idx >= max_batches:
                    break
                inputs, targets = self.prepare_batch(batch, device=device)
                logits = model(inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100, reduction="sum")
                v_tokens = (targets != -100).sum().item()
                total_loss += float(loss.item())
                total_tokens += v_tokens
        mean_loss = total_loss / max(1, total_tokens)
        return {"loss": mean_loss, "val_tokens": total_tokens}


class SequenceToSequenceTaskAdapter(BaseTaskAdapter):
    """Sequence-to-Sequence translation and summarization."""
    task_name = "seq2seq"
    description = "Sequence to Sequence encoder-decoder transformation"

    def prepare_batch(
        self,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            inp = batch.get("input_ids", batch.get("inputs"))
            lbl = batch.get("labels", batch.get("targets"))
        else:
            inp, lbl = batch[0], batch[1]
        return inp.to(device=device, dtype=torch.long), lbl.to(device=device, dtype=torch.long)

    def forward_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        valid_normalizer: Optional[int] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        logits = model(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        valid_tokens = (targets != -100).sum().item()
        return loss, max(1, valid_tokens), {"loss": float(loss.item())}

    def evaluate(
        self,
        model: nn.Module,
        val_loader: Any,
        device: torch.device,
        autocast_dtype: Optional[torch.dtype] = None,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if idx >= max_batches:
                    break
                inputs, targets = self.prepare_batch(batch, device=device)
                logits = model(inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100, reduction="sum")
                v_tokens = (targets != -100).sum().item()
                total_loss += float(loss.item())
                total_tokens += v_tokens
        return {"loss": total_loss / max(1, total_tokens)}


class ClassificationTaskAdapter(BaseTaskAdapter):
    """Classification for MLPs, CNNs, and Encoders."""
    task_name = "classification"
    description = "Categorical Classification with Cross-Entropy Loss"

    def prepare_batch(
        self,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            inp = batch.get("features", batch.get("inputs", batch.get("x")))
            lbl = batch.get("labels", batch.get("targets", batch.get("y")))
        elif isinstance(batch, (tuple, list)):
            inp, lbl = batch[0], batch[1]
        else:
            raise ValueError(f"Invalid batch format for classification: {type(batch)}")

        # Features can be float tensors [B, S, D] / [B, D] or integer token IDs [B, S]
        if inp.dtype in (torch.float32, torch.float16, torch.bfloat16):
            input_tensor = inp.to(device=device, dtype=dtype)
        else:
            input_tensor = inp.to(device=device, dtype=torch.long)

        target_tensor = lbl.to(device=device, dtype=torch.long)
        return input_tensor, target_tensor

    def forward_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        valid_normalizer: Optional[int] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        device_type = inputs.device.type
        cast_dtype = autocast_dtype or (torch.bfloat16 if device_type == "cuda" else torch.float32)
        enable_autocast = (device_type == "cuda") or (device_type == "cpu" and cast_dtype in (torch.bfloat16, torch.float16))

        with torch.autocast(device_type=device_type, dtype=cast_dtype, enabled=enable_autocast):
            out = model(inputs)
            if isinstance(out, tuple):
                out = out[0]

            # Pool sequence dimension if [Batch, Seq, Classes] -> [Batch, Classes]
            if out.dim() == 3 and targets.dim() == 1:
                logits = out.mean(dim=1)
            else:
                logits = out

            loss = F.cross_entropy(logits, targets)

        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        batch_size = targets.size(0)

        return loss, batch_size, {"loss": float(loss.item()), "accuracy": round(acc, 4)}

    def evaluate(
        self,
        model: nn.Module,
        val_loader: Any,
        device: torch.device,
        autocast_dtype: Optional[torch.dtype] = None,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        device_type = device.type
        cast_dtype = autocast_dtype or (torch.bfloat16 if device_type == "cuda" else torch.float32)

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if idx >= max_batches:
                    break
                inputs, targets = self.prepare_batch(batch, device=device)
                with torch.autocast(device_type=device_type, dtype=cast_dtype):
                    out = model(inputs)
                    if isinstance(out, tuple):
                        out = out[0]
                    if out.dim() == 3 and targets.dim() == 1:
                        logits = out.mean(dim=1)
                    else:
                        logits = out
                    loss = F.cross_entropy(logits, targets, reduction="sum")
                    preds = logits.argmax(dim=-1)
                    total_correct += (preds == targets).sum().item()
                    total_loss += float(loss.item())
                    total_samples += targets.size(0)

        mean_loss = total_loss / max(1, total_samples)
        acc = total_correct / max(1, total_samples)
        return {"loss": mean_loss, "accuracy": round(acc, 4), "samples": total_samples}


class RegressionTaskAdapter(BaseTaskAdapter):
    """Continuous numerical feature regression."""
    task_name = "regression"
    description = "Continuous Feature Regression with Mean Squared Error Loss"

    def prepare_batch(
        self,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            inp = batch.get("features", batch.get("inputs", batch.get("x")))
            lbl = batch.get("targets", batch.get("labels", batch.get("y")))
        elif isinstance(batch, (tuple, list)):
            inp, lbl = batch[0], batch[1]
        else:
            raise ValueError(f"Invalid batch format for regression: {type(batch)}")

        return inp.to(device=device, dtype=dtype), lbl.to(device=device, dtype=dtype)

    def forward_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
        valid_normalizer: Optional[int] = None,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        device_type = inputs.device.type
        cast_dtype = autocast_dtype or (torch.bfloat16 if device_type == "cuda" else torch.float32)
        enable_autocast = (device_type == "cuda") or (device_type == "cpu" and cast_dtype in (torch.bfloat16, torch.float16))

        with torch.autocast(device_type=device_type, dtype=cast_dtype, enabled=enable_autocast):
            out = model(inputs)
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 3 and targets.dim() <= 2:
                preds = out.mean(dim=1)
            else:
                preds = out

            preds_flat = preds.squeeze(-1)
            targets_flat = targets.squeeze(-1)
            loss = F.mse_loss(preds_flat, targets_flat)

        mae = F.l1_loss(preds_flat, targets_flat).item()
        batch_size = targets.size(0)
        return loss, batch_size, {"loss": float(loss.item()), "mae": round(mae, 4)}

    def evaluate(
        self,
        model: nn.Module,
        val_loader: Any,
        device: torch.device,
        autocast_dtype: Optional[torch.dtype] = None,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if idx >= max_batches:
                    break
                inputs, targets = self.prepare_batch(batch, device=device)
                out = model(inputs)
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 3 and targets.dim() <= 2:
                    preds = out.mean(dim=1)
                else:
                    preds = out
                preds_flat = preds.squeeze(-1)
                targets_flat = targets.squeeze(-1)
                total_loss += float(F.mse_loss(preds_flat, targets_flat, reduction="sum").item())
                total_mae += float(F.l1_loss(preds_flat, targets_flat, reduction="sum").item())
                total_samples += targets.size(0)

        return {
            "loss": total_loss / max(1, total_samples),
            "mae": total_mae / max(1, total_samples),
            "samples": total_samples,
        }


class TaskAdapterFactory:
    """Registry and resolver for task adapters."""
    _ADAPTERS: Dict[str, BaseTaskAdapter] = {
        "causal_lm": CausalLMTaskAdapter(),
        "pretrain": CausalLMTaskAdapter(),
        "midtrain": CausalLMTaskAdapter(),
        "chat_sft": CausalLMTaskAdapter(),
        "persona": CausalLMTaskAdapter(),
        "masked_lm": MaskedLMTaskAdapter(),
        "seq2seq": SequenceToSequenceTaskAdapter(),
        "classification": ClassificationTaskAdapter(),
        "regression": RegressionTaskAdapter(),
    }

    @classmethod
    def get_adapter(cls, task_name: Optional[str] = None) -> BaseTaskAdapter:
        if not task_name:
            return cls._ADAPTERS["causal_lm"]
        key = task_name.lower().strip()
        if key in cls._ADAPTERS:
            return cls._ADAPTERS[key]
        return cls._ADAPTERS["causal_lm"]

    @classmethod
    def register_adapter(cls, name: str, adapter: BaseTaskAdapter) -> None:
        cls._ADAPTERS[name.lower().strip()] = adapter

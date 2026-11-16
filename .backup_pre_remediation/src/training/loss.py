"""
Fused and Chunked Cross-Entropy Loss implementations for RoCanon-48.5M.

Provides memory-efficient loss calculation that avoids allocating full
[B * S, V] intermediate logit tensors in GPU VRAM (saving over 1.3 GB VRAM).
Supports Liger-Kernel when available, with mathematically exact chunked
autograd fallback and token-normalized loss accumulation across micro-batches.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # type: ignore
    HAS_LIGER = True
except (ImportError, RuntimeError, TypeError):
    HAS_LIGER = False
    LigerFusedLinearCrossEntropyLoss = None


class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    """
    Autograd Function that computes cross-entropy over chunks of tokens.

    Recomputes logits chunk-by-chunk during backward pass, strictly bounding
    peak memory consumption to a single chunk of size [chunk_size, vocab_size]
    throughout both forward and backward passes.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,
        weight: Tensor,
        targets: Tensor,
        chunk_size: int,
        ignore_index: int,
        valid_normalizer: int | None,
    ) -> Tensor:
        """
        Forward pass computing cross-entropy loss over token chunks.

        Args:
            hidden: Input hidden states [N, D].
            weight: LM head projection weight [V, D].
            targets: Target token IDs [N].
            chunk_size: Maximum tokens per chunk (default: 2048).
            ignore_index: Token ID to ignore in loss calculation (default: -100).
            valid_normalizer: Optional global token count to divide by (for gradient accumulation).
        """
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        n_tokens = hidden.size(0)

        valid_mask = (targets != ignore_index)
        local_valid = valid_mask.sum().item()

        if valid_normalizer is not None and valid_normalizer > 0:
            normalizer = valid_normalizer
        else:
            normalizer = max(local_valid, 1)

        ctx.normalizer = normalizer

        # Accumulate cross entropy sum across chunks
        total_loss = torch.zeros((), device=hidden.device, dtype=torch.float32)

        for i in range(0, n_tokens, chunk_size):
            end_idx = min(i + chunk_size, n_tokens)
            h_chunk = hidden[i:end_idx]
            t_chunk = targets[i:end_idx]
            l_chunk = F.linear(h_chunk, weight).float()
            loss_chunk = F.cross_entropy(
                l_chunk,
                t_chunk,
                ignore_index=ignore_index,
                reduction="sum",
            )
            total_loss += loss_chunk

        ctx.save_for_backward(hidden, weight, targets)
        loss = total_loss / float(normalizer)
        return loss.to(dtype=hidden.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, None, None, None, None]:
        hidden, weight, targets = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        ignore_index = ctx.ignore_index
        normalizer = ctx.normalizer
        n_tokens = hidden.shape[0]

        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight)

        scale = grad_output.float() / float(normalizer)

        for i in range(0, n_tokens, chunk_size):
            end_idx = min(i + chunk_size, n_tokens)
            h_chunk = hidden[i:end_idx]
            t_chunk = targets[i:end_idx]

            # Recompute logits for this chunk
            l_chunk = F.linear(h_chunk, weight).float()
            probs = torch.softmax(l_chunk, dim=-1)

            # Gradient of cross-entropy wrt logits: probs - one_hot(target)
            mask = (t_chunk != ignore_index)
            t_clamped = t_chunk.masked_fill(~mask, 0)

            # Subtract 1.0 from true class positions
            probs.scatter_add_(
                -1,
                t_clamped.unsqueeze(-1),
                torch.where(
                    mask.unsqueeze(-1),
                    torch.tensor(-1.0, device=probs.device, dtype=probs.dtype),
                    torch.tensor(0.0, device=probs.device, dtype=probs.dtype),
                ),
            )
            probs.masked_fill_(~mask.unsqueeze(-1), 0.0)

            d_logits = (probs * scale).to(dtype=hidden.dtype)

            # Accumulate gradients into hidden states and linear projection weight
            grad_hidden[i:end_idx] = d_logits @ weight
            grad_weight.addmm_(d_logits.T, h_chunk)

        return grad_hidden, grad_weight, None, None, None, None


class ChunkedCrossEntropyLoss(nn.Module):
    """
    Chunked Cross-Entropy Loss module.

    Computes cross-entropy over token chunks (default chunk size: 2048)
    to prevent materializing large [B * S, V] logits in GPU memory.
    """

    def __init__(
        self,
        chunk_size: int = 2048,
        ignore_index: int = -100,
        prefer_liger: bool = False,
    ) -> None:
        """
        Args:
            chunk_size: Maximum tokens per chunk (default: 2048).
            ignore_index: Target token ID to ignore (default: -100).
            prefer_liger: Whether to use Liger kernel if installed.
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        self.prefer_liger = prefer_liger and HAS_LIGER

        if self.prefer_liger:
            self.liger_loss = LigerFusedLinearCrossEntropyLoss(
                ignore_index=ignore_index,
                reduction="sum",
            )
        else:
            self.liger_loss = None

    def forward(
        self,
        hidden_or_logits: Tensor,
        targets: Tensor,
        weight: Tensor | None = None,
        valid_normalizer: int | None = None,
        shift_labels: bool = False,
    ) -> tuple[Tensor, int]:
        """
        Computes chunked cross-entropy loss.

        Args:
            hidden_or_logits: Hidden states [B, S, D] / [N, D] if weight is provided,
                              or Logits [B, S, V] / [N, V] if weight is None.
            targets: Target labels [B, S] or [N].
            weight: LM head linear weight [V, D] (optional).
            valid_normalizer: Optional cumulative valid token normalizer across micro-batches.
            shift_labels: Whether to shift inputs and targets for causal language modeling.

        Returns:
            Tuple of (scalar_loss_tensor, num_valid_tokens_int).
        """
        # Handle shifting for causal LM
        if shift_labels:
            if hidden_or_logits.ndim == 3:
                hidden_or_logits = hidden_or_logits[:, :-1, :].contiguous()
            elif hidden_or_logits.ndim == 2 and weight is None:
                pass  # Already flattened or 2D
            if targets.ndim == 2:
                targets = targets[:, 1:].contiguous()

        # Flatten sequence dimensions
        if hidden_or_logits.ndim == 3:
            hidden = hidden_or_logits.reshape(-1, hidden_or_logits.size(-1))
        else:
            hidden = hidden_or_logits.contiguous()

        targets_flat = targets.reshape(-1)
        valid_tokens = int((targets_flat != self.ignore_index).sum().item())

        if valid_tokens == 0:
            # Avoid division by zero if all tokens are ignored
            dummy_loss = (hidden.sum() * 0.0)
            return dummy_loss, 0

        # Case 1: Projection weight provided -> Use memory-efficient chunked linear cross entropy
        if weight is not None:
            if self.prefer_liger and self.liger_loss is not None and hidden.device.type == "cuda":
                # Liger fused linear CE expects [N, D] and [N]
                loss_sum = self.liger_loss(hidden, weight, targets_flat)
                norm = valid_normalizer if (valid_normalizer is not None and valid_normalizer > 0) else valid_tokens
                return loss_sum / float(norm), valid_tokens

            loss = ChunkedLinearCrossEntropyFunction.apply(
                hidden,
                weight,
                targets_flat,
                self.chunk_size,
                self.ignore_index,
                valid_normalizer,
            )
            return loss, valid_tokens

        # Case 2: Logits provided directly -> Chunk over sequence dimension
        n_tokens = hidden.size(0)
        norm = valid_normalizer if (valid_normalizer is not None and valid_normalizer > 0) else valid_tokens

        total_loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for i in range(0, n_tokens, self.chunk_size):
            end_idx = min(i + self.chunk_size, n_tokens)
            l_chunk = hidden[i:end_idx]
            t_chunk = targets_flat[i:end_idx]
            chunk_loss = F.cross_entropy(
                l_chunk,
                t_chunk,
                ignore_index=self.ignore_index,
                reduction="sum",
            )
            total_loss = total_loss + chunk_loss

        final_loss = (total_loss / float(norm)).to(dtype=hidden.dtype)
        return final_loss, valid_tokens


class TokenLossAccumulator:
    """
    Accumulator for tracking loss sums and valid token counts across
    micro-batches during gradient accumulation.
    """

    def __init__(self) -> None:
        self.total_loss_sum: float = 0.0
        self.total_valid_tokens: int = 0
        self.micro_batch_count: int = 0

    def reset(self) -> float:
        val = self.mean_loss
        self.total_loss_sum = 0.0
        self.total_valid_tokens = 0
        self.micro_batch_count = 0
        return val

    def add(self, loss_sum_value: float, valid_tokens: int) -> None:
        """Adds a micro-batch's loss contribution and valid token count."""
        self.total_loss_sum += float(loss_sum_value)
        self.total_valid_tokens += int(valid_tokens)
        self.micro_batch_count += 1

    @property
    def mean_loss(self) -> float:
        """Computes true global token-normalized loss: sum(L) / sum(T_valid)."""
        if self.total_valid_tokens == 0:
            return 0.0
        return self.total_loss_sum / float(self.total_valid_tokens)

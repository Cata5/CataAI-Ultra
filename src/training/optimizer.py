"""
Hybrid Optimizer implementation for RoCanon-48.5M.

Partitions model parameters into:
1. Muon: 112 2D hidden layer linear weight matrices (41,943,040 params, 86.42%)
   optimized with Quintic Newton-Schulz orthogonalization (lr=3e-3, wd=0.01, momentum=0.95, nesterov=True).
2. AdamW: Remaining 6,588,928 params (13.58%):
   - Tied token embeddings: wd = 0.01
   - 1D RMSNorm scale parameters: wd = 0.0
   - 3D Canon Conv1d filters: wd = 0.0
   (lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=True on CUDA).
"""

import collections.abc
import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def batched_zeropower_via_newtonschulz(
    grad_stack: Tensor,
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    """
    Batched Quintic Newton-Schulz iteration across a stack of 2D matrices [B, M, N].
    Accelerates Muon by grouping same-shape weight matrices into batched GEMM operations.
    """
    a, b, c = ns_coefficients
    ortho_grad = grad_stack.bfloat16() if grad_stack.device.type == "cuda" else grad_stack.float()
    transposed = ortho_grad.size(1) > ortho_grad.size(2)
    if transposed:
        ortho_grad = ortho_grad.transpose(1, 2)

    norm = ortho_grad.norm(dim=(-2, -1), keepdim=True)
    ortho_grad = ortho_grad / norm.clamp(min=eps)

    for _ in range(ns_steps):
        gram_matrix = torch.bmm(ortho_grad, ortho_grad.transpose(1, 2))
        gram_update = torch.baddbmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.baddbmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if transposed:
        ortho_grad = ortho_grad.transpose(1, 2)

    return ortho_grad.to(dtype=grad_stack.dtype)


def zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    """
    Quintic Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k
    where a=3.4445, b=-4.775, c=2.0315 maximize the derivative at zero.

    Args:
        grad: 2D gradient tensor [M, N].
        ns_coefficients: Quintic polynomial coefficients (a, b, c).
        ns_steps: Number of polynomial iterations (default: 5).
        eps: Small epsilon for spectral norm stabilization.

    Returns:
        Orthogonalized gradient tensor with the same shape as grad.
    """
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if grad.ndim != 2:
        raise ValueError(f"Input tensor gradient must be a 2D matrix, got ndim={grad.ndim}")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")

    a, b, c = ns_coefficients
    # Compute in bfloat16 or float32 matching device capability
    ortho_grad = grad.bfloat16() if grad.device.type == "cuda" else grad.float()
    transposed = ortho_grad.size(0) > ortho_grad.size(1)
    if transposed:
        ortho_grad = ortho_grad.T

    # Ensure spectral norm is at most 1
    norm = ortho_grad.norm()
    ortho_grad = ortho_grad / norm.clamp(min=eps)

    # Perform quintic Newton-Schulz iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if transposed:
        ortho_grad = ortho_grad.T

    return ortho_grad.to(dtype=grad.dtype)


def adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """
    Aspect-ratio learning rate scaling for rectangular 2D weight matrices in Muon.

    Args:
        lr: Base learning rate.
        adjust_lr_fn: Method name ('original', 'match_rms_adamw', or None).
        param_shape: Shape of 2D parameter tensor (A, B).

    Returns:
        Adjusted learning rate.
    """
    a, b = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1, a / b))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(a, b))
    else:
        ratio = 1.0
    return lr * ratio


class Muon(Optimizer):
    """
    Mathematically authentic Muon (MomentUm Orthogonalized by Newton-schulz) Optimizer.

    Features:
    - Strictly enforces 2D matrices (rejects 1D vectors and 3D tensors).
    - Nesterov momentum with decoupled weight decay.
    - Quintic Newton-Schulz iteration (k=5) with Keller Jordan's coefficients (3.4445, -4.775, 2.0315).
    - Aspect-ratio learning rate adjustment: lr * sqrt(max(1, A/B)).
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, object]],
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        ns_steps: int = 5,
        eps: float = 1e-7,
        adjust_lr_fn: str | None = "original",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "ns_steps": ns_steps,
            "eps": eps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: collections.abc.Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            adjust_lr_fn = group["adjust_lr_fn"]

            # Group parameters by shape for batched Newton-Schulz
            shape_groups: dict[tuple[int, int], list[Tensor]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    raise ValueError(
                        f"Muon strictly requires 2D weight matrices! Received tensor with shape {p.shape} (ndim={p.grad.ndim})"
                    )
                shape_groups.setdefault(tuple(p.shape), []).append(p)

            for shape, p_list in shape_groups.items():
                adj_lr = adjust_lr(lr, adjust_lr_fn, p_list[0].shape)
                updates = []
                for p in p_list:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    buf.lerp_(p.grad, 1.0 - momentum)
                    u = p.grad.lerp(buf, momentum) if nesterov else buf
                    updates.append(u)

                if len(updates) > 1 and updates[0].is_cuda:
                    ortho_updates = batched_zeropower_via_newtonschulz(
                        torch.stack(updates),
                        ns_coefficients=ns_coefficients,
                        ns_steps=ns_steps,
                        eps=eps,
                    )
                    for i, p in enumerate(p_list):
                        if weight_decay != 0.0:
                            p.mul_(1.0 - lr * weight_decay)
                        p.add_(ortho_updates[i], alpha=-adj_lr)
                else:
                    for i, p in enumerate(p_list):
                        ortho = zeropower_via_newtonschulz(
                            updates[i],
                            ns_coefficients=ns_coefficients,
                            ns_steps=ns_steps,
                            eps=eps,
                        )
                        if weight_decay != 0.0:
                            p.mul_(1.0 - lr * weight_decay)
                        p.add_(ortho, alpha=-adj_lr)

        return loss


def partition_parameters(
    model: nn.Module,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """
    Partitions model parameters into:
    1. muon_params: 2D hidden layer linear weight matrices (41,943,040 params in RoCanon-48.5M).
    2. adamw_decay_params: Tied token embeddings (6,291,456 params, wd=0.01).
    3. adamw_no_decay_params: 1D RMSNorm scales (18,944 params, wd=0.0) and
       3D Canon Conv1d filters (278,528 params, wd=0.0).

    Args:
        model: PyTorch model instance (e.g. RoCanonForCausalLM).

    Returns:
        Tuple of (muon_params, adamw_decay_params, adamw_no_decay_params).
    """
    muon_params: list[Tensor] = []
    adamw_decay_params: list[Tensor] = []
    adamw_no_decay_params: list[Tensor] = []

    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Avoid double-counting tied weights (e.g. lm_head.weight == embed_tokens.weight)
        if id(param) in seen:
            continue
        seen.add(id(param))

        # Check for token embedding / lm_head
        if "embed" in name or "lm_head" in name:
            adamw_decay_params.append(param)
        elif param.ndim == 1:
            # RMSNorm scales, 1D biases
            adamw_no_decay_params.append(param)
        elif param.ndim == 3 or "canon" in name:
            # Canon depthwise 1D conv filters [C, 1, K]
            adamw_no_decay_params.append(param)
        elif param.ndim == 2:
            # 2D hidden layer linear projections (W_Q, W_K, W_V, W_O, W_gate, W_up, W_down)
            muon_params.append(param)
        else:
            # Fallback for unexpected shapes
            adamw_no_decay_params.append(param)

    return muon_params, adamw_decay_params, adamw_no_decay_params


def configure_optimizers(
    model: nn.Module,
    muon_lr: float = 3e-3,
    muon_wd: float = 0.01,
    adamw_lr: float = 6e-4,
    adamw_wd: float = 0.01,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    fused: bool = True,
) -> tuple[Optimizer, Optimizer]:
    """
    Configures and instantiates the hybrid (Muon + AdamW) optimizer pair for RoCanon-48.5M.

    - Muon: 112 2D hidden linear weights totaling exactly 41,943,040 parameters (86.42%).
      Uses torch.optim.Muon if available, with automatic fallback to authentic Muon implementation.
    - AdamW: non-2D hidden parameters totaling 6,588,928 parameters (13.58%):
      * Embeddings: weight_decay = adamw_wd (0.01)
      * RMSNorm scales: weight_decay = 0.0
      * Canon Conv1d filters: weight_decay = 0.0
      Uses fused=True on CUDA when supported, with graceful fallback to CPU AdamW.

    Args:
        model: Target PyTorch model (e.g. RoCanonForCausalLM).
        muon_lr: Learning rate for Muon (default: 3e-3).
        muon_wd: Weight decay for Muon (default: 0.01, overriding PyTorch default 0.1).
        adamw_lr: Learning rate for AdamW (default: 6e-4).
        adamw_wd: Weight decay for token embeddings (default: 0.01).
        adamw_betas: Betas for AdamW (default: (0.9, 0.95), beta2=0.95 not 0.999).
        adamw_eps: Epsilon for AdamW numerical stability (default: 1e-8).
        fused: Whether to attempt fused AdamW on CUDA (default: True).

    Returns:
        Tuple of (muon_optimizer, adamw_optimizer).
    """
    muon_params, adamw_decay_params, adamw_no_decay_params = partition_parameters(model)

    # Compute parameter counts
    muon_count = sum(p.numel() for p in muon_params)
    adamw_decay_count = sum(p.numel() for p in adamw_decay_params)
    adamw_no_decay_count = sum(p.numel() for p in adamw_no_decay_params)
    adamw_total_count = adamw_decay_count + adamw_no_decay_count
    total_count = muon_count + adamw_total_count

    # If initializing on standard RoCanon-48.5M architecture, enforce strict parameter counts
    if total_count == 48531968:
        assert muon_count == 41943040, (
            f"Muon parameter count mismatch: expected 41,943,040, got {muon_count}"
        )
        assert adamw_total_count == 6588928, (
            f"AdamW parameter count mismatch: expected 6,588,928, got {adamw_total_count}"
        )
        assert total_count == 48531968, (
            f"Total parameter count mismatch: expected 48,531,968, got {total_count}"
        )

    # 1. Instantiate Muon optimizer
    if hasattr(torch.optim, "Muon"):
        try:
            muon_optimizer: Optimizer = torch.optim.Muon(
                muon_params,
                lr=muon_lr,
                weight_decay=muon_wd,
                momentum=0.95,
                nesterov=True,
                ns_coefficients=(3.4445, -4.775, 2.0315),
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn="original",
            )
        except (RuntimeError, ValueError, TypeError):
            muon_optimizer = Muon(
                muon_params,
                lr=muon_lr,
                weight_decay=muon_wd,
                momentum=0.95,
                nesterov=True,
                ns_coefficients=(3.4445, -4.775, 2.0315),
                ns_steps=5,
                eps=1e-7,
                adjust_lr_fn="original",
            )
    else:
        muon_optimizer = Muon(
            muon_params,
            lr=muon_lr,
            weight_decay=muon_wd,
            momentum=0.95,
            nesterov=True,
            ns_coefficients=(3.4445, -4.775, 2.0315),
            ns_steps=5,
            eps=1e-7,
            adjust_lr_fn="original",
        )

    # 2. Instantiate AdamW optimizer with differentiated weight decays
    adamw_groups = [
        {"params": adamw_decay_params, "weight_decay": adamw_wd},
        {"params": adamw_no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = fused and torch.cuda.is_available()
    if use_fused:
        try:
            adamw_optimizer: Optimizer = torch.optim.AdamW(
                adamw_groups,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                fused=True,
            )
        except (RuntimeError, ValueError, TypeError):
            adamw_optimizer = torch.optim.AdamW(
                adamw_groups,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                fused=False,
            )
    else:
        adamw_optimizer = torch.optim.AdamW(
            adamw_groups,
            lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            fused=False,
        )

    return muon_optimizer, adamw_optimizer


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "muon_adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    momentum: float = 0.9,
    eps: float = 1e-8,
    fused: bool = False,
    muon_lr: float = 3e-3,
    muon_wd: float = 0.01,
) -> tuple[Optimizer, ...]:
    """
    Universal optimizer factory supporting:
    - muon_adamw: Dual hybrid optimizer (Muon on 2D weights, AdamW on 1D/embeddings)
    - adamw: PyTorch AdamW with decay/no-decay parameter grouping
    - adam: PyTorch Adam
    - sgd: PyTorch SGD with momentum
    - rmsprop: PyTorch RMSprop
    - adagrad: PyTorch Adagrad
    """
    opt_name = optimizer_type.lower()

    if opt_name == "muon_adamw":
        return configure_optimizers(
            model=model,
            muon_lr=muon_lr,
            muon_wd=muon_wd,
            adamw_lr=lr,
            adamw_wd=weight_decay,
            adamw_betas=betas,
            adamw_eps=eps,
            fused=fused,
        )

    # Standard decay vs no-decay parameter grouping
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and not any(nd in name.lower() for nd in ("norm", "bias", "embedding")):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if opt_name == "adamw":
        use_fused = fused and torch.cuda.is_available()
        try:
            opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=use_fused)
        except Exception:
            opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=False)
        return (opt,)

    elif opt_name == "adam":
        return (torch.optim.Adam(groups, lr=lr, betas=betas, eps=eps),)

    elif opt_name == "sgd":
        return (torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=(momentum > 0)),)

    elif opt_name == "rmsprop":
        return (torch.optim.RMSprop(groups, lr=lr, eps=eps, momentum=momentum),)

    elif opt_name == "adagrad":
        return (torch.optim.Adagrad(groups, lr=lr, eps=eps),)

    # Default fallback
    return (torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps),)

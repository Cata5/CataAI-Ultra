"""
RoCanon-48.5M Training Engine and Optimization Module.

Exports:
- configure_optimizers: Hybrid Muon + AdamW parameter partition and configuration
- Muon: Authentic Zeroth-power orthogonalization optimizer
- WSDScheduler: Warmup-Stable-Decay learning rate scheduler
- StageBatchScheduler: Stage-wise gradient accumulation and context size manager
- clip_gradient_norm: Global gradient clipping utility at max_norm=1.0
- ChunkedCrossEntropyLoss: Chunked / fused cross-entropy with token normalization
- TokenLossAccumulator: Multi-microbatch token loss tracker
- Trainer: Mixed-precision training engine
- TrainingConfig: Hyperparameter configuration dataclass
"""

from .loss import (
    ChunkedCrossEntropyLoss,
    ChunkedLinearCrossEntropyFunction,
    TokenLossAccumulator,
)
from .optimizer import (
    Muon,
    configure_optimizers,
    partition_parameters,
    zeropower_via_newtonschulz,
)
from .scheduler import (
    StageBatchScheduler,
    WSDScheduler,
    clip_gradient_norm,
)
from .trainer import (
    Trainer,
    TrainingConfig,
)

__all__ = [
    "ChunkedCrossEntropyLoss",
    "ChunkedLinearCrossEntropyFunction",
    "Muon",
    "StageBatchScheduler",
    "TokenLossAccumulator",
    "Trainer",
    "TrainingConfig",
    "WSDScheduler",
    "clip_gradient_norm",
    "configure_optimizers",
    "partition_parameters",
    "zeropower_via_newtonschulz",
]

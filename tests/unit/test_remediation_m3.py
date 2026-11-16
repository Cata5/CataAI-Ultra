"""
tests/unit/test_remediation_m3.py
Automated acceptance tests for Milestone 3:
- BUG-05: Task Adapter Architecture (Causal LM, Masked LM, Seq2Seq, Classification, Regression)
- BUG-09: Generic Architecture & Model Factory Training (RoCanon, Custom Transformer, MLP, CNN)
"""

import pytest
import torch
import torch.nn as nn

from src.networks.factory import ModelFactory
from src.networks.registry import NetworkRegistry
from src.training.task_adapter import (
    BaseTaskAdapter,
    CausalLMTaskAdapter,
    ClassificationTaskAdapter,
    MaskedLMTaskAdapter,
    RegressionTaskAdapter,
    SequenceToSequenceTaskAdapter,
    TaskAdapterFactory,
)
from src.training.trainer import Trainer, TrainingConfig


class TestMilestone3TaskAdapters:
    def test_task_adapter_factory(self):
        """TaskAdapterFactory resolves all required task types."""
        tasks = ["causal_lm", "masked_lm", "seq2seq", "classification", "regression"]
        for t in tasks:
            adapter = TaskAdapterFactory.get_adapter(t)
            assert adapter is not None
            assert isinstance(adapter, BaseTaskAdapter)

    def test_causal_lm_adapter_forward_loss(self):
        """CausalLMTaskAdapter prepares shifted tokens and chunked cross-entropy."""
        adapter = CausalLMTaskAdapter()
        vocab_size = 128
        model = nn.Sequential(nn.Embedding(vocab_size, 32), nn.Linear(32, vocab_size))

        tokens = torch.randint(0, vocab_size, (2, 16))
        inputs, targets = adapter.prepare_batch((tokens, tokens), device=torch.device("cpu"))
        loss, valid_tokens, metrics = adapter.forward_loss(model, inputs, targets)

        assert loss > 0
        assert valid_tokens == 2 * 15
        assert "perplexity" in metrics

    def test_classification_adapter_forward_loss(self):
        """ClassificationTaskAdapter computes cross-entropy and accuracy on logits."""
        adapter = ClassificationTaskAdapter()
        num_classes = 4
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, num_classes))

        x = torch.randn(8, 16)
        y = torch.randint(0, num_classes, (8,))

        inputs, targets = adapter.prepare_batch((x, y), device=torch.device("cpu"))
        loss, count, metrics = adapter.forward_loss(model, inputs, targets)

        assert loss > 0
        assert count == 8
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_regression_adapter_forward_loss(self):
        """RegressionTaskAdapter computes MSE and MAE on continuous values."""
        adapter = RegressionTaskAdapter()
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))

        x = torch.randn(4, 8)
        y = torch.randn(4, 1)

        inputs, targets = adapter.prepare_batch((x, y), device=torch.device("cpu"))
        loss, count, metrics = adapter.forward_loss(model, inputs, targets)

        assert loss >= 0
        assert count == 4
        assert "mae" in metrics

    def test_masked_lm_adapter_forward_loss(self):
        """MaskedLMTaskAdapter computes loss over unmasked positions (-100 ignored)."""
        adapter = MaskedLMTaskAdapter()
        vocab_size = 64
        model = nn.Sequential(nn.Embedding(vocab_size, 16), nn.Linear(16, vocab_size))

        x = torch.randint(0, vocab_size, (2, 8))
        y = x.clone()
        y[:, :4] = -100  # unmasked tokens ignored

        inputs, targets = adapter.prepare_batch((x, y), device=torch.device("cpu"))
        loss, valid_tokens, metrics = adapter.forward_loss(model, inputs, targets)

        assert loss > 0
        assert valid_tokens == 2 * 4

    def test_mlp_classification_training_step(self):
        """Deep MLP 5M template successfully steps through Trainer with ClassificationTaskAdapter."""
        reg = NetworkRegistry()
        cfg = reg.get_network("deep_mlp_5m")
        assert cfg is not None
        model = ModelFactory.create(dict(cfg))

        adapter = ClassificationTaskAdapter()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizers=(opt, opt),
            config=TrainingConfig(device="cpu", dtype=torch.float32, micro_batch_size=4, seq_len=16),
            task_adapter=adapter,
        )

        input_dim = int(cfg.get("input_dim", 64))
        num_classes = int(cfg.get("output_head", {}).get("num_classes", 2))

        batch_x = torch.randn(4, 16, input_dim)
        batch_y = torch.randint(0, num_classes, (4,))

        loss_val, count = trainer.accumulate_micro_batch(input_ids=batch_x, labels=batch_y)
        assert loss_val > 0
        assert count == 4

        grad_norm = trainer.optimizer_step()
        assert grad_norm >= 0

    def test_cnn_classification_training_step(self):
        """Conv1D Network 4M template successfully steps through Trainer with ClassificationTaskAdapter."""
        reg = NetworkRegistry()
        cfg = reg.get_network("conv1d_network_4m")
        assert cfg is not None
        model = ModelFactory.create(dict(cfg))

        adapter = ClassificationTaskAdapter()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizers=(opt, opt),
            config=TrainingConfig(device="cpu", dtype=torch.float32, micro_batch_size=4, seq_len=16),
            task_adapter=adapter,
        )

        input_dim = int(cfg.get("input_dim", 64))
        num_classes = int(cfg.get("output_head", {}).get("num_classes", 2))

        batch_x = torch.randn(4, 16, input_dim)
        batch_y = torch.randint(0, num_classes, (4,))

        loss_val, count = trainer.accumulate_micro_batch(input_ids=batch_x, labels=batch_y)
        assert loss_val > 0
        assert count == 4

        grad_norm = trainer.optimizer_step()
        assert grad_norm >= 0

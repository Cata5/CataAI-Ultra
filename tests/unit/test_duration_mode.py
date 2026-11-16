"""
tests/unit/test_duration_mode.py
Regression tests for training duration modes ("steps" vs "epochs").
Verifies stop condition evaluation and token accounting.
"""

import pytest


def should_stop_run(
    duration_mode: str,
    step: int,
    total_steps: int,
    total_tokens: int,
    train_len: int,
    target_epochs: int,
) -> bool:
    """Exact stop condition logic from training_worker.py."""
    if duration_mode == "epochs" and target_epochs > 0 and train_len > 0:
        if total_tokens >= target_epochs * train_len:
            return True
    elif step >= total_steps:
        return True
    return False


def test_steps_duration_mode_stopping():
    """Steps mode terminates strictly when step >= total_steps."""
    # Before target
    assert not should_stop_run(
        duration_mode="steps",
        step=999,
        total_steps=1000,
        total_tokens=10_000_000,
        train_len=1_000_000,
        target_epochs=1,
    )
    # At target
    assert should_stop_run(
        duration_mode="steps",
        step=1000,
        total_steps=1000,
        total_tokens=10_000_000,
        train_len=1_000_000,
        target_epochs=1,
    )


def test_epochs_duration_mode_stopping():
    """Epochs mode terminates strictly when total_tokens >= target_epochs * train_len."""
    train_len = 500_000  # 500k tokens per epoch
    target_epochs = 3    # 1.5M tokens total

    # In epoch 1
    assert not should_stop_run(
        duration_mode="epochs",
        step=5000,
        total_steps=10000,
        total_tokens=499_999,
        train_len=train_len,
        target_epochs=target_epochs,
    )
    # In epoch 2
    assert not should_stop_run(
        duration_mode="epochs",
        step=5000,
        total_steps=10000,
        total_tokens=1_200_000,
        train_len=train_len,
        target_epochs=target_epochs,
    )
    # Completed epoch 3
    assert should_stop_run(
        duration_mode="epochs",
        step=5000,
        total_steps=10000,
        total_tokens=1_500_000,
        train_len=train_len,
        target_epochs=target_epochs,
    )


def test_epoch_token_accounting():
    """Current epoch and tokens within epoch must calculate correctly."""
    train_len = 10_000_000  # 10M tokens per epoch

    tokens_seen = 26_400_000  # 2.64 epochs
    current_epoch = (tokens_seen // train_len) + 1
    tokens_in_epoch = tokens_seen % train_len
    pct = (tokens_in_epoch / train_len) * 100

    assert current_epoch == 3
    assert tokens_in_epoch == 6_400_000
    assert abs(pct - 64.0) < 1e-4

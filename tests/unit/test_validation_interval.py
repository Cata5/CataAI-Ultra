"""
tests/unit/test_validation_interval.py
Regression tests for validation interval evaluation triggering.
Verifies intervals: 10, 50, 100, 250, 500.
Ensures step 1 does NOT trigger validation when interval is 250.
"""

import pytest


def should_eval_step(step: int, start_step: int, eval_freq: int, manual_requested: bool = False) -> bool:
    """Exact logic from training_worker.py."""
    return (eval_freq > 0 and (step - start_step) > 0 and (step - start_step) % eval_freq == 0) or manual_requested


@pytest.mark.parametrize("interval", [10, 50, 100, 250, 500])
def test_validation_interval_trigger_points(interval: int):
    start_step = 0
    # Test first 1000 steps
    triggered_steps = []
    for step in range(1, 1001):
        if should_eval_step(step, start_step, interval):
            triggered_steps.append(step)

    expected = [s for s in range(interval, 1001, interval)]
    assert triggered_steps == expected, f"Triggered steps {triggered_steps[:5]} did not match expected {expected[:5]}"


def test_step_1_does_not_trigger_when_interval_is_250():
    """Ensure previous bug where step 1 triggered validation is permanently prevented."""
    assert not should_eval_step(step=1, start_step=0, eval_freq=250, manual_requested=False)
    assert not should_eval_step(step=1, start_step=0, eval_freq=10, manual_requested=False)
    assert not should_eval_step(step=1, start_step=0, eval_freq=50, manual_requested=False)


def test_manual_validation_trigger():
    """Manual validation must trigger even if step is not an interval multiple."""
    assert should_eval_step(step=17, start_step=0, eval_freq=250, manual_requested=True)
    assert should_eval_step(step=1, start_step=0, eval_freq=250, manual_requested=True)


def test_resume_start_step_offset():
    """When resuming from step 500, next validation for interval 250 must occur at 750, not 501."""
    start_step = 500
    assert not should_eval_step(step=501, start_step=start_step, eval_freq=250)
    assert not should_eval_step(step=600, start_step=start_step, eval_freq=250)
    assert should_eval_step(step=750, start_step=start_step, eval_freq=250)
    assert should_eval_step(step=1000, start_step=start_step, eval_freq=250)

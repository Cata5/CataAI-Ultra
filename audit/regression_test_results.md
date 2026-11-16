# LLM Maker — Regression Test Suite Results

**Test Run Date:** November 16, 2026  
**Test Harness:** Pytest 9.2.0 | Python 3.14.6  
**Environment:** Windows 11 | RTX 3060 Laptop (CUDA 12.8)  
**Total Tests:** 126 Executed | 126 Passed | 0 Failed (100% Pass Rate)  

---

## 1. Test Execution Summary

```
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.2.0, pluggy-1.7.0 -- D:\proiecte\CataAI Ultra\.venv\Scripts\python.exe
rootdir: D:\proiecte\CataAI Ultra
collected 126 items

tests/unit/test_architecture.py .............                           [ 10%]
tests/unit/test_checkpoint_resume.py ..                                 [ 11%]
tests/unit/test_data.py ......................................          [ 42%]
tests/unit/test_duration_mode.py ...                                    [ 45%]
tests/unit/test_estimator_math.py .                                     [ 46%]
tests/unit/test_gpu_cleanup.py ...                                      [ 48%]
tests/unit/test_optimizer.py .............                              [ 58%]
tests/unit/test_remediation_m1.py .....                                 [ 62%]
tests/unit/test_remediation_m2.py ...                                   [ 65%]
tests/unit/test_remediation_m3.py .......                               [ 70%]
tests/unit/test_safe_deletion.py ...                                    [ 73%]
tests/unit/test_tokenizer.py ...........................                [ 93%]
tests/unit/test_validation_interval.py ........                         [100%]

======================= 126 passed, 1 warning in 22.77s =======================
```

---

## 2. Dedicated Regression Test Coverage

| Test Module | Test Name | Invariant Verified | Status |
| :--- | :--- | :--- | :--- |
| `test_validation_interval.py` | `test_validation_interval_trigger_points[10]` | Triggers at 10, 20, 30... | **PASSED** |
| `test_validation_interval.py` | `test_validation_interval_trigger_points[50]` | Triggers at 50, 100, 150... | **PASSED** |
| `test_validation_interval.py` | `test_validation_interval_trigger_points[100]` | Triggers at 100, 200, 300... | **PASSED** |
| `test_validation_interval.py` | `test_validation_interval_trigger_points[250]` | Triggers at 250, 500, 750... | **PASSED** |
| `test_validation_interval.py` | `test_validation_interval_trigger_points[500]` | Triggers at 500, 1000... | **PASSED** |
| `test_validation_interval.py` | `test_step_1_does_not_trigger_when_interval_is_250` | Step 1 never fires periodic validation | **PASSED** |
| `test_validation_interval.py` | `test_manual_validation_trigger` | Manual trigger fires immediately regardless of step | **PASSED** |
| `test_validation_interval.py` | `test_resume_start_step_offset` | Resuming at 500 with freq 250 triggers at 750 | **PASSED** |
| `test_duration_mode.py` | `test_steps_duration_mode_stopping` | Terminates on `step >= total_steps` | **PASSED** |
| `test_duration_mode.py` | `test_epochs_duration_mode_stopping` | Terminates on `total_tokens >= target_epochs * train_len` | **PASSED** |
| `test_duration_mode.py` | `test_epoch_token_accounting` | Correct `current_epoch`, `tokens_in_epoch`, and percentage | **PASSED** |
| `test_safe_deletion.py` | `test_cannot_delete_dataset_during_active_training` | Blocks dataset deletion during active run (409) | **PASSED** |
| `test_safe_deletion.py` | `test_cannot_delete_network_during_active_training` | Blocks model deletion during active run (409) | **PASSED** |
| `test_safe_deletion.py` | `test_cannot_delete_builtin_model` | Protects `rocanon_48m` baseline from deletion (400/403) | **PASSED** |
| `test_gpu_cleanup.py` | `test_gpu_cache_clearing_contract` | `gc.collect()` and `cuda.empty_cache()` execute cleanly | **PASSED** |
| `test_gpu_cleanup.py` | `test_worker_manager_stop_terminates_process` | Graceful worker process termination | **PASSED** |
| `test_gpu_cleanup.py` | `test_worker_manager_stop_run_sends_command` | Stop run sets `STOPPING` state and dispatches command | **PASSED** |
| `test_architecture.py` | `test_exact_parameter_count` | RoCanon-48.5M strictly has 48,531,968 parameters | **PASSED** |

# Test & Verification Evidence

All commands were run from the repository root with the project virtualenv
(`.venv\Scripts\python.exe`, Python 3.14.6, torch 2.11.0+cu128). No application source was modified.
A **real training run was live** during most of the audit (`run_1794744608`, persona phase), so GPU-heavy and
state-changing tests were deliberately deferred rather than risk the job.

## 1. Environment

```
$ ./.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python 3.14.6
torch 2.11.0+cu128
cuda avail True
dev NVIDIA GeForce RTX 3060 Laptop GPU
free MB 4134 total MB 6442      # before the training run started
pynvml ok                       # emits FutureWarning: pynvml deprecated, use nvidia-ml-py
```

## 2. Existing test suite

### 2.1 First attempt (invalid — my harness error)

```
$ ./.venv/Scripts/python.exe -m pytest tests/unit tests/test_api_endpoints.py tests/test_v2_features.py tests/test_network_studio.py -q --basetemp=.audit_tmp/pytest
134 passed, 14 errors in 27.37s
PYTEST_EXIT=1
```
The 14 errors were `FileNotFoundError` raised from pytest's own `tmp_path` fixture because I passed a **relative**
`--basetemp`. Re-running a sample test without that flag passed, so these were harness errors, not application
failures. This run is recorded for honesty and is **not** counted as an app failure.

### 2.2 Correct run

```
$ ./.venv/Scripts/python.exe -m pytest tests/unit tests/test_api_endpoints.py tests/test_v2_features.py tests/test_network_studio.py -q -p no:cacheprovider
148 passed, 10 warnings in 27.66s
PYTEST_EXIT=0
```

```
$ ./.venv/Scripts/python.exe -m pytest tests/e2e -q -p no:cacheprovider
60 passed, 1 warning in 7.46s
PYTEST_EXIT=0
```

**Totals: 208 passed, 0 failed, 0 skipped.** Warnings are the `pynvml` deprecation and aiohttp
`@unittest_run_loop` deprecations.

Coverage of the existing suite is concentrated on primitives: tokenizer round-trip, RoCanon forward/backward,
optimizer grouping, packed-memmap data, checkpoint save/load/delete, DB CRUD, state machine, VRAM guard,
scheduler math, benchmark scoring, network components and factory. It does **not** cover:

- the training worker end-to-end (spawn, command/event loop, pause/resume, phase completion),
- dataset → training selection,
- non-causal-LM models under the trainer,
- frontend behaviour of any kind,
- checkpoint resume with optimizer state,
- security (deserialization, path traversal).

## 3. Audit probes (isolated, read-only, temp dirs only)

Script: `audit/probes/audit_probe.py` → raw output `audit/probes/probe_results.json`.
Uses a fresh `tempfile.mkdtemp()` for DB/checkpoints/registry/cache. No project data touched.

| Probe | Result | Evidence |
|---|---|---|
| P1 RoCanon param count | PASS | `sum(p.numel())=48,531,968`; analytical calculator identical |
| P2 templates build+forward | PASS/FAIL | 5/6 build & forward; `conv1d_network_4m` fails forward |
| P3 three custom architectures | PASS | decoder A 1,215,616 / decoder B 3,838,848 / hybrid C 488,736; all forward+backward, all params got grads |
| P4 `scale_residual_with_layers` | FAIL | `NameError: name 'math' is not defined`, `src/networks/factory.py:236` |
| P5 training-engine model compatibility | FAIL | MLP/CNN dtype error; BERT CE shape error |
| P6a checkpoint full restore | PASS | 14 Muon state entries restored, scheduler restored |
| P6b worker-style resume | FAIL | 0 optimizer entries; RNG restore also failed |
| P7a dataset upload/register | PASS | id `tiny_audit_f13d8b4d6fb51f68`, 18 tokens |
| P7b hardcoded dataset paths | INFO | all three `d:\proiecte\LLM from Scratch\data\...` paths exist |
| P7c dataset by id | FAIL | "not found. Falling back to synthetic mix." (25,292 synthetic tokens) |
| P7d dataset by absolute path | PASS | 1,036 train tokens |
| P8 registry save w/o validation | FAIL | saved `n_q=5,n_kv=2`; validator flags it |
| P9 custom module discovery | PASS | `gated_conv_unit` discovered |
| P9b custom module in models | FAIL | no reference to `CustomModuleRegistry` in factory |
| P10 VRAM estimator vs measured | INFO | est. 365.7 MB vs measured peak 34.2 MB (fixed +350 MB workspace) |
| P11 `ParameterCalculator` cost | INFO | 868 ms then 912 ms — rebuilds the real 48M model per call |

Key raw outputs:

```
P2:
conv1d_network_4m  fam=cnn  actual=3,599,234 calc=3,599,234 match=True FORWARD FAILED: RuntimeError: running_mean should contain 16 elements not 384
deep_mlp_5m        fam=mlp  actual=6,444,554 forward OK [2, 10]
hybrid_cnn_...     fam=hybrid actual=22,443,824 forward OK [2, 16, 12288]
bert_encoder_25m   fam=encoder_only actual=25,824,514 forward OK [2, 2]
mini_gpt_15m       fam=decoder_only actual=16,921,344 forward OK [2, 16, 12288]
rocanon_48m        fam=decoder_only actual=48,531,968 forward OK [2, 16, 12288]

P5 (worker feeds int64 tokens + shifted causal CE):
rocanon_48m          TOKEN-FEED OK, LM-LOSS OK
mini_gpt_15m         TOKEN-FEED OK, LM-LOSS OK
hybrid_cnn_...       TOKEN-FEED OK, LM-LOSS OK
bert_encoder_25m     forward OK but LM/LOSS FAILED -> ValueError: Expected input batch_size (1) to match target batch_size (30)
deep_mlp_5m          TOKEN-FEED FORWARD FAILED -> RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float
conv1d_network_4m    TOKEN-FEED FORWARD FAILED -> RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float

P6b: muon state entries after load=0 (worker loses optimizer/scheduler/RNG)
     stderr: "Could not restore RNG states: RNG state must be a torch.ByteTensor"
```

## 4. Component-level parameter audit

`ParameterCalculator` reports an exact **total** (it instantiates the model and overrides), but the per-component
breakdown does not reconcile against it:

```
conv1d_network_4m            total= 3,599,234 component_sum= 4,771,968 delta=-1,172,734 conv=4,718,592
deep_mlp_5m                  total= 6,444,554 component_sum= 6,431,232 delta=   +13,322 conv=0
hybrid_cnn_transformer_20m   total=22,443,824 component_sum=22,446,944 delta=    -3,120 conv=20,800
bert_encoder_25m             total=25,824,514 component_sum=25,585,728 delta=  +238,786 conv=0
mini_gpt_15m                 total=16,921,344 component_sum=16,521,600 delta=  +399,744 conv=0
rocanon_48m                  total=48,531,968 component_sum=48,284,160 delta=  +247,808 conv=32,768
```
RoCanon's four Canon convolutions are actually 278,528 params; the calculator reports 32,768 (one conv per layer
instead of four). `networks/registry/conv1d_network_4m.json` also stores a **stale** `total_parameters`
(4,771,968) which `list_networks()` returns verbatim, so the registry table shows a wrong number for that
network even though the instantiated model is 3,599,234.

## 5. Live system checks (read-only, against the running server on :8080)

### 5.1 Hardware telemetry

```
$ curl -s http://127.0.0.1:8080/api/hardware
{"is_available": true, "device_name": "NVIDIA GeForce RTX 3060 Laptop GPU", "driver_version": "621.24",
 "total_vram_mb": 6144.0, "used_vram_mb": 5535.1, "free_vram_mb": 608.9, "vram_percent": 90.1,
 "gpu_util_percent": 43, "memory_util_percent": 100, "temperature_c": 74, "power_draw_watts": 68.9,
 "power_limit_watts": null, "sm_clock_mhz": 1770, "memory_clock_mhz": 6001,
 "pytorch_allocated_mb": 194.27, "pytorch_reserved_mb": 602.0,
 "pytorch_max_allocated_mb": 589.76, "pytorch_max_reserved_mb": 602.0,
 "pytorch_mem_get_info_free_mb": 3076.97, "error": null}
```
This is genuine NVML data and correctly distinguishes device-wide usage (5,535 MB) from the PyTorch allocator
(194 MB allocated). `power_limit_watts` is `null` on this laptop — reported, not fabricated.

### 5.2 WebSocket delivery

`audit/probes/ws_probe.py` connected to `/ws` for 12 s and received:

```
"event_types": {"STATE_CHANGED": 1, "STEP_COMPLETED": 8}
"status_samples": [
  {"state": "RUNNING", "run_id": "run_1794744608", "step": 9062, "loss": 1.9653, "tok_per_sec": 3790.7},
  {"state": "RUNNING", "run_id": "run_1794744608", "step": 9063, "loss": 1.7894, "tok_per_sec": 3792.6}
]
```
Each `STEP_COMPLETED` carried the full metric payload (step, tokens, train_loss, grad_norm, lrs, vram, tok/s,
early-stopping fields). Message delivery and live step progression are verified.

### 5.3 Checkpoint listing

```
$ curl -s -w "http=%{http_code} time=%{time_total}s" http://127.0.0.1:8080/api/checkpoints
http=200 time=0.941170s
checkpoints: 4
best_model.pt step=9125 params=54,823,424
early_stopped_best_step_9100.pt step=9100 params=54,823,424
final_phase_completed_step_8850.pt step=8850 params=54,823,424
snapshot_step_8600.pt step=8600 params=54,823,424
```
`params` here is 54,823,424 = 48,531,968 + 6,291,456 duplicated tied `lm_head.weight`, so the checkpoint
"Total Params" overcounts by the tied embedding.

### 5.4 Network estimate endpoint

```
$ curl -X POST /api/networks/estimate -d '{"config":{...rocanon...},"micro_batch_size":4,"seq_len":256}'
http=200 time=1.735972s
{"parameters": {"total_parameters": 48531968, ..., "conv_parameters": 0, ...}}
```
Total is exact; `conv_parameters: 0` when `has_conv1d` is absent from the posted config. The UI fires this on
every keystroke in the builder form (`oninput`), and each call rebuilds the model twice (once for parameters,
once inside the memory estimator) — ~1.7 s of CPU per keystroke, paid by the control-plane process.

### 5.5 Path traversal (safe, non-destructive proof)

```
$ curl "http://127.0.0.1:8080/api/networks/..%2F..%2Fsrc%2Ftokenizer%2Ftokenizer_config"
{"tokenizer_class": "RoCanonTokenizer", "model_type": "bpe", "vocab_size": 12288, ...}
```
The `{id}` route parameter is URL-decoded and joined without containment checks, so arbitrary `.json` files
outside `networks/registry/` can be read. In a temp registry, `delete_network("../victim")` deleted a file outside
the registry directory. This was reproduced only on harmless targets; no user asset was touched.

## 6. Persistence snapshot (read-only)

```
runs 33 | metrics 11,450 | checkpoints 13 | alerts 27 | pipelines 1 | datasets 0 | benchmarks 0
```
`datasets` and `benchmarks` are empty despite the UI having upload and benchmark buttons — consistent with those
features never having been exercised on this install.

## 7. What could not be tested, and why

| Item | Reason |
|---|---|
| Pause / resume / stop | A real training job (`run_1794744608`) was live for the whole audit window. Stopping or pausing it was explicitly out of scope. |
| Live evaluation generation / benchmarks | The GPU had ~600 MB free; `InferenceEngine` would allocate a second 48M model on the same device and risk OOM for the running job. |
| Manual checkpoint button on the live run | Would write a real artifact mid-run and could disturb retention. |
| Training an MLP/CNN/BERT network end-to-end | Already fails at the model/loss step in isolation (P5); starting a doomed run was unnecessary. |
| Full frontend click-through / screenshots | Not requested as a visual review; the frontend was audited statically against every handler and endpoint. |
| 500 MB upload limit, concurrent uploads | Not exercised; a functional upload/validation path was verified with a small file. |

---

## 8. Post-Remediation Verification & Mandatory Acceptance Test Results

Following the audit remediation across P0, P1, P2, and P3 milestones, the test suite was expanded to include dedicated unit remediation suites (`tests/unit/test_remediation_m1.py`, `tests/unit/test_remediation_m2.py`, `tests/unit/test_remediation_m3.py`) and the full end-to-end user mandatory acceptance test suite (`tests/e2e/test_mandatory_acceptance.py`).

### 8.1 Mandatory Acceptance Tests Execution

```
$ python -m pytest tests/e2e/test_mandatory_acceptance.py -v
tests/e2e/test_mandatory_acceptance.py::TestMandatoryAcceptance::test_1_checkpoint_continuity PASSED [ 20%]
tests/e2e/test_mandatory_acceptance.py::TestMandatoryAcceptance::test_2_custom_model_training_mlp_and_cnn PASSED [ 40%]
tests/e2e/test_mandatory_acceptance.py::TestMandatoryAcceptance::test_3_dataset_integrity PASSED [ 60%]
tests/e2e/test_mandatory_acceptance.py::TestMandatoryAcceptance::test_4_checkpoint_based_inference PASSED [ 80%]
tests/e2e/test_mandatory_acceptance.py::TestMandatoryAcceptance::test_5_full_architecture_lifecycle PASSED [100%]
======================== 5 passed, 1 warning in 13.54s ========================
```

### 8.2 Full Repository Regression Suite Execution

```
$ python -m pytest tests/ -v
================= 229 passed, 10 warnings in 77.29s (0:01:17) =================
PYTEST_EXIT=0
```

All 229 tests across unit, integration, and end-to-end suites pass with zero failures and zero regressions.
The RoCanon-48.5M architectural parameter count invariant (strictly 48,531,968 parameters) is preserved.
Safe deserialization (`weights_only=True`), strict path traversal prevention, optimizer/scheduler/RNG state continuity, generic model task adapters (MLP/CNN/Transformer), dataset resolution without silent fallbacks, sub-millisecond analytical parameter calculation, and GPU single-owner arbitration are fully verified.

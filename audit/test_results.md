# Test Results

All commands were executed in `D:\proiecte\CataAI Ultra` on Windows with the project's `.venv`.
"Not executed" items are listed explicitly at the end; nothing below is inferred.

## 1. Python unit suite

```
.venv\Scripts\python.exe -m pytest tests\unit -q -p no:cacheprovider
```

Result: **142 passed, 0 failed** (exit 0), run again after the final CORS/security change.
Includes the 8 new audit regression tests in `tests/unit/test_audit_regressions.py` (unsafe-pickle
refusal, plain tensor load, interval parsing, `final_loss` persistence, legacy-schema migration, dataset
preset resolution, orphan checkpoint cleanup, missing entry handling).

## 2. HTTP API suite

```
.venv\Scripts\python.exe -m pytest tests\test_api_endpoints.py -q -p no:cacheprovider
```

Result: **18 passed, 0 failed** (exit 0). This suite boots the real aiohttp `create_app()` and covers the
index, status, hardware, checkpoints, pipelines, scheduler curve, plus seven new security tests:
localhost origin reflected; foreign origin cannot POST (403); foreign origin GET has no CORS header;
foreign `Host` rejected (403); IPv6 loopback accepted; `Origin: null` gets no CORS header and cannot
POST (403).

Note: the suite creates temporary databases and loopback sockets, so on this host it must run outside
the file sandbox — a workspace-only run fails at collection with `unable to open database file`.

## 3. Remaining backend suites

```
.venv\Scripts\python.exe -m pytest tests\test_frontend_integration.py tests\test_network_studio.py tests\test_v2_features.py -q -p no:cacheprovider
```

Result: **58 passed, 0 failed** (exit 0). Covers the frontend/backend contract, Network Studio
validation and the v2 run-state machine.

Total Python tests executed in this audit: 142 + 18 + 58 = **218**.

## 4. Worker lifecycle integration (real training, real SQLite)

```
.venv\Scripts\python.exe tests\integration\audit_worker_lifecycle.py
```

Result: **18/18 checks passed**, three independent worker processes:

| Session | Scenario | Observed |
| --- | --- | --- |
| 1 | start, stream 4 steps, graceful stop | exit 0 in 7.9 s; `RUN_STOPPED`; row `STOPPED`; `final_loss` persisted; step matches last metric; exactly 1 resumable snapshot |
| 2 | resume from that snapshot | exit 0 in 9.8 s; first step > stop step; validation fired; run `COMPLETED` |
| 3 | resume from a truncated checkpoint | exit 0 in 6.9 s; `RUN_FAILED`; row `FAILED`; `CHECKPOINT_LOAD_FAILED` alert; **0** training steps executed |

Session 3 is the regression that matters most: before the fix the worker silently trained from random
weights and reported success.

## 5. Throughput benchmark

```
.venv\Scripts\python.exe tests\integration\audit_worker_benchmark.py 120
```

```
steps: 120 (warm 115) in 18.6s wall
step_time_ms: median 48.36  min 45.64  max 96.50
tokens_per_sec: median 10546.6  max 11175.0
tflops_estimate: median 0.119
vram_peak_gb: 0.11
final_loss: 0.1245
trainable_parameters: 1876736
```

A previous run produced 47.97 ms, 10,632.6 tokens/s, 0.120 TFLOPs, final loss 0.1157 — consistent within
~1 %. Details and caveats in `performance_audit.md`.

## 6. Frontend

```
cd frontend; npx tsc --noEmit --incremental false   → exit 0
cd frontend; npx next build                         → exit 0, 16 routes compiled/prerendered
```

No frontend test runner is configured (AUD-027), so type checking plus a production build are the
available automated gates. Both pass with the final code.

## 7. Desktop shell (Rust)

```
cargo check --offline   → Finished dev profile in 24.66s
cargo build --offline   → Finished dev profile in 17.66s, debug binary produced
cargo build --offline   → Finished dev profile in 7.48s (incremental, after the final stdin-lock fix)
```

Startup smoke test (production DB **copied** to a temp directory, `LLMMAKER_PROJECT_ROOT` pointed there):

```
[TRACE] Resolved project root: "\\?\C:\Users\Cata\AppData\Local\Temp\llmmaker-rust-smoke"
[TRACE] Initializing DatabaseManager at \\?\...\studio.db...
[TRACE] Recovered 12 stale run(s) from a previous session
[TRACE] WorkerSupervisor initialized: project_root=..., python_path="python", db_path=...
```

The seeded `RUNNING` row (`run_1795506751`) became `STOPPED`; the application stayed alive until
terminated. No user database was modified.

## 8. HTTP server smoke test (live process)

`dashboard.py --port 8123`, then:

| Endpoint | Result |
| --- | --- |
| `GET /api/system/status` | 200, `state=IDLE`, active network `rocanon_48m` |
| `GET /api/runs` | 200, `[]` |
| `GET /api/checkpoints` | 200, `[]` |
| `GET /api/networks` | 200, 3,666 bytes, presets listed |
| `GET /api/datasets` | 200, `[]` |
| `GET /api/dataset_info` | 200, `exists=true` for the legacy preset path |

The server was stopped afterwards; no runs, metrics or settings were written to the browser database.

Re-run after the final CORS/Host changes (port 8124): the same six endpoints returned 200 and
`/api/dataset_info` still reported `exists=True`, confirming the middleware does not block ordinary
same-origin/no-`Origin` requests.

## 9. Static checks

* `ast.parse` over all 11 modified Python files → `AST OK: 11 files`.
* No TypeScript errors; no Rust warnings that block the build.
* `git status` reviewed: only intended source/test/audit files plus build artifacts.

## 10. Coverage gaps (not executed)

| Gap | Why |
| --- | --- |
| Interactive GUI walkthrough | No display automation for the Tauri webview in this environment; UI was verified by type check, build and code tracing |
| Release `tauri build` / installer | Debug build + startup smoke covered; bundling/installer not built |
| 48.5 M-parameter benchmark on a real dataset | Time/scope; the 1.9 M smoke config validates plumbing, not GPU efficiency |
| Multi-GPU / cluster training | No multi-GPU hardware available |
| macOS/Linux | Windows-only environment |
| Frontend component/e2e tests | No test runner configured (documented as AUD-027) |

## 11. Pillar 1/2 verification (schema + data safety)

Re-run after replacing the duplicated DDL with shared migrations and adding
pre-migration backups:

* `pytest tests/unit -q -p no:cacheprovider` → **150 passed** (the 8 new tests are
  in `tests/unit/test_migrations.py`: legacy upgrade, backup contents/retention,
  idempotency, fresh DB, absolute default path, safe mode, refused migration
  without a backup, refusal of a database from a newer build).
* `pytest tests/test_api_endpoints.py tests/test_frontend_integration.py tests/test_network_studio.py tests/test_v2_features.py -q` → **76 passed**.
* `cargo build --offline` (after adding the `backup` feature) → exit 0 in 22.6 s.
* Desktop smoke on a legacy DB root (`LLMMAKER_PROJECT_ROOT`): app started,
  `schema_migrations` reached 3, `final_loss`/`workspace_id` added, `run_legacy`
  (step 99) preserved, `backups/studio-1795528248.db` created before migration,
  log showed migrations 1–3.
* Python `StudioDatabase` opened the same Rust-migrated file: version 3, run
  readable, no second backup.
* `python -m compileall -q src tests` → exit 0; `npx tsc --noEmit --incremental false` → exit 0.
* Newer-schema guard (desktop): a DB stamped version 99 made the app exit with
  `FATAL: database initialization failed: ... newer than this build supports (3)`
  instead of running against an unknown schema.
* `.github/workflows/ci.yml` parses (`yaml.safe_load`, jobs: python/frontend/
  desktop) but has **not** been executed on GitHub-hosted runners.

Total Python tests now: 150 unit + 76 API/backend = **226**.
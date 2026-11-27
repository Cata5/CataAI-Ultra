# Security Audit

Scope: desktop shell, Python backend, worker protocol, checkpoint/dataset file handling, local HTTP/WS
surface. Defensive, source-first review; no live third-party system was probed.

## 1. Trust boundaries

| # | Boundary | Attacker-controlled input | Principal at risk |
| --- | --- | --- | --- |
| B1 | Checkpoint file → `torch.load` | Any `.pt` the user imports or downloads | Local user / OS account |
| B2 | `workspace_id`, ids in SQL | Webview / HTTP client | Database integrity |
| B3 | `127.0.0.1:8080` HTTP + `/ws` | Any website open in the user's browser | Local data, training control |
| B4 | Hugging Face import | Remote repository contents | Local user (remote code execution by design) |
| B5 | Worker `python` executable lookup | Files next to the executable / `PATH` | Local user |
| B6 | Dataset upload filename / network id | Multipart filename, JSON | Filesystem outside data dir |

## 2. Findings

### AUD-001 — Arbitrary code execution via checkpoint deserialization (Critical, fixed)

* **Evidence:** resume/listing called `torch.load(path, weights_only=True)` inside a `try` whose
  `except` re-loaded the same file with `weights_only=False`; package creation passed
  `weights_only=False` unconditionally. So in practice every path could fall back to full pickle.
* **Impact:** a crafted `.pt` executes arbitrary Python with the user's privileges the moment it is
  resumed or previewed. Checkpoints are routinely shared/downloaded, so this is a realistic vector.
* **Fix:** single `safe_torch_load()` helper using `weights_only=True`, with an explicit, actionable
  error and **no** unsafe fallback; all call sites routed through it.
* **Verification:** regression test writes a checkpoint whose `__reduce__` creates a marker file; load
  raises `RuntimeError` and the marker is absent (`tests/unit/test_audit_regressions.py`).

### AUD-011 — SQL injection in the Rust data layer (High, fixed)

* **Evidence:** `list_runs`/`list_checkpoints` formatted `workspace_id` into the SQL string.
* **Impact:** a workspace id containing a quote altered the query; with a crafted id, arbitrary
  `SELECT`/`DELETE` semantics against the local database. Inputs come from the webview, so a compromised
  renderer or a malicious workspace name is enough.
* **Fix:** bound parameters (`WHERE (?1 IS NULL OR workspace_id = ?1)`), covered by the real desktop run
  list check.

### AUD-031 — Local API readable and controllable by any website (High, fixed)

* **Evidence:** `cors_middleware` returned `Access-Control-Allow-Origin: *` on every route and no
  `Origin`/`Host` validation existed; `/api/settings` returns the stored Discord webhook URL and the API
  exposes `start/stop/delete` actions without authentication.
* **Impact:** any page the user visits could `fetch('http://127.0.0.1:8080/api/settings')` and read it
  (webhook secret), or `fetch(..., {method:'POST'})` to start/stop training and delete datasets. No
  authentication is involved, so no credential is needed.
* **Fix:** middleware now (a) rejects requests whose `Host` is not local unless explicitly allow-listed
  via `LLMMAKER_ALLOWED_HOSTS`, (b) reflects only localhost `Origin`s, (c) never treats `Origin: null`
  (sandboxed iframe / `data:` page) as local, (d) returns 403 for non-GET requests from foreign
  origins, and (e) `handle_websocket` rejects foreign origins. This also blocks DNS-rebinding reads
  because the rebound hostname fails the `Host` check.
* **Verification:** `tests/test_api_endpoints.py` — local origin is reflected, remote origin cannot POST
  (403), remote origin reads get no CORS header, foreign `Host` is rejected (403), IPv6 loopback works,
  `null` origin gets no CORS permission and cannot POST. 18/18 pass.
* **Residual:** a non-browser local process can still call the API; that is the same trust level as the
  user's own shell and is out of scope for a loopback-only service.

### AUD-032 — `trust_remote_code=True` on foundation-model import (Medium, accepted)

* **Evidence:** `src/networks/foundation_loader.py:112,204` pass `trust_remote_code=True` to
  `AutoConfig`/`AutoModel`.
* **Impact:** importing a Hugging Face repository can execute its Python modules locally. This is a
  product feature (custom architectures), not an accident.
* **Recommendation:** require an explicit per-import confirmation in the UI that names the repository,
  and default to `False` when the model card declares a standard architecture. Not changed in this
  audit because it alters the import contract and the UI copy.

### AUD-033 — Secrets stored in plaintext (Medium, mitigated)

* **Evidence:** `settings` table stores `discord_webhook_url` (and notification credentials) unencrypted;
  the database is a normal user file.
* **Impact:** any process/malware with read access to `studio.db` obtains a working webhook URL (this is
  also why AUD-031 mattered).
* **Recommendation:** store notification secrets with Windows DPAPI (`CryptProtectData`) and redact them
  from `GET /api/settings`. Not implemented to avoid migrating existing user data without a versioned
  schema change.

### AUD-034 — Worker interpreter resolution falls back to `PATH` (Low, accepted)

* **Evidence:** `WorkerSupervisor` probes venv paths and finally `python`/`python3` on `PATH`.
* **Impact:** if the app is stored in a user-writable directory and launched with a poisoned `PATH`, a
  different interpreter runs. The fallback is necessary for portable installs.
* **Recommendation:** record the resolved interpreter in the startup log (already done) and prefer a
  bundled runtime for release builds.

### AUD-035 — AUD-001 fix was incomplete: unrestricted unpickling remained in two loaders (High, fixed)
* **Component:** `src/eval/inference.py` (checkpoint inspection before evaluation), `src/data/custom_loader.py` (pre-tokenized `.pt` cache loading, 2 call sites), `src/infer/generate.py` (CLI and generator loaders).
* **Evidence:** `safe_torch_load` documents that the `weights_only=False` fallback must never be used, but `inference.py` retried any failed load with `weights_only=False`, and both cache loaders did the same. A `.pt` file with a `__reduce__` payload placed where a cache is looked up (or any checkpoint evaluated) would execute code during load.
* **Fix:** `inference.py` now calls `safe_torch_load`. The dataset loader uses a new `_load_token_cache` helper that only allows `weights_only=True`, and on failure or missing keys logs a warning and re-tokenizes from source. `generate.py` passes `weights_only=True` explicitly.
* **Verification:** `tests/unit/test_audit_regressions.py::test_token_cache_loader_never_unpickles_payloads` (marker file not created), `::test_token_cache_loader_returns_valid_cache_and_rejects_missing_keys`. Full Python suite: 414 passed.

## 3. Areas checked and found sound

* **Path traversal:** network ids are validated against `^[A-Za-z0-9_-]{1,64}$` and resolved inside the
  registry with `is_relative_to` (`src/networks/registry.py:106-118`); uploads use
  `Path(field.filename).name`, discarding directories.
* **Command execution:** no `subprocess`, `os.system` or `shell=True` anywhere in `src/`; the only
  process spawn is the Tauri shell, with an argument vector (no shell).
* **Destructive guards:** deleting a network/dataset/checkpoint used by an active run is rejected with
  409 (`tests/unit/test_safe_deletion.py`).
* **Capabilities:** the Tauri capability file grants only `core:default` plus application commands; no
  remote origin is allowed, so IPC is not reachable from web content.
* **Secrets in source:** none found (patterns for `sk-…`, `hf_…`, Discord webhook URLs).
* **Checkpoint path confinement:** `resolve_checkpoint_path` is bounded to `checkpoints_v2/` and
  `workspaces/`, so a crafted id cannot read arbitrary files.
* **Upload size limit:** 500 MB enforced by aiohttp; no zip extraction or archive handling.

## 4. Risk summary

| ID | Severity | Status |
| --- | --- | --- |
| AUD-001 | Critical | Fixed, tested |
| AUD-011 | High | Fixed |
| AUD-031 | High | Fixed, tested |
| AUD-032 | Medium | Accepted, documented |
| AUD-033 | Medium | Mitigated by AUD-031, encryption pending |
| AUD-034 | Low | Accepted |

The overall posture is reasonable for a single-user desktop application: no network services except a
loopback API, no credentials in source, and the two highest-impact defects (pickle RCE, cross-site API
access) are now closed and covered by regression tests.
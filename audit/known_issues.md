# Known Issues (open after this audit)

Everything in `FULL_AUDIT_REPORT.md` marked **Fixed** has been changed and verified. The items below are
deliberately **not** fixed: either the fix requires a product decision, hardware, or an external service.
Each entry states the evidence, the impact, a workaround, and the recommended fix.

## Summary

| ID | Severity | Component | Title |
| --- | --- | --- | --- |
| AUD-022 | Medium | Build config | `.cargo/config.toml` hardcodes a user target dir and the GNU target |
| AUD-027 | Medium | Frontend | No automated frontend tests |

**Resolved since this list was written:** AUD-020, AUD-021, AUD-023, AUD-029, AUD-030, AUD-032, AUD-033,
AUD-034. Details below; see `audit/pillar_1_2_status.md` for the tests that cover them.

---

## Resolved in the Pillar 1/2 work

| ID | What changed |
| --- | --- |
| AUD-020 | `create_run` and `create_workspace` use plain `INSERT` and raise `ValueError` on an id collision, so an existing run/workspace is never replaced. `PRAGMA foreign_keys = ON` is set on every Python connection and by the Rust shell. Workspace deletion removes metrics, runs, checkpoints and datasets before the workspace row. |
| AUD-021 | `DEFAULT_DB_PATH` resolves against the repository root; Python and Rust apply the same numbered SQL migrations and version table. |
| AUD-023 | The worker sums the `valid_tokens` returned by `accumulate_micro_batch` and uses that for `total_tokens` and the scheduler. Throughput and TFLOP/s still use processed positions, which is the hardware-work number. |
| AUD-029 | The `+ 10_000_000_000_000` target inflation was removed. Completion uses `session_target_tokens`, which is derived from the current session, so a resumed run no longer reports a 10-trillion-token target. |
| AUD-030 | Metrics emit `vram_used_gb` as current `torch.cuda.memory_allocated()` and add `vram_peak_gb`. The VRAM guard still evaluates peak, the conservative value. |
| AUD-032 | Hugging Face inspection and loading default to `trust_remote_code=False`. The import API accepts `allow_remote_code`, and the Network Studio asks for explicit confirmation before retrying with custom code. |
| AUD-033 | Notification secrets are encrypted with AES-256-GCM by both the Python control plane and the Rust desktop shell, using the same key file and ciphertext format. The key is protected by Windows DPAPI (or stored in a 0600 file elsewhere) next to the database. `GET /api/settings` returns only `*_configured` flags and a sentinel; saving the sentinel keeps the stored value, and an empty string clears it. Legacy plaintext values are upgraded on first open; a missing key is never silently replaced. |
| AUD-034 | Release builds no longer fall back to `PATH`. `LLMMAKER_PYTHON` is checked first, followed by the project virtualenv; the `PATH` fallback exists only in debug builds. |

---

## AUD-022 — Cargo config pins a machine-specific target directory

* **Evidence:** `.cargo/config.toml` sets `target-dir = "C:/Users/Cata/.cargo/target_llm_maker"` and
  `target = "x86_64-pc-windows-gnu"`.
* **Impact:** a fresh clone on another machine writes build output to a non-existent path (or fails
  outright), and the MSVC default toolchain breaks `windres` on repository paths containing spaces.
* **Workaround:** keep the file on this machine; delete it on machines without a space in the path, or set
  `CARGO_TARGET_DIR` and pass `--target x86_64-pc-windows-msvc`.
* **Fix:** move the redirection into a machine-local `config.toml` outside the repo (or a documented
  `scripts/setup-windows.ps1`), and build release artifacts into `src-tauri/target`.
* **Status:** CI overrides both settings, but the local clone still depends on this file;
  `scripts/bootstrap.ps1` inherits it. Not fixed.

## AUD-027 — No automated frontend tests

* **Evidence:** `frontend/package.json` has no test runner; CI relies on `tsc` and `next build`.
* **Impact:** UI regressions (for example the silent `validateRun` no-op) can only be caught by a human.
* **Fix:** add Vitest + Testing Library for hooks/views and Playwright for the three core flows
  (start → stop, dataset upload → tokenize, network save → start).

---

## Verification gaps (not product defects)

| Gap | Consequence |
| --- | --- |
| No interactive GUI automation | Click-through behaviour verified by code + build only; see `test_results.md` |
| Debug build only, no installer | Packaging/upgrade path unverified |
| 48.5 M-parameter benchmark not run | Throughput numbers apply to the 1.9 M smoke configuration |
| Multi-GPU not exercised | Cluster/multi-GPU code paths are code-reviewed only |
| Legacy dataset directory fallback retained | `D:\proiecte\LLM from Scratch\data` remains a last-resort search path; it is logged, not silent |

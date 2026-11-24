# Pillar 1 & 2 implementation status

Scope: the first two roadmap pillars — "make installation a non-event" and "make
data safety non-negotiable". This file records what shipped, the evidence, and
what remains. It is deliberately separate from `FULL_AUDIT_REPORT.md`, which
describes the earlier correctness audit.

## Shipped

### Single versioned schema (AUD-020/AUD-021 root cause)

* `src/db/migrations/001_baseline.sql` — complete schema for all nine tables and
  indexes.
* `002_final_loss.sql`, `003_workspace_columns.sql` — the two ad-hoc column
  upgrades that previously existed only as `try/except ALTER` code in each
  language.
* `src/db/migrations.py` — statement splitter, `schema_migrations` bookkeeping,
  idempotent duplicate-column handling, and automatic pre-migration backup.
* `src-tauri/src/database.rs` — embeds the same SQL files with `include_str!` and
  applies them through the identical version table; `rusqlite` uses the `backup`
  feature for the online snapshot.
* Python `_init_db` no longer contains its own DDL: it calls the shared runner.

### Pre-migration backups

Both layers take a consistent copy through SQLite's online backup API before
applying any pending version to a non-empty database:

```
backups/studio-YYYYMMDD-HHMMSS.db      # Python
backups/studio-<epoch-seconds>.db      # Rust
```

The ten newest snapshots are retained. Fresh databases need no snapshot. On a
populated database, if the snapshot cannot be written the upgrade is refused
(startup fails with an explanatory error) instead of risking data; a failed
migration likewise aborts rather than continuing on a half-upgraded schema.

### Data safety

* **Secrets at rest (AUD-033).** `src/security/secrets.py` and
  `src-tauri/src/secrets.rs` encrypt `discord_webhook_url` and
  `telegram_bot_token` with AES-256-GCM using the same key file and ciphertext
  format. The data key is protected with Windows DPAPI (`CryptProtectData`) and
  stored next to the database; non-Windows uses a raw 32-byte key file with `0600`
  permissions. Legacy plaintext rows are encrypted on first database open by both
  the Python and Rust paths. A missing key is never replaced while encrypted rows
  exist, and a wrong key returns an explicit error instead of an empty value.
* **Redacted API.** `GET /api/settings` returns configured booleans and the
  sentinel `__LLMMAKER_SECRET_SET__` instead of secret values. A `POST` that omits
  the field leaves it unchanged, the sentinel keeps it, an empty string clears it,
  and any other value replaces it.
* **Duplicate ids rejected (AUD-020).** `create_run` and `create_workspace` use a
  plain `INSERT` and raise `ValueError` instead of replacing history.
* **Foreign keys and child-first deletion.** Every Python connection enables
  `PRAGMA foreign_keys = ON`; the Rust shell does the same. Workspace deletion
  removes metrics, runs, checkpoints and datasets before the workspace row.
* **Integrity check.** `_init_db` runs `PRAGMA quick_check` before migrations and
  raises `DatabaseIntegrityError` with the database path instead of continuing on
  corruption. The Rust shell performs the same check.
* **Workspace archives.** `src/workspace/archive.py` exports a ZIP containing a
  versioned `manifest.json` (`llmmaker.workspace/1`), workspace/run/metric/
  checkpoint/dataset rows, and only the files those rows reference. Every file has
  a SHA-256 checksum and size. Import validates the schema, rejects absolute or
  `..` paths, verifies checksums, and always creates a new workspace id; it never
  overwrites. The 4 GiB / 20,000-entry limits are enforced before extraction.
* **Safe mode has a UI.** The persisted `safe_mode` setting is applied at startup;
  `LLMMAKER_SAFE_MODE` still wins and the API returns 409 when the environment
  locks the toggle. Settings shows Recovery & Diagnostics with the safe-mode
  switch and the workspace export/import controls.
* **Remote model code is opt-in (AUD-032).** Hugging Face inspection and loading
  default to `trust_remote_code=False`. The import API accepts
  `allow_remote_code`, and Network Studio asks for an explicit confirmation naming
  the repository before retrying with custom code. The choice is stored in the
  network config.

### Training correctness and telemetry

* **AUD-023.** The worker sums the `valid_tokens` returned by
  `accumulate_micro_batch` and uses that for `total_tokens` and the scheduler.
  Throughput and TFLOP/s still use raw processed positions, which is the
  hardware-work number.
* **AUD-029.** The `+ 10_000_000_000_000` target inflation was removed. Completion
  uses `session_target_tokens`, derived from the current session, so a resumed run
  does not report a 10-trillion-token target.
* **AUD-030.** Metrics emit `vram_used_gb` as current
  `torch.cuda.memory_allocated()` and add `vram_peak_gb`. The VRAM guard still
  evaluates peak, the conservative value.

### Install, CI and website

* `scripts/bootstrap.ps1`: prerequisite check, virtualenv, dependency install,
  frontend build, desktop build, safe to re-run.
* `.github/workflows/ci.yml`: Python tests (CPU torch), frontend `tsc` + build,
  website `tsc` + build, Windows desktop `cargo check` with machine-specific Cargo
  settings overridden.
* `.github/workflows/pages.yml`: builds `website/` with the base path supplied by
  `actions/configure-pages` and deploys `website/out` through GitHub Pages Actions.
* The website now lives in `website/` in this repository; its old standalone
  folder is retired.
* `docs/INSTALL.md`, `docs/DATA_SAFETY.md`.

## Verification

| Check | Command | Result |
| --- | --- | --- |
| Unit suite | `pytest tests/unit -q` | **162 passed** |
| API + backend suites | `pytest tests/test_api_endpoints.py tests/test_frontend_integration.py tests/test_network_studio.py tests/test_v2_features.py -q` | **76 passed** |
| Combined Python | same targets plus `tests/unit` | **238 passed** |
| Migration tests | `pytest tests/unit/test_migrations.py -q` | 8 passed |
| Python compile | `python -m compileall -q src tests` | exit 0 |
| TypeScript | `npx tsc --noEmit --incremental false` | exit 0 |
| Frontend build | `npm run build` (frontend) | exit 0, 16 static routes |
| Website build | `NEXT_PUBLIC_BASE_PATH=/CataAI-Ultra npm run build` (website) | exit 0, base path and screenshots rewritten |
| Rust check | `cargo check --manifest-path src-tauri/Cargo.toml` | exit 0 |
| Rust/Python crypto interop | standalone probe using `src-tauri/src/secrets.rs` | AES-GCM ciphertext decrypted both ways; Rust DPAPI key read by Python |
| Data-safety tests | `pytest tests/unit/test_data_safety.py -q` | 12 passed |
| Workflow YAML | `yaml.safe_load` | four CI jobs and the Pages workflow parsed |

The Tauri test binary cannot start on this host (`STATUS_ENTRYPOINT_NOT_FOUND`), so
Rust crypto was exercised through a standalone probe importing the same
`secrets.rs`; `cargo check` covers the shell integration. The Windows CI job runs
the MSVC target on a clean runner.

The desktop smoke used a legacy database (no `final_loss`, no `workspace_id`,
missing tables) containing `run_legacy` at step 99. The Rust shell applied
migrations 1–3, wrote `backups/studio-1795528248.db`, preserved the run, and the
log recorded each migration. Python then opened the same file and returned the
run with `STOPPED` status, proving both layers agree on the version table.

## Remaining (not done, with reasons)

| Item | Why / next step |
| --- | --- |
| Signed installer and auto-update | Needs a code-signing certificate and a release pipeline; cannot be produced or validated in this environment. Do not ship a placeholder updater. |
| Bundled Python/CUDA runtime | Needs a packaging decision (embeddable Python + pinned wheels per GPU vendor) and multi-GB artifacts; bootstrap currently installs into `.venv`. |
| Portable release data directory | Debug/source installs keep data in the project root. A release build should use `%LOCALAPPDATA%` with explicit migration; that is a product decision because it moves existing user data. |
| Machine-specific Cargo config (AUD-022) | CI overrides it; local clones still depend on `.cargo/config.toml`. |
| Frontend component tests (AUD-027) | `tsc` and `next build` cover types and rendering, not interaction. Vitest/Playwright remain to be added. |
| CI execution | The workflows have not run on GitHub-hosted runners; only their YAML and the underlying commands were validated locally. |

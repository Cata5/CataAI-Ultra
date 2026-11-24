# Frontend Functionality Audit

Stack: Next.js App Router, React 19, TypeScript, TanStack Query, Tailwind CSS v4.
14 routes, 9 feature views, `api.ts` (54 methods) as the single data layer, WebSocket + polling fallback.

Verification performed:

* `npx tsc --noEmit --incremental false` → exit 0, no type errors.
* `npx next build` → exit 0, all 16 routes compiled and statically prerendered.
* Every `api.*` call was traced to its Rust command and/or HTTP route (see `api_endpoint_inventory.md`).
* No frontend unit/component tests exist (AUD-027); UI behaviour was verified by code tracing plus the
  live backend/worker checks described in `test_results.md`.

## 1. Route → feature → backend map

| Route | View | Backend calls |
| --- | --- | --- |
| `/` | `OverviewView` | `checkpointRun`, `pauseRun`, `resumeRun`, `stopRun`, `updateLr`, `validateRun`, `useSystemStatus` (`getSystemStatus`), `useHardwareTelemetry` |
| `/training` | `TrainingView` | `startRun`, `pauseRun`, `resumeRun`, `stopRun`, `checkpointRun`, `validateRun`, `getActiveWorkspace`, `listNetworks`, `listDatasets`, `listCheckpoints`, `tokenizeDataset`, `getComputeTopology` |
| `/runs` | `RunsView` (tab `experiments`) | `listRuns`, `deleteCheckpoint`, live metrics via `useWebSocket`/`getRunMetrics` |
| `/analytics` | `RunsView` (tab `analytics`) | `listRuns` + metric history |
| `/checkpoints` | `RunsView` (tab `checkpoints`) | `listRuns`, `deleteCheckpoint`, `SaveModelModal` → `saveLocalModel` |
| `/models` | `RunsView` (tab `models`) | `listRuns`, checkpoint listing |
| `/projects` | `RunsView` (tab `projects`) | runs grouped by workspace |
| `/studio` | `NetworkStudioView` (dynamic import + error boundary) | `listNetworks`, `getNetworkTemplates`, `getCustomModules`*, `getNetwork`, `saveNetwork`, `cloneNetwork`, `deleteNetwork`, `getFoundationPresets`, `importFoundationModel`, `estimateNetwork`*, `validateNetwork`*, `recommendBudget`* |
| `/datasets` | `DatasetsView` | `listDatasets`, `uploadDataset`, `tokenizeDataset`, `curateDataset`, `deleteDataset`, `getDatasetStats`, `inspectTokenizer` |
| `/evaluation` | `EvaluationView` | `generateEvaluation`, `compareEvaluation`, `runBenchmarkSuite`, `listBenchmarkResults`, `listCheckpoints` |
| `/hardware` | `HardwareView` | `getHardwareTelemetry` (polling) |
| `/logs` | `LogsView` | `getSystemLogs`, `clearSystemLogs` |
| `/settings` | `SettingsView` | `getSettings`, `saveSettings`, `testNotification` |

\* reached through helper modules rather than directly in the view file.

Global chrome: `Topbar` (workspace switcher: `listWorkspaces`, `createWorkspace`, `deleteWorkspace`,
`getActiveWorkspace`, `setActiveWorkspace`), `Sidebar`, `SetupWizardModal` (`createWorkspace`,
`getHardwareTelemetry`), status pills fed by `useSystemStatus` and `useWebSocket`.

## 2. Feature-by-feature findings

### Overview (`/`)

* Live state, current run, loss/LR/tokens cards, GPU panel, control bar (pause/resume/stop/save
  checkpoint/validate/apply LR).
* **Fixed:** `handleTriggerVal` called `api.validateRun()`, which in Tauri mode was a no-op that
  resolved `{success: true}` without contacting the worker. The UI reported success while nothing
  happened. It now invokes the new `validate_run` command and surfaces worker rejections (AUD-010).
* **Fixed upstream:** the Stop button previously appeared to work but the run was killed after 800 ms
  and its final state/snapshot were lost (AUD-002); the button now waits for a real graceful stop.

### Training (`/training`)

* Largest view: network/dataset/optimizer/scheduler form, live loss chart, metrics grid, LR controls,
  checkpoint actions, resume selector, LoRA and multi-GPU options.
* Payload construction (`TrainingView.tsx:461-517`) is the contract with the worker: `workspace_id`,
  `network_id`, `dataset_id`, `duration_mode`, step/token targets, optimizer and scheduler fields,
  `precision`, `grad_clip`, intervals, early stopping, device selection, `start_fresh`,
  `resume_checkpoint`.
* **Fixed:** interval fields of `0` ("disable") were silently replaced by defaults inside the worker
  (AUD-012), so a user could not turn interval checkpoints/validation off.
* **Fixed:** a tokenizer/network vocabulary mismatch used to reach the CUDA kernel and abort with
  `device-side assert`; the worker now rejects the combination before step 1 with an actionable
  message (AUD-018).
* Error path uses `alert()`; functional but inconsistent with the rest of the design system (cosmetic).

### Runs / Analytics / Checkpoints / Models / Projects (`RunsView`)

* One view, five tabs; destructive checkpoint deletion is confirmed in a modal.
* **Fixed upstream:** run rows could be missing `final_loss`, which made the desktop `list_runs` query
  fail outright (AUD-009); stale `RUNNING/STOPPING` rows from killed sessions are now repaired at
  startup (AUD-015).
* Checkpoint deletion previously left the database row behind when the file was already gone; the
  worker now clears the row (AUD-019).

### Network Studio (`/studio`)

* Dynamic import with an error boundary; block-based architecture editor, parameter/VRAM estimates,
  template gallery, clone/delete, foundation preset import, Ollama export.
* Backend contract is the `networks` table and `NetworkRegistry`; saving a network writes JSON that the
  worker later consumes through `network_id`. The vocabulary guard above also protects this path.

### Datasets (`/datasets`)

* Upload, tokenize (with progress), curate, stats, delete, tokenizer inspector.
* **Fixed:** built-in presets (`tech_wiki_ro`, `chat_persona_ro`, `alpaca_ro`) were bound to absolute
  paths on the original developer machine; resolution is now env/project-relative with a documented
  legacy fallback (AUD-017). `/api/dataset_info` was re-checked live and still resolves the existing
  legacy file.

### Evaluation (`/evaluation`)

* Generate (single checkpoint), compare (two checkpoints side by side), benchmark suite, results list.
* Generation runs in the aiohttp server process; in desktop mode it goes through Tauri commands. No
  functional defect found in the UI trace.

### Hardware (`/hardware`), Logs (`/logs`), Settings (`/settings`)

* Hardware polls telemetry every second; Logs supports filtering and clearing; Settings persists to the
  `settings` table and can send a test notification.
* No functional defect found; Settings writes are handled by both backends consistently.

## 3. Cross-cutting observations

1. **API parity.** `api.ts` is the only file that knows about the two backends. Every method has a
   Tauri branch and an HTTP branch; the audit added `validateRun`'s real Tauri branch. Methods without a
   Tauri branch (evaluation streams, pipeline advance, `/v1/*`) are never called in desktop mode.
2. **Live data.** `useWebSocket` merges `STEP_COMPLETED` payloads into React Query caches and falls back
   to polling when the socket is not connected, so the UI degrades instead of freezing.
3. **Swallowed errors.** Several `catch {}` blocks (workspace fetch, optional telemetry) hide failures;
   acceptable for optional chrome, but they make real regressions invisible. Left unchanged to avoid
   changing UX behaviour.
4. **No automated UI tests.** This is the largest remaining quality gap on the frontend (AUD-027).
5. **Accessibility/perf.** No obvious blocking issue; charts re-render on every metric frame but the
   build and type check are clean.
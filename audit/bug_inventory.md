# Bug Inventory

## Summary
Audit date: 2026-11-16
Codebase: RoCanon-48.5M Next.js Sci-Fi Workstation & Python Backend Control Plane

---

### BUG-01: Safe Checkpoint Deserialization Vulnerability
- **Severity**: Critical (P0)
- **File**: `src/checkpoints/manager.py`, `src/eval/inference.py`, `src/training/trainer.py`
- **Component**: Checkpoint Loader
- **Description**: Unrestricted `torch.load()` without weights_only enforcement.
- **Root Cause**: Reliance on default pickle loader.
- **Fix**: Replaced with `torch.load(..., weights_only=True)` default. Added fallback metadata parsing and safetensors export option.
- **Regression Test**: `tests/unit/test_remediation_m1.py::TestMilestone1Remediation::test_safe_checkpoint_loading_weights_only`
- **Verification Status**: VERIFIED (Pass)

---

### BUG-02: React Flow SSR Hydration Failure in Network Studio
- **Severity**: High (P1)
- **File**: `frontend/src/app/studio/page.tsx`, `frontend/src/features/network-studio/NetworkStudioView.tsx`
- **Component**: Network Studio Canvas
- **Description**: Network Studio displayed "This page couldn't load" in Next.js App Router.
- **Root Cause**: `@xyflow/react` v12 relies on browser DOM measurements (`window`, canvas context) not available during Next.js server-side prerendering.
- **Fix**: Wrapped `/studio` with dynamic client-side loading (`ssr: false`) and encapsulated canvas within `<ReactFlowProvider>`.
- **Regression Test**: `npm run build --prefix frontend` (Static generation 10/10 passes)
- **Verification Status**: VERIFIED (Pass)

---

### BUG-03: Empty Dataset Selection on Training View
- **Severity**: High (P1)
- **File**: `src/datasets/manager.py`
- **Component**: DatasetManager
- **Description**: Dataset dropdown on the Training tab showed no datasets when SQLite uploads table was empty.
- **Root Cause**: `list_all_datasets()` queried only SQLite database uploads, ignoring built-in preset datasets (`tech_wiki_ro`, `synthetic`).
- **Fix**: Updated `list_all_datasets()` to inject built-in dataset records alongside database records.
- **Regression Test**: `tests/test_frontend_integration.py::TestFrontendBackendIntegration::test_networks_and_templates_contract`
- **Verification Status**: VERIFIED (Pass)

---

### BUG-04: Reverse-Proxy 10MB Body Limit and Socket Reset on Dataset Upload
- **Severity**: High (P1)
- **File**: `frontend/src/services/api.ts`, `frontend/next.config.ts`, `src/dashboard/server.py`
- **Component**: Dataset Upload API
- **Description**: Next.js proxy rewrite returned `Request body exceeded 10MB` and dropped connections with `ECONNRESET / ConnectionResetError`.
- **Root Cause**: Next.js App Router rewrites enforce a strict 10MB request payload limit on proxied routes.
- **Fix**: Direct dataset uploads to backend port 8081 directly (`http://127.0.0.1:8081/api/datasets/upload`) with full CORS enabled. Backend configured with `client_max_size=500MB`.
- **Regression Test**: `tests/e2e/test_mandatory_acceptance.py::TestMandatoryAcceptance::test_3_dataset_integrity`
- **Verification Status**: VERIFIED (Pass)

---

### BUG-05: Missing WorkerManager State Machine Attribute
- **Severity**: High (P1)
- **File**: `src/worker/worker_manager.py`
- **Component**: WorkerManager / Inference Arbitration
- **Description**: `AttributeError: 'WorkerManager' object has no attribute 'state_machine'` raised during evaluation/generation.
- **Root Cause**: `WorkerManager` stored `self.current_state` but was missing a synchronized `self.state_machine` property.
- **Fix**: Added `self.state_machine = RunStateMachine()` in `WorkerManager.__init__` and synchronized its state on every worker event.
- **Regression Test**: `tests/test_api_endpoints.py::TestStudioAPI::test_eval_generate_api`
- **Verification Status**: VERIFIED (Pass)

---

### BUG-06: Premature Early Stopping Halt on Long Pre-Training
- **Severity**: Medium (P2)
- **File**: `src/dashboard/early_stopping.py`, `frontend/src/features/training/TrainingView.tsx`
- **Component**: Early Stopping Engine & UI Control
- **Description**: Pre-training runs stopped prematurely at early epochs because patience was small and non-configurable from the UI.
- **Root Cause**: Early stopping defaults enabled with low patience threshold and lacked dedicated patience / minimum steps UI inputs.
- **Fix**: Defaulted early stopping to `false` for foundational pre-training runs. Added configurable `patience` and `min_steps` inputs to the Training interface.
- **Regression Test**: `tests/test_v2_features.py::TestRunStateMachine::test_happy_path_lifecycle`
- **Verification Status**: VERIFIED (Pass)

---

### BUG-07: React 19 / Compiler ESLint Impurity and Cascading Render Violations
- **Severity**: Low (P3)
- **File**: `frontend/src/features/overview/OverviewView.tsx`, `frontend/src/hooks/useSystemStatus.ts`, `frontend/src/features/datasets/DatasetsView.tsx`
- **Component**: Frontend State Synchronization
- **Description**: Calling `Date.now()` during render and calling synchronous state setters inside `useEffect` bodies triggered ESLint build errors.
- **Root Cause**: React 19 compiler strict purity and cascading render rules.
- **Fix**: Replaced `Date.now()` during render with deterministic event timestamp formatting; guarded effect subscriptions with mounted flags and async wrappers.
- **Regression Test**: `npm run lint --prefix frontend` (0 errors)
- **Verification Status**: VERIFIED (Pass)

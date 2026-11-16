# Dead Code Inventory

## Classification Scheme
- **CONFIRMED DEAD**: Proven to be unused and unnecessary. Removed.
- **PROBABLY DEAD**: No references found, verified safe or candidate for deprecation.
- **LEGACY**: Retained for backward compatibility or rollback safety.
- **DUPLICATED**: Functionality exists in multiple places; consolidated.
- **ORPHANED**: Implementation exists but disconnected from main loop.
- **INCOMPLETE**: Function exists but unfinished.
- **ACTIVE**: In production use.

---

### Findings

| File Path | Symbol / Feature | Type | Status | References Found | Reason for Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `src/dashboard/ui.py` | `INDEX_HTML` | HTML template | **LEGACY** | `server.py` | Retained as emergency fallback when running backend port 8081 directly without Node/Next.js. |
| `frontend/src/components/layout/StatusBar.tsx` | `Cpu`, `currentState` | Imports / vars | **CONFIRMED DEAD** | None | Removed unused icon import and redundant state binding. |
| `frontend/src/components/ui/ChartPanel.tsx` | `RotateCcw` | Import | **CONFIRMED DEAD** | None | Removed unused icon import. |
| `frontend/src/components/ui/CommandPalette.tsx` | `cn` | Import | **CONFIRMED DEAD** | None | Removed unused utility import. |
| `frontend/src/features/datasets/DatasetsView.tsx` | `formatNumber`, `Search`, `AlertCircle`, `isLoading` | Imports / vars | **CONFIRMED DEAD** | None | Removed unused variables and imports. |
| `frontend/src/hooks/useWebSocket.ts` | `useRef` | Import | **CONFIRMED DEAD** | None | Removed unused React hook import. |
| `frontend/src/features/overview/OverviewView.tsx` | `useEffect`, `isConnected`, `valSteps` | Imports / vars | **CONFIRMED DEAD** | None | Cleaned up unused variables and unused React hook. |
| `frontend/src/features/evaluation/EvaluationView.tsx` | `CheckCircle2`, `Layers` | Imports | **CONFIRMED DEAD** | None | Removed unused icon imports. |
| `frontend/src/features/network-studio/NetworkStudioView.tsx` | `PALETTE_ITEMS`, `Cpu`, `RotateCcw`, `Sparkles`, `cn` | Imports | **CONFIRMED DEAD** | None | Removed unused palette and icon imports. |
| `frontend/src/features/network-studio/components/ArchitectureNode.tsx` | `ArrowRightLeft` | Import | **CONFIRMED DEAD** | None | Removed unused icon import. |
| `frontend/src/features/network-studio/components/ComponentPalette.tsx` | `Sparkles`, `Binary`, `Cpu`, `ShieldCheck` | Imports | **CONFIRMED DEAD** | None | Removed unused icon imports. |
| `frontend/src/features/network-studio/components/ConfigInspector.tsx` | `CheckCircle2`, `AlertTriangle` | Imports | **CONFIRMED DEAD** | None | Removed unused icon imports. |
| `frontend/src/features/runs/RunsView.tsx` | `Sparkles`, `TrendingDown`, `isLoadingMetrics` | Imports / vars | **CONFIRMED DEAD** | None | Removed unused state and icon imports. |
| `frontend/src/features/settings/SettingsView.tsx` | `HardDrive`, `Sliders`, `Palette`, `isLoading` | Imports / vars | **CONFIRMED DEAD** | None | Removed unused state and icon imports. |
| `frontend/src/features/training/TrainingView.tsx` | `Settings`, `Flame`, `latestMetric` | Imports / vars | **CONFIRMED DEAD** | None | Removed unused imports. Setters for early stopping wired to UI inputs. |
| `tests/test_api_endpoints.py` | `@unittest_run_loop` | Decorator | **CONFIRMED DEAD** | Tests | Removed deprecated decorator (aiohttp 3.8+ native async test support). |
| `tests/test_frontend_integration.py` | `@unittest_run_loop` | Decorator | **CONFIRMED DEAD** | Tests | Removed deprecated decorator (aiohttp 3.8+ native async test support). |

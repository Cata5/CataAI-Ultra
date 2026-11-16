"""Builds audit/functional_matrix.json and the markdown table used in AUDIT_REPORT.md."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]

# columns: feature, frontend, api, backend, runtime, persistence, overall, evidence
ROWS = [
    ("Start training — RoCanon decoder", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS",
     "Live run_1794744608 advanced 8996→9063 at ~3.7-5.4k tok/s; worker is a real spawn process writing metrics"),
    ("Start training — custom decoder/hybrid network", "PASS", "PASS", "PASS", "NOT TESTED", "N/A", "NOT TESTED",
     "ModelFactory builds + forward/backward OK (probe P3); not started live to protect the active job"),
    ("Start training — MLP / CNN / classifier networks", "PASS", "PASS", "FAIL", "N/A", "N/A", "FAIL",
     "Worker feeds int64 token IDs and shifted causal CE; deep_mlp_5m/conv1d fail on dtype, bert_encoder_25m fails CE shape (probe P5)"),
    ("Pause training", "PASS", "PASS", "PASS", "NOT TESTED", "N/A", "NOT TESTED",
     "State machine + PAUSE_RUN command implemented; not exercised because a real job is running"),
    ("Resume training", "PASS", "PASS", "PASS", "NOT TESTED", "N/A", "NOT TESTED",
     "RESUME_RUN implemented; not exercised (active job protected)"),
    ("Stop training", "PASS", "PASS", "PASS", "NOT TESTED", "N/A", "NOT TESTED",
     "STOP_RUN + stopped_snapshot implemented; not exercised (active job protected)"),
    ("Manual checkpoint button", "PASS", "PASS", "PASS", "NOT TESTED", "PASS", "NOT TESTED",
     "manual_chk_requested handled in worker; not triggered on the live run"),
    ("Automatic / best / periodic checkpoints", "N/A", "N/A", "PASS", "PASS", "PASS", "PASS",
     "4 real .pt files incl. best_model.pt step 9125, DB checkpoint rows present"),
    ("Checkpoint listing", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS",
     "GET /api/checkpoints returned 4 entries in 0.94 s"),
    ("Checkpoint weight loading", "PARTIAL", "PASS", "PASS", "NOT TESTED", "N/A", "PARTIAL",
     "load_checkpoint(strict=False) + shape-adapt; used by evaluation compare; not run live"),
    ("Optimizer / scheduler / RNG restoration", "N/A", "N/A", "FAIL", "NOT TESTED", "N/A", "FAIL",
     "training_worker passes restore_training_state=False (probe P6b: 0 optimizer entries). RNG restore also fails: 'RNG state must be a torch.ByteTensor' (map_location=cuda moves it to GPU)"),
    ("Delete checkpoint", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS",
     "Unit test deletes a temp checkpoint successfully; DELETE route + UI button wired; not run on real artifacts"),
    ("Dataset upload + validation + analysis", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS",
     "Probe P7a: upload/register OK; .txt/.jsonl/.json validator + tokenizer analyzer are real"),
    ("Dataset list / stats API", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS",
     "GET /api/datasets and /api/datasets/{id}/stats read SQLite"),
    ("Dataset preview in UI", "NOT IMPLEMENTED", "PASS", "PASS", "N/A", "PASS", "PARTIAL",
     "Validator returns previews and they are stored in stats_json, but the UI only renders counts"),
    ("Select an uploaded dataset for training", "FAIL", "PASS", "FAIL", "N/A", "PASS", "FAIL",
     "config-dataset has only 3 hardcoded preset keys; passing the dataset id falls back to synthetic data (probe P7c)"),
    ("Sample a dataset in training", "N/A", "PASS", "FAIL", "N/A", "PASS", "FAIL",
     "max_samples is in StateManager but the Start payload never sends it; no UI control"),
    ("Tokenizer inspector", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS",
     "Exact round-trip asserted; pytest + /api/tokenizer/inspect"),
    ("Tokenizer corpus / vocab", "N/A", "N/A", "PASS", "PASS", "N/A", "PASS",
     "12288 vocab verified by tests; special tokens load"),
    ("Live text generation", "PASS", "PASS", "PASS", "BLOCKED", "N/A", "PARTIAL",
     "Real autoregressive sampler, but always the default randomly-initialised RoCanon; no model/checkpoint selector. Not run live: a second 48M model on a GPU with ~600 MB free would risk OOM for the active job"),
    ("Evaluation model / checkpoint selection", "NOT IMPLEMENTED", "FAIL", "PARTIAL", "N/A", "N/A", "FAIL",
     "handle_eval_generate ignores model/checkpoint fields; InferenceEngine is constructed with RoCanonConfig()"),
    ("Side-by-side checkpoint comparison", "PASS", "PASS", "PASS", "BLOCKED", "N/A", "PARTIAL",
     "Sequential single-instance compare implemented correctly; not run during active training"),
    ("Benchmark suite execution", "PASS", "PASS", "PASS", "BLOCKED", "PASS", "PARTIAL",
     "Real generation + keyword/repetition scoring, persists to benchmarks table; not run during active training"),
    ("Benchmark history retrieval", "NOT IMPLEMENTED", "PASS", "PASS", "N/A", "PASS", "FAIL",
     "GET /api/evaluation/results exists but no UI element calls it; benchmarks table currently empty"),
    ("NVML hardware telemetry", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS",
     "Live: RTX 3060 Laptop, 74°C, 5535 MB used device-wide, 43% util, 1770/6001 MHz, 68.9 W"),
    ("PyTorch allocator statistics", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS",
     "Live: allocated 194 MB / reserved 602 MB vs device-wide 5535 MB — correctly distinguished"),
    ("VRAM alert evaluation", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS",
     "VRAMGuard thresholds + worker logging; live free VRAM 609 MB > 600 MB warning"),
    ("Alert delivery over WebSocket", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS",
     "Loss-spike/gradient events broadcast and rendered; alerts persisted in DB"),
    ("Alert history from DB", "NOT IMPLEMENTED", "PASS", "PASS", "N/A", "PASS", "PARTIAL",
     "GET /api/alerts is never called by the UI; alerts container is WS-only"),
    ("Architecture templates instantiate + forward", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS",
     "rocanon/mini_gpt/hybrid/bert/mlp build and forward; parameter totals match the module exactly"),
    ("CNN template forward pass", "PASS", "PASS", "FAIL", "N/A", "N/A", "FAIL",
     "conv1d_network_4m saved with norm_type=batchnorm; BatchNorm1d receives [B,S,D] → 'running_mean should contain 16 elements not 384'"),
    ("Add / remove / reorder / connect components", "NOT IMPLEMENTED", "N/A", "N/A", "N/A", "N/A", "NOT IMPLEMENTED",
     "Palette entries are inert <span>s; Visual Builder is a dimension form; no graph editing"),
    ("Visual computation graph", "PASS", "N/A", "PASS", "N/A", "N/A", "PASS",
     "renderComputationGraph derives blocks from config (read-only, not editable)"),
    ("Save architecture to registry", "PASS", "PASS", "PARTIAL", "PASS", "PASS",
     "PARTIAL", "Persists atomically, but ParameterCalculator swallows errors so invalid configs save (probe P8 saved n_q=5/n_kv=2)"),
    ("Load / clone / delete architecture", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS",
     "Verified via API + registry; rocanon_48m delete is blocked"),
    ("Export / import architecture", "NOT IMPLEMENTED", "NOT IMPLEMENTED", "NOT IMPLEMENTED", "N/A", "N/A",
     "NOT IMPLEMENTED", "Only a copy/paste JSON textarea; no file import/export"),
    ("Architecture versioning", "NOT IMPLEMENTED", "N/A", "NOT IMPLEMENTED", "N/A", "PARTIAL",
     "NOT IMPLEMENTED", "A static 'version': '1.0' string; no revision history"),
    ("Parameter count (total)", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS",
     "48,531,968 exact for RoCanon; matches sum(p.numel()) for all 6 templates"),
    ("Parameter breakdown per component", "PASS", "PASS", "PARTIAL", "N/A", "N/A", "PARTIAL",
     "Components do not sum to total for any template; RoCanon conv reported 32,768 vs ~278,528 actual; registry list shows stale 4,771,968 for conv1d"),
    ("VRAM estimator", "PASS", "PASS", "PARTIAL", "PASS", "N/A", "PARTIAL",
     "Conservative heuristic (+350 MB fixed workspace); tiny-model estimate 366 MB vs measured 34 MB; UI has no gradient-checkpointing toggle; each call rebuilds the model (~0.9 s, called twice per keystroke)"),
    ("Architecture preflight (forward/backward)", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS",
     "CPU instantiate + forward + backward + gradient coverage check"),
    ("Custom module discovery", "PASS", "PASS", "PASS", "PASS", "N/A", "PARTIAL",
     "gated_conv_unit is discovered; but registry.reload() re-executes module files on every GET"),
    ("Use a custom module in a model", "NOT IMPLEMENTED", "NOT IMPLEMENTED", "NOT IMPLEMENTED", "N/A", "N/A",
     "NOT IMPLEMENTED", "ModelFactory/GenericModularNetwork never references CustomModuleRegistry"),
    ("Generic ModelFactory build/forward/backward", "N/A", "N/A", "PASS", "PASS", "N/A", "PASS",
     "5 families build, forward and propagate gradients (probe P2/P3)"),
    ("Create / list pipelines", "NOT IMPLEMENTED", "PASS", "PASS", "PASS", "PASS", "PARTIAL",
     "CRUD + readiness checks work and persist; no UI and no execution engine"),
    ("Multi-stage pipeline execution", "NOT IMPLEMENTED", "NOT IMPLEMENTED", "NOT IMPLEMENTED", "N/A",
     "NOT IMPLEMENTED", "NOT IMPLEMENTED",
     "No start endpoint, no worker handling of LOAD_PIPELINE/PHASE_COMPLETED, no automatic phase transition"),
    ("Scheduler curve visualizer", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS",
     "WSD/cosine/linear/constant curves computed server-side and charted"),
    ("Runs history table", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS",
     "refreshRunsTable() is called on tab switch; GET /api/runs returns 33 persisted runs"),
    ("Live metric charts", "PASS", "PASS", "PASS", "PASS", "PARTIAL", "PARTIAL",
     "WS STEP_COMPLETED drives charts in real time; after a page refresh only the latest point is seeded — /api/runs/{id}/metrics is never called"),
    ("Settings persistence", "FAIL", "PASS", "PASS", "N/A", "PARTIAL", "PARTIAL",
     "Settings tab is read-only diagnostics; /api/config POST exists but no UI writes it"),
    ("Live learning-rate adjustment", "PASS", "PASS", "PASS", "NOT TESTED", "N/A", "NOT TESTED",
     "SET_LR_MULTIPLIER + dynamic base LR applied each step; not exercised on the live run"),
    ("WebSocket event delivery", "PASS", "PASS", "PASS", "PASS", "N/A", "PASS",
     "Probe: 1 STATE_CHANGED + 8 STEP_COMPLETED in 12 s with full metric payloads; auto-reconnect on close"),
    ("State persistence across restart", "N/A", "N/A", "PASS", "PASS", "PASS", "PASS",
     "SQLite WAL (33 runs, 11,450 metrics, 13 checkpoints) + JSON network registry"),
    ("Security: checkpoint deserialization", "N/A", "N/A", "FAIL", "N/A", "N/A", "FAIL",
     "torch.load(..., weights_only=False) in inspect_checkpoint; load_checkpoint uses default torch.load"),
    ("Security: path traversal in network ids", "N/A", "FAIL", "PASS", "PASS", "N/A", "FAIL",
     "GET /api/networks/..%2F..%2Fsrc%2Ftokenizer%2Ftokenizer_config returned the tokenizer JSON; delete_network('../victim') deleted an out-of-registry file"),
]

STATUSES = ["PASS", "PARTIAL", "FAIL", "MOCKED", "NOT IMPLEMENTED", "NOT TESTED", "BLOCKED"]


def main():
    rows = []
    for r in ROWS:
        rows.append(dict(zip(
            ["feature", "frontend", "api", "backend", "runtime", "persistence", "overall", "evidence"], r)))

    overall = Counter(r["overall"] for r in rows)
    total = len(rows)
    excluded = overall["NOT TESTED"] + overall["BLOCKED"]
    verified_den = total - excluded
    pass_count = overall["PASS"]

    summary = {
        "total_features_audited": total,
        "overall_counts": {k: overall.get(k, 0) for k in STATUSES},
        "denominator_excludes": ["NOT TESTED", "BLOCKED"],
        "verified_denominator": verified_den,
        "fully_functional": pass_count,
        "verified_pass_percent": round(100.0 * pass_count / verified_den, 1) if verified_den else 0.0,
        "raw_pass_percent_of_all_features": round(100.0 * pass_count / total, 1),
    }
    out = {"summary": summary, "features": rows}
    (OUT / "functional_matrix.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = ["| Feature | Frontend | API | Backend | Runtime | Persistence | Overall |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {feature} | {frontend} | {api} | {backend} | {runtime} | {persistence} | **{overall}** |".format(**r))
    (OUT / "matrix_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

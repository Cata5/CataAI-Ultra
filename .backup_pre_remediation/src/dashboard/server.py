"""
src/dashboard/server.py
Real-time aiohttp web server hosting RoCanon-48.5M Studio & Experimentation Platform v2.
Integrates SQLite persistence, decoupled worker process management, NVML hardware telemetry,
dynamic VRAM protection, dataset management, scheduler visualizer, and evaluation lab.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from aiohttp import WSMsgType, web

from ..checkpoints.manager import CheckpointManager
from ..datasets.manager import DatasetManager
from ..db.database import StudioDatabase
from ..eval.benchmarks import BenchmarkManager
from ..eval.inference import InferenceEngine
from ..infer.export import export_to_safetensors
from ..model.architecture import RoCanonConfig, RoCanonForCausalLM
from ..networks import (
    ALL_TEMPLATES,
    ArchitectureValidator,
    BudgetAssistant,
    CustomModuleRegistry,
    MemoryEstimator,
    ModelFactory,
    NetworkRegistry,
    ParameterCalculator,
    get_template_by_id,
)
from ..pipeline.manager import PipelineManager
from ..pipeline.scheduler_math import calculate_schedule_curve
from ..telemetry.hardware import HardwareMonitor
from ..training.factory import calculate_analytical_parameters, list_checkpoints
from ..worker.protocol import CommandType, EventType, RunState, WorkerCommand, WorkerEvent
from ..worker.worker_manager import WorkerManager
from .state import STATE, MetricPoint
from .ui import INDEX_HTML

logger = logging.getLogger(__name__)

# Singletons for backend control plane
DB = StudioDatabase()
WORKER_MGR = WorkerManager(db=DB)
CHECKPOINT_MGR = CheckpointManager(db=DB)
DATASET_MGR = DatasetManager(db=DB)
PIPELINE_MGR = PipelineManager(db=DB)
BENCHMARK_MGR = BenchmarkManager(db=DB)
HW_MONITOR = HardwareMonitor()
NETWORK_REGISTRY = NetworkRegistry()
CUSTOM_MODULE_REG = CustomModuleRegistry()
_INFERENCE_ENGINE: Optional[InferenceEngine] = None

# Active WebSockets
CONNECTED_CLIENTS: Set[web.WebSocketResponse] = set()
MAIN_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


def get_inference_engine() -> InferenceEngine:
    """Lazy initializer for inference engine to avoid holding GPU memory when not in use."""
    global _INFERENCE_ENGINE
    if _INFERENCE_ENGINE is None:
        _INFERENCE_ENGINE = InferenceEngine()
    return _INFERENCE_ENGINE


def broadcast_to_websockets(msg_dict: Dict[str, Any]) -> None:
    """Thread-safe WebSocket broadcaster."""
    global MAIN_EVENT_LOOP
    if MAIN_EVENT_LOOP is None or MAIN_EVENT_LOOP.is_closed():
        try:
            MAIN_EVENT_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            return
    if not CONNECTED_CLIENTS:
        return

    msg_str = json.dumps(msg_dict)

    async def _send():
        for ws in list(CONNECTED_CLIENTS):
            if not ws.closed:
                try:
                    await ws.send_str(msg_str)
                except Exception:
                    pass

    asyncio.run_coroutine_threadsafe(_send(), MAIN_EVENT_LOOP)


# Register worker event broadcaster
def on_worker_event(event: WorkerEvent) -> None:
    broadcast_to_websockets(event.to_dict())


WORKER_MGR.subscribe(on_worker_event)


# ================= HTTP HANDLERS =================

async def handle_index(request: web.Request) -> web.Response:
    """Serves the primary single-page dark technical UI."""
    try:
        from . import ui
        return web.Response(text=ui.INDEX_HTML, content_type="text/html", charset="utf-8")
    except Exception:
        return web.Response(text=INDEX_HTML, content_type="text/html", charset="utf-8")


# ---------------- System & Telemetry ----------------

async def handle_system_status(request: web.Request) -> web.Response:
    """Returns current system and run state machine status."""
    metrics = WORKER_MGR.latest_metrics
    if not metrics and WORKER_MGR.current_run_id:
        db_metric = DB.get_latest_metric(WORKER_MGR.current_run_id)
        if db_metric:
            metrics = db_metric

    state_val = WORKER_MGR.current_state.value
    if WORKER_MGR.current_run_id and state_val == "STOPPING":
        db_run = DB.get_run(WORKER_MGR.current_run_id)
        if db_run and db_run.get("status") in ("STOPPED", "COMPLETED", "FAILED"):
            state_val = db_run["status"]
            try:
                WORKER_MGR.current_state = RunState(state_val)
            except Exception:
                pass

    return web.json_response({
        "state": state_val,
        "run_id": WORKER_MGR.current_run_id,
        "latest_metrics": metrics,
        "alerts_count": len(WORKER_MGR.latest_alerts),
        "timestamp": time.time(),
    })


async def handle_hardware_telemetry(request: web.Request) -> web.Response:
    """Returns genuine NVML & PyTorch hardware telemetry."""
    telem = HW_MONITOR.read_telemetry()
    return web.json_response(telem.to_dict())


async def handle_get_alerts(request: web.Request) -> web.Response:
    """Returns latest alerts from database."""
    run_id = request.query.get("run_id")
    limit = int(request.query.get("limit", 50))
    alerts = DB.list_alerts(run_id=run_id, limit=limit)
    return web.json_response(alerts)


# ---------------- Runs & Training Controls ----------------

async def handle_list_runs(request: web.Request) -> web.Response:
    """Lists runs from database."""
    limit = int(request.query.get("limit", 50))
    runs = DB.list_runs(limit=limit)
    return web.json_response(runs)


async def handle_create_run(request: web.Request) -> web.Response:
    """Starts a new training session with payload."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    success = WORKER_MGR.start_run(data)
    if success:
        return web.json_response({"status": "started", "payload": data})
    else:
        return web.json_response({"error": "Failed to start run; worker busy or invalid state"}, status=409)


async def handle_run_pause(request: web.Request) -> web.Response:
    success = WORKER_MGR.pause_run()
    return web.json_response({"success": success})


async def handle_run_resume(request: web.Request) -> web.Response:
    success = WORKER_MGR.resume_run()
    return web.json_response({"success": success})


async def handle_run_stop(request: web.Request) -> web.Response:
    success = WORKER_MGR.stop_run()
    return web.json_response({"success": success})


async def handle_run_checkpoint(request: web.Request) -> web.Response:
    success = WORKER_MGR.request_checkpoint()
    return web.json_response({"success": success})


async def handle_run_validate(request: web.Request) -> web.Response:
    success = WORKER_MGR.trigger_validation()
    return web.json_response({"success": success})


async def handle_update_lr(request: web.Request) -> web.Response:
    """Updates dynamic learning rate multiplier or values on the fly."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    cmd = WorkerCommand(command_type=CommandType.SET_LR_MULTIPLIER, payload=data)
    success = WORKER_MGR.send_command(cmd)
    return web.json_response({"success": success, "updated": data})


async def handle_run_metrics(request: web.Request) -> web.Response:
    run_id = request.match_info.get("id")
    start_step = int(request.query.get("start_step", 0))
    limit = int(request.query.get("limit", 5000))
    metrics = DB.get_metrics(run_id=run_id, start_step=start_step, max_points=limit)
    return web.json_response(metrics)


# ---------------- Checkpoints ----------------

async def handle_get_checkpoints(request: web.Request) -> web.Response:
    """Scans and returns all checkpoints on disk."""
    list_files = CHECKPOINT_MGR.list_checkpoints_on_disk()
    return web.json_response(list_files)


async def handle_delete_checkpoint(request: web.Request) -> web.Response:
    chk_id = request.match_info.get("id")
    success = CHECKPOINT_MGR.delete_checkpoint(chk_id)
    return web.json_response({"success": success, "id": chk_id})


# ---------------- Pipelines & Schedulers ----------------

async def handle_get_pipelines(request: web.Request) -> web.Response:
    pipes = PIPELINE_MGR.list_pipelines()
    return web.json_response(pipes)


async def handle_create_pipeline(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    name = data.get("name", "Custom Pipeline")
    phases = data.get("phases")
    pid = PIPELINE_MGR.create_pipeline(name=name, phases=phases)
    return web.json_response({"pipeline_id": pid})


async def handle_scheduler_curve(request: web.Request) -> web.Response:
    """Calculates mathematical schedule curve for visualizer."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    curve = calculate_schedule_curve(
        schedule_type=data.get("schedule_type", "wsd"),
        total_steps=int(data.get("total_steps", 1000)),
        peak_lr=float(data.get("peak_lr", 0.003)),
        min_lr=float(data.get("min_lr", 0.0)),
        warmup_steps=int(data.get("warmup_steps", 200)),
        stable_steps=int(data.get("stable_steps", 600)),
        decay_style=data.get("decay_style", "cosine"),
    )
    return web.json_response(curve)


# ---------------- Datasets & Tokenizer ----------------

async def handle_upload_dataset(request: web.Request) -> web.Response:
    """Handles multipart upload of .txt, .jsonl, .json files."""
    reader = await request.multipart()
    field = await reader.next()
    if not field or not field.filename:
        return web.json_response({"error": "No file field found"}, status=400)

    filename = Path(field.filename).name
    temp_dir = Path("data/uploads/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / filename

    with open(temp_path, "wb") as f:
        while chunk := await field.read_chunk():
            f.write(chunk)

    try:
        res = DATASET_MGR.register_uploaded_file(temp_path, name=filename)
        return web.json_response(res)
    finally:
        if temp_path.exists():
            temp_path.unlink()


async def handle_list_datasets(request: web.Request) -> web.Response:
    datasets = DATASET_MGR.list_all_datasets()
    return web.json_response(datasets)


async def handle_get_dataset_stats(request: web.Request) -> web.Response:
    ds_id = request.match_info.get("id")
    ds = DATASET_MGR.get_dataset_details(ds_id)
    if not ds:
        return web.json_response({"error": "Dataset not found"}, status=404)
    return web.json_response(ds)


async def handle_tokenizer_inspect(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    text = data.get("text", "")
    res = DATASET_MGR.inspect_tokenizer_sample(text)
    return web.json_response(res)


# ---------------- Evaluation Lab ----------------

async def handle_eval_generate(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    prompt = data.get("prompt", "")
    max_tokens = int(data.get("max_new_tokens", 128))
    temp = float(data.get("temperature", 0.7))
    top_p = float(data.get("top_p", 0.9))
    top_k = int(data.get("top_k", 50))
    rep_pen = float(data.get("repetition_penalty", 1.1))

    engine = get_inference_engine()
    res = engine.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=rep_pen,
    )
    return web.json_response(res)


async def handle_eval_compare(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    prompt = data.get("prompt", "")
    paths = data.get("checkpoint_paths", [])
    engine = get_inference_engine()
    comparison = engine.compare_checkpoints_sequential(
        prompt=prompt,
        checkpoint_paths=paths,
    )
    return web.json_response({"prompt": prompt, "results": comparison})


async def handle_eval_benchmark(request: web.Request) -> web.Response:
    engine = get_inference_engine()
    results = BENCHMARK_MGR.run_benchmark_suite(engine=engine)
    return web.json_response(results)


async def handle_eval_results(request: web.Request) -> web.Response:
    limit = int(request.query.get("limit", 100))
    res = DB.list_benchmarks(limit=limit)
    return web.json_response(res)


# ---------------- Network Studio & Registry ----------------

async def handle_list_networks(request: web.Request) -> web.Response:
    networks = NETWORK_REGISTRY.list_networks()
    return web.json_response(networks)


async def handle_get_network_templates(request: web.Request) -> web.Response:
    return web.json_response(ALL_TEMPLATES)


async def handle_get_network(request: web.Request) -> web.Response:
    net_id = request.match_info.get("id")
    net = NETWORK_REGISTRY.get_network(net_id)
    if not net:
        return web.json_response({"error": "Network not found"}, status=404)
    return web.json_response(net)


async def handle_save_network(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    net_id = data.get("id", f"network_{int(time.time())}")
    saved = NETWORK_REGISTRY.save_network(net_id, data)
    return web.json_response(saved)


async def handle_clone_network(request: web.Request) -> web.Response:
    orig_id = request.match_info.get("id")
    try:
        data = await request.json()
    except Exception:
        data = {}
    new_id = data.get("new_id", f"{orig_id}_clone")
    new_name = data.get("new_name", f"Clone of {orig_id}")
    cloned = NETWORK_REGISTRY.clone_network(orig_id, new_id, new_name)
    if not cloned:
        return web.json_response({"error": "Failed to clone network"}, status=400)
    return web.json_response(cloned)


async def handle_delete_network(request: web.Request) -> web.Response:
    net_id = request.match_info.get("id")
    success = NETWORK_REGISTRY.delete_network(net_id)
    return web.json_response({"success": success, "id": net_id})


async def handle_validate_network(request: web.Request) -> web.Response:
    try:
        config = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    res = ArchitectureValidator.run_preflight_check(config, device="cpu", batch_size=2, seq_len=16)
    return web.json_response(res)


async def handle_estimate_network(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    config = data.get("config", data)
    bs = int(data.get("micro_batch_size", 4))
    sl = int(data.get("seq_len", 256))
    dt = data.get("dtype", "bfloat16")
    opt = data.get("optimizer", "muon_adamw")
    gc = bool(data.get("gradient_checkpointing", False))

    params = ParameterCalculator.calculate(config)
    mem = MemoryEstimator.estimate(
        config=config,
        micro_batch_size=bs,
        seq_len=sl,
        dtype=dt,
        optimizer=opt,
        gradient_checkpointing=gc,
    )
    return web.json_response({"parameters": params, "memory": mem})


async def handle_budget_recommend(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        data = {}
    target = int(data.get("target_params", 15_000_000))
    rec = BudgetAssistant.recommend(target)
    return web.json_response(rec)


async def handle_custom_modules(request: web.Request) -> web.Response:
    CUSTOM_MODULE_REG.reload()
    mods = CUSTOM_MODULE_REG.list_modules()
    return web.json_response(mods)


# ---------------- Legacy Compatibility Routes ----------------

async def handle_get_state(request: web.Request) -> web.Response:
    """Legacy state snapshot handler."""
    last_pt = STATE.metrics_history[-1] if STATE.metrics_history else None
    return web.json_response({
        "status": {
            "is_training": WORKER_MGR.current_state == RunState.RUNNING,
            "is_paused": WORKER_MGR.current_state == RunState.PAUSED,
            "current_step": WORKER_MGR.latest_metrics.get("step", 0),
            "total_tokens_processed": WORKER_MGR.latest_metrics.get("tokens", 0),
            "current_stage": "RoCanon Studio v2",
            "active_device": "cuda" if HW_MONITOR.read_telemetry().cuda_available else "cpu",
        },
        "latest_metric": WORKER_MGR.latest_metrics,
        "overfitting": {
            "status": "HEALTHY",
            "best_val": None,
            "best_step": None,
            "counter": 0,
            "patience": 5,
        }
    })


async def handle_get_config(request: web.Request) -> web.Response:
    return web.json_response(STATE.get_full_config())


async def handle_post_config(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        section = data.get("section")
        updates = data.get("updates", {})
        if section and updates:
            return web.json_response(STATE.update_config_section(section, updates))
    except Exception:
        pass
    return web.json_response(STATE.get_full_config())


async def handle_calculate_params(request: web.Request) -> web.Response:
    q = request.query
    stats = calculate_analytical_parameters(
        vocab_size=int(q.get("vocab_size", 12288)),
        d_model=int(q.get("d_model", 512)),
        n_layers=int(q.get("n_layers", 16)),
        n_q=int(q.get("n_q", 8)),
        n_kv=int(q.get("n_kv", 2)),
        d_ff=int(q.get("d_ff", 1280)),
        kernel_size=int(q.get("kernel_size", 4)),
        tie_word_embeddings=q.get("tied", "true").lower() == "true",
    )
    return web.json_response(stats)


async def handle_get_dataset_info(request: web.Request) -> web.Response:
    from ..data.custom_loader import CustomDatasetEngine
    src = STATE.dataset_source
    engine = CustomDatasetEngine()
    p = Path(engine.KNOWN_DATASETS.get(src, src))
    exists = p.exists()
    sub_tag = "_full"
    cache_file = engine.cache_dir / f"{p.stem}{sub_tag}_tokens.pt"
    is_cached = cache_file.exists()
    cache_size_mb = round(cache_file.stat().st_size / (1024 * 1024), 2) if is_cached else 0
    return web.json_response({
        "name": p.name if exists else src,
        "exists": exists,
        "path": str(p.resolve()) if exists else str(p),
        "size_mb": round(p.stat().st_size / (1024 * 1024), 2) if exists else 0,
        "chars": p.stat().st_size if exists else 0,
        "cached": is_cached,
        "cache_filename": cache_file.name if is_cached else "None",
        "cache_size_mb": cache_size_mb,
        "preview": "Pre-tokenized dataset cache ready" if is_cached else ("Dataset ready" if exists else "File not found"),
        "total_tokens": 48419639 if is_cached else STATE.dataset_total_tokens,
        "train_tokens": 48419639 if is_cached else STATE.dataset_train_tokens,
        "val_tokens": 2548402 if is_cached else STATE.dataset_val_tokens,
    })


async def handle_action(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    action = data.get("action")
    if action == "start_run":
        WORKER_MGR.start_run(data)
        return web.json_response({"status": "started"})
    elif action in ("pause", "toggle_pause"):
        WORKER_MGR.pause_run()
        return web.json_response({"status": "paused"})
    elif action == "resume":
        WORKER_MGR.resume_run()
        return web.json_response({"status": "resumed"})
    elif action == "stop":
        WORKER_MGR.stop_run()
        return web.json_response({"status": "stopped"})
    elif action == "manual_checkpoint":
        WORKER_MGR.request_checkpoint()
        return web.json_response({"status": "checkpoint_requested"})
    elif action == "trigger_val":
        WORKER_MGR.trigger_validation()
        return web.json_response({"status": "validation_triggered"})
    elif action == "set_lr_multiplier":
        mult = float(data.get("multiplier", 1.0))
        WORKER_MGR.set_lr_multiplier(mult)
        return web.json_response({"lr_multiplier": mult})
    elif action == "chat_message":
        msg = data.get("message", "")
        engine = get_inference_engine()
        res = engine.generate(prompt=msg, max_new_tokens=64)
        return web.json_response({"reply": res["generated_text"]})
    return web.json_response({"error": f"Unknown action: {action}"}, status=400)


# ---------------- WebSocket ----------------

async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    CONNECTED_CLIENTS.add(ws)

    # Send initial status
    try:
        await ws.send_str(json.dumps({
            "event_type": "STATE_CHANGED",
            "payload": {
                "new_state": WORKER_MGR.current_state.value,
                "run_id": WORKER_MGR.current_run_id,
            }
        }))
    except Exception:
        pass

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                pass
            elif msg.type == WSMsgType.ERROR:
                logger.error("WebSocket error: %s", ws.exception())
    finally:
        CONNECTED_CLIENTS.discard(ws)

    return ws


def create_app() -> web.Application:
    """Builds and returns the configured aiohttp Application."""
    app = web.Application(client_max_size=1024 * 1024 * 500)  # Support up to 500MB dataset uploads

    async def _on_startup(app_instance: web.Application) -> None:
        global MAIN_EVENT_LOOP
        MAIN_EVENT_LOOP = asyncio.get_running_loop()
        STATE.set_event_loop(MAIN_EVENT_LOOP)

    app.on_startup.append(_on_startup)

    # UI
    app.router.add_get("/", handle_index)

    # Core REST APIs
    app.router.add_get("/api/system/status", handle_system_status)
    app.router.add_get("/api/hardware", handle_hardware_telemetry)
    app.router.add_get("/api/alerts", handle_get_alerts)

    app.router.add_get("/api/runs", handle_list_runs)
    app.router.add_post("/api/runs", handle_create_run)
    app.router.add_post("/api/runs/{id}/start", handle_create_run)
    app.router.add_post("/api/runs/{id}/pause", handle_run_pause)
    app.router.add_post("/api/runs/{id}/resume", handle_run_resume)
    app.router.add_post("/api/runs/{id}/stop", handle_run_stop)
    app.router.add_post("/api/runs/{id}/checkpoint", handle_run_checkpoint)
    app.router.add_post("/api/runs/{id}/validate", handle_run_validate)
    app.router.add_post("/api/runs/lr", handle_update_lr)
    app.router.add_post("/api/runs/{id}/lr", handle_update_lr)
    app.router.add_get("/api/runs/{id}/metrics", handle_run_metrics)

    app.router.add_get("/api/checkpoints", handle_get_checkpoints)
    app.router.add_delete("/api/checkpoints/{id}", handle_delete_checkpoint)

    app.router.add_get("/api/pipelines", handle_get_pipelines)
    app.router.add_post("/api/pipelines", handle_create_pipeline)
    app.router.add_post("/api/scheduler/curve", handle_scheduler_curve)

    app.router.add_post("/api/datasets/upload", handle_upload_dataset)
    app.router.add_get("/api/datasets", handle_list_datasets)
    app.router.add_get("/api/datasets/{id}/stats", handle_get_dataset_stats)
    app.router.add_post("/api/tokenizer/inspect", handle_tokenizer_inspect)

    app.router.add_post("/api/evaluation/generate", handle_eval_generate)
    app.router.add_post("/api/evaluation/compare", handle_eval_compare)
    app.router.add_post("/api/evaluation/benchmark", handle_eval_benchmark)
    app.router.add_get("/api/evaluation/results", handle_eval_results)

    # Network Studio & Registry APIs
    app.router.add_get("/api/networks", handle_list_networks)
    app.router.add_get("/api/networks/templates", handle_get_network_templates)
    app.router.add_get("/api/networks/custom_modules", handle_custom_modules)
    app.router.add_post("/api/networks/validate", handle_validate_network)
    app.router.add_post("/api/networks/estimate", handle_estimate_network)
    app.router.add_post("/api/networks/budget_recommend", handle_budget_recommend)
    app.router.add_get("/api/networks/{id}", handle_get_network)
    app.router.add_post("/api/networks", handle_save_network)
    app.router.add_post("/api/networks/{id}/clone", handle_clone_network)
    app.router.add_delete("/api/networks/{id}", handle_delete_network)

    # Legacy routes
    app.router.add_get("/api/state", handle_get_state)
    app.router.add_get("/api/dataset_info", handle_get_dataset_info)
    app.router.add_get("/api/config", handle_get_config)
    app.router.add_post("/api/config", handle_post_config)
    app.router.add_get("/api/calculate_params", handle_calculate_params)
    app.router.add_post("/api/action", handle_action)

    # WebSocket
    app.router.add_get("/ws", handle_websocket)

    return app


async def start_dashboard_server(host: str = "127.0.0.1", port: int = 8080) -> web.AppRunner:
    """Starts the studio server asynchronously on the given host and port."""
    global MAIN_EVENT_LOOP
    MAIN_EVENT_LOOP = asyncio.get_running_loop()
    STATE.set_event_loop(MAIN_EVENT_LOOP)

    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("RoCanon Studio v2 listening at http://%s:%d", host, port)
    return runner

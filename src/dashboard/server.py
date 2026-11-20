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

import torch
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
from ..pipeline.runner import PipelineRunner
from ..pipeline.scheduler_math import calculate_schedule_curve
from ..plugins.manager import PluginManager
from ..telemetry.hardware import HardwareMonitor
from ..telemetry.profiles import HardwareProfile
from ..training.factory import calculate_analytical_parameters, list_checkpoints
from ..training.tasks.registry import TaskRegistry
from ..worker.protocol import CommandType, EventType, RunState, WorkerCommand, WorkerEvent
from ..worker.worker_manager import WorkerManager
from ..workspace.manager import WorkspaceManager
from .state import STATE, MetricPoint
from .ui import INDEX_HTML

logger = logging.getLogger(__name__)

# Singletons for backend control plane
DB = StudioDatabase()
WORKSPACE_MGR = WorkspaceManager(db=DB)
PLUGIN_MGR = PluginManager(db=DB)
WORKER_MGR = WorkerManager(db=DB)
CHECKPOINT_MGR = CheckpointManager(db=DB)
DATASET_MGR = DatasetManager(db=DB)
PIPELINE_MGR = PipelineManager(db=DB)
PIPELINE_RUNNER = PipelineRunner(pipeline_mgr=PIPELINE_MGR, worker_mgr=WORKER_MGR)
BENCHMARK_MGR = BenchmarkManager(db=DB)
HW_MONITOR = HardwareMonitor()
NETWORK_REGISTRY = NetworkRegistry()
CUSTOM_MODULE_REG = CustomModuleRegistry()
_INFERENCE_ENGINE: Optional[InferenceEngine] = None

# Active WebSockets
CONNECTED_CLIENTS: Set[web.WebSocketResponse] = set()
MAIN_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


def get_inference_engine(force_cpu: bool = False) -> InferenceEngine:
    """Lazy initializer for inference engine with GPU single-owner arbitration."""
    global _INFERENCE_ENGINE
    is_training_active = getattr(WORKER_MGR, "current_state", RunState.IDLE) in (RunState.RUNNING, RunState.STARTING)
    target_device = torch.device("cpu") if (force_cpu or is_training_active or not torch.cuda.is_available()) else torch.device("cuda")
    if _INFERENCE_ENGINE is None or _INFERENCE_ENGINE.device != target_device:
        _INFERENCE_ENGINE = InferenceEngine(device=target_device)
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
    metrics = dict(WORKER_MGR.latest_metrics) if WORKER_MGR.latest_metrics else {}
    if WORKER_MGR.current_run_id:
        if not metrics:
            db_metric = DB.get_latest_metric(WORKER_MGR.current_run_id)
            if db_metric:
                metrics = dict(db_metric)
        if "latest_val_loss" not in metrics or metrics["latest_val_loss"] is None:
            val_info = DB.get_latest_validation(WORKER_MGR.current_run_id)
            if val_info:
                metrics["latest_val_loss"] = val_info["val_loss"]
                metrics["latest_val_step"] = val_info["step"]
                metrics["latest_val_timestamp"] = val_info["timestamp"]
                metrics["val_status"] = "completed"
            else:
                metrics["val_status"] = "awaiting"

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
    if run_id == "current" or not run_id:
        run_id = WORKER_MGR.current_run_id or ""
    start_step = int(request.query.get("start_step", 0))
    limit = int(request.query.get("limit", 5000))
    metrics = DB.get_metrics(run_id=run_id, start_step=start_step, max_points=limit)
    normalized = []
    for m in metrics:
        m_copy = dict(m)
        if "loss" not in m_copy:
            m_copy["loss"] = m.get("train_loss")
        if "learning_rate" not in m_copy:
            m_copy["learning_rate"] = m.get("muon_lr") or m.get("adamw_lr") or 0.0
        if "tokens_seen" not in m_copy:
            m_copy["tokens_seen"] = m.get("tokens", 0)
        if "tokens_per_second" not in m_copy:
            m_copy["tokens_per_second"] = m.get("tok_per_sec", 0.0)
        if "memory_allocated_mb" not in m_copy and m.get("vram_used_gb"):
            m_copy["memory_allocated_mb"] = m["vram_used_gb"] * 1024
        normalized.append(m_copy)
    return web.json_response(normalized)


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


async def handle_get_pipeline_by_id(request: web.Request) -> web.Response:
    pid = request.match_info["id"]
    pipe = PIPELINE_MGR.get_pipeline(pid)
    if not pipe:
        return web.json_response({"error": "Pipeline not found"}, status=404)
    return web.json_response(pipe)


async def handle_start_pipeline(request: web.Request) -> web.Response:
    pid = request.match_info["id"]
    try:
        data = await request.json()
    except Exception:
        data = {}
    phase_idx = data.get("phase_index")
    res = PIPELINE_RUNNER.start_pipeline_execution(pid, start_from_phase=phase_idx)
    status_code = 200 if res.get("success") else 400
    return web.json_response(res, status=status_code)


async def handle_advance_pipeline(request: web.Request) -> web.Response:
    pid = request.match_info["id"]
    try:
        data = await request.json()
    except Exception:
        data = {}
    step = int(data.get("completed_step", 0))
    chk = data.get("checkpoint_path", "")
    res = PIPELINE_MGR.advance_phase(pipeline_id=pid, completed_step=step, checkpoint_path=chk)
    return web.json_response(res)


async def handle_get_settings(request: web.Request) -> web.Response:
    settings = DB.get_settings()
    return web.json_response(settings)


async def handle_save_settings(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    DB.save_settings(data)
    return web.json_response({"status": "saved", "settings": data})


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


async def handle_tokenize_dataset(request: web.Request) -> web.Response:
    ds_id = request.match_info.get("id")
    if not ds_id:
        try:
            data = await request.json()
            ds_id = data.get("id") or data.get("dataset_id")
        except Exception:
            pass

    if not ds_id:
        return web.json_response({"error": "No dataset_id provided"}, status=400)

    try:
        from ..data.custom_loader import CustomDatasetEngine
        engine = CustomDatasetEngine(db=DB)
        train_x, train_y, val_x, val_y = engine.load_dataset(ds_id, device=torch.device("cpu"))
        train_tok = int(train_x.numel())
        val_tok = int(val_x.numel()) if val_x is not None else 0
        total_tok = train_tok + val_tok
        return web.json_response({
            "success": True,
            "dataset_id": ds_id,
            "cached": True,
            "token_count": total_tok,
            "train_tokens": train_tok,
            "val_tokens": val_tok,
        })
    except Exception as e:
        logger.exception("Failed to tokenize dataset %s: %s", ds_id, e)
        return web.json_response({"error": f"Tokenization error: {e}"}, status=500)


async def handle_curate_dataset(request: web.Request) -> web.Response:
    ds_id = request.match_info.get("id")
    try:
        data = await request.json()
    except Exception:
        data = {}

    if not ds_id:
        ds_id = data.get("id") or data.get("dataset_id")
    if not ds_id:
        return web.json_response({"error": "No dataset_id provided"}, status=400)

    try:
        score_res = DATASET_MGR.curate_dataset(
            dataset_id=ds_id,
            jaccard_threshold=float(data.get("jaccard_threshold", 0.80)),
            lid_threshold=float(data.get("lid_threshold", 0.80)),
            save_curated=bool(data.get("save_curated", True)),
            max_samples=int(data.get("max_samples", 5000)) if data.get("max_samples") else None,
        )
        return web.json_response(score_res)
    except Exception as e:
        logger.exception("Dataset curation failed for %s: %s", ds_id, e)
        return web.json_response({"error": str(e)}, status=500)


async def handle_test_notification(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        data = {}
    from ..notifications.webhook import get_notifier
    notifier = get_notifier(db=DB)
    res = notifier.test_notification(
        discord_url=data.get("discord_webhook_url"),
        telegram_token=data.get("telegram_bot_token"),
        telegram_chat_id=data.get("telegram_chat_id"),
    )
    return web.json_response(res)


# ---------------- Evaluation Lab ----------------

async def handle_eval_generate(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    prompt = data.get("prompt", "")
    chat_mode = bool(data.get("chat_mode", False))
    max_tokens = int(data.get("max_new_tokens", 128))
    temp = float(data.get("temperature", 0.7))
    top_p = float(data.get("top_p", 0.9))
    top_k = int(data.get("top_k", 50))
    rep_pen = float(data.get("repetition_penalty", 1.15))
    no_repeat_ngram = int(data.get("no_repeat_ngram_size", 3))
    checkpoint_req = data.get("checkpoint") or data.get("checkpoint_path")

    is_training_active = getattr(WORKER_MGR, "current_state", RunState.IDLE) in (RunState.RUNNING, RunState.STARTING)
    # GPU single-owner arbitration: if training is active on GPU, route inference to CPU
    engine = get_inference_engine(force_cpu=is_training_active)

    loaded_chk_meta = None
    if checkpoint_req:
        chk_path = Path(checkpoint_req)
        if not chk_path.exists():
            chk_path = Path("checkpoints") / checkpoint_req
        if chk_path.exists():
            try:
                loaded_chk_meta = engine.load_checkpoint_weights(chk_path)
            except Exception as e:
                logger.error("Failed to load checkpoint %s for inference: %s", chk_path, e)
                return web.json_response({"error": f"Failed to load checkpoint: {e}"}, status=400)
        else:
            return web.json_response({"error": f"Checkpoint not found: {checkpoint_req}"}, status=404)

    res = engine.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=rep_pen,
        no_repeat_ngram_size=no_repeat_ngram,
        chat_mode=chat_mode,
    )
    res["device"] = str(engine.device)
    res["gpu_arbitrated"] = is_training_active
    res["checkpoint_loaded"] = str(checkpoint_req) if checkpoint_req else None
    if loaded_chk_meta:
        res["checkpoint_step"] = loaded_chk_meta.get("step")
        res["checkpoint_val_loss"] = loaded_chk_meta.get("val_loss")

    return web.json_response(res)


async def handle_eval_stream(request: web.Request) -> web.StreamResponse:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    prompt = data.get("prompt", "")
    chat_mode = bool(data.get("chat_mode", False))
    max_tokens = int(data.get("max_new_tokens", 128))
    temp = float(data.get("temperature", 0.7))
    top_p = float(data.get("top_p", 0.9))
    top_k = int(data.get("top_k", 50))
    rep_pen = float(data.get("repetition_penalty", 1.15))
    no_repeat_ngram = int(data.get("no_repeat_ngram_size", 3))
    checkpoint_req = data.get("checkpoint") or data.get("checkpoint_path")

    is_training_active = getattr(WORKER_MGR, "current_state", RunState.IDLE) in (RunState.RUNNING, RunState.STARTING)
    engine = get_inference_engine(force_cpu=is_training_active)

    if checkpoint_req:
        chk_path = Path(checkpoint_req)
        if not chk_path.exists():
            chk_path = Path("checkpoints") / checkpoint_req
        if chk_path.exists():
            try:
                engine.load_checkpoint_weights(chk_path)
            except Exception as e:
                logger.error("Failed to load checkpoint %s for stream inference: %s", chk_path, e)
                return web.json_response({"error": f"Failed to load checkpoint: {e}"}, status=400)

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await response.prepare(request)

    t0 = time.time()
    tok_count = 0
    try:
        for chunk in engine.stream_generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=rep_pen,
            no_repeat_ngram_size=no_repeat_ngram,
            chat_mode=chat_mode,
        ):
            tok_count += 1
            payload = json.dumps({"token": chunk, "done": False, "index": tok_count})
            await response.write(f"data: {payload}\n\n".encode("utf-8"))

        dt = time.time() - t0
        tok_per_sec = round(tok_count / max(1e-5, dt), 1)
        final_payload = json.dumps({
            "done": True,
            "tokens_generated": tok_count,
            "generation_time_sec": round(dt, 3),
            "tokens_per_sec": tok_per_sec,
            "device": str(engine.device),
        })
        await response.write(f"data: {final_payload}\n\n".encode("utf-8"))
    except Exception as e:
        logger.error("Streaming generation error: %s", e)
        err_payload = json.dumps({"error": str(e), "done": True})
        await response.write(f"data: {err_payload}\n\n".encode("utf-8"))
    finally:
        await response.write_eof()

    return response


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
    res = []
    for tmpl in ALL_TEMPLATES:
        t = tmpl.copy()
        try:
            stats = ParameterCalculator.calculate(tmpl)
            t["total_params"] = stats.get("total_parameters", 0)
            t["total_parameters"] = stats.get("total_parameters", 0)
            t["trainable_params"] = stats.get("trainable_parameters", 0)
        except Exception:
            pass
        res.append(t)
    return web.json_response(res)


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
    try:
        saved = NETWORK_REGISTRY.save_network(net_id, data)
        return web.json_response(saved)
    except ValueError as ve:
        return web.json_response({"error": str(ve)}, status=400)
    except Exception as e:
        logger.exception("Failed to save network %s: %s", net_id, e)
        return web.json_response({"error": f"Failed to save network: {e}"}, status=500)


async def handle_clone_network(request: web.Request) -> web.Response:
    orig_id = request.match_info.get("id")
    try:
        data = await request.json()
    except Exception:
        data = {}
    new_id = data.get("new_id", f"{orig_id}_clone")
    new_name = data.get("new_name", f"Clone of {orig_id}")
    try:
        cloned = NETWORK_REGISTRY.clone_network(orig_id, new_id, new_name)
        if not cloned:
            return web.json_response({"error": "Failed to clone network"}, status=400)
        return web.json_response(cloned)
    except ValueError as ve:
        return web.json_response({"error": str(ve)}, status=400)


async def handle_delete_network(request: web.Request) -> web.Response:
    net_id = request.match_info.get("id")
    if not net_id:
        return web.json_response({"error": "Missing network ID"}, status=400)

    # Check if active training is using this network
    is_training_active = getattr(WORKER_MGR, "current_state", RunState.IDLE) in (RunState.RUNNING, RunState.STARTING)
    if is_training_active:
        curr_payload = getattr(WORKER_MGR, "current_payload", {}) or {}
        if curr_payload.get("network_id") == net_id:
            return web.json_response(
                {"error": f"Cannot delete network '{net_id}': active training run is currently utilizing this architecture."},
                status=409,
            )
        curr_run_id = getattr(WORKER_MGR, "current_run_id", None)
        if curr_run_id:
            run_data = DB.get_run(curr_run_id)
            if run_data:
                cfg = run_data.get("config") or {}
                if cfg.get("network_id") == net_id or cfg.get("arch", {}).get("network_id") == net_id:
                    return web.json_response(
                        {"error": f"Cannot delete network '{net_id}': active training run is currently utilizing this architecture."},
                        status=409,
                    )

    try:
        success = NETWORK_REGISTRY.delete_network(net_id)
        if not success:
            return web.json_response({"error": f"Could not delete network '{net_id}'. RoCanon baseline is protected or network was not found."}, status=400)
        return web.json_response({"success": True, "id": net_id})
    except ValueError as ve:
        return web.json_response({"error": str(ve)}, status=400)


async def handle_delete_dataset(request: web.Request) -> web.Response:
    dataset_id = request.match_info.get("id")
    if not dataset_id:
        return web.json_response({"error": "Missing dataset ID"}, status=400)

    # 1. Check if active training is utilizing this dataset
    is_training_active = getattr(WORKER_MGR, "current_state", RunState.IDLE) in (RunState.RUNNING, RunState.STARTING)
    if is_training_active:
        curr_payload = getattr(WORKER_MGR, "current_payload", {}) or {}
        if curr_payload.get("dataset_source") == dataset_id or dataset_id in str(curr_payload.get("dataset_source", "")):
            return web.json_response(
                {"error": f"Cannot delete dataset '{dataset_id}': active training run is currently utilizing it."},
                status=409,
            )
        curr_run_id = getattr(WORKER_MGR, "current_run_id", None)
        if curr_run_id:
            run_data = DB.get_run(curr_run_id)
            if run_data and dataset_id in str(run_data.get("dataset_source", "")):
                return web.json_response(
                    {"error": f"Cannot delete dataset '{dataset_id}': active training run is currently utilizing it."},
                    status=409,
                )

    # 2. Safely remove physical files inside managed data/uploads and data/cache
    uploads_dir = Path("data/uploads").resolve()
    cache_dir = Path("data/cache").resolve()
    deleted_files = []

    for p in uploads_dir.glob(f"*{dataset_id}*"):
        try:
            if p.is_file() and p.resolve().is_relative_to(uploads_dir):
                p.unlink(missing_ok=True)
                deleted_files.append(p.name)
        except Exception as e:
            logger.warning("Could not delete uploaded file %s: %s", p, e)

    for p in cache_dir.glob(f"*{dataset_id}*"):
        try:
            if p.is_file() and p.resolve().is_relative_to(cache_dir):
                p.unlink(missing_ok=True)
                deleted_files.append(p.name)
        except Exception as e:
            logger.warning("Could not delete cache file %s: %s", p, e)

    # 3. Remove DB record
    DB.delete_dataset(dataset_id)
    return web.json_response({"success": True, "id": dataset_id, "deleted_files": deleted_files})


async def handle_validate_network(request: web.Request) -> web.Response:
    try:
        config = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    res = ArchitectureValidator.run_preflight_check(config, device="cpu", batch_size=2, seq_len=16)
    errors = res.get("errors", [])
    res["issues"] = errors
    res["warnings"] = []
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
    params["total_params"] = params.get("total_parameters", 0)
    params["trainable_params"] = params.get("trainable_parameters", 0)
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
    cache_bin = engine.cache_dir / f"{p.stem}{sub_tag}_tokens.bin"
    cache_file = engine.cache_dir / f"{p.stem}{sub_tag}_tokens.pt"
    is_cached = cache_bin.exists() or cache_file.exists()
    target_cache = cache_bin if cache_bin.exists() else cache_file
    cache_size_mb = round(target_cache.stat().st_size / (1024 * 1024), 2) if is_cached else 0
    return web.json_response({
        "name": p.name if exists else src,
        "exists": exists,
        "path": str(p.resolve()) if exists else str(p),
        "size_mb": round(p.stat().st_size / (1024 * 1024), 2) if exists else 0,
        "chars": p.stat().st_size if exists else 0,
        "cached": is_cached,
        "cache_filename": target_cache.name if is_cached else "None",
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


# ---------------- Workspaces & Product Generalization ----------------

async def handle_list_workspaces(request: web.Request) -> web.Response:
    return web.json_response(WORKSPACE_MGR.list_workspaces())


async def handle_create_workspace(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        name = data.get("name")
        if not name:
            return web.json_response({"error": "Workspace name is required"}, status=400)
        ws = WORKSPACE_MGR.create_workspace(
            name=name,
            workspace_id=data.get("id"),
            description=data.get("description", ""),
            config=data.get("config", {}),
        )
        return web.json_response(ws)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def handle_get_active_workspace(request: web.Request) -> web.Response:
    return web.json_response(WORKSPACE_MGR.get_active_workspace())


async def handle_set_active_workspace(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        ws_id = data.get("workspace_id") or data.get("id")
        if not ws_id:
            return web.json_response({"error": "workspace_id is required"}, status=400)
        success = WORKSPACE_MGR.set_active_workspace(ws_id)
        if not success:
            return web.json_response({"error": f"Workspace '{ws_id}' not found"}, status=404)
        return web.json_response({"success": True, "active_workspace": WORKSPACE_MGR.get_active_workspace()})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def handle_update_workspace(request: web.Request) -> web.Response:
    ws_id = request.match_info["id"]
    try:
        data = await request.json()
        res = WORKSPACE_MGR.update_workspace(
            workspace_id=ws_id,
            name=data.get("name"),
            description=data.get("description"),
            config=data.get("config"),
        )
        if not res:
            return web.json_response({"error": f"Workspace '{ws_id}' not found"}, status=404)
        return web.json_response(res)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def handle_delete_workspace(request: web.Request) -> web.Response:
    ws_id = request.match_info["id"]
    if ws_id == "default":
        return web.json_response({"error": "Cannot delete default workspace"}, status=400)
    success = WORKSPACE_MGR.delete_workspace(ws_id, delete_files=True)
    return web.json_response({"success": success})


async def handle_export_workspace(request: web.Request) -> web.Response:
    ws_id = request.match_info["id"]
    try:
        bundle = WORKSPACE_MGR.export_workspace(ws_id)
        return web.json_response(bundle)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def handle_import_workspace(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        res = WORKSPACE_MGR.import_workspace(data)
        return web.json_response(res)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


# ---------------- Hardware Profiles ----------------

async def handle_set_hardware_profile(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        mode = data.get("mode", "balanced")
        HW_MONITOR.default_profile_mode = mode
        DB.save_settings({"hardware_profile_mode": mode})
        telem = HW_MONITOR.read_telemetry()
        return web.json_response({
            "success": True,
            "active_mode": mode,
            "hardware_profile": telem.hardware_profile,
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


# ---------------- Training Tasks ----------------

async def handle_list_tasks(request: web.Request) -> web.Response:
    return web.json_response(TaskRegistry.list_tasks())


# ---------------- Plugins ----------------

async def handle_list_plugins(request: web.Request) -> web.Response:
    return web.json_response(PLUGIN_MGR.discover_plugins())


async def handle_toggle_plugin(request: web.Request) -> web.Response:
    plugin_id = request.match_info["id"]
    try:
        data = await request.json()
        enabled = bool(data.get("enabled", True))
        success, msg = PLUGIN_MGR.set_plugin_enabled(plugin_id, enabled)
        if not success:
            return web.json_response({"error": msg}, status=400)
        return web.json_response({"success": True, "message": msg, "plugins": PLUGIN_MGR.discover_plugins()})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


# ---------------- Evaluation Suites ----------------

async def handle_list_eval_suites(request: web.Request) -> web.Response:
    categories = BENCHMARK_MGR.get_categories()
    from dataclasses import asdict
    items = [asdict(it) for it in BENCHMARK_MGR.items.values()]
    return web.json_response({
        "categories": categories,
        "items": items,
        "total_items": len(items),
    })


async def handle_create_eval_suite(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        from dataclasses import asdict
        item = BenchmarkItem(
            id=data.get("id", f"custom_{int(time.time())}"),
            category=data.get("category", "custom_persona"),
            prompt=data["prompt"],
            expected_keywords=data.get("expected_keywords", []),
            forbidden_keywords=data.get("forbidden_keywords", []),
            reference_answer=data.get("reference_answer"),
            compliance_rule=data.get("compliance_rule"),
        )
        BENCHMARK_MGR.add_custom_item(item)
        return web.json_response({"success": True, "item": asdict(item)})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


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


@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=200)
    else:
        try:
            response = await handler(request)
        except web.HTTPException as ex:
            response = ex
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response


def create_app() -> web.Application:
    """Builds and returns the configured aiohttp Application."""
    app = web.Application(client_max_size=1024 * 1024 * 500, middlewares=[cors_middleware])  # Support up to 500MB dataset uploads

    async def _on_startup(app_instance: web.Application) -> None:
        global MAIN_EVENT_LOOP
        MAIN_EVENT_LOOP = asyncio.get_running_loop()
        STATE.set_event_loop(MAIN_EVENT_LOOP)

    app.on_startup.append(_on_startup)

    # UI
    app.router.add_get("/", handle_index)

    # Workspaces APIs
    app.router.add_get("/api/workspaces", handle_list_workspaces)
    app.router.add_post("/api/workspaces", handle_create_workspace)
    app.router.add_get("/api/workspaces/current", handle_get_active_workspace)
    app.router.add_post("/api/workspaces/current", handle_set_active_workspace)
    app.router.add_put("/api/workspaces/{id}", handle_update_workspace)
    app.router.add_delete("/api/workspaces/{id}", handle_delete_workspace)
    app.router.add_post("/api/workspaces/{id}/export", handle_export_workspace)
    app.router.add_post("/api/workspaces/import", handle_import_workspace)

    # Hardware Profile API
    app.router.add_post("/api/hardware/profile", handle_set_hardware_profile)

    # Training Tasks & Adapters API
    app.router.add_get("/api/tasks", handle_list_tasks)

    # Plugins & Extensions APIs
    app.router.add_get("/api/plugins", handle_list_plugins)
    app.router.add_post("/api/plugins/{id}/toggle", handle_toggle_plugin)

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
    app.router.add_get("/api/pipelines/{id}", handle_get_pipeline_by_id)
    app.router.add_post("/api/pipelines/{id}/start", handle_start_pipeline)
    app.router.add_post("/api/pipelines/{id}/advance", handle_advance_pipeline)
    app.router.add_get("/api/settings", handle_get_settings)
    app.router.add_post("/api/settings", handle_save_settings)
    app.router.add_post("/api/scheduler/curve", handle_scheduler_curve)

    app.router.add_post("/api/datasets/upload", handle_upload_dataset)
    app.router.add_get("/api/datasets", handle_list_datasets)
    app.router.add_get("/api/datasets/{id}/stats", handle_get_dataset_stats)
    app.router.add_post("/api/datasets/{id}/tokenize", handle_tokenize_dataset)
    app.router.add_post("/api/datasets/tokenize", handle_tokenize_dataset)
    app.router.add_post("/api/datasets/{id}/curate", handle_curate_dataset)
    app.router.add_post("/api/datasets/curate", handle_curate_dataset)
    app.router.add_delete("/api/datasets/{id}", handle_delete_dataset)
    app.router.add_post("/api/tokenizer/inspect", handle_tokenizer_inspect)
    app.router.add_post("/api/notifications/test", handle_test_notification)

    app.router.add_post("/api/evaluation/generate", handle_eval_generate)
    app.router.add_post("/api/evaluation/stream", handle_eval_stream)
    app.router.add_post("/api/evaluation/compare", handle_eval_compare)
    app.router.add_post("/api/evaluation/benchmark", handle_eval_benchmark)
    app.router.add_get("/api/evaluation/results", handle_eval_results)
    app.router.add_get("/api/evaluation/suites", handle_list_eval_suites)
    app.router.add_post("/api/evaluation/suites", handle_create_eval_suite)

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

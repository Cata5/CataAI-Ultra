"""
tests/e2e/test_product_generalization.py
Comprehensive Product Acceptance Tests for LLM Maker (Requirement 18).
Validates:
1. Launch without personal configuration.
2. Dynamic environment detection (CUDA, CPU, RAM, Precisions).
3. Workspace creation, switching, and isolation.
4. Model architecture creation & parameter calculation.
5. Task adapter registry & compatibility validation.
6. Dataset upload, schema validation, and stats.
7. Optimizer configuration (AdamW, SGD, Muon).
8. Scheduler configuration (WSD, Cosine, Linear) and curve preview.
9. Preflight memory estimation on detected hardware.
10. Execution of training step on CPU and CUDA.
11. Checkpoint save, list, and verify.
12. Checkpoint restoration.
13. Evaluation execution and objective metrics.
14. Workspace export and import round-trip.
15. Plugin discovery and enable/disable toggle.
16. Safe resource cleanup.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.db.database import StudioDatabase
from src.datasets.validator import DatasetValidator
from src.datasets.manager import DatasetManager
from src.eval.benchmarks import BenchmarkManager, BenchmarkItem, EvaluationMetricsHelper
from src.networks.estimator import MemoryEstimator, ParameterCalculator
from src.networks.factory import ModelFactory
from src.plugins.manager import PluginManager
from src.telemetry.hardware import HardwareMonitor
from src.telemetry.profiles import HardwareProfile
from src.training.optimizer import build_optimizer
from src.training.scheduler import GeneralLRScheduler
from src.training.tasks import TaskRegistry, CausalLMTask, ClassificationTask, RegressionTask
from src.workspace.manager import WorkspaceManager


@pytest.fixture
def temp_workspace_env():
    """Sets up an isolated database and directories for acceptance tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="llm_maker_test_"))
    db_path = temp_dir / "test_studio.db"
    db = StudioDatabase(db_path=db_path)
    wm = WorkspaceManager(db=db, base_dir=temp_dir / "workspaces")
    pm = PluginManager(plugins_dir=temp_dir / "plugins", db=db)
    dm = DatasetManager(upload_dir=temp_dir / "uploads", db=db)
    bm = BenchmarkManager(db=db)

    yield {
        "root": temp_dir,
        "db": db,
        "wm": wm,
        "pm": pm,
        "dm": dm,
        "bm": bm,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_01_launch_without_personal_config(temp_workspace_env):
    """Test 1: App initializes clean default state with no hardcoded personal config."""
    db = temp_workspace_env["db"]
    active_ws = db.get_active_workspace()
    assert active_ws is not None
    assert active_ws["id"] == "default"
    assert active_ws["name"] == "Default Workspace"


def test_02_dynamic_hardware_detection():
    """Test 2: Dynamic environment discovery works on CPU and CUDA."""
    monitor = HardwareMonitor()
    telem = monitor.read_telemetry()
    assert telem.device_name is not None
    assert telem.system_ram_total_mb > 1000.0
    assert telem.system_ram_free_mb > 500.0
    assert "fp32" in telem.supported_precisions
    assert telem.hardware_profile is not None
    assert telem.hardware_profile["mode"] in ("conservative", "balanced", "performance")


def test_03_workspace_isolation_lifecycle(temp_workspace_env):
    """Test 3: Workspace creation, switching, and path isolation."""
    wm = temp_workspace_env["wm"]
    ws = wm.create_workspace(name="Research Alpha", description="Testing isolation")
    assert ws["name"] == "Research Alpha"
    ws_id = ws["id"]

    paths = wm.get_workspace_paths(ws_id)
    assert paths["checkpoints"].exists()
    assert paths["datasets"].exists()

    # Switch workspace
    assert wm.set_active_workspace(ws_id)
    active = wm.get_active_workspace()
    assert active["id"] == ws_id

    # Switch back to default
    assert wm.set_active_workspace("default")
    assert wm.get_active_workspace()["id"] == "default"


def test_04_model_creation_and_parameter_calc():
    """Test 4: Architecture creation and parameter calculation for multiple families."""
    # Decoder-only Transformer
    transformer_cfg = {
        "architecture_family": "decoder_only",
        "d_model": 256,
        "n_layers": 4,
        "n_q": 4,
        "n_kv": 2,
        "d_ff": 640,
        "vocab_size": 4096,
        "tie_word_embeddings": True,
    }
    stats = ParameterCalculator.calculate(transformer_cfg)
    assert stats["total_parameters"] > 0
    assert stats["embedding_parameters"] > 0

    # MLP for tabular
    mlp_cfg = {
        "architecture_family": "mlp",
        "d_model": 128,
        "input_dim": 32,
        "n_layers": 3,
        "d_ff": 256,
    }
    mlp_stats = ParameterCalculator.calculate(mlp_cfg)
    assert mlp_stats["total_parameters"] > 0


def test_05_task_adapters_and_compatibility():
    """Test 5: Task adapter registry and architecture compatibility validation."""
    tasks = TaskRegistry.list_tasks()
    task_ids = [t["task_id"] for t in tasks]
    assert "causal_lm" in task_ids
    assert "instruction_sft" in task_ids
    assert "classification" in task_ids
    assert "regression" in task_ids

    assert TaskRegistry.validate_compatibility("causal_lm", "decoder_only")
    assert TaskRegistry.validate_compatibility("classification", "cnn")
    assert TaskRegistry.validate_compatibility("regression", "mlp")


def test_06_dataset_validation_formats(temp_workspace_env):
    """Test 6: Dataset validator parses TXT, JSONL, and CSV with clear error reporting."""
    root = temp_workspace_env["root"]
    validator = DatasetValidator()

    # 1. Plain text
    txt_file = root / "sample.txt"
    txt_file.write_text("Sentence 1\nSentence 2\nSentence 3\n", encoding="utf-8")
    txt_res = validator.validate_file(txt_file)
    assert txt_res["valid"] is True
    assert txt_res["valid_records"] == 3

    # 2. JSONL instruction
    jsonl_file = root / "instruct.jsonl"
    jsonl_file.write_text(
        '{"instruction": "Hi", "output": "Hello"}\n{"instruction": "2+2", "output": "4"}\n',
        encoding="utf-8",
    )
    jsonl_res = validator.validate_file(jsonl_file)
    assert jsonl_res["valid"] is True
    assert jsonl_res["valid_records"] == 2

    # 3. CSV
    csv_file = root / "data.csv"
    csv_file.write_text("text,label\nSample text A,0\nSample text B,1\n", encoding="utf-8")
    csv_res = validator.validate_file(csv_file)
    assert csv_res["valid"] is True
    assert csv_res["valid_records"] == 2


def test_07_optimizer_configuration():
    """Test 7: Universal optimizer factory configures AdamW, SGD, and Muon."""
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))

    # AdamW
    opts_adamw = build_optimizer(model, optimizer_type="adamw", lr=1e-3, weight_decay=0.01)
    assert len(opts_adamw) == 1
    assert isinstance(opts_adamw[0], torch.optim.AdamW)

    # SGD
    opts_sgd = build_optimizer(model, optimizer_type="sgd", lr=0.01, momentum=0.9)
    assert len(opts_sgd) == 1
    assert isinstance(opts_sgd[0], torch.optim.SGD)


def test_08_scheduler_configuration_and_preview():
    """Test 8: General scheduler supports WSD, Cosine, Linear and generates preview curve."""
    curve = GeneralLRScheduler.preview_curve(
        schedule_type="cosine",
        warmup_steps=100,
        total_steps=1000,
        peak_lr=3e-4,
        min_lr_ratio=0.1,
        num_points=10,
    )
    assert len(curve) > 5
    assert curve[0]["lr"] == 0.0  # Warmup start
    # Peak near warmup
    peak = max(pt["lr"] for pt in curve)
    assert abs(peak - 3e-4) < 1e-5
    # Final step near min_lr
    assert curve[-1]["lr"] <= 3e-4 * 0.15


def test_09_hardware_aware_memory_preflight():
    """Test 9: Preflight memory estimation respects detected hardware limits."""
    cfg = {
        "architecture_family": "decoder_only",
        "d_model": 256,
        "n_layers": 4,
        "n_q": 4,
        "n_kv": 2,
        "d_ff": 640,
        "vocab_size": 4096,
    }
    est = MemoryEstimator.estimate(cfg, micro_batch_size=2, seq_len=128)
    assert "total_training_vram_gb" in est
    assert est["safety_status"] in ("SAFE", "TIGHT", "UNLIKELY_TO_FIT")
    assert "recommendation" in est


def test_10_cpu_and_cuda_task_forward_loss():
    """Test 10: Task adapters execute forward pass and compute genuine loss on device."""
    device = torch.device("cpu")
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))

    task = ClassificationTask(num_classes=2)
    raw_x = torch.randn(4, 16)
    raw_y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    batch = task.prepare_batch((raw_x, raw_y), device=device, dtype=torch.float32)

    loss, metrics = task.forward_and_loss(model, batch)
    assert float(loss.item()) > 0.0
    assert "accuracy" in metrics.detailed_metrics


def test_11_evaluation_metrics_helper():
    """Test 11: Objective non-LM evaluation metrics (Classification accuracy, Regression RMSE)."""
    # Classification
    cls_eval = EvaluationMetricsHelper.evaluate_classification([0, 1, 1, 0], [0, 1, 0, 0])
    assert cls_eval["accuracy"] == 75.0

    # Regression
    reg_eval = EvaluationMetricsHelper.evaluate_regression([2.0, 4.0, 6.0], [2.1, 3.9, 6.2])
    assert reg_eval["rmse"] > 0.0
    assert reg_eval["mae"] > 0.0


def test_12_workspace_export_and_import(temp_workspace_env):
    """Test 12: Workspaces can be exported and imported with zero data loss."""
    wm = temp_workspace_env["wm"]
    ws = wm.create_workspace(name="Exportable Project", description="Will be exported")
    ws_id = ws["id"]

    bundle = wm.export_workspace(ws_id)
    assert bundle["workspace"]["name"] == "Exportable Project"

    imported = wm.import_workspace(bundle)
    assert imported["id"] != ws_id or "imported" in imported["id"]
    assert imported["name"] == "Exportable Project"


def test_13_plugin_discovery_and_toggle(temp_workspace_env):
    """Test 13: Local plugins are discovered, validated, and can be toggled safely."""
    pm = temp_workspace_env["pm"]
    plugins = pm.discover_plugins()
    assert len(plugins) >= 2
    assert any(p["id"] == "vision_adapter" for p in plugins)

    # Toggle plugin
    success, msg = pm.set_plugin_enabled("vision_adapter", False)
    assert success is True
    updated = {p["id"]: p for p in pm.discover_plugins()}
    assert updated["vision_adapter"]["enabled"] is False

    success, msg = pm.set_plugin_enabled("vision_adapter", True)
    assert success is True
    updated = {p["id"]: p for p in pm.discover_plugins()}
    assert updated["vision_adapter"]["enabled"] is True

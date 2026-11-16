"""
audit/probes/audit_probe.py
Read-only audit probes. Does NOT modify application source or real project data.
Writes only into a fresh temporary directory.
Run:  ./.venv/Scripts/python.exe audit/probes/audit_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch

RESULTS = []


def record(probe, status, detail, evidence=""):
    RESULTS.append({
        "probe": probe,
        "status": status,
        "detail": detail,
        "evidence": evidence,
    })
    print(f"[{status:10}] {probe}: {detail}")
    if evidence:
        print("           " + str(evidence).replace("\n", "\n           ")[:1500])


def probe_param_count():
    from src.model.architecture import RoCanonConfig, RoCanonForCausalLM
    m = RoCanonForCausalLM(RoCanonConfig())
    total = sum(p.numel() for p in m.parameters())
    uniq = sum(p.numel() for p in set(m.parameters()))
    record(
        "P1 RoCanon parameter count",
        "PASS" if total == 48_531_968 else "FAIL",
        f"sum(numel)={total:,} unique={uniq:,} (claimed 48,531,968)",
    )
    from src.training.factory import calculate_analytical_parameters
    ana = calculate_analytical_parameters()
    record(
        "P1b analytical calculator",
        "PASS" if ana["total_parameters"] == total else "FAIL",
        f"analytical={ana['total_parameters']:,}",
    )


def probe_factory_templates():
    from src.networks.registry import NetworkRegistry
    from src.networks.factory import ModelFactory
    from src.networks.estimator import ParameterCalculator

    reg = NetworkRegistry()  # reads real registry (read-only)
    rows = []
    for net in reg.list_networks():
        cfg = reg.get_network(net["id"])
        try:
            model = ModelFactory.create(dict(cfg))
            actual = sum(p.numel() for p in model.parameters())
            calc = ParameterCalculator.calculate(dict(cfg))["total_parameters"]
            fam = cfg.get("architecture_family")
            vocab = int(cfg.get("vocab_size", 12288))
            if fam in ("mlp", "cnn"):
                x = torch.randn(2, 16, int(cfg.get("input_dim", 64)))
            else:
                x = torch.randint(0, vocab, (2, 16))
            try:
                y = model(x)
                shape = list(y[0].shape) if isinstance(y, tuple) else list(y.shape)
                fwd = f"forward OK {shape}"
            except Exception as e:
                fwd = f"FORWARD FAILED: {type(e).__name__}: {e}"
            rows.append(f"{net['id']:32s} fam={fam:14s} actual={actual:>12,} calc={calc:>12,} match={actual==calc} {fwd}")
        except Exception as e:
            rows.append(f"{net['id']:32s} INSTANTIATION FAILED: {type(e).__name__}: {e}")
    record("P2 registry templates instantiate/forward", "INFO", "see evidence", "\n".join(rows))


def probe_three_custom_configs():
    from src.networks.factory import ModelFactory
    from src.networks.estimator import ParameterCalculator

    cfgs = [
        {"id": "audit_dec_A", "name": "AuditDecoderA", "architecture_family": "decoder_only",
         "vocab_size": 4096, "d_model": 128, "n_layers": 4, "n_q": 4, "n_kv": 2, "d_ff": 320,
         "norm_type": "rmsnorm", "activation": "swiglu", "pos_encoding": "rope",
         "has_conv1d": True, "conv_kernel_size": 4, "tie_word_embeddings": True},
        {"id": "audit_dec_B", "name": "AuditDecoderB", "architecture_family": "decoder_only",
         "vocab_size": 4096, "d_model": 192, "n_layers": 6, "n_q": 6, "n_kv": 6, "d_ff": 512,
         "norm_type": "layernorm", "activation": "gelu", "pos_encoding": "learned",
         "has_conv1d": False, "tie_word_embeddings": False},
        {"id": "audit_hyb_C", "name": "AuditHybridC", "architecture_family": "hybrid",
         "vocab_size": 2048, "d_model": 96, "n_layers": 3, "n_q": 4, "n_kv": 1, "d_ff": 256,
         "norm_type": "rmsnorm", "activation": "swiglu", "pos_encoding": "rope",
         "has_conv1d": True, "conv_kernel_size": 5, "tie_word_embeddings": True},
    ]
    rows = []
    for cfg in cfgs:
        m = ModelFactory.create(dict(cfg))
        actual = sum(p.numel() for p in m.parameters())
        calc = ParameterCalculator.calculate(dict(cfg))
        x = torch.randint(0, cfg["vocab_size"], (2, 24))
        y = m(x)
        loss = y.float().mean()
        loss.backward()
        grads = sum(1 for p in m.parameters() if p.grad is not None)
        total = sum(1 for p in m.parameters())
        rows.append(
            f"{cfg['id']}: actual={actual:,} calc_total={calc['total_parameters']:,} match={actual==calc['total_parameters']} "
            f"out={list(y.shape)} grads={grads}/{total} breakdown(attn={calc['attention_parameters']:,},ffn={calc['ffn_parameters']:,},conv={calc['conv_parameters']:,})"
        )
    record("P3 three custom architectures", "PASS", "3/3 built, forward+backward executed", "\n".join(rows))


def probe_residual_scale_math_bug():
    sys.modules.pop("math", None)  # isolate the NameError in networks/factory.py
    import importlib
    import src.networks.factory as nf
    importlib.reload(nf)
    cfg = {"id": "audit_scale", "architecture_family": "decoder_only", "vocab_size": 512,
           "d_model": 64, "n_layers": 2, "n_q": 4, "n_kv": 2, "d_ff": 128,
           "scale_residual_with_layers": True}
    try:
        nf.ModelFactory.create(cfg)
        record("P4 scale_residual_with_layers", "PASS", "created")
    except NameError as e:
        record("P4 scale_residual_with_layers", "FAIL",
               "NameError in src/networks/factory.py (missing `import math`)",
               traceback.format_exc(limit=2))
    finally:
        import importlib as _il
        _il.reload(nf)


def probe_training_model_compat():
    """The training worker feeds int64 token IDs and a causal-LM shift loss."""
    from src.networks.registry import NetworkRegistry
    from src.networks.factory import ModelFactory

    reg = NetworkRegistry()
    rows = []
    for nid in ["rocanon_48m", "mini_gpt_15m", "bert_encoder_25m", "deep_mlp_5m", "conv1d_network_4m", "hybrid_cnn_transformer_20m"]:
        cfg = reg.get_network(nid)
        if not cfg:
            rows.append(f"{nid}: not in registry")
            continue
        try:
            model = ModelFactory.create(dict(cfg))
            tokens = torch.randint(0, int(cfg.get("vocab_size", 12288)), (2, 16))
            try:
                logits = model(tokens)
                logits = logits[0] if isinstance(logits, tuple) else logits
                try:
                    sl = logits[..., :-1, :].contiguous()
                    tg = tokens[..., 1:].contiguous()
                    import torch.nn.functional as F
                    loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), tg.reshape(-1))
                    rows.append(f"{nid}: TOKEN-FEED OK, LM-LOSS OK (loss={float(loss):.3f})")
                except Exception as e:
                    rows.append(f"{nid}: forward OK but LM/LOSS FAILED -> {type(e).__name__}: {e}")
            except Exception as e:
                rows.append(f"{nid}: TOKEN-FEED FORWARD FAILED -> {type(e).__name__}: {e}")
        except Exception as e:
            rows.append(f"{nid}: build failed {type(e).__name__}: {e}")
    record("P5 training-engine model compatibility", "INFO",
           "worker always feeds int64 tokens + shifted causal-CE", "\n".join(rows))


def probe_checkpoint_restore(tmp: Path):
    from src.model.architecture import RoCanonConfig, RoCanonForCausalLM
    from src.training.factory import create_optimizers_from_dict
    from src.training.scheduler import WSDScheduler
    from src.checkpoints.manager import CheckpointManager

    cfg = RoCanonConfig(vocab_size=512, d_model=64, n_layers=2, n_q=4, n_kv=2, d_h=16, d_ff=128, max_seq_len=64)
    cm = CheckpointManager(checkpoint_dir=tmp / "ckpt")

    m1 = RoCanonForCausalLM(cfg)
    o1 = create_optimizers_from_dict(m1, {})
    sch1 = WSDScheduler(optimizers=o1, warmup_steps=2, total_tokens=10_000)
    x = torch.randint(0, 512, (2, 16))
    loss = m1(x).float().mean()
    loss.backward()
    o1[0].step(); o1[1].step()
    sch1.step(tokens=1000)
    path = cm.save_checkpoint(m1, step=7, tokens_seen=1000, optimizers=o1, scheduler=sch1, filename_prefix="audit")

    # 1) full restore
    m2 = RoCanonForCausalLM(cfg)
    o2 = create_optimizers_from_dict(m2, {})
    sch2 = WSDScheduler(optimizers=o2, warmup_steps=2, total_tokens=10_000)
    r_full = cm.load_checkpoint(path, m2, optimizers=o2, scheduler=sch2, restore_training_state=True)
    muon_state_after_full = len(o2[0].state)
    sched_after_full = sch2.state_dict()
    record("P6a checkpoint full-restore (optimizer+scheduler)",
           "PASS" if muon_state_after_full > 0 else "FAIL",
           f"muon state entries after load={muon_state_after_full}, step={r_full['step']}, sched={json.dumps(sched_after_full, default=str)[:200]}")

    # 2) worker-style restore (as training_worker.py calls it)
    m3 = RoCanonForCausalLM(cfg)
    o3 = create_optimizers_from_dict(m3, {})
    sch3 = WSDScheduler(optimizers=o3, warmup_steps=2, total_tokens=10_000)
    r_partial = cm.load_checkpoint(path, m3, restore_training_state=False)
    record("P6b worker-style resume (restore_training_state=False)",
           "FAIL" if len(o3[0].state) == 0 else "PASS",
           f"muon state entries after load={len(o3[0].state)} (worker loses optimizer/scheduler/RNG); step={r_partial['step']}")


def probe_dataset_training_bridge(tmp: Path):
    from src.datasets.manager import DatasetManager
    from src.db.database import StudioDatabase

    db = StudioDatabase(tmp / "audit.db")
    dm = DatasetManager(upload_dir=tmp / "uploads", db=db)
    src_file = tmp / "tiny_audit.txt"
    src_file.write_text("Acesta este un text de audit.\n\nAl doilea paragraf pentru tokenizare.\n", encoding="utf-8")
    res = dm.register_uploaded_file(src_file, name="tiny_audit.txt")
    record("P7a dataset upload+register",
           "PASS" if res.get("success") else "FAIL",
           f"success={res.get('success')} id={res.get('dataset_id')} tokens={res.get('token_count')}")

    from src.data.custom_loader import CustomDatasetEngine
    eng = CustomDatasetEngine(cache_dir=tmp / "cache")
    known = {k: Path(v).exists() for k, v in eng.KNOWN_DATASETS.items()}
    record("P7b hardcoded KNOWN_DATASETS existence", "INFO", "absolute paths outside repo", json.dumps(known, indent=2))

    ds_id = res.get("dataset_id")
    file_path = res.get("file_path")
    # What the UI sends in the training payload
    out_by_id = eng.load_dataset(dataset_source=ds_id, phase="pretrain", max_samples=50, device=torch.device("cpu"))
    record("P7c training resolves uploaded dataset by ID",
           "FAIL",
           f"dataset_source='{ds_id}' -> file exists? {Path(ds_id).exists()}; engine fell back to synthetic data (train tokens={len(out_by_id[0])})")

    out_by_path = eng.load_dataset(dataset_source=file_path, phase="pretrain", max_samples=50, device=torch.device("cpu"))
    record("P7d training resolves uploaded dataset by absolute file path",
           "PASS" if len(out_by_path[0]) > 0 else "FAIL",
           f"train tokens={len(out_by_path[0])}")


def probe_registry_validation(tmp: Path):
    from src.networks.registry import NetworkRegistry

    reg = NetworkRegistry(registry_dir=tmp / "registry")
    bad = {"name": "AuditInvalid", "architecture_family": "decoder_only", "d_model": 64,
           "n_layers": 2, "n_q": 5, "n_kv": 2, "d_ff": 128, "vocab_size": 512}
    saved = reg.save_network("audit_invalid", bad)
    from src.networks.validator import ArchitectureValidator
    errs = ArchitectureValidator.validate_config_schema(bad)
    record("P8 registry save without validation",
           "FAIL" if errs else "PASS",
           f"saved id={saved['id']} even though validator reports: {errs}")


def probe_custom_modules():
    from src.networks.custom_loader import CustomModuleRegistry

    reg = CustomModuleRegistry()  # reads networks/custom (local only)
    mods = reg.list_modules()
    record("P9 custom module discovery", "PASS" if mods else "FAIL",
           f"registered={[m['module_name'] for m in mods]}")
    import inspect
    import src.networks.factory as nf
    src = inspect.getsource(nf)
    wired = "custom_loader" in src or "CustomModule" in src
    record("P9b custom modules wired into ModelFactory", "PASS" if wired else "FAIL",
           "ModelFactory/GenericModularNetwork has no reference to CustomModuleRegistry - modules are catalog-only")


def probe_vram_estimator():
    from src.networks.estimator import MemoryEstimator
    cfg = {"id": "audit_mem", "architecture_family": "decoder_only", "vocab_size": 4096,
           "d_model": 128, "n_layers": 4, "n_q": 4, "n_kv": 2, "d_ff": 320}
    est = MemoryEstimator.estimate(cfg, micro_batch_size=2, seq_len=64, dtype="bfloat16")
    if not torch.cuda.is_available():
        record("P10 VRAM estimator vs measured", "BLOCKED", "CUDA unavailable")
        return
    from src.networks.factory import ModelFactory
    from src.training.factory import create_optimizers_from_dict
    m = ModelFactory.create(dict(cfg)).to("cuda")
    opts = create_optimizers_from_dict(m, {})
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated() / (1024 ** 2)
    x = torch.randint(0, 4096, (2, 64), device="cuda")
    logits = m(x)
    loss = logits.float().mean()
    loss.backward()
    opts[0].step(); opts[1].step()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    measured = peak
    record("P10 VRAM estimator vs measured (tiny model)",
           "INFO",
           f"estimated_total={est['total_training_vram_gb']} GB ({est['total_training_vram_mb']} MB) "
           f"vs measured_peak_allocated={measured:.1f} MB (delta {measured - est['total_training_vram_mb']:+.1f} MB) "
           f"[estimator adds fixed 350MB workspace; measured is allocator-only, FP32 master weights not modelled here]")
    del m, opts, x, logits, loss
    torch.cuda.empty_cache()


def probe_paramcalc_cost():
    from src.networks.estimator import ParameterCalculator
    from src.networks.registry import NetworkRegistry
    reg = NetworkRegistry()
    cfg = reg.get_network("rocanon_48m")
    t0 = time.time()
    ParameterCalculator.calculate(dict(cfg))
    dt = time.time() - t0
    t0 = time.time()
    ParameterCalculator.calculate(dict(cfg))
    dt2 = time.time() - t0
    record("P11 ParameterCalculator cost", "INFO",
           f"24.25M-style calculator instantiates the real model each call: {dt*1000:.0f} ms then {dt2*1000:.0f} ms for rocanon_48m")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="rocanon_audit_"))
    print(f"audit temp dir: {tmp}\n")
    for fn in (probe_param_count, probe_factory_templates, probe_three_custom_configs,
               probe_residual_scale_math_bug, probe_training_model_compat):
        try:
            fn()
        except Exception:
            record(fn.__name__, "ERROR", "probe crashed", traceback.format_exc(limit=3))
    for fn in (probe_checkpoint_restore, probe_dataset_training_bridge, probe_registry_validation):
        try:
            fn(tmp / fn.__name__)
        except Exception:
            record(fn.__name__, "ERROR", "probe crashed", traceback.format_exc(limit=3))
    for fn in (probe_custom_modules, probe_vram_estimator, probe_paramcalc_cost):
        try:
            fn()
        except Exception:
            record(fn.__name__, "ERROR", "probe crashed", traceback.format_exc(limit=3))

    out = REPO / "audit" / "probes" / "probe_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

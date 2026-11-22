"""
src/checkpoints/manager.py
Robust, atomic checkpoint management for RoCanon-48.5M.
Preserves model, dual optimizers, scheduler, RNG state, phase, and validation metrics.
Ensures backward compatibility with existing legacy checkpoints.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..db.database import StudioDatabase

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


class CheckpointManager:
    """Manages atomic saving, loading, validation, comparison, and deletion of checkpoints."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = DEFAULT_CHECKPOINT_DIR,
        db: Optional[StudioDatabase] = None,
        max_checkpoints: int = 4,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self.max_checkpoints = max(1, max_checkpoints)

    def save_checkpoint(
        self,
        model: nn.Module,
        step: int,
        tokens_seen: int,
        optimizers: Optional[Tuple[torch.optim.Optimizer, torch.optim.Optimizer]] = None,
        scheduler: Optional[Any] = None,
        phase: str = "pretrain",
        dataset_rev: str = "v1",
        tokenizer_ver: str = "rocanon-12k-v1",
        training_config: Optional[Dict[str, Any]] = None,
        val_loss: Optional[float] = None,
        train_loss: Optional[float] = None,
        run_id: Optional[str] = None,
        filename_prefix: str = "checkpoint",
        is_best: bool = False,
        workspace_id: str = "default",
    ) -> Path:
        """
        Atomically saves training state to disk.
        Writes to a temporary file first, flushes, and renames atomically to prevent corrupted files.
        """
        timestamp = time.time()
        chk_id = f"{filename_prefix}_step_{step}" if not is_best else "best_model"
        target_name = f"{chk_id}.pt"
        target_path = self.checkpoint_dir / target_name
        temp_path = self.checkpoint_dir / f".tmp_{uuid.uuid4().hex}_{target_name}"

        # Collect RNG states for exact reproducibility using safe primitives/tensors
        safe_rng_states: Dict[str, Any] = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
        }
        try:
            import numpy as np
            np_st = np.random.get_state()
            safe_rng_states["numpy"] = (np_st[0], torch.from_numpy(np_st[1].copy()), int(np_st[2]), int(np_st[3]), float(np_st[4]))
        except Exception:
            pass

        # Optimizer states
        muon_sd = None
        adamw_sd = None
        if optimizers is not None:
            muon_opt, adamw_opt = optimizers
            if muon_opt is not None:
                muon_sd = muon_opt.state_dict()
            if adamw_opt is not None:
                adamw_sd = adamw_opt.state_dict()

        # Scheduler state
        scheduler_sd = scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None

        checkpoint_payload = {
            "version": "2.0",
            "step": step,
            "global_step": step,
            "tokens_seen": tokens_seen,
            "phase": phase,
            "dataset_revision": dataset_rev,
            "tokenizer_version": tokenizer_ver,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "model_state_dict": model.state_dict(),
            "muon_state_dict": muon_sd,
            "adamw_state_dict": adamw_sd,
            "scheduler_state_dict": scheduler_sd,
            "rng_states": safe_rng_states,
            "training_config": training_config or {},
            "workspace_id": workspace_id,
            "timestamp": timestamp,
        }

        # Atomic write with Windows retry loop & fallback
        try:
            torch.save(checkpoint_payload, temp_path)
            replaced = False
            last_err = None
            for attempt in range(8):
                try:
                    os.replace(temp_path, target_path)
                    replaced = True
                    break
                except (PermissionError, OSError) as e:
                    last_err = e
                    time.sleep(0.15 * (attempt + 1))
                    if attempt >= 2 and target_path.exists():
                        try:
                            target_path.unlink(missing_ok=True)
                        except Exception:
                            pass
            if not replaced:
                try:
                    shutil.copyfile(temp_path, target_path)
                    replaced = True
                except Exception as fb_err:
                    logger.error("Failed to replace checkpoint atomically: %s, fallback failed: %s", last_err, fb_err)
                    raise last_err or fb_err

            file_size = target_path.stat().st_size
            logger.info("Saved atomic checkpoint to %s (%d bytes)", target_path, file_size)

            # Save companion metadata JSON for instant zero-deserialization inspection
            meta_json = {
                "checkpoint_id": chk_id,
                "run_id": run_id,
                "workspace_id": workspace_id,
                "version": "2.0",
                "step": step,
                "global_step": step,
                "tokens_seen": tokens_seen,
                "phase": phase,
                "dataset_revision": dataset_rev,
                "tokenizer_version": tokenizer_ver,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "timestamp": timestamp,
                "is_best": is_best,
                "file_size_bytes": file_size,
                "total_params": sum(p.numel() for p in model.parameters()),
                "network_id": (training_config or {}).get("network_id"),
                "config_arch": (training_config or {}).get("arch") or (training_config or {}).get("config_arch"),
            }
            meta_path = target_path.with_suffix(".json")
            try:
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(meta_json, mf, indent=2)
            except Exception as me:
                logger.warning("Could not write companion checkpoint metadata JSON: %s", me)

            # Record in database if available
            if self.db is not None:
                self.db.log_checkpoint(
                    checkpoint_id=chk_id,
                    run_id=run_id,
                    step=step,
                    tokens=tokens_seen,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    file_path=str(target_path.resolve()),
                    file_size_bytes=file_size,
                    phase=phase,
                    workspace_id=workspace_id,
                    metadata={
                        "version": "2.0",
                        "dataset_revision": dataset_rev,
                        "tokenizer_version": tokenizer_ver,
                        "is_best": is_best,
                    },
                )

            # Automatically enforce rolling checkpoint retention (keep max N step checkpoints)
            if not is_best:
                try:
                    self.prune_checkpoints(keep_last_n=self.max_checkpoints)
                except Exception as prune_err:
                    logger.warning("Error during checkpoint pruning: %s", prune_err)

            return target_path
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def resolve_checkpoint_path(self, checkpoint_path: Union[str, Path]) -> Path:
        """
        Universally resolves a checkpoint path from:
        1. Exact absolute or relative path on disk
        2. Filename inside self.checkpoint_dir
        3. Filename inside models/ (or any subdirectory of models/)
        4. Filename inside checkpoints/ (or any subdirectory of checkpoints/)
        """
        p = Path(checkpoint_path)
        if p.is_file():
            return p.resolve()

        # Check self.checkpoint_dir
        cand = self.checkpoint_dir / p.name
        if cand.is_file():
            return cand.resolve()

        # Check models directory and subdirectories
        models_root = Path("models")
        if models_root.exists():
            cand_m = models_root / p
            if cand_m.is_file():
                return cand_m.resolve()
            cand_m2 = models_root / p.name
            if cand_m2.is_file():
                return cand_m2.resolve()
            for found in models_root.glob(f"**/{p.name}"):
                if found.is_file():
                    return found.resolve()

        # Check checkpoints directory subdirectories
        for found in self.checkpoint_dir.glob(f"**/{p.name}"):
            if found.is_file():
                return found.resolve()

        # Check legacy project location if present
        cand_legacy = Path(r"d:\proiecte\LLM from Scratch\checkpoints") / p.name
        if cand_legacy.is_file():
            return cand_legacy.resolve()

        # Fallback search across repository root
        for found in Path(".").glob(f"**/{p.name}"):
            p_str = str(found)
            if found.is_file() and not p_str.startswith(".venv") and not p_str.startswith(".git"):
                return found.resolve()

        raise FileNotFoundError(f"Checkpoint not found on disk: {checkpoint_path}")

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizers: Optional[Tuple[torch.optim.Optimizer, torch.optim.Optimizer]] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        restore_training_state: bool = True,
    ) -> Dict[str, Any]:
        """
        Loads checkpoint weights and optionally restores optimizer, scheduler, and RNG state.
        Preserves backward compatibility with legacy checkpoints (v1, LLM from Scratch, safetensors).
        """
        path = self.resolve_checkpoint_path(checkpoint_path)

        target_device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info("Loading checkpoint from %s onto %s", path, target_device)

        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(str(path), device=str(target_device))
            raw_payload = {"model_state_dict": sd, "step": 0, "tokens_seen": 0}
        else:
            # Safe checkpoint deserialization with fallback for numpy/custom RNG states
            try:
                raw_payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                raw_payload = torch.load(path, map_location="cpu", weights_only=False)

        # Extract model state dict
        if isinstance(raw_payload, dict) and "model_state_dict" in raw_payload:
            sd = raw_payload["model_state_dict"]
        elif isinstance(raw_payload, dict) and any("weight" in k for k in raw_payload.keys()):
            sd = raw_payload
        else:
            raise ValueError(f"Unrecognized checkpoint format in {path}")

        # Transfer model weights to model parameters
        # Strip torch.compile / module prefixes if model doesn't use them
        target_keys = set(model.state_dict().keys())
        first_k = next(iter(sd.keys())) if sd else ""
        if first_k and first_k not in target_keys:
            clean_sd = {}
            for k, v in sd.items():
                ck = k
                for pfx in ("_orig_mod.", "module.", "model."):
                    if ck.startswith(pfx) and ck[len(pfx):] in target_keys:
                        ck = ck[len(pfx):]
                        break
                clean_sd[ck] = v
            sd = clean_sd

        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            exact_match = len(missing) == 0 and len(unexpected) == 0
        except RuntimeError as r_err:
            logger.warning("Shape mismatch in load_state_dict: %s. Performing safe shape-filtered loading.", r_err)
            exact_match = False
            missing = ["shape_mismatch"]
            unexpected = []

        if not exact_match:
            # Vocabulary adaptation and shape-compatible loading for legacy or mismatched checkpoints
            model_sd = model.state_dict()
            transferred = 0
            adapted_emb = 0
            for k, v in sd.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k].copy_(v)
                    transferred += 1
                elif ("tok_embeddings" in k or "embed_tokens" in k or "head" in k) and k in model_sd:
                    try:
                        min_v = min(model_sd[k].shape[0], v.shape[0])
                        if len(model_sd[k].shape) > 1 and len(v.shape) > 1:
                            min_dim = min(model_sd[k].shape[1], v.shape[1])
                            model_sd[k][:min_v, :min_dim].copy_(v[:min_v, :min_dim])
                        else:
                            model_sd[k][:min_v].copy_(v[:min_v])
                        adapted_emb += 1
                    except Exception:
                        pass
            model.load_state_dict(model_sd)
            logger.info("Adapted checkpoint loaded: %d exact keys, %d adapted embeddings (missing=%d, unexpected=%d)", transferred, adapted_emb, len(missing), len(unexpected))
        else:
            logger.info("Exact checkpoint match loaded (%d keys) from %s", len(sd), path)

        # Restore optimizers, scheduler, and RNG if requested
        if restore_training_state and isinstance(raw_payload, dict):
            if optimizers is not None:
                muon_opt, adamw_opt = optimizers
                if muon_opt is not None and raw_payload.get("muon_state_dict"):
                    try:
                        muon_opt.load_state_dict(raw_payload["muon_state_dict"])
                        # Ensure optimizer tensor states are transferred to the target compute device
                        for state_entry in muon_opt.state.values():
                            for sk, sv in state_entry.items():
                                if isinstance(sv, torch.Tensor):
                                    state_entry[sk] = sv.to(device=target_device)
                    except Exception as e:
                        logger.warning("Could not restore Muon optimizer state: %s", e)
                if adamw_opt is not None and raw_payload.get("adamw_state_dict"):
                    try:
                        adamw_opt.load_state_dict(raw_payload["adamw_state_dict"])
                        # Ensure optimizer tensor states are transferred to the target compute device
                        for state_entry in adamw_opt.state.values():
                            for sk, sv in state_entry.items():
                                if isinstance(sv, torch.Tensor):
                                    state_entry[sk] = sv.to(device=target_device)
                    except Exception as e:
                        logger.warning("Could not restore AdamW optimizer state: %s", e)

            if scheduler is not None and raw_payload.get("scheduler_state_dict"):
                try:
                    if hasattr(scheduler, "load_state_dict"):
                        scheduler.load_state_dict(raw_payload["scheduler_state_dict"])
                except Exception as e:
                    logger.warning("Could not restore Scheduler state: %s", e)

            # Restore RNG states accurately (ensuring CPU ByteTensors remain on CPU)
            rng = raw_payload.get("rng_states")
            if rng:
                try:
                    if rng.get("torch_cpu") is not None:
                        cpu_rng = rng["torch_cpu"]
                        if isinstance(cpu_rng, torch.Tensor):
                            cpu_rng = cpu_rng.cpu()
                        torch.set_rng_state(cpu_rng)
                    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                        cuda_rng = rng["torch_cuda"]
                        if isinstance(cuda_rng, list):
                            cuda_rng = [s.cpu() if isinstance(s, torch.Tensor) else s for s in cuda_rng]
                        elif isinstance(cuda_rng, torch.Tensor):
                            cuda_rng = cuda_rng.cpu()
                        torch.cuda.set_rng_state_all(cuda_rng)
                    if rng.get("python") is not None:
                        random.setstate(rng["python"])
                    if rng.get("numpy") is not None:
                        try:
                            import numpy as np
                            np_st = rng["numpy"]
                            if isinstance(np_st[1], torch.Tensor):
                                np_st = (np_st[0], np_st[1].numpy(), int(np_st[2]), int(np_st[3]), float(np_st[4]))
                            np.random.set_state(np_st)
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning("Could not restore RNG states: %s", e)

        return {
            "path": str(path.resolve()),
            "step": raw_payload.get("step", raw_payload.get("global_step", 0)),
            "tokens_seen": raw_payload.get("tokens_seen", 0),
            "phase": raw_payload.get("phase", "pretrain"),
            "val_loss": raw_payload.get("val_loss"),
            "train_loss": raw_payload.get("train_loss"),
            "exact_match": exact_match,
            "version": raw_payload.get("version", "1.0"),
        }

    def inspect_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Reads metadata and parameter statistics safely without loading tensors into GPU or running unsafe pickles."""
        try:
            path = self.resolve_checkpoint_path(checkpoint_path)
        except Exception:
            path = Path(checkpoint_path)
            if not path.exists():
                cand = self.checkpoint_dir / path.name
                if cand.exists():
                    path = cand
                else:
                    return {"error": f"File not found: {checkpoint_path}"}

        stat = path.stat()
        is_saved_model = "models" in path.parts or (path.parent.name != "checkpoints" and path.parent.name != "checkpoints_v2")
        default_model_name = None
        if is_saved_model:
            if path.parent.name and path.parent.name != "models":
                default_model_name = path.parent.name
            else:
                default_model_name = path.stem.replace("_full", "").replace("_stripped", "")
        res: Dict[str, Any] = {
            "filename": path.name,
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
            "is_saved_model": is_saved_model,
            "model_name": default_model_name,
        }

        # Check for companion sidecar JSON metadata first (instant, 0% CPU memory overhead)
        meta_path = path.with_suffix(".json")
        if not meta_path.exists() and (path.parent / "metadata.json").exists():
            meta_path = path.parent / "metadata.json"

        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                res.update({
                    "run_id": meta.get("run_id"),
                    "workspace_id": meta.get("workspace_id", "default"),
                    "version": meta.get("version", "2.0"),
                    "step": meta.get("step", meta.get("global_step", 0)),
                    "tokens_seen": meta.get("tokens_seen", 0),
                    "val_loss": meta.get("val_loss") or meta.get("best_val_loss"),
                    "train_loss": meta.get("train_loss"),
                    "phase": meta.get("phase", "unknown"),
                    "total_params": meta.get("total_params", 0),
                    "format": "pytorch",
                    "model_name": meta.get("custom_name") or meta.get("name") or default_model_name or meta.get("model_id"),
                })
                if self.db is not None:
                    try:
                        with self.db._lock, self.db._get_connection() as conn:
                            row = conn.execute(
                                "SELECT run_id, workspace_id FROM checkpoints WHERE id = ? OR file_path LIKE ? ORDER BY created_at DESC LIMIT 1",
                                (path.stem, f"%{path.name}%"),
                            ).fetchone()
                            if row:
                                if row["run_id"] and not res.get("run_id"):
                                    res["run_id"] = row["run_id"]
                                if row["workspace_id"] and not res.get("workspace_id"):
                                    res["workspace_id"] = row["workspace_id"]
                    except Exception:
                        pass
                res.setdefault("workspace_id", "default")
                return res
            except Exception:
                pass

        try:
            if path.suffix == ".safetensors":
                res["format"] = "safetensors"
            else:
                res["format"] = "pytorch"
                try:
                    chk = torch.load(path, map_location="cpu", weights_only=True)
                except Exception as e:
                    # Record error safely without executing untrusted payload
                    res["load_error"] = str(e)
                    chk = None

                if isinstance(chk, dict):
                    res["version"] = chk.get("version", "1.0")
                    res["step"] = chk.get("step", chk.get("global_step", 0))
                    res["tokens_seen"] = chk.get("tokens_seen", 0)
                    res["val_loss"] = chk.get("val_loss")
                    res["train_loss"] = chk.get("train_loss")
                    res["phase"] = chk.get("phase", "unknown")
                    sd = chk.get("model_state_dict", chk)
                    res["num_tensors"] = len(sd)
                    res["total_params"] = sum(p.numel() for p in sd.values() if isinstance(p, torch.Tensor))
        except Exception as e:
            res["load_error"] = str(e)
        if self.db is not None:
            try:
                with self.db._lock, self.db._get_connection() as conn:
                    row = conn.execute(
                        "SELECT run_id, workspace_id FROM checkpoints WHERE id = ? OR file_path LIKE ? ORDER BY created_at DESC LIMIT 1",
                        (path.stem, f"%{path.name}%"),
                    ).fetchone()
                    if row:
                        if row["run_id"] and not res.get("run_id"):
                            res["run_id"] = row["run_id"]
                        if row["workspace_id"] and not res.get("workspace_id"):
                            res["workspace_id"] = row["workspace_id"]
            except Exception:
                pass
        res.setdefault("workspace_id", "default")
        return res

    def list_checkpoints_on_disk(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Scans checkpoint and model directories for all .pt and .safetensors files, optionally filtered by workspace."""
        files = []
        # Scan standard checkpoints directory
        for ext in ("*.pt", "*.bin", "*.safetensors"):
            for p in self.checkpoint_dir.glob(ext):
                if p.name.startswith(".tmp"):
                    continue
                info = self.inspect_checkpoint(p)
                files.append(info)
        # Scan models directory for saved local packages
        models_root = Path("models")
        if models_root.exists():
            for ext in ("*.pt", "*.bin", "*.safetensors"):
                for p in models_root.glob(f"**/{ext}"):
                    if p.name.startswith(".tmp"):
                        continue
                    info = self.inspect_checkpoint(p)
                    files.append(info)
        files.sort(key=lambda x: x.get("step", 0) or 0, reverse=True)
        if workspace_id:
            files = [f for f in files if f.get("workspace_id", "default") == workspace_id]
        return files

    def save_local_model_package(
        self,
        model_id: str = "rocanon_48m",
        custom_name: Optional[str] = None,
        save_mode: str = "best",
        active_model: Optional[nn.Module] = None,
        active_optimizers: Optional[Tuple[torch.optim.Optimizer, torch.optim.Optimizer]] = None,
        active_scheduler: Optional[Any] = None,
        training_config: Optional[Dict[str, Any]] = None,
        metrics_history: Optional[List[Dict[str, Any]]] = None,
        output_base_dir: Union[str, Path] = "models",
        workspace_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Saves model weights locally under a custom name or {model_id} name:
        1. {name}_full.pt: Complete weights with optimizer/scheduler/RNG states for resuming training.
        2. {name}_stripped.pt: Stripped compact weights for inference (no optimizer moments, minimal size).
        3. High-resolution validation loss & training dynamics charts in the model folder.
        4. Companion metadata.json.
        """
        raw_name = (custom_name.strip() if custom_name and custom_name.strip() else model_id).strip()
        target_name = re.sub(r"[^\w\-_.]", "_", raw_name)
        out_folder = Path(output_base_dir) / target_name
        out_folder.mkdir(parents=True, exist_ok=True)

        logger.info("Saving local model package '%s' into %s (mode=%s)", target_name, out_folder, save_mode)

        # 1. Determine model weights & training states
        payload_to_use: Dict[str, Any] = {}
        best_chk_path = self.checkpoint_dir / "best_model.pt"

        if save_mode == "best" and best_chk_path.exists():
            try:
                payload_to_use = torch.load(best_chk_path, map_location="cpu", weights_only=False)
                logger.info("Loaded best_model.pt for local package '%s'", target_name)
            except Exception as e:
                logger.warning("Could not load best_model.pt (%s), falling back to active model or latest", e)

        if not payload_to_use and active_model is not None:
            # Active model in memory
            muon_sd = None
            adamw_sd = None
            if active_optimizers:
                m_opt, a_opt = active_optimizers
                if m_opt: muon_sd = m_opt.state_dict()
                if a_opt: adamw_sd = a_opt.state_dict()
            sched_sd = active_scheduler.state_dict() if active_scheduler and hasattr(active_scheduler, "state_dict") else None
            
            payload_to_use = {
                "version": "2.0",
                "step": getattr(active_model, "global_step", 0),
                "tokens_seen": 0,
                "model_state_dict": {k: v.cpu() for k, v in active_model.state_dict().items()},
                "muon_state_dict": muon_sd,
                "adamw_state_dict": adamw_sd,
                "scheduler_state_dict": sched_sd,
                "training_config": training_config or {},
            }
        elif not payload_to_use:
            # Check latest checkpoint in self.checkpoint_dir
            chk_files = [p for p in self.checkpoint_dir.glob("*.pt") if not p.name.startswith(".tmp")]
            if chk_files:
                chk_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                payload_to_use = torch.load(chk_files[0], map_location="cpu", weights_only=False)

        if not payload_to_use or "model_state_dict" not in payload_to_use:
            raise RuntimeError(f"No checkpoint or active model weights found to save package '{target_name}'.")

        model_sd = payload_to_use["model_state_dict"]
        step = payload_to_use.get("step", 0)
        tokens_seen = payload_to_use.get("tokens_seen", 0)
        val_loss = payload_to_use.get("val_loss")
        train_loss = payload_to_use.get("train_loss")
        config = payload_to_use.get("training_config") or training_config or {}

        # 2. Save Full Keep-Training Weights ({target_name}_full.pt)
        full_pt_name = f"{target_name}_full.pt"
        full_pt_path = out_folder / full_pt_name
        full_payload = {
            "version": "2.0",
            "type": "keep_training",
            "model_id": model_id,
            "name": target_name,
            "step": step,
            "tokens_seen": tokens_seen,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "model_state_dict": model_sd,
            "muon_state_dict": payload_to_use.get("muon_state_dict"),
            "adamw_state_dict": payload_to_use.get("adamw_state_dict"),
            "scheduler_state_dict": payload_to_use.get("scheduler_state_dict"),
            "rng_states": payload_to_use.get("rng_states"),
            "training_config": config,
            "timestamp": time.time(),
        }
        torch.save(full_payload, full_pt_path)
        full_size_bytes = full_pt_path.stat().st_size

        # 3. Save Stripped Smaller Weights for Inference ({target_name}_stripped.pt)
        # Cast floating tensors to float16 or keep as-is, exclude all optimizer states
        stripped_pt_name = f"{target_name}_stripped.pt"
        stripped_pt_path = out_folder / stripped_pt_name
        stripped_sd = {
            k: (v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
            for k, v in model_sd.items()
        }
        stripped_payload = {
            "version": "2.0",
            "type": "stripped_inference",
            "model_id": model_id,
            "name": target_name,
            "step": step,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "total_params": sum(p.numel() for p in model_sd.values() if isinstance(p, torch.Tensor)),
            "model_state_dict": stripped_sd,
            "training_config": config,
            "timestamp": time.time(),
        }
        torch.save(stripped_payload, stripped_pt_path)
        stripped_size_bytes = stripped_pt_path.stat().st_size

        # 4. Generate validation loss and training dynamics charts in model folder
        chart_paths: Dict[str, str] = {}
        history = metrics_history
        if not history and self.db is not None:
            # Query recent metrics from database
            try:
                run_id = payload_to_use.get("run_id")
                if run_id:
                    history = self.db.get_metrics_history(run_id, limit=2000)
                else:
                    # Query latest run metrics
                    with self.db._lock, self.db._get_connection() as conn:
                        row = conn.execute("SELECT id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
                        if row:
                            history = self.db.get_metrics_history(row["id"], limit=2000)
            except Exception as e:
                logger.warning("Could not retrieve metrics for charts: %s", e)

        if history:
            try:
                from .charts import generate_training_charts
                chart_paths = generate_training_charts(
                    metrics=history,
                    output_dir=out_folder,
                    model_id=target_name,
                    best_step=step,
                    best_val_loss=val_loss,
                )
            except Exception as ce:
                logger.warning("Failed to render charts for model package: %s", ce)

        # 5. Write metadata.json
        resolved_id = target_name if (custom_name and custom_name.strip()) else model_id
        meta_info = {
            "model_id": resolved_id,
            "workspace_id": workspace_id or "default",
            "name": target_name,
            "custom_name": custom_name.strip() if (custom_name and custom_name.strip()) else target_name,
            "saved_at": time.time(),
            "save_mode": save_mode,
            "step": step,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "tokens_seen": tokens_seen,
            "total_parameters": sum(p.numel() for p in model_sd.values() if isinstance(p, torch.Tensor)),
            "files": {
                "full_weights": {
                    "file_name": full_pt_name,
                    "path": str(full_pt_path.resolve()),
                    "size_bytes": full_size_bytes,
                    "size_mb": round(full_size_bytes / (1024 * 1024), 2),
                    "description": "Full weights with Muon & AdamW optimizer states for continuing training",
                },
                "stripped_weights": {
                    "file_name": stripped_pt_name,
                    "path": str(stripped_pt_path.resolve()),
                    "size_bytes": stripped_size_bytes,
                    "size_mb": round(stripped_size_bytes / (1024 * 1024), 2),
                    "reduction_pct": round((1.0 - stripped_size_bytes / max(1, full_size_bytes)) * 100, 1),
                    "description": "Compact stripped weights for high-efficiency inference",
                },
                "charts": {k: str(Path(v).name) for k, v in chart_paths.items()},
            },
            "training_config": config,
        }
        meta_json_path = out_folder / "metadata.json"
        with open(meta_json_path, "w", encoding="utf-8") as mf:
            json.dump(meta_info, mf, indent=2)

        # 6. Auto-register network config in registry so it appears in LLM Builder & Training selectors
        try:
            reg_dir = Path("networks/registry")
            reg_dir.mkdir(parents=True, exist_ok=True)
            net_path = reg_dir / f"{target_name}.json"
            if not net_path.exists():
                from ..networks.registry import NetworkRegistry
                base_cfg = None
                try:
                    base_cfg = NetworkRegistry().get_network(model_id)
                except Exception:
                    pass
                if not base_cfg:
                    try:
                        base_cfg = NetworkRegistry().get_network("rocanon_48m")
                    except Exception:
                        pass
                if base_cfg:
                    new_cfg = dict(base_cfg)
                    new_cfg["id"] = target_name
                    new_cfg["name"] = target_name
                    new_cfg["workspace_id"] = workspace_id or "default"
                    new_cfg["description"] = f"Saved local model package {target_name} (Trained checkpoint step {step}, val loss {val_loss})."
                    with open(net_path, "w", encoding="utf-8") as nf:
                        json.dump(new_cfg, nf, indent=2)
        except Exception as n_err:
            logger.warning("Could not auto-register network for saved model: %s", n_err)

        return meta_info

    def delete_checkpoint(self, filename_or_path: str) -> bool:
        """Deletes checkpoint from disk and database."""
        path = Path(filename_or_path)
        if not path.is_absolute():
            path = self.checkpoint_dir / path.name
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning("Could not unlink %s: %s", path, e)
                return False

            # Delete companion JSON if present
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                try:
                    meta_path.unlink()
                except Exception:
                    pass

            if self.db is not None:
                # Remove from database by matching path suffix or id
                chk_id = path.stem
                try:
                    self.db.delete_checkpoint(chk_id)
                except Exception:
                    pass
            return True
        return False
        return False

    def prune_checkpoints(self, keep_last_n: int = 4) -> List[str]:
        """
        Retains at most keep_last_n step-based checkpoint files (.pt).
        Preserves 'best_model.pt' unconditionally.
        Deletes older step checkpoints from disk and database.
        Returns list of deleted filenames.
        """
        if keep_last_n <= 0:
            return []

        step_files: List[Tuple[int, float, Path]] = []
        for p in self.checkpoint_dir.glob("*.pt"):
            if p.name.startswith(".tmp") or p.name == "best_model.pt":
                continue

            # Extract step number if available from filename or fallback to mtime
            step_num = 0
            m = re.findall(r"step_(\d+)", p.name)
            if m:
                step_num = int(m[-1])
            else:
                try:
                    step_num = int(p.stat().st_mtime)
                except Exception:
                    pass

            step_files.append((step_num, p.stat().st_mtime, p))

        # Sort ascending by file mtime (oldest first, so newly saved runs/steps are never purged over old runs)
        step_files.sort(key=lambda x: x[1])

        deleted: List[str] = []
        num_to_delete = len(step_files) - keep_last_n
        if num_to_delete > 0:
            for step_num, mtime, file_path in step_files[:num_to_delete]:
                try:
                    logger.info("Pruning old checkpoint: %s (step %d)", file_path.name, step_num)
                    if self.delete_checkpoint(file_path.name):
                        deleted.append(file_path.name)
                except Exception as err:
                    logger.warning("Failed to prune checkpoint %s: %s", file_path.name, err)

        return deleted


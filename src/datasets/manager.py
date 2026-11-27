"""
src/datasets/manager.py
Central dataset management service supporting upload, validation, statistical analysis,
immutable versioning, and tokenizer inspection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..db.database import StudioDatabase
from ..tokenizer.tokenizer import RoCanonTokenizer
from ..utils.paths import cache_dir, datasets_dir, uploads_dir
from .analyzer import DatasetAnalyzer
from .validator import DatasetValidator

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_DIR = uploads_dir()


class DatasetManager:
    """Manages dataset files, immutable revisions, and statistical exploration."""

    def __init__(
        self,
        upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
        db: Optional[StudioDatabase] = None,
        tokenizer: Optional[RoCanonTokenizer] = None,
    ) -> None:
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db = db or StudioDatabase()
        self.tokenizer = tokenizer or RoCanonTokenizer()
        self.validator = DatasetValidator()
        self.analyzer = DatasetAnalyzer(tokenizer=self.tokenizer)
        self.deleted_dataset_ids: set[str] = set()

    def compute_file_hash(self, file_path: Path) -> str:
        """Computes SHA-256 revision hash of dataset content."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def register_uploaded_file(
        self,
        file_path: Path | str,
        name: Optional[str] = None,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        workspace_id: str = "default",
    ) -> Dict[str, Any]:
        """Validates and analyzes an uploaded dataset, saving it with an immutable revision."""
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        ds_name = name or path.name
        revision = self.compute_file_hash(path)
        dataset_id = f"{path.stem}_{revision}"

        # Destination in upload dir
        dest_filename = f"{path.stem}_{revision}{path.suffix}"
        dest_path = self.upload_dir / dest_filename
        if path.resolve() != dest_path.resolve():
            shutil.copy2(path, dest_path)

        # Validate
        validator = DatasetValidator(
            instruction_field=instruction_field,
            input_field=input_field,
            output_field=output_field,
        )
        val_res = validator.validate_file(dest_path)
        if not val_res.get("valid"):
            return {
                "success": False,
                "dataset_id": dataset_id,
                "error": "Dataset validation failed",
                "validation_report": val_res,
            }

        # Analyze
        extracted_texts = val_res.get("extracted_texts", [])
        stats = self.analyzer.analyze_samples(extracted_texts)

        file_size = dest_path.stat().st_size
        token_count = stats.get("estimated_total_tokens", 0)
        sample_count = len(extracted_texts)

        # Save metadata to DB
        self.db.save_dataset(
            dataset_id=dataset_id,
            name=ds_name,
            file_path=str(dest_path.resolve()),
            file_size_bytes=file_size,
            format_type=val_res.get("format", "unknown"),
            revision=revision,
            token_count=token_count,
            sample_count=sample_count,
            stats={
                "validation": {
                    "total_records": val_res.get("total_records"),
                    "valid_records": val_res.get("valid_records"),
                    "invalid_records": val_res.get("invalid_records"),
                    "errors_sample": val_res.get("errors", [])[:10],
                },
                "analysis": stats,
                "preview": val_res.get("samples_preview", [])[:5],
            },
            workspace_id=workspace_id,
        )

        return {
            "success": True,
            "dataset_id": dataset_id,
            "name": ds_name,
            "revision": revision,
            "file_path": str(dest_path.resolve()),
            "token_count": token_count,
            "sample_count": sample_count,
            "stats": stats,
            "preview": val_res.get("samples_preview", [])[:5],
        }

    def inspect_tokenizer_sample(self, text: str) -> Dict[str, Any]:
        """Exposes Tokenizer Inspector metrics."""
        return self.analyzer.inspect_tokenization(text)

    def get_dataset_details(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        ds = self.db.get_dataset(dataset_id)
        if ds:
            return ds
        for preset in self.list_all_datasets():
            if preset["id"] == dataset_id:
                cache_p = cache_dir() / f"{dataset_id}_tokens.pt"
                if not cache_p.exists():
                    cache_p = cache_dir() / f"{dataset_id}_full_tokens.pt"
                raw_txt = datasets_dir() / f"{dataset_id}.txt"
                raw_path = str(raw_txt.resolve()) if raw_txt.exists() else str(raw_txt)
                return {
                    "id": preset["id"],
                    "name": preset["name"],
                    "path": raw_path,
                    "file_path": raw_path,
                    "size_mb": preset.get("size_mb", 0.0),
                    "file_size_bytes": int(preset.get("size_mb", 0.0) * 1024 * 1024),
                    "total_tokens": preset.get("total_tokens", 0),
                    "train_tokens": preset.get("train_tokens", 0),
                    "val_tokens": preset.get("val_tokens", 0),
                    "token_count": preset.get("total_tokens", 0),
                    "sample_count": 50000,
                    "format": "raw_text_stream",
                    "cached": cache_p.exists(),
                    "cache_file": str(cache_p) if cache_p.exists() else None,
                    "is_uploaded": False,
                }
        return None

    def list_all_datasets(self, workspace_id: str = "default") -> List[Dict[str, Any]]:
        presets = []
        if workspace_id == "default":
            presets = [
                {
                    "id": "tech_wiki_ro",
                    "name": "RoCanon Pretrain Corpus (tech_wiki_ro)",
                    "source": "preset",
                    "status": "ready",
                    "size_mb": 115.4,
                    "total_tokens": 48419639,
                    "train_tokens": 48419639,
                    "val_tokens": 2548402,
                    "is_uploaded": False,
                },
                {
                    "id": "synthetic",
                    "name": "Fast Synthetic Benchmark (synthetic)",
                    "source": "synthetic",
                    "status": "ready",
                    "size_mb": 1.0,
                    "total_tokens": 100000,
                    "train_tokens": 80000,
                    "val_tokens": 20000,
                    "is_uploaded": False,
                },
            ]
        uploaded = self.db.list_datasets(workspace_id=workspace_id)
        for u in uploaded:
            u["is_uploaded"] = True
            if "size_mb" not in u and "file_size_bytes" in u:
                u["size_mb"] = round(u["file_size_bytes"] / (1024 * 1024), 2)
            if "total_tokens" not in u and "token_count" in u:
                u["total_tokens"] = u["token_count"]
        combined = presets + uploaded
        return [d for d in combined if d["id"] not in self.deleted_dataset_ids]

    def delete_dataset(self, dataset_id: str) -> bool:
        """Deletes dataset from memory filter, DB, and unlinks physical storage."""
        self.deleted_dataset_ids.add(dataset_id)
        self.db.delete_dataset(dataset_id)
        # Safely remove files
        for search_dir in (self.upload_dir, cache_dir(), datasets_dir()):
            if search_dir.exists():
                for p in search_dir.glob(f"*{dataset_id}*"):
                    try:
                        if p.is_file():
                            p.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning("Could not unlink dataset file %s: %s", p, e)
        return True

    def _resolve_dataset_path(self, dataset_id: str, source_path: Path | str | None = None) -> Path:
        if source_path is not None:
            path = Path(source_path)
        else:
            ds_info = self.get_dataset_details(dataset_id) or {}
            source = ds_info.get("file_path")
            if not source:
                raise FileNotFoundError(f"Dataset '{dataset_id}' not found or has no valid file path.")
            path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist on disk: {path}")
        return path

    def _invalidate_token_cache(self, stem: str) -> None:
        cache = cache_dir()
        if not cache.is_dir():
            return
        patterns = (
            f"{stem}_full_tokens.bin",
            f"{stem}_full_tokens.pt",
            f"{stem}.tokenized",
            f"{stem}_sft*.pt",
            f"{stem}_train*.bin",
            f"{stem}_val*.bin",
        )
        for pattern in patterns:
            for cached in cache.glob(pattern):
                try:
                    cached.unlink(missing_ok=True)
                except OSError as exc:
                    logger.debug("Could not remove stale token cache %s: %s", cached, exc)

    def clean_dataset(
        self,
        dataset_id: str,
        options: Dict[str, Any] | None = None,
        source_path: Path | str | None = None,
        output_suffix: str = "_clean",
        save_cleaned: bool = True,
        name: str | None = None,
        workspace_id: str = "default",
        progress: Callable[[str, int, int | None, str], None] | None = None,
    ) -> Dict[str, Any]:
        """Clean a dataset into a sibling file using the shared format-aware pipeline."""
        from ..data.dataset_cleaning import CleanOptions, clean_dataset_file

        opts = options or {}
        path = self._resolve_dataset_path(dataset_id, source_path)
        output_path = self.upload_dir / f"{path.stem}{output_suffix}{path.suffix}"
        clean_options = CleanOptions(
            quality=bool(opts.get("quality", True)),
            language=bool(opts.get("language", True)),
            deduplicate=bool(opts.get("deduplicate", True)),
            mask_pii=bool(opts.get("mask_pii", True)),
            lid_threshold=float(opts.get("lid_threshold", 0.80)),
            jaccard_threshold=float(opts.get("jaccard_threshold", 0.80)),
            target_languages=tuple(opts.get("target_languages", ("ron", "eng", "ro", "en"))),
            max_docs=opts.get("max_samples") or None,
        )
        result = clean_dataset_file(path, output_path, clean_options, progress=progress)
        self._invalidate_token_cache(path.stem)
        self._invalidate_token_cache(output_path.stem)

        cleaned_dataset_id = None
        if result.get("records_out", 0) > 0 and save_cleaned:
            registered = self.register_uploaded_file(
                output_path,
                name=name or f"{dataset_id} (Clean)",
                workspace_id=workspace_id,
            )
            if registered.get("success"):
                cleaned_dataset_id = registered.get("dataset_id")
        cleaned_path = str(output_path) if result.get("records_out", 0) > 0 else None
        result.update({
            "dataset_id": dataset_id,
            "cleaned_dataset_id": cleaned_dataset_id,
            "cleaned_file_path": cleaned_path,
            "curated_dataset_id": cleaned_dataset_id,
            "curated_file_path": cleaned_path,
        })
        return result

    def curate_dataset(
        self,
        dataset_id: str,
        jaccard_threshold: float = 0.80,
        lid_threshold: float = 0.80,
        target_languages: tuple[str, ...] = ("ron", "eng", "ro", "en"),
        save_curated: bool = True,
        max_samples: Optional[int] = None,
        source_path: Path | str | None = None,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper around the shared cleaning pipeline."""
        result = self.clean_dataset(
            dataset_id,
            options={
                "quality": True,
                "language": True,
                "deduplicate": True,
                "mask_pii": True,
                "jaccard_threshold": jaccard_threshold,
                "lid_threshold": lid_threshold,
                "target_languages": target_languages,
                "max_samples": max_samples,
            },
            source_path=source_path,
            output_suffix="_curated",
            save_cleaned=save_curated,
            name=f"{dataset_id} (Curated)",
        )
        result.update({
            "total_samples_analyzed": result.get("records_in", 0),
            "total_samples": result.get("records_in", 0),
            "accepted_samples": result.get("records_out", 0),
            "curated_samples": result.get("records_out", 0),
            "duplicates_purged": result.get("duplicates_removed", 0),
            "exact_duplicates_dropped": result.get("exact_duplicates", 0),
            "minhash_duplicates_dropped": result.get("near_duplicates", 0),
            "pii_masked_count": result.get("pii_documents", 0),
            "quality_pass_rate_pct": result.get("retention_pct", 0.0),
            "retention_rate": result.get("retention_pct", 0.0),
            "duplicate_rate_pct": round(
                result.get("duplicates_removed", 0) / max(1, result.get("records_in", 0)) * 100, 2
            ),
            "lid_threshold": lid_threshold,
            "jaccard_threshold": jaccard_threshold,
            "preview_samples": [],
        })
        return result

    def import_cleaned_file(
        self,
        file_path: Path | str,
        name: str | None = None,
        clean: bool = True,
        options: Dict[str, Any] | None = None,
        workspace_id: str = "default",
        progress: Callable[[str, int, int | None, str], None] | None = None,
    ) -> Dict[str, Any]:
        """Import a file, optionally cleaning it and hiding the raw source under raw/."""
        source = Path(file_path)
        if not source.is_file():
            return {"success": False, "error": f"File not found: {source}"}
        safe_name = Path(name or source.name).name
        if not safe_name or safe_name in {".", ".."}:
            return {"success": False, "error": "Invalid dataset filename"}
        raw_dir = self.upload_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / safe_name
        if raw_path.exists():
            raw_path = raw_dir / f"{Path(safe_name).stem}_{time.time_ns()}{Path(safe_name).suffix}"
        shutil.copy2(source, raw_path)

        if not clean:
            target = self.upload_dir / safe_name
            if source.resolve() != target.resolve():
                shutil.copy2(raw_path, target)
            result = self.register_uploaded_file(target, name=safe_name, workspace_id=workspace_id)
            result["cleaning"] = None
            return result

        try:
            from ..data.dataset_cleaning import CleanOptions, clean_dataset_file

            opts = options or {}
            output = self.upload_dir / safe_name
            result = clean_dataset_file(
                raw_path,
                output,
                CleanOptions(
                    quality=bool(opts.get("quality", True)),
                    language=bool(opts.get("language", True)),
                    deduplicate=bool(opts.get("deduplicate", True)),
                    mask_pii=bool(opts.get("mask_pii", True)),
                    lid_threshold=float(opts.get("lid_threshold", 0.80)),
                    jaccard_threshold=float(opts.get("jaccard_threshold", 0.80)),
                    target_languages=tuple(opts.get("target_languages", ("ron", "eng", "ro", "en"))),
                    max_docs=opts.get("max_samples") or None,
                ),
                progress=progress,
            )
            self._invalidate_token_cache(output.stem)
            if result.get("records_out", 0) <= 0:
                raise ValueError("Cleaning removed every record")
            registered = self.register_uploaded_file(output, name=safe_name, workspace_id=workspace_id)
            if not registered.get("success"):
                raise ValueError(registered.get("error") or "Cleaned dataset validation failed")
            registered["cleaning"] = result
            return registered
        except Exception as exc:
            logger.warning("Dataset cleaning failed for %s; importing raw file: %s", safe_name, exc)
            target = self.upload_dir / safe_name
            if raw_path.resolve() != target.resolve():
                shutil.copy2(raw_path, target)
            registered = self.register_uploaded_file(target, name=safe_name, workspace_id=workspace_id)
            registered["cleaning"] = {"success": False, "error": str(exc)}
            return registered


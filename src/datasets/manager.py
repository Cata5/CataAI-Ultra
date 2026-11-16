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
from typing import Any, Dict, List, Optional, Tuple

from ..db.database import StudioDatabase
from ..tokenizer.tokenizer import RoCanonTokenizer
from .analyzer import DatasetAnalyzer
from .validator import DatasetValidator

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_DIR = Path("data/uploads")


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
                cache_p = Path("data/cache") / f"{dataset_id}_tokens.pt"
                if not cache_p.exists():
                    cache_p = Path("data/cache") / f"{dataset_id}_full_tokens.pt"
                raw_txt = Path(f"data/{dataset_id}.txt")
                raw_path = str(raw_txt.resolve()) if raw_txt.exists() else f"data/{dataset_id}.txt"
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

    def list_all_datasets(self) -> List[Dict[str, Any]]:
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
        uploaded = self.db.list_datasets()
        for u in uploaded:
            u["is_uploaded"] = True
            if "size_mb" not in u and "file_size_bytes" in u:
                u["size_mb"] = round(u["file_size_bytes"] / (1024 * 1024), 2)
            if "total_tokens" not in u and "token_count" in u:
                u["total_tokens"] = u["token_count"]
        return presets + uploaded


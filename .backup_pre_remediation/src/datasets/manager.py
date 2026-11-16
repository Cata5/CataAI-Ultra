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
        return self.db.get_dataset(dataset_id)

    def list_all_datasets(self) -> List[Dict[str, Any]]:
        return self.db.list_datasets()


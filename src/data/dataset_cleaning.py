"""Format-aware cleaning pipeline used by imports, curation, and the worker."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .curation import (
    MinHashLSHDeduplicator,
    check_text_quality,
    classify_language_statistical,
    mask_pii_and_count,
    mask_pii_in_text,
    strip_html_and_boilerplate,
)
from .document_io import TrainingDocument, count_training_documents, iter_training_documents


ProgressCallback = Callable[[str, int, int | None, str], None]


@dataclass
class CleanOptions:
    quality: bool = True
    language: bool = True
    deduplicate: bool = True
    mask_pii: bool = True
    min_chars_text: int = 50
    min_words_text: int = 5
    min_chars_record: int = 20
    min_words_record: int = 3
    lid_threshold: float = 0.80
    target_languages: tuple[str, ...] = ("ron", "eng", "ro", "en")
    jaccard_threshold: float = 0.80
    num_perm: int = 64
    b_bands: int = 16
    k_shingle: int = 5
    max_docs: int | None = None


@dataclass
class CleanReport:
    records_in: int = 0
    records_out: int = 0
    quality_rejected: int = 0
    language_rejected: int = 0
    language_uncertain: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    pii_documents: int = 0
    empty_documents: int = 0
    pii_occurrences: dict[str, int] = field(default_factory=dict)

    @property
    def duplicates_removed(self) -> int:
        return self.exact_duplicates + self.near_duplicates

    @property
    def hard_rejected(self) -> int:
        return self.records_in - self.records_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_in": self.records_in,
            "records_out": self.records_out,
            "hard_rejected": self.hard_rejected,
            "quality_rejected": self.quality_rejected,
            "language_rejected": self.language_rejected,
            "language_uncertain": self.language_uncertain,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "duplicates_removed": self.duplicates_removed,
            "pii_documents": self.pii_documents,
            "pii_occurrences": dict(self.pii_occurrences),
            "empty_documents": self.empty_documents,
            "retention_pct": round(
                (self.records_out / self.records_in * 100.0) if self.records_in else 0.0, 2
            ),
        }


_CHATML_MARKER = re.compile(r"<\|[^|>]+\|>")
_CONTENT_KEYS = {
    "text",
    "content",
    "instruction",
    "input",
    "output",
    "prompt",
    "question",
    "answer",
    "response",
    "chosen",
    "rejected",
    "completion",
    "context",
    "system",
}



def _quality_text(text: str) -> str:
    """Remove structural markers for quality measurements, not from the output."""
    if "<|" not in text:
        return text
    return _CHATML_MARKER.sub(" ", text)


def _looks_code_or_math(text: str) -> bool:
    if "```" in text:
        return True
    if "<" in text and "<code>" in text.lower():
        return True
    compact = text.strip()
    if not compact:
        return False
    symbols = sum(1 for ch in compact if ch in "{}[]();=<>+-*/|&")
    return symbols / len(compact) > 0.10


def _clean_string(value: str) -> str:
    return mask_pii_in_text(strip_html_and_boilerplate(value))


def _is_nan(value: Any) -> bool:
    """True for a float NaN coming from CSV, Parquet or a JSON NaN literal."""
    return isinstance(value, float) and math.isnan(value)


def _clean_payload(
    value: Any,
    memo: dict[str, str] | None = None,
    *,
    clean_strings: bool = True,
) -> Any:
    """Clean content fields and replace NaN with JSON-safe values.

    ``clean_strings`` is only false below a non-content key, so IDs and labels
    are preserved while the text fields that feed training are cleaned.
    """
    if value is None:
        return "" if clean_strings else None
    if _is_nan(value):
        # A missing Alpaca input or a null message content must become an empty
        # string, not the literal "nan" that poisoned the SFT corpus.
        return "" if clean_strings else None
    if isinstance(value, str):
        if not clean_strings:
            return value
        if memo is not None:
            cached = memo.get(value)
            if cached is not None:
                return cached
        cleaned = _clean_string(value)
        if memo is not None:
            memo[value] = cleaned
        return cleaned
    if isinstance(value, list):
        return [_clean_payload(item, memo, clean_strings=clean_strings) for item in value]
    if not isinstance(value, dict):
        return value
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        is_content = key in _CONTENT_KEYS or key == "messages"
        cleaned[key] = _clean_payload(item, memo, clean_strings=is_content)
    return cleaned


class _Cleaner:
    def __init__(
        self,
        options: CleanOptions,
        progress: ProgressCallback | None = None,
        total: int | None = None,
    ):
        self.options = options
        self.progress = progress
        self.total = total
        self.report = CleanReport()
        self._target_languages = set(options.target_languages)
        self._seen_exact: set[str] = set()
        self._deduper = MinHashLSHDeduplicator(
            num_perm=options.num_perm,
            k_shingle=options.k_shingle,
            jaccard_threshold=options.jaccard_threshold,
            b_bands=options.b_bands,
        )

    def stage(self, phase: str, message: str) -> None:
        self._notify(phase, self.report.records_in, message)

    def _notify(self, phase: str, current: int, message: str) -> None:
        if self.progress is None:
            return
        try:
            self.progress(phase, current, self.total, message)
        except Exception:
            pass

    def process(self, document: TrainingDocument) -> tuple[str, Any] | None:
        options = self.options
        self.report.records_in += 1
        if self.report.records_in == 1 or self.report.records_in % 25 == 0 or (
            self.total is not None and self.report.records_in >= self.total
        ):
            self._notify(
                "filtering",
                self.report.records_in,
                f"Processed {self.report.records_in:,} records",
            )
        cleaned = strip_html_and_boilerplate(document.text or "")
        if not cleaned.strip():
            self.report.empty_documents += 1
            return None

        quality_text = _quality_text(cleaned)
        min_chars = options.min_chars_record if document.kind == "record" else options.min_chars_text
        min_words = options.min_words_record if document.kind == "record" else options.min_words_text
        if options.quality and not check_text_quality(
            quality_text, min_chars=min_chars, min_words=min_words
        ):
            self.report.quality_rejected += 1
            return None

        if options.language and not _looks_code_or_math(quality_text):
            language, confidence = classify_language_statistical(
                quality_text, options.target_languages, normalized=True
            )
            if language not in self._target_languages:
                if confidence >= options.lid_threshold:
                    self.report.language_rejected += 1
                    return None
                self.report.language_uncertain += 1

        if options.deduplicate:
            exact_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            if exact_hash in self._seen_exact:
                self.report.exact_duplicates += 1
                return None
            self._seen_exact.add(exact_hash)
            if self._deduper.is_duplicate(cleaned, normalized=True):
                self.report.near_duplicates += 1
                return None

        if options.mask_pii:
            cleaned, pii = mask_pii_and_count(cleaned)
            if pii:
                self.report.pii_documents += 1
                for key, count in pii.items():
                    self.report.pii_occurrences[key] = self.report.pii_occurrences.get(key, 0) + count

        if document.kind == "text":
            payload = cleaned
        else:
            # A JSON/CSV row usually contains the same text field that was just
            # cleaned above. Reuse that result instead of stripping and masking
            # the original string a second time.
            memo = {document.text: cleaned} if document.text else {}
            payload = _clean_payload(document.payload, memo)
        self.report.records_out += 1
        return cleaned, payload


def _atomic_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")


def _finish_atomic(tmp: Path, output: Path, records_out: int) -> None:
    if records_out <= 0:
        tmp.unlink(missing_ok=True)
        return
    os.replace(tmp, output)


def _write_text_dataset(source: Path, output: Path, cleaner: _Cleaner, suffix: str) -> None:
    tmp = _atomic_path(output)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            for document in iter_training_documents(source, cleaner.options.max_docs):
                result = cleaner.process(document)
                if not result:
                    continue
                _cleaned, payload = result
                if suffix == ".txt":
                    handle.write(str(payload))
                    handle.write("\n\n")
                elif suffix == ".jsonl":
                    handle.write(json.dumps(payload, ensure_ascii=False))
                    handle.write("\n")
        cleaner.stage("writing", "Writing cleaned dataset...")
        _finish_atomic(tmp, output, cleaner.report.records_out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _write_json_dataset(source: Path, output: Path, cleaner: _Cleaner) -> None:
    rows: list[Any] = []
    for document in iter_training_documents(source, cleaner.options.max_docs):
        result = cleaner.process(document)
        if result:
            rows.append(result[1])
    cleaner.stage("writing", "Writing cleaned dataset...")
    tmp = _atomic_path(output)
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        _finish_atomic(tmp, output, cleaner.report.records_out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _write_csv_dataset(source: Path, output: Path, cleaner: _Cleaner) -> None:
    import csv

    with open(source, "r", encoding="utf-8", errors="replace", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames or []
    tmp = _atomic_path(output)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for document in iter_training_documents(source, cleaner.options.max_docs):
                result = cleaner.process(document)
                if result:
                    writer.writerow(result[1])
        cleaner.stage("writing", "Writing cleaned dataset...")
        _finish_atomic(tmp, output, cleaner.report.records_out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _write_arrow_dataset(source: Path, output: Path, cleaner: _Cleaner) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    suffix = source.suffix.lower()
    if suffix == ".parquet":
        schema = pq.ParquetFile(str(source)).schema_arrow
    else:
        with pa.memory_map(str(source), "r") as handle:
            try:
                schema = pa.ipc.open_stream(handle).schema
            except Exception:
                handle.seek(0)
                schema = pa.ipc.open_file(handle).schema
    rows: list[dict[str, Any]] = []
    for document in iter_training_documents(source, cleaner.options.max_docs):
        result = cleaner.process(document)
        if result:
            rows.append(result[1])
    table = pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()
    cleaner.stage("writing", "Writing cleaned dataset...")
    tmp = _atomic_path(output)
    try:
        if suffix == ".parquet":
            pq.write_table(table, tmp)
        else:
            with pa.OSFile(str(tmp), "wb") as sink:
                with pa.ipc.new_file(sink, schema) as writer:
                    writer.write_table(table)
        _finish_atomic(tmp, output, cleaner.report.records_out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def clean_dataset_file(
    source: str | Path,
    output: str | Path | None = None,
    options: CleanOptions | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Clean a dataset into a new file without modifying the source."""
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source_path}")
    options = options or CleanOptions()
    suffix = source_path.suffix.lower()
    if suffix not in {".txt", ".jsonl", ".json", ".csv", ".parquet", ".arrow"}:
        raise ValueError(f"Unsupported dataset format for cleaning: {suffix or '<none>'}")
    output_path = Path(output) if output else source_path.with_name(
        f"{source_path.stem}_clean{source_path.suffix}"
    )
    if output_path.resolve() == source_path.resolve():
        raise ValueError("Cleaning output must differ from the source file")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = None
    if progress is not None:
        progress("counting", 0, None, "Counting records...")
        total = count_training_documents(source_path, options.max_docs)
        progress("filtering", 0, total, "Reading and cleaning records...")
    cleaner = _Cleaner(options, progress=progress, total=total)
    if suffix == ".txt":
        _write_text_dataset(source_path, output_path, cleaner, suffix)
    elif suffix == ".jsonl":
        _write_text_dataset(source_path, output_path, cleaner, suffix)
    elif suffix == ".json":
        _write_json_dataset(source_path, output_path, cleaner)
    elif suffix == ".csv":
        _write_csv_dataset(source_path, output_path, cleaner)
    else:
        _write_arrow_dataset(source_path, output_path, cleaner)

    cleaner.stage("finalizing", "Finalizing cleaned dataset...")
    report = cleaner.report.to_dict()
    report.update({
        "success": True,
        "source": str(source_path),
        "output": str(output_path),
        "format": suffix,
    })
    return report
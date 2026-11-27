"""Format-aware document iteration for dataset cleaning and auditing."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class TrainingDocument:
    """One logical training document with its original payload."""

    index: int
    text: str
    payload: Any
    kind: str = "text"
    text_key: str | None = None


def _scalar_text(value: Any) -> str:
    """Stringify a scalar without turning a missing/NaN field into "nan"."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, dict):
                parts.append(_scalar_text(part.get("text", "")))
            else:
                parts.append(_scalar_text(part))
        return " ".join(p for p in parts if p.strip())
    return _scalar_text(value)


def extract_record_text(obj: Any) -> str:
    """Extract the training text from a JSON/table record."""
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return _scalar_text(obj)

    messages = obj.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = _content_text(message.get("content", ""))
            if content.strip():
                parts.append(content)
        if parts:
            return "\n".join(parts)

    prompt_parts = [
        obj.get(key)
        for key in ("instruction", "system", "input", "prompt", "question", "context")
        if isinstance(obj.get(key), str) and obj.get(key).strip()
    ]
    answer_parts = [
        obj.get(key)
        for key in ("output", "response", "answer", "completion", "chosen")
        if isinstance(obj.get(key), str) and obj.get(key).strip()
    ]
    if prompt_parts or answer_parts:
        return "\n".join([*prompt_parts, *answer_parts])

    for key in ("text", "content", "document", "body"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value

    strings = [value for value in obj.values() if isinstance(value, str) and value.strip()]
    if strings:
        return "\n".join(strings)
    return json.dumps(obj, ensure_ascii=False, default=str)


_TEXT_COLUMNS = (
    "text",
    "content",
    "document",
    "body",
    "sentence",
    "instruction",
    "prompt",
    "question",
    "output",
    "response",
    "answer",
)


def _pick_text_key(row: dict[str, Any]) -> str | None:
    for key in _TEXT_COLUMNS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return key
    for key, value in row.items():
        if isinstance(value, str) and value.strip():
            return key
    return None


def _best_text(row: dict[str, Any]) -> tuple[str, str | None]:
    key = _pick_text_key(row)
    if key is not None:
        return str(row.get(key) or ""), key
    return extract_record_text(row), None


def _iter_txt(path: Path, max_docs: int | None) -> Iterator[TrainingDocument]:
    index = 0
    buffer: list[str] = []

    def emit(lines: list[str]) -> Iterator[TrainingDocument]:
        nonlocal index
        text = "\n".join(lines).strip()
        if not text:
            return
        if max_docs is not None and index >= max_docs:
            return
        yield TrainingDocument(index=index, text=text, payload=text, kind="text")
        index += 1

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.rstrip("\r\n")
            if raw.strip():
                buffer.append(raw)
                # A file without blank lines should not become one giant document.
                if len("\n".join(buffer)) > 50_000:
                    yield from emit(buffer)
                    buffer = []
            elif buffer:
                yield from emit(buffer)
                buffer = []
    if buffer:
        yield from emit(buffer)


def _iter_jsonl(path: Path, max_docs: int | None) -> Iterator[TrainingDocument]:
    index = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if max_docs is not None and index >= max_docs:
                return
            text = extract_record_text(payload)
            yield TrainingDocument(index=index, text=text, payload=payload, kind="record")
            index += 1


def _iter_json(path: Path, max_docs: int | None) -> Iterator[TrainingDocument]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, dict):
        items = data.get("data") if isinstance(data.get("data"), list) else [data]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]
    for index, payload in enumerate(items):
        if max_docs is not None and index >= max_docs:
            return
        text = payload if isinstance(payload, str) else extract_record_text(payload)
        yield TrainingDocument(index=index, text=text, payload=payload, kind="record")


def _iter_csv(path: Path, max_docs: int | None) -> Iterator[TrainingDocument]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if max_docs is not None and index >= max_docs:
                return
            text, key = _best_text(row)
            yield TrainingDocument(index=index, text=text, payload=row, kind="record", text_key=key)


def _iter_parquet(path: Path, max_docs: int | None) -> Iterator[TrainingDocument]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(str(path))
    index = 0
    for batch in parquet_file.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if max_docs is not None and index >= max_docs:
                return
            text, key = _best_text(row)
            yield TrainingDocument(index=index, text=text, payload=row, kind="record", text_key=key)
            index += 1


def _iter_arrow(path: Path, max_docs: int | None) -> Iterator[TrainingDocument]:
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        try:
            reader = pa.ipc.open_stream(source)
        except Exception:
            source.seek(0)
            reader = pa.ipc.open_file(source)
        table = reader.read_all()
    for index, row in enumerate(table.to_pylist()):
        if max_docs is not None and index >= max_docs:
            return
        text, key = _best_text(row)
        yield TrainingDocument(index=index, text=text, payload=row, kind="record", text_key=key)


def count_training_documents(path: Path | str, max_docs: int | None = None) -> int | None:
    """Best-effort record count used for progress reporting.

    Text formats are counted without parsing JSON payloads; malformed JSONL lines
    are included in the estimate. Formats where an extra full read would be
    expensive return ``None`` and the caller shows indeterminate progress.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    total: int | None = None
    try:
        if suffix == ".txt":
            total = 0
            in_block = False
            block_chars = 0
            with open(source, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    raw = line.rstrip("\r\n")
                    if raw.strip():
                        if not in_block:
                            total += 1
                            in_block = True
                        block_chars += len(raw)
                        if block_chars > 50_000:
                            in_block = False
                            block_chars = 0
                    elif in_block:
                        in_block = False
                        block_chars = 0
        elif suffix == ".jsonl":
            total = 0
            with open(source, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.strip():
                        total += 1
        elif suffix == ".csv":
            with open(source, "r", encoding="utf-8", errors="replace", newline="") as handle:
                total = max(0, sum(1 for line in handle if line.strip()) - 1)
        elif suffix == ".parquet":
            import pyarrow.parquet as pq

            total = int(pq.ParquetFile(str(source)).metadata.num_rows)
    except Exception:
        total = None
    if total is not None and max_docs is not None:
        total = min(total, max_docs)
    return total

def iter_training_documents(path: Path | str, max_docs: int | None = None) -> Iterator[TrainingDocument]:
    """Yield logical documents for every supported dataset format."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".txt":
        yield from _iter_txt(source, max_docs)
    elif suffix == ".jsonl":
        yield from _iter_jsonl(source, max_docs)
    elif suffix == ".json":
        yield from _iter_json(source, max_docs)
    elif suffix == ".csv":
        yield from _iter_csv(source, max_docs)
    elif suffix == ".parquet":
        yield from _iter_parquet(source, max_docs)
    elif suffix == ".arrow":
        yield from _iter_arrow(source, max_docs)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix or '<none>'}")
"""Read-only dataset quality audit."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


ProgressCallback = Callable[[str, int, int | None, str], None]

from .curation import (
    API_KEY_PATTERNS,
    CNP_REGEX,
    EMAIL_REGEX,
    PHONE_REGEX,
    MinHashLSHDeduplicator,
    TextCurator,
    check_text_quality,
    classify_language_statistical,
    extract_13grams,
    is_valid_cnp,
    mask_pii_in_text,
    strip_html_and_boilerplate,
)
from .document_io import iter_training_documents

MAX_AUDIT_DOCS = 20000
MAX_PREVIEW_DOCS = 40


def _extract_text(raw: str, is_jsonl: bool) -> str:
    if not is_jsonl:
        return raw
    try:
        obj = json.loads(raw)
    except Exception:
        return raw
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return raw

    messages = obj.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            if str(content).strip():
                parts.append(str(content))
        if parts:
            return "\n".join(parts)

    # SFT/alpace-style records carry the prompt and the answer in separate
    # fields. Auditing only one field under-counts characters and tokens.
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
    return raw


def _iter_arrow_rows(path: Path, max_docs: int, suffix: str) -> Iterable[Tuple[int, str]]:
    import pyarrow as pa

    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(str(path))
        batches = parquet_file.iter_batches(batch_size=256)
    else:
        try:
            table = pa.ipc.open_file(str(path)).read_all()
        except Exception:
            table = pa.ipc.open_stream(str(path)).read_all()
        batches = table.to_batches(max_chunksize=256)

    delivered = 0
    for batch in batches:
        for row in batch.to_pylist():
            if delivered >= max_docs:
                return
            delivered += 1
            yield delivered - 1, _extract_text(json.dumps(row, ensure_ascii=False, default=str), True)


def _iter_documents(path: Path, max_docs: int) -> Iterable[Tuple[int, str]]:
    """Yield logical documents using the same segmentation as cleaning."""
    for document in iter_training_documents(path, max_docs):
        yield document.index, document.text

def _pii_counts(text: str) -> Dict[str, int]:
    counts = {
        "email": len(EMAIL_REGEX.findall(text)),
        "phone": len(PHONE_REGEX.findall(text)),
        "cnp": len([m.group(0) for m in CNP_REGEX.finditer(text) if is_valid_cnp(m.group(0))]),
        "api_key": sum(len(pattern.findall(text)) for pattern in API_KEY_PATTERNS),
    }
    return {key: value for key, value in counts.items() if value}


def audit_documents(
    documents: Iterable[Tuple[int, str]],
    *,
    eval_documents: Sequence[str] | None = None,
    tokenizer: Any = None,
    duplicate_threshold: float = 0.80,
    max_docs: int = MAX_AUDIT_DOCS,
    progress: ProgressCallback | None = None,
) -> Dict[str, Any]:
    """Audits documents without modifying the source file."""
    max_docs = max(1, min(int(max_docs), MAX_AUDIT_DOCS))
    curator = TextCurator(jaccard_threshold=duplicate_threshold)
    if eval_documents:
        curator.add_benchmark_documents(list(eval_documents)[:5000])
    deduper = MinHashLSHDeduplicator(
        num_perm=64,
        k_shingle=5,
        jaccard_threshold=duplicate_threshold,
        b_bands=16,
    )

    total = 0
    hard_rejected_docs = 0
    quality_rejected = 0
    language_rejected = 0
    language_uncertain = 0
    duplicates = 0
    contaminated = 0
    empty_documents = 0
    pii_docs = 0
    pii_totals: Dict[str, int] = {}
    token_total = 0
    word_total = 0
    char_total = 0
    unknown_tokens = 0
    flagged: List[Dict[str, Any]] = []

    for index, raw in documents:
        if total >= max_docs:
            break
        total += 1
        if progress is not None and (
            total == 1 or total % 25 == 0 or total >= max_docs
        ):
            try:
                progress(
                    "auditing",
                    total,
                    max_docs,
                    f"Audited {total:,} of up to {max_docs:,} documents",
                )
            except Exception:
                pass
        text = strip_html_and_boilerplate(raw or "")
        words = re.findall(r"\S+", text)
        char_total += len(text)
        word_total += len(words)
        reasons: List[str] = []
        hard_rejected = False

        if not text.strip():
            reasons.append("empty")
            empty_documents += 1
            hard_rejected = True
        if text and not check_text_quality(text):
            reasons.append("quality")
            quality_rejected += 1
            hard_rejected = True
        if text:
            language, confidence = classify_language_statistical(
                text, ("ron", "eng", "ro", "en")
            )
            if language not in {"ron", "eng", "ro", "en"}:
                if confidence >= 0.80:
                    reasons.append("language")
                    language_rejected += 1
                    hard_rejected = True
                else:
                    language_uncertain += 1
        if text and not curator.decontaminate(text):
            reasons.append("contamination")
            contaminated += 1
            hard_rejected = True

        pii = _pii_counts(text) if text else {}
        if pii:
            pii_docs += 1
            reasons.append("pii")
            for key, count in pii.items():
                pii_totals[key] = pii_totals.get(key, 0) + count

        if text and deduper.is_duplicate(text):
            reasons.append("duplicate")
            duplicates += 1
            hard_rejected = True

        if hard_rejected:
            hard_rejected_docs += 1

        doc_tokens = 0
        if tokenizer is not None and text:
            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
                doc_tokens = len(ids)
                token_total += doc_tokens
                unk_id = getattr(tokenizer, "unk_token_id", None)
                if unk_id is not None:
                    unknown_tokens += sum(1 for token_id in ids if int(token_id) == int(unk_id))
            except Exception:
                doc_tokens = 0

        if reasons and len(flagged) < MAX_PREVIEW_DOCS:
            flagged.append({
                "index": index,
                "reasons": reasons,
                "characters": len(text),
                "words": len(words),
                "tokens": doc_tokens,
                "snippet": mask_pii_in_text(text[:240]),
            })

    hard_rejected = hard_rejected_docs
    accepted = max(0, total - hard_rejected)
    fertility = (token_total / word_total) if word_total and token_total else None
    unk_rate = (unknown_tokens / token_total * 100.0) if token_total else None
    return {
        "documents_analyzed": total,
        "accepted_for_training": accepted,
        "hard_rejected": hard_rejected,
        "quality_rejected": quality_rejected,
        "language_rejected": language_rejected,
        "language_uncertain": language_uncertain,
        "empty_documents": empty_documents,
        "duplicates": duplicates,
        "contaminated": contaminated,
        "pii_documents": pii_docs,
        "pii_occurrences": pii_totals,
        "characters": char_total,
        "words": word_total,
        "tokens": token_total,
        "tokens_per_word": round(fertility, 3) if fertility is not None else None,
        "unknown_token_rate_pct": round(unk_rate, 4) if unk_rate is not None else None,
        "duplicate_threshold": duplicate_threshold,
        "eval_documents": len(eval_documents or []),
        "decontamination_enabled": bool(curator.decontamination_13grams),
        "flagged_preview": flagged,
        "flagged_preview_truncated": len(flagged) >= MAX_PREVIEW_DOCS,
    }


def audit_dataset_file(
    path: str | Path,
    *,
    eval_paths: Sequence[str | Path] | None = None,
    tokenizer: Any = None,
    max_docs: int = MAX_AUDIT_DOCS,
    duplicate_threshold: float = 0.80,
    progress: ProgressCallback | None = None,
) -> Dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source}")
    eval_texts: List[str] = []
    for eval_path in eval_paths or []:
        candidate = Path(eval_path)
        if candidate.is_file():
            eval_texts.extend(text for _idx, text in _iter_documents(candidate, 5000))
    return audit_documents(
        _iter_documents(source, max_docs),
        eval_documents=eval_texts,
        tokenizer=tokenizer,
        duplicate_threshold=duplicate_threshold,
        max_docs=max_docs,
        progress=progress,
    )

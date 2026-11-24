"""Normalize Alpaca, ShareGPT, and ChatML records into masked-SFT JSONL.

The trainer consumes one JSON object per line with a canonical ``messages`` list.
Formatting is deliberately conservative: malformed records are reported with their
line number instead of being silently dropped, and empty conversations never reach
the tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..tokenizer.chat_template import ChatTemplateRenderer

ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "prompter": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
}


class SFTFormatError(ValueError):
    """Raised when no valid record can be produced."""


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_messages(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        raise SFTFormatError("messages must be a list")
    messages: List[Dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise SFTFormatError("each message must be an object")
        role = _text(entry.get("role") or entry.get("from")).strip().lower()
        if role not in ROLE_ALIASES:
            raise SFTFormatError(f"unsupported role: {role or '<empty>'}")
        content = _text(entry.get("content") if "content" in entry else entry.get("value"))
        if not content.strip():
            raise SFTFormatError("message content is empty")
        message = {"role": ROLE_ALIASES[role], "content": content}
        thought = entry.get("thought") or entry.get("reasoning")
        if thought is not None and _text(thought).strip():
            message["thought"] = _text(thought)
        messages.append(message)
    return messages


def normalize_record(raw: Any) -> List[Dict[str, str]]:
    """Converts one supported record shape into canonical chat messages."""
    if not isinstance(raw, dict):
        raise SFTFormatError("record must be a JSON object")
    if "messages" in raw:
        messages = _normalize_messages(raw["messages"])
    elif "conversations" in raw:
        messages = _normalize_messages(raw["conversations"])
    elif "instruction" in raw or "output" in raw:
        instruction = _text(raw.get("instruction")).strip()
        extra = _text(raw.get("input")).strip()
        output = _text(raw.get("output"))
        user = instruction + (f"\n\n{extra}" if extra else "")
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": output}]
    elif "prompt" in raw and "completion" in raw:
        messages = [
            {"role": "user", "content": _text(raw.get("prompt"))},
            {"role": "assistant", "content": _text(raw.get("completion"))},
        ]
    else:
        raise SFTFormatError("unrecognized record; expected messages, conversations, instruction/output, or prompt/completion")

    if not messages:
        raise SFTFormatError("record has no messages")
    if not any(m["role"] == "assistant" for m in messages):
        raise SFTFormatError("record has no assistant message")
    if not any(m["role"] == "user" for m in messages):
        raise SFTFormatError("record has no user message")
    return messages


def load_records(path: Path) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Reads JSON array, JSONL, or a JSON object containing an array field."""
    errors: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SFTFormatError(f"invalid JSON array: {exc}") from exc
        if not isinstance(payload, list):
            raise SFTFormatError("top-level JSON must be an array or object")
        return payload, errors
    if stripped.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("data", "records", "examples", "rows"):
                if isinstance(payload.get(key), list):
                    return payload[key], errors
            if any(k in payload for k in ("messages", "conversations", "instruction", "prompt")):
                return [payload], errors

    records: List[Any] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append({"line": number, "error": f"invalid JSON: {exc.msg}"})
    return records, errors


def format_dataset(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    template: str = "chatml",
    max_records: Optional[int] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Writes canonical JSONL and returns counts, errors, and a SHA-256 digest."""
    source = Path(input_path)
    if not source.is_file():
        raise SFTFormatError(f"input file not found: {source}")
    raw_records, errors = load_records(source)
    renderer = ChatTemplateRenderer(template)
    limit = max_records if max_records and max_records > 0 else None
    formatted: List[Dict[str, Any]] = []
    token_total = 0
    for index, raw in enumerate(raw_records):
        if limit is not None and len(formatted) >= limit:
            break
        try:
            messages = normalize_record(raw)
            text = renderer.render(messages, add_generation_prompt=False)
            if not text.strip():
                raise SFTFormatError("rendered text is empty")
            item: Dict[str, Any] = {"messages": messages, "text": text}
            for key in ("id", "source", "language", "license"):
                if isinstance(raw, dict) and key in raw:
                    item[key] = raw[key]
            if tokenizer is not None:
                count = len(tokenizer.encode(text))
                item["token_count"] = count
                token_total += count
            formatted.append(item)
        except SFTFormatError as exc:
            errors.append({"line": index + 1, "error": str(exc)})

    if not formatted:
        detail = errors[0]["error"] if errors else "no records found"
        raise SFTFormatError(f"no valid SFT records: {detail}")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".sft_", suffix=".jsonl", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for item in formatted:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "success": True,
        "input_path": str(source),
        "output_path": str(target.resolve()),
        "format": template,
        "valid_records": len(formatted),
        "invalid_records": len(errors),
        "total_records": len(raw_records),
        "token_count": token_total if tokenizer is not None else None,
        "sha256": digest,
        "errors": errors[:50],
    }

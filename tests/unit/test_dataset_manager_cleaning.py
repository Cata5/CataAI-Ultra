"""Tests for cleaning during import through DatasetManager."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.datasets.manager import DatasetManager


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_import_cleans_and_keeps_raw_backup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manager = DatasetManager(upload_dir=root / "uploads")
        source = root / "sample.jsonl"
        good = {
            "messages": [
                {"role": "user", "content": "Explică pe scurt arhitectura unui procesor modern."},
                {"role": "assistant", "content": "Un procesor modern are nuclee, cache și un controler de memorie."},
            ]
        }
        _write_jsonl(source, [good, good, {"messages": [{"role": "user", "content": "x"}]}])

        events: list[tuple[str, int, int | None, str]] = []
        result = manager.import_cleaned_file(
            source,
            name="sample.jsonl",
            clean=True,
            progress=lambda phase, current, total, message: events.append(
                (phase, current, total, message)
            ),
        )

        assert any(phase == "filtering" for phase, _, _, _ in events)
        assert result.get("success") is not False, result
        assert (root / "uploads" / "raw" / "sample.jsonl").is_file()
        assert result["cleaning"]["success"] is True
        assert result["cleaning"]["records_out"] == 1
        cleaned = Path(result["file_path"]).read_text(encoding="utf-8")
        assert "procesor modern" in cleaned


def test_import_falls_back_to_raw_when_all_records_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manager = DatasetManager(upload_dir=root / "uploads")
        source = root / "short.jsonl"
        _write_jsonl(source, [{"messages": [{"role": "user", "content": "x"}]}])

        result = manager.import_cleaned_file(source, name="short.jsonl", clean=True)

        assert result.get("success") is not False, result
        assert result["cleaning"]["success"] is False
        assert Path(result["file_path"]).read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_clean_existing_dataset_leaves_source_unchanged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manager = DatasetManager(upload_dir=root / "uploads")
        source = root / "raw.jsonl"
        _write_jsonl(source, [
            {"text": "Aceasta este o propoziție suficient de lungă despre rețele neuronale."},
            {"text": "Aceasta este o propoziție suficient de lungă despre rețele neuronale."},
        ])
        meta = manager.register_uploaded_file(source, name="raw.jsonl")
        original = Path(meta["file_path"]).read_text(encoding="utf-8")

        result = manager.clean_dataset(meta["dataset_id"])

        assert result["records_in"] == 2
        assert result["records_out"] == 1
        assert Path(meta["file_path"]).read_text(encoding="utf-8") == original
        assert Path(result["cleaned_file_path"]).name.endswith("_clean.jsonl")
        assert result["cleaned_dataset_id"]


def test_import_txt_preserves_chatml_markers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manager = DatasetManager(upload_dir=root / "uploads")
        source = root / "chat.txt"
        source.write_text(
            "<|im_start|>user\nExplică rețelele neuronale în limba română.\n<|im_end|>\n"
            "<|im_start|>assistant\nO rețea neuronală învață ponderi din datele de antrenare.\n<|im_end|>\n\n"
            "<|im_start|>user\nCe face un GPU?\n<|im_end|>\n"
            "<|im_start|>assistant\nAccelerează calculele paralele folosite de rețele.\n<|im_end|>\n",
            encoding="utf-8",
        )

        result = manager.import_cleaned_file(source, name="chat.txt", clean=True)

        assert result.get("success") is True, result
        assert result["cleaning"]["success"] is True
        cleaned = Path(result["file_path"]).read_text(encoding="utf-8")
        assert cleaned.count("<|im_start|>") == 4
        assert "<|im_start|>assistant" in cleaned
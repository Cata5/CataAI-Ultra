"""Tests for format-aware import cleaning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.data.dataset_cleaning import CleanOptions, clean_dataset_file


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_txt_blocks_preserve_chatml(tmp_path: Path) -> None:
    source = tmp_path / "chat.txt"
    source.write_text(
        "<|im_start|>user\nSalut, explică rețelele neuronale.\n<|im_end|>\n"
        "<|im_start|>assistant\nO rețea neuronală învață ponderi din date.\n<|im_end|>\n\n"
        "<|im_start|>user\nCe este un GPU?\n<|im_end|>\n"
        "<|im_start|>assistant\nUn procesor optimizat pentru calcule paralele.\n<|im_end|>\n",
        encoding="utf-8",
    )
    output = tmp_path / "chat_clean.txt"
    report = clean_dataset_file(
        source,
        output,
        CleanOptions(quality=False, language=False, deduplicate=False, mask_pii=False),
    )
    cleaned = output.read_text(encoding="utf-8")
    assert report["records_in"] == 2
    assert report["records_out"] == 2
    assert cleaned.count("<|im_start|>") == 4
    assert "<|im_start|>assistant" in cleaned


def test_txt_rejects_short_and_duplicate_blocks(tmp_path: Path) -> None:
    source = tmp_path / "blocks.txt"
    valid = (
        "Intel și AMD au schimbat arhitectura procesoarelor de-a lungul istoriei. "
        "Această secțiune explică instrucțiunile CISC, RISC și tranziția către x86-64."
    )
    source.write_text(f"{valid}\n\n{valid}\n\nScurt\n", encoding="utf-8")
    output = tmp_path / "blocks_clean.txt"
    report = clean_dataset_file(
        source,
        output,
        CleanOptions(deduplicate=True, quality=True, language=True),
    )
    assert report["records_in"] == 3
    assert report["records_out"] == 1
    assert report["exact_duplicates"] == 1
    assert report["quality_rejected"] == 1


def test_jsonl_preserves_messages_and_masks_pii(tmp_path: Path) -> None:
    source = tmp_path / "sft.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "Scrie-mi la adresa user@example.com."},
            {"role": "assistant", "content": "Am notat adresa de email."},
        ],
        "id": "keep-this-id",
    }
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "sft_clean.jsonl"
    report = clean_dataset_file(
        source,
        output,
        CleanOptions(quality=False, language=False, deduplicate=False, mask_pii=True),
    )
    cleaned = json.loads(output.read_text(encoding="utf-8").strip())
    assert report["pii_documents"] == 1
    assert cleaned["id"] == "keep-this-id"
    assert cleaned["messages"][0]["role"] == "user"
    assert "[EMAIL]" in cleaned["messages"][0]["content"]
    assert "user@example.com" not in cleaned["messages"][0]["content"]


def test_low_confidence_short_line_is_kept(tmp_path: Path) -> None:
    source = tmp_path / "short.txt"
    source.write_text("qwerty zxcvbn asdfgh hjkl poiu\n", encoding="utf-8")
    output = tmp_path / "short_clean.txt"
    report = clean_dataset_file(
        source,
        output,
        CleanOptions(quality=False, language=True, deduplicate=False, mask_pii=False),
    )
    assert report["records_out"] == 1
    assert report["language_uncertain"] >= 1


def test_clean_rejects_identical_output_path(tmp_path: Path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("Conținut suficient de lung pentru a fi păstrat în setul de date.", encoding="utf-8")
    before = _sha(source)
    with pytest.raises(ValueError, match="must differ"):
        clean_dataset_file(source, source)
    assert _sha(source) == before


def test_jsonl_short_record_uses_relaxed_threshold(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Salut, ce faci astăzi?"}, {"role": "assistant", "content": "Bine, mulțumesc!"}]})
        + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]})
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "records_clean.jsonl"
    report = clean_dataset_file(
        source,
        output,
        CleanOptions(quality=True, language=False, deduplicate=False, mask_pii=False),
    )
    assert report["records_out"] == 1


def test_jsonl_nan_fields_become_empty_not_literal_nan(tmp_path: Path) -> None:
    source = tmp_path / "nan_fields.jsonl"
    rows = [
        {"instruction": "Ce este un GPU?", "input": float("nan"), "output": "Un procesor grafic."},
        {
            "messages": [
                {"role": "user", "content": float("nan")},
                {"role": "assistant", "content": "Un răspuns valid."},
            ]
        },
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "nan_fields_clean.jsonl"
    clean_dataset_file(
        source,
        output,
        CleanOptions(quality=False, language=False, deduplicate=False, mask_pii=False),
    )
    text = output.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "nan" not in text.lower()
    cleaned = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert cleaned[0]["input"] == ""
    assert cleaned[1]["messages"][0]["content"] == ""


def test_csv_preserves_columns(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_text(
        "instruction,output\n"
        + '"Explică rețelele neuronale suficient de detaliat.","O rețea neuronală învață din date."\n',
        encoding="utf-8",
    )
    output = tmp_path / "data_clean.csv"
    clean_dataset_file(
        source,
        output,
        CleanOptions(quality=True, language=False, deduplicate=False, mask_pii=False),
    )
    text = output.read_text(encoding="utf-8")
    assert text.startswith("instruction,output")
    assert "rețelele neuronale" in text


def test_zero_record_clean_does_not_clobber_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "short.txt"
    source.write_text("Scurt\n", encoding="utf-8")
    output = tmp_path / "existing_clean.txt"
    output.write_text("KEEP THIS CLEANED FILE\n", encoding="utf-8")

    report = clean_dataset_file(
        source,
        output,
        CleanOptions(quality=True, language=False, deduplicate=False, mask_pii=False),
    )

    assert report["records_out"] == 0
    assert output.read_text(encoding="utf-8") == "KEEP THIS CLEANED FILE\n"
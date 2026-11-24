"""Tests for SFT record normalization and checksum sidecars."""

import json
from pathlib import Path

import pytest

from src.checkpoints.manager import checksum_path, file_sha256, verify_checksum, write_checksum
from src.data.sft_formatter import SFTFormatError, format_dataset


def test_formats_sharegpt_and_instruction_records(tmp_path: Path) -> None:
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        "\n".join([
            json.dumps({"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}),
            json.dumps({"instruction": "Add", "input": "2+2", "output": "4"}),
        ]),
        encoding="utf-8",
    )
    result = format_dataset(source, tmp_path / "out.jsonl", template="chatml")
    assert result["valid_records"] == 2
    assert result["invalid_records"] == 0
    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["messages"][0] == {"role": "user", "content": "Hi"}
    assert rows[1]["messages"][0]["content"] == "Add\n\n2+2"
    assert "<|im_start|>assistant" in rows[0]["text"]


def test_reports_invalid_records_and_rejects_empty_output(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(
        "\n".join([
            json.dumps({"messages": [{"role": "user", "content": "only user"}]}),
            "not json",
        ]),
        encoding="utf-8",
    )
    with pytest.raises(SFTFormatError, match="no valid SFT records"):
        format_dataset(source, tmp_path / "out.jsonl")


def test_checksum_detects_corruption(tmp_path: Path) -> None:
    artefact = tmp_path / "best_model.pt"
    artefact.write_bytes(b"checkpoint-bytes")
    digest = write_checksum(artefact)
    assert checksum_path(artefact).read_text(encoding="ascii").strip() == digest
    assert file_sha256(artefact) == digest
    assert verify_checksum(artefact) == digest
    artefact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_checksum(artefact)

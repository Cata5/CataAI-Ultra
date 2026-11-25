import json

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.quality_audit import audit_dataset_file, audit_documents


def test_parquet_is_read_as_rows_not_binary(tmp_path) -> None:
    path = tmp_path / "data.parquet"
    rows = [
        "This document explains how distributed training works across multiple graphics processing units. It covers gradient accumulation, mixed precision, checkpoint sharding, deterministic data loading, validation splits, learning rate schedules and reproducible evaluation.",
        "This guide describes tokenizer construction, vocabulary coverage, dataset deduplication, language identification and benchmark decontamination. It explains how to measure tokens per word and unknown token rates before training.",
        "The audit reads tabular corpora without rewriting them. Parquet columns are decoded into text before quality, language, contamination and duplicate checks run.",
    ]
    pq.write_table(pa.table({"text": rows}), path)

    result = audit_dataset_file(path, max_docs=10)

    assert result["documents_analyzed"] == 3
    assert result["accepted_for_training"] >= 2
    assert result["characters"] >= 600


def test_sft_audit_includes_instruction_and_input(tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    instruction = "Explain how a compiler lowers a loop into vector operations."
    input_text = "Target hardware: AVX2, four lanes, no aliasing."
    output = "The compiler first proves the loop is vectorizable, then widens the body."
    path.write_text(json.dumps({"instruction": instruction, "input": input_text, "output": output}), encoding="utf-8")

    result = audit_dataset_file(path, max_docs=10)

    expected_words = len(instruction.split()) + len(input_text.split()) + len(output.split())
    assert result["words"] == expected_words
    assert result["characters"] > len(output)


def test_overlapping_reasons_are_counted_once() -> None:
    sentence = "The quick brown fox jumps over the lazy dog while the compiler optimizes the kernel carefully today."
    docs = [
        (0, sentence),
        (1, sentence),
        (2, "Contact user@example.com or +40 712 345 678 for the private dataset export details."),
    ]
    result = audit_documents(docs)  # no eval docs, so contamination is not the overlap here

    # Duplicate + quality/language overlap still counts the document once.
    assert result["accepted_for_training"] + result["hard_rejected"] == 3
    assert result["hard_rejected"] >= 1
    assert any("pii" in row["reasons"] for row in result["flagged_preview"])

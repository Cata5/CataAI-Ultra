"""Dataset quality audit reports before it deletes."""

import json

from src.data.quality_audit import audit_dataset_file, audit_documents


class FakeTokenizer:
    unk_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 if len(word) < 3 else 0 for word in text.split()]


def test_audit_flags_duplicates_pii_and_contamination() -> None:
    sentence = "The quick brown fox jumps over the lazy dog while the compiler optimizes the kernel carefully today."
    docs = [
        (0, sentence),
        (1, sentence),
        (2, "Contact user@example.com or +40 712 345 678 for the private dataset export details."),
    ]
    result = audit_documents(docs, eval_documents=[sentence], tokenizer=FakeTokenizer(), duplicate_threshold=0.8)
    assert result["documents_analyzed"] == 3
    assert result["duplicates"] >= 1
    assert result["contaminated"] >= 1
    assert result["pii_documents"] == 1
    assert result["pii_occurrences"]["email"] == 1
    assert result["tokens"] > 0
    assert result["unknown_token_rate_pct"] is not None
    snippets = " ".join(row["snippet"] for row in result["flagged_preview"])
    assert "user@example.com" not in snippets
    assert "[EMAIL]" in snippets


def test_audit_dataset_file_reads_jsonl_and_masks_preview(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    rows = [
        {"text": "This document explains how distributed training works across multiple graphics processing units. It covers gradient accumulation, mixed precision, checkpoint sharding, deterministic data loading, validation splits, learning rate schedules and reproducible evaluation. Engineers can use these notes to configure a local cluster and compare throughput."},
        {"text": "This guide describes tokenizer construction, vocabulary coverage, dataset deduplication, language identification and benchmark decontamination. It explains how to measure tokens per word and unknown token rates before training. The goal is a reproducible audit that never rewrites the source corpus."},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = audit_dataset_file(path, max_docs=10)
    assert result["documents_analyzed"] == 2
    assert result["accepted_for_training"] >= 1
    assert result["characters"] > 0
    assert result["tokens_per_word"] is None

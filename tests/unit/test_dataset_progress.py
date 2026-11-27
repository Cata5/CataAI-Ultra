"""Tests for persisted dataset operation progress."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.dataset_cleaning import CleanOptions, clean_dataset_file
from src.data.dataset_progress import DatasetProgress, normalize_job_id, read_job_status


def test_progress_status_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMMAKER_DATA_DIR", str(tmp_path))
    progress = DatasetProgress("job-demo", "clean", "dataset-1", "Demo")
    progress.update("filtering", 5, 10, "Processed records")
    status = read_job_status("job-demo")
    assert status is not None
    assert status["status"] == "running"
    assert status["percent"] == 50.0
    assert status["current"] == 5
    assert status["total"] == 10
    assert status["dataset_name"] == "Demo"

    progress.complete({"records_out": 5})
    completed = read_job_status("job-demo")
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["percent"] == 100.0
    assert completed["result"]["records_out"] == 5


def test_progress_sanitizes_unsafe_job_ids(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMMAKER_DATA_DIR", str(tmp_path))
    safe = normalize_job_id("../../escape")
    assert "/" not in safe and "\\" not in safe
    DatasetProgress("../../escape", "quality", "dataset-1").fail("boom")
    status = read_job_status(safe)
    assert status is not None
    assert status["status"] == "failed"


def test_cleaning_emits_progress_with_counts(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    records = [
        {
            "text": (
                "A long and useful training sentence about local neural networks, "
                "tokenizers, and efficient data pipelines."
            )
        }
        for _ in range(30)
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    events: list[tuple[str, int, int | None, str]] = []
    report = clean_dataset_file(
        source,
        tmp_path / "records_clean.jsonl",
        CleanOptions(quality=False, language=False, deduplicate=False, mask_pii=False),
        progress=lambda phase, current, total, message: events.append(
            (phase, current, total, message)
        ),
    )
    assert report["records_out"] == 30
    assert events[0][0] == "counting"
    assert any(phase == "filtering" and total == 30 for phase, _, total, _ in events)
    assert events[-1][0] == "finalizing"
    assert max(current for _, current, _, _ in events) == 30
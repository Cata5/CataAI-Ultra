"""Dataset worker path resolution and token cache integrity."""

import json
from pathlib import Path

from src.worker import dataset_ops


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dataset_ops, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(dataset_ops, "uploads_dir", lambda: tmp_path / "data" / "uploads")
    monkeypatch.setattr(dataset_ops, "workspaces_dir", lambda: tmp_path / "workspaces")
    monkeypatch.setattr(dataset_ops, "app_cache_dir", lambda: tmp_path / "data" / "cache")
    monkeypatch.chdir(tmp_path)


def test_find_dataset_file_searches_active_workspace(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    target = tmp_path / "workspaces" / "cataai" / "datasets" / "alpaca_ro.txt"
    target.parent.mkdir(parents=True)
    target.write_text("alpha beta gamma delta", encoding="utf-8")

    assert dataset_ops.find_dataset_file("alpaca_ro", "cataai") == target


def test_find_dataset_file_searches_every_workspace(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    target = tmp_path / "workspaces" / "cataai" / "datasets" / "alpaca_ro.txt"
    target.parent.mkdir(parents=True)
    target.write_text("alpha beta gamma delta", encoding="utf-8")

    # The desktop audit defaults to the default workspace, while the dataset
    # list may show a file from another workspace.
    assert dataset_ops.find_dataset_file("alpaca_ro") == target
    assert dataset_ops.find_dataset_file("alpaca_ro", "default") == target


def test_find_dataset_file_accepts_exact_filename_and_rejects_missing(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    target = (tmp_path / "data" / "uploads" / "corpus.jsonl")
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")

    assert dataset_ops.find_dataset_file("corpus.jsonl") == target
    assert dataset_ops.find_dataset_file("does-not-exist") is None


def test_tokenize_rebuilds_cache_with_stale_fallback_marker(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    target = tmp_path / "workspaces" / "cataai" / "datasets" / "corpus.txt"
    target.parent.mkdir(parents=True)
    target.write_text("Aceasta este o linie de test.\n" * 200, encoding="utf-8")

    first = dataset_ops.run_tokenize("corpus", "cataai")
    assert first["success"] is True
    assert first["cached"] is False
    assert first["token_count"] > 0

    marker = tmp_path / "data" / "cache" / "corpus.tokenized"
    marker.write_text(
        json.dumps({"is_tokenized": True, "token_count": 1_144_246}),
        encoding="utf-8",
    )

    second = dataset_ops.run_tokenize("corpus", "cataai")
    assert second["success"] is True
    assert second["cached"] is False
    assert second["token_count"] == first["token_count"]
    assert json.loads(marker.read_text(encoding="utf-8"))["token_count"] == first["token_count"]
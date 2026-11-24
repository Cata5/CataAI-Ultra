"""Data-safety invariants: encryption at rest, no silent replacement, integrity."""

import shutil
import sqlite3
from pathlib import Path

import pytest

from src.db.database import DatabaseIntegrityError, StudioDatabase
from src.security.secrets import REDACTED, SecretStore, SecretsKeyError


def _db(tmp_path: Path) -> StudioDatabase:
    return StudioDatabase(tmp_path / "studio.db")


def test_secrets_are_encrypted_at_rest_and_redacted_on_read(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.save_settings({"discord_webhook_url": "https://discord.com/api/webhooks/123/abc"})

    raw = sqlite3.connect(str(tmp_path / "studio.db")).execute(
        "SELECT value_json FROM settings WHERE key = ?", ("discord_webhook_url",)
    ).fetchone()[0]
    assert raw.startswith(chr(34) + "enc:v1:")
    assert "discord.com" not in raw

    assert db.get_settings()["discord_webhook_url"].endswith("/abc")
    public = db.get_settings(redact=True)
    assert public["discord_webhook_url"] == REDACTED
    assert public["discord_webhook_url_configured"] is True
    assert "discord.com" not in str(public)


def test_redacted_sentinel_keeps_existing_secret_and_empty_clears(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.save_settings({"telegram_bot_token": "123:secret"})
    db.save_settings({"telegram_bot_token": REDACTED, "temp_threshold_c": 77})
    assert db.get_settings()["telegram_bot_token"] == "123:secret"
    db.save_settings({"telegram_bot_token": ""})
    assert db.get_settings()["telegram_bot_token"] == ""
    assert db.get_settings(redact=True)["telegram_bot_token_configured"] is False


def test_missing_key_is_not_silently_regenerated(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.save_settings({"discord_webhook_url": "https://discord.com/api/webhooks/keep"})
    key_path = tmp_path / ".secrets.key"
    key_path.unlink()

    with pytest.raises(SecretsKeyError, match="Restore the matching"):
        StudioDatabase(tmp_path / "studio.db")
    assert not key_path.exists()


def test_wrong_key_does_not_return_empty_secret(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.save_settings({"telegram_bot_token": "123:secret"})
    other = SecretStore(tmp_path / "other.key")
    assert other.key_path.exists()
    shutil.copyfile(other.key_path, tmp_path / ".secrets.key")

    reopened = StudioDatabase(tmp_path / "studio.db")
    with pytest.raises(SecretsKeyError, match="authentication"):
        reopened.get_settings()


def test_legacy_plaintext_secret_is_upgraded_on_open(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with sqlite3.connect(str(tmp_path / "studio.db")) as conn:
        conn.execute(
            "INSERT INTO settings (key, value_json, updated_at) VALUES (?, ?, 0)",
            ("discord_webhook_url", '"https://discord.com/api/webhooks/legacy"'),
        )
    reopened = StudioDatabase(tmp_path / "studio.db")
    raw = sqlite3.connect(str(tmp_path / "studio.db")).execute(
        "SELECT value_json FROM settings WHERE key = ?", ("discord_webhook_url",)
    ).fetchone()[0]
    assert raw.startswith(chr(34) + "enc:v1:")
    assert reopened.get_settings()["discord_webhook_url"].endswith("/legacy")


def test_duplicate_run_id_is_rejected_without_losing_history(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.create_run("run_1", "original")
    db.log_metric("run_1", {"step": 1, "train_loss": 1.5})
    with pytest.raises(ValueError, match="already exists"):
        db.create_run("run_1", "replacement")
    assert db.get_run("run_1")["name"] == "original"
    assert len(db.get_metrics("run_1")) == 1


def test_duplicate_workspace_id_is_rejected(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.create_workspace("ws", "first")
    with pytest.raises(ValueError, match="already exists"):
        db.create_workspace("ws", "second")
    assert db.get_workspace("ws")["name"] == "first"


def test_delete_workspace_removes_children_first(tmp_path: Path) -> None:
    db = _db(tmp_path)
    db.create_workspace("ws", "temporary")
    db.create_run("run_ws", "run", workspace_id="ws")
    db.log_metric("run_ws", {"step": 1, "train_loss": 1.0})
    db.log_checkpoint("chk_ws", "run_ws", 1, 10, 1.0, None, "x.pt", 5, workspace_id="ws")
    assert db.delete_workspace("ws") is True
    assert db.get_workspace("ws") is None
    with sqlite3.connect(str(tmp_path / "studio.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0


def test_corrupt_database_is_refused_before_migration(tmp_path: Path) -> None:
    path = tmp_path / "studio.db"
    path.write_bytes(b"this is not a sqlite database" * 100)
    with pytest.raises(DatabaseIntegrityError):
        StudioDatabase(path)


def test_key_file_is_created(tmp_path: Path) -> None:
    store = SecretStore(tmp_path / ".secrets.key")
    assert store.key_path.exists()
    assert store.key_path.stat().st_size > 0
def test_workspace_archive_roundtrip_and_checksum(tmp_path: Path, monkeypatch) -> None:
    from src.workspace.archive import WorkspaceArchive, WorkspaceArchiveError

    db = _db(tmp_path)
    checkpoint_file = tmp_path / "model.pt"
    checkpoint_file.write_bytes(b"weights" * 100)
    dataset_file = tmp_path / "data.jsonl"
    dataset_file.write_text('{"text": "hello"}\n', encoding="utf-8")

    db.create_workspace("source", "Source")
    db.create_run("run_src", "run", workspace_id="source")
    db.log_metric("run_src", {"step": 1, "train_loss": 2.0})
    db.log_checkpoint("chk_src", "run_src", 1, 10, 2.0, None, str(checkpoint_file), 700, workspace_id="source")
    db.save_dataset(
        dataset_id="ds_src", name="toy", file_path=str(dataset_file), file_size_bytes=dataset_file.stat().st_size,
        format_type="jsonl", revision="1", workspace_id="source",
    )

    archive = WorkspaceArchive(db, base_dir=tmp_path / "workspaces")
    path, name = archive.export("source")
    assert name.endswith(".zip") and path.exists()

    result = archive.import_archive(path)
    assert result["workspace"]["id"].startswith("source")
    assert result["runs_imported"] == 1
    imported = result["workspace"]["id"]
    checkpoints = db.list_checkpoints(workspace_id=imported)
    datasets = db.list_datasets(workspace_id=imported)
    assert len(checkpoints) == 1 and len(datasets) == 1
    assert Path(checkpoints[0]["file_path"]).read_bytes() == checkpoint_file.read_bytes()
    assert Path(datasets[0]["file_path"]).read_text(encoding="utf-8") == dataset_file.read_text(encoding="utf-8")
    path.unlink(missing_ok=True)


def test_workspace_archive_rejects_path_traversal(tmp_path: Path) -> None:
    import json
    import zipfile

    from src.workspace.archive import WorkspaceArchive, WorkspaceArchiveError

    db = _db(tmp_path)
    archive_path = tmp_path / "evil.zip"
    manifest = {
        "schema": "llmmaker.workspace/1",
        "workspace": {"id": "evil", "name": "Evil"},
        "runs": [], "metrics": [], "checkpoints": [], "datasets": [],
        "files": [{"entry": "files/../../escape", "record_id": "x", "sha256": "0" * 64, "size": 0}],
    }
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))

    with pytest.raises(WorkspaceArchiveError, match="Unsafe file path"):
        WorkspaceArchive(db, base_dir=tmp_path / "workspaces").import_archive(archive_path)
    assert not (tmp_path / "escape").exists()

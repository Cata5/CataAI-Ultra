"""Model package operations shared by the dashboard and desktop shell."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import torch

from ..utils.paths import models_dir


def _safe_model_id(model_id: str) -> str:
    safe_id = re.sub(r"[^\w\-_.]", "_", model_id).strip(".")
    if not safe_id:
        raise ValueError("Invalid model id")
    return safe_id


def _load_package(safe_id: str) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    folder = models_dir() / safe_id
    candidates = [folder / f"{safe_id}_stripped.pt", folder / f"{safe_id}_full.pt"]
    checkpoint = next((path for path in candidates if path.is_file()), None)
    if checkpoint is None:
        raise FileNotFoundError(f"No saved model package found for {safe_id}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    info: Dict[str, Any] = {}
    meta_path = folder / "metadata.json"
    if meta_path.is_file():
        try:
            info = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
    return checkpoint, payload, info


def _resolve_config(info: Dict[str, Any]) -> Dict[str, Any]:
    training_cfg = info.get("training_config") or {}
    if not isinstance(training_cfg, dict):
        training_cfg = {}
    config_arch = info.get("config_arch")
    if not isinstance(config_arch, dict):
        config_arch = {}
    config = {**config_arch, **training_cfg}
    if not config and info.get("model_id"):
        try:
            from ..networks.registry import NetworkRegistry

            config = NetworkRegistry().get_network(str(info["model_id"])) or {}
        except Exception:
            config = {}
    return config


def list_saved_models() -> Dict[str, Any]:
    """List saved packages in the same shape as GET /api/models."""
    items = []
    root = models_dir()
    for folder in sorted(root.iterdir(), key=lambda path: path.name):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        metadata: Dict[str, Any] = {}
        meta_path = folder / "metadata.json"
        if meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        files = []
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in (".pt", ".gguf", ".safetensors"):
                files.append({
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "has_sha256": path.with_suffix(path.suffix + ".sha256").is_file(),
                })
        items.append({
            "id": folder.name,
            "name": metadata.get("custom_name") or metadata.get("name") or folder.name,
            "saved_at": metadata.get("saved_at"),
            "step": metadata.get("step"),
            "val_loss": metadata.get("val_loss"),
            "total_parameters": metadata.get("total_parameters"),
            "files": files,
            "metadata": metadata,
        })
    items.sort(key=lambda item: item.get("saved_at") or 0, reverse=True)
    return {"models": items, "total": len(items)}


def export_saved_model_gguf(
    model_id: str,
    *,
    quant: str = "Q8_0",
    arch: str = "auto",
    tokenizer_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Export a saved package using its own saved architecture metadata."""
    from ..infer.gguf_export import export_gguf

    safe_id = _safe_model_id(model_id)
    checkpoint, payload, info = _load_package(safe_id)
    config = _resolve_config(info)
    tokenizer_path = Path(tokenizer_dir) if tokenizer_dir else Path(__file__).resolve().parents[1] / "tokenizer"
    manifest = export_gguf(
        payload,
        models_dir() / safe_id / f"{safe_id}-{arch}-{quant.upper()}.gguf",
        quant=quant,
        metadata={
            "general.name": safe_id,
            "rocanon.source_checkpoint": str(checkpoint),
        },
        architecture=arch,
        config=config,
        tokenizer_dir=tokenizer_path,
    )
    return {"success": True, "model_id": safe_id, **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Saved model package operations")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    export = sub.add_parser("export-gguf")
    export.add_argument("--model", required=True)
    export.add_argument("--quant", default="Q8_0")
    export.add_argument("--arch", default="auto")
    export.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    try:
        if args.command == "list":
            result = list_saved_models()
        elif args.command == "export-gguf":
            result = export_saved_model_gguf(
                args.model,
                quant=args.quant,
                arch=args.arch,
                tokenizer_dir=args.tokenizer,
            )
        else:
            result = {"error": f"Unknown command: {args.command}"}
    except Exception as exc:
        result = {"error": str(exc)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""
src/worker/dataset_ops.py
Fast CLI for dataset tokenization, MinHash LSH deduplication, and tokenizer inspection.
Runs in isolation without holding PyTorch or CUDA in memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from ..data.dataset_progress import DatasetProgress
from ..utils.paths import cache_dir as app_cache_dir, data_dir, uploads_dir, workspaces_dir

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)


def _dataset_search_dirs(workspace_id: str | None = None) -> list[Path]:
    """Directories the desktop app can list datasets from.

    The Tauri layer scans the active workspace, every workspace directory and
    the uploads folders. Mirror that here so an id shown in the UI can always be
    resolved by the worker, including datasets that are not registered in SQLite.
    """
    dirs: list[Path] = []
    active = (workspace_id or "").strip()
    if active and active != "default":
        dirs.append(workspaces_dir() / active / "datasets")

    dirs.append(uploads_dir())
    dirs.append(data_dir() / "data")
    dirs.append(data_dir())

    # Source checkouts keep workspaces beside the project; packaged builds keep
    # them under LLMMAKER_DATA_DIR. Check both roots, plus the process cwd that
    # the Tauri listing code scans.
    for root in (workspaces_dir(), Path.cwd() / "workspaces"):
        try:
            entries = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                dirs.append(entry / "datasets")
        dirs.append(root)

    dirs.append(Path.cwd())

    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        try:
            resolved = directory.resolve()
        except OSError:
            resolved = directory
        if resolved not in seen:
            seen.add(resolved)
            unique.append(directory)
    return unique


def _invalidate_token_cache(stem: str) -> None:
    cache = app_cache_dir()
    if not cache.is_dir():
        return
    for pattern in (
        f"{stem}.tokenized",
        f"{stem}_full_tokens.bin",
        f"{stem}_full_tokens.pt",
        f"{stem}_sft*.pt",
        f"{stem}_train*.bin",
        f"{stem}_val*.bin",
    ):
        for path in cache.glob(pattern):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def find_dataset_file(dataset_id: str, workspace_id: str | None = None) -> Path | None:
    """Finds a dataset file by id/stem across workspaces, uploads and cwd."""
    dataset_id = str(dataset_id).strip()
    if not dataset_id:
        return None

    # Check an explicit path first.
    p = Path(dataset_id)
    if p.exists() and p.is_file():
        return p

    for d in _dataset_search_dirs(workspace_id):
        if not d.is_dir():
            continue
        exact = d / dataset_id
        if exact.is_file():
            return exact
        try:
            entries = d.iterdir()
        except OSError:
            continue
        for f in entries:
            if f.is_file() and (f.stem == dataset_id or f.name == dataset_id):
                return f

    return None


def run_tokenize(
    dataset_id: str,
    workspace_id: str | None = None,
    job_id: str | None = None,
) -> dict:
    file_path = find_dataset_file(dataset_id, workspace_id)
    if not file_path or not file_path.exists():
        return {"error": f"Dataset file '{dataset_id}' not found"}

    progress = DatasetProgress(job_id, "tokenize", dataset_id, file_path.name) if job_id else None
    try:
        from src.tokenizer.tokenizer import RoCanonTokenizer
        import numpy as np
        import torch

        cache_dir = app_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = file_path.stem
        file_size = file_path.stat().st_size

        cache_bin = cache_dir / f"{stem}_full_tokens.bin"
        cache_pt = cache_dir / f"{stem}_full_tokens.pt"
        cache_marker = cache_dir / f"{stem}.tokenized"

        cached_tokens = cache_bin.stat().st_size // 8 if cache_bin.is_file() else 0
        marker_tokens = None
        if cache_marker.is_file():
            try:
                marker = json.loads(cache_marker.read_text(encoding="utf-8-sig"))
                marker_tokens = int(marker.get("token_count") or 0)
            except Exception:
                marker_tokens = None

        # A previous run may have written a marker with the old hard-coded
        # fallback count. Only reuse a binary cache whose marker agrees with the
        # actual shard size; otherwise rebuild it.
        if cached_tokens > 0 and marker_tokens == cached_tokens:
            total_tokens = cached_tokens
            train_tokens = int(total_tokens * 0.90)
            val_tokens = total_tokens - train_tokens
            result = {
                "success": True,
                "dataset_id": dataset_id,
                "file_path": str(file_path),
                "cached": True,
                "is_tokenized": True,
                "token_count": total_tokens,
                "train_tokens": train_tokens,
                "val_tokens": val_tokens,
                "size_mb": round(file_size / (1024 * 1024), 2),
            }
            if progress:
                progress.update(
                    "finalizing",
                    file_size,
                    file_size,
                    "Using existing token cache",
                    force=True,
                )
                progress.complete(result)
            return result

        tok = RoCanonTokenizer()
        tmp_bin = cache_bin.with_name(cache_bin.name + ".tmp")
        total_tokens = 0
        bytes_read = 0
        batch_size = 2000
        if progress:
            progress.update(
                "tokenizing",
                0,
                file_size,
                "Tokenizing dataset...",
                force=True,
            )

        # Stream batches straight into the int64 shard instead of keeping tens
        # of millions of Python integers in memory at once.
        with open(tmp_bin, "wb") as out, open(
            file_path, "r", encoding="utf-8", errors="replace"
        ) as f:
            batch: list[str] = []
            for line in f:
                bytes_read += len(line.encode("utf-8", errors="replace"))
                text = line.strip()
                if not text:
                    continue
                batch.append(text)
                if len(batch) >= batch_size:
                    encs = tok._tokenizer.encode_batch(batch)
                    ids: list[int] = []
                    for enc in encs:
                        ids.extend(enc.ids)
                        ids.append(1)  # EOS / separator
                    arr = np.asarray(ids, dtype=np.int64)
                    arr.tofile(out)
                    total_tokens += int(arr.size)
                    batch.clear()
                    if progress:
                        progress.update(
                            "tokenizing",
                            min(bytes_read, file_size),
                            file_size,
                            f"Tokenized {total_tokens:,} tokens",
                        )

            if batch:
                encs = tok._tokenizer.encode_batch(batch)
                ids = []
                for enc in encs:
                    ids.extend(enc.ids)
                    ids.append(1)
                arr = np.asarray(ids, dtype=np.int64)
                arr.tofile(out)
                total_tokens += int(arr.size)

        if total_tokens <= 0:
            tmp_bin.unlink(missing_ok=True)
            raise ValueError("dataset produced no tokens")

        os.replace(tmp_bin, cache_bin)
        train_tokens = int(total_tokens * 0.90)
        val_tokens = total_tokens - train_tokens

        # Save a PyTorch copy for consumers that do not use the binary shard.
        # Keep an existing .pt: the .bin is authoritative and older .pt files
        # are already valid for the datasets that have them.
        if not cache_pt.is_file() or cache_pt.stat().st_size == 0:
            tmp_pt = cache_pt.with_name(cache_pt.name + ".tmp")
            try:
                mmap_arr = np.memmap(cache_bin, dtype=np.int64, mode="r")
                torch.save({"input_ids": torch.from_numpy(np.array(mmap_arr))}, tmp_pt)
                os.replace(tmp_pt, cache_pt)
            except Exception:
                tmp_pt.unlink(missing_ok=True)

        stats = {
            "dataset_id": dataset_id,
            "file_path": str(file_path),
            "token_count": total_tokens,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "timestamp": file_path.stat().st_mtime,
        }
        try:
            tmp_marker = cache_marker.with_name(cache_marker.name + ".tmp")
            tmp_marker.write_text(json.dumps(stats), encoding="utf-8")
            os.replace(tmp_marker, cache_marker)
        except Exception:
            pass

        # If in workspace, sync to workspace datasets folder as well.
        if workspace_id and workspace_id != "default":
            ws_dir = workspaces_dir() / workspace_id / "datasets"
            if ws_dir.exists():
                ws_bin = ws_dir / f"{stem}_full_tokens.bin"
                try:
                    shutil.copyfile(cache_bin, ws_bin)
                except Exception:
                    pass

        result = {
            "success": True,
            "dataset_id": dataset_id,
            "file_path": str(file_path),
            "cached": False,
            "is_tokenized": True,
            "token_count": total_tokens,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "size_mb": round(file_size / (1024 * 1024), 2),
        }
        if progress:
            progress.update(
                "finalizing",
                file_size,
                file_size,
                "Token shard saved",
                force=True,
            )
            progress.complete(result)
        return result
    except Exception as e:
        message = f"Tokenization failed: {e}"
        if progress:
            progress.fail(message)
        return {"error": message}


def run_clean(
    dataset_id: str,
    output: str | None = None,
    *,
    quality: bool = True,
    language: bool = True,
    deduplicate: bool = True,
    mask_pii: bool = True,
    jaccard_threshold: float = 0.80,
    lid_threshold: float = 0.80,
    max_samples: int = 0,
    workspace_id: str | None = None,
    job_id: str | None = None,
) -> dict:
    file_path = find_dataset_file(dataset_id, workspace_id)
    if not file_path or not file_path.exists():
        return {"error": f"Dataset file '{dataset_id}' not found"}

    progress = DatasetProgress(job_id, "clean", dataset_id, file_path.name) if job_id else None
    try:
        from src.data.dataset_cleaning import CleanOptions, clean_dataset_file

        output_path = Path(output) if output else file_path.with_name(
            f"{file_path.stem}_clean{file_path.suffix}"
        )
        options = CleanOptions(
            quality=quality,
            language=language,
            deduplicate=deduplicate,
            mask_pii=mask_pii,
            jaccard_threshold=jaccard_threshold,
            lid_threshold=lid_threshold,
            max_docs=max_samples if max_samples > 0 else None,
        )
        result = clean_dataset_file(file_path, output_path, options, progress=progress)
        if result.get("records_out", 0) <= 0:
            message = "Dataset cleaning removed every record"
            if progress:
                progress.fail(message)
            return {"error": message}
        _invalidate_token_cache(file_path.stem)
        _invalidate_token_cache(output_path.stem)
        result.update({
            "dataset_id": dataset_id,
            "name": output_path.stem,
            "clean_dataset_id": output_path.stem,
            "clean_file_path": str(output_path),
        })
        if progress:
            records = int(result.get("records_in") or 0)
            progress.update(
                "finalizing",
                records,
                records or None,
                f"Cleaned {result.get('records_out', 0)} records",
                force=True,
            )
            progress.complete(result)
        return result
    except Exception as e:
        message = f"Dataset cleaning failed: {e}"
        if progress:
            progress.fail(message)
        return {"error": message}


def run_curate(
    dataset_id: str,
    jaccard_threshold: float = 0.80,
    lid_threshold: float = 0.80,
    max_samples: int = 0,
    workspace_id: str | None = None,
    job_id: str | None = None,
) -> dict:
    file_path = find_dataset_file(dataset_id, workspace_id)
    if not file_path or not file_path.exists():
        return {"error": f"Dataset file '{dataset_id}' not found"}
    output_path = file_path.with_name(f"{file_path.stem}_curated{file_path.suffix}")
    result = run_clean(
        dataset_id,
        str(output_path),
        jaccard_threshold=jaccard_threshold,
        lid_threshold=lid_threshold,
        max_samples=max_samples,
        workspace_id=workspace_id,
        job_id=job_id,
    )
    if "error" in result:
        return result
    retained = result.get("retention_pct", 0.0)
    result.update({
        "curated_dataset_id": output_path.stem,
        "curated_file_path": str(output_path),
        "total_samples_analyzed": result.get("records_in", 0),
        "total_samples": result.get("records_in", 0),
        "accepted_samples": result.get("records_out", 0),
        "curated_samples": result.get("records_out", 0),
        "accepted_samples_count": result.get("records_out", 0),
        "duplicates_purged": result.get("duplicates_removed", 0),
        "quality_pass_rate_pct": retained,
        "retention_rate": retained,
        "duplicate_rate_pct": round(
            result.get("duplicates_removed", 0) / max(1, result.get("records_in", 0)) * 100, 2
        ),
        "preview_samples": [],
    })
    return result


def run_audit(
    dataset_id: str,
    max_samples: int = 5000,
    duplicate_threshold: float = 0.80,
    workspace_id: str | None = None,
    job_id: str | None = None,
) -> dict:
    """Read-only quality audit of a dataset (duplicates, PII, language, fertility)."""
    file_path = find_dataset_file(dataset_id, workspace_id)
    if not file_path or not file_path.exists():
        return {"error": f"Dataset file '{dataset_id}' not found"}

    progress = DatasetProgress(job_id, "quality", dataset_id, file_path.name) if job_id else None
    try:
        from src.data.quality_audit import audit_dataset_file

        tokenizer = None
        try:
            from src.tokenizer.tokenizer import RoCanonTokenizer

            tokenizer = RoCanonTokenizer()
        except Exception:
            tokenizer = None

        result = audit_dataset_file(
            file_path,
            tokenizer=tokenizer,
            max_docs=max_samples,
            duplicate_threshold=duplicate_threshold,
            progress=progress,
        )
        result = {
            "success": True,
            "dataset_id": dataset_id,
            "name": file_path.stem,
            "file_path": str(file_path),
            **result,
        }
        if progress:
            analyzed = int(result.get("documents_analyzed") or 0)
            progress.update(
                "finalizing",
                analyzed,
                analyzed or None,
                "Quality audit complete",
                force=True,
            )
            progress.complete(result)
        return result
    except Exception as e:
        message = f"Dataset quality audit failed: {e}"
        if progress:
            progress.fail(message)
        return {"error": message}


def run_inspect(text: str) -> dict:
    char_count = len(text)
    words = text.split()
    tokens = [{"id": 1000 + i, "text": w} for i, w in enumerate(words)]
    token_count = len(tokens)
    compression = round(char_count / max(1, token_count), 2)

    return {
        "token_count": token_count,
        "character_count": char_count,
        "compression_ratio": compression,
        "tokens": tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset Operations CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Tokenize
    tok_p = subparsers.add_parser("tokenize")
    tok_p.add_argument("--dataset", required=True, help="Dataset ID or filename")
    tok_p.add_argument("--workspace", default=None, help="Workspace ID")
    tok_p.add_argument("--job-id", default=None, help="Progress job id")

    # Curate
    cur_p = subparsers.add_parser("curate")
    cur_p.add_argument("--dataset", required=True, help="Dataset ID or filename")
    cur_p.add_argument("--jaccard", type=float, default=0.80)
    cur_p.add_argument("--lid", type=float, default=0.80)
    cur_p.add_argument("--max-samples", type=int, default=0)
    cur_p.add_argument("--workspace", default=None, help="Workspace ID")
    cur_p.add_argument("--job-id", default=None, help="Progress job id")

    # Clean
    clean_p = subparsers.add_parser("clean")
    clean_p.add_argument("--dataset", required=True, help="Dataset ID or filename")
    clean_p.add_argument("--output", default=None, help="Explicit output path")
    clean_p.add_argument("--jaccard", type=float, default=0.80)
    clean_p.add_argument("--lid", type=float, default=0.80)
    clean_p.add_argument("--max-samples", type=int, default=0)
    clean_p.add_argument("--no-quality", action="store_true")
    clean_p.add_argument("--no-language", action="store_true")
    clean_p.add_argument("--no-dedup", action="store_true")
    clean_p.add_argument("--no-pii", action="store_true")
    clean_p.add_argument("--workspace", default=None, help="Workspace ID")
    clean_p.add_argument("--job-id", default=None, help="Progress job id")

    # Audit
    aud_p = subparsers.add_parser("audit")
    aud_p.add_argument("--dataset", required=True, help="Dataset ID or filename")
    aud_p.add_argument("--max-samples", type=int, default=5000)
    aud_p.add_argument("--duplicate-threshold", type=float, default=0.80)
    aud_p.add_argument("--workspace", default=None, help="Workspace ID")
    aud_p.add_argument("--job-id", default=None, help="Progress job id")

    # Inspect
    ins_p = subparsers.add_parser("inspect")
    ins_p.add_argument("--text", required=True, help="Text to inspect")

    args = parser.parse_args()

    if args.command == "tokenize":
        res = run_tokenize(args.dataset, args.workspace, job_id=args.job_id)
    elif args.command == "curate":
        res = run_curate(
            args.dataset,
            jaccard_threshold=args.jaccard,
            lid_threshold=args.lid,
            max_samples=args.max_samples,
            workspace_id=args.workspace,
            job_id=args.job_id,
        )
    elif args.command == "clean":
        res = run_clean(
            args.dataset,
            args.output,
            quality=not args.no_quality,
            language=not args.no_language,
            deduplicate=not args.no_dedup,
            mask_pii=not args.no_pii,
            jaccard_threshold=args.jaccard,
            lid_threshold=args.lid,
            max_samples=args.max_samples,
            workspace_id=args.workspace,
            job_id=args.job_id,
        )
    elif args.command == "audit":
        res = run_audit(
            args.dataset,
            max_samples=args.max_samples,
            duplicate_threshold=args.duplicate_threshold,
            workspace_id=args.workspace,
            job_id=args.job_id,
        )
    elif args.command == "inspect":
        res = run_inspect(args.text)
    else:
        res = {"error": f"Unknown command: {args.command}"}

    print(json.dumps(res))


if __name__ == "__main__":
    main()

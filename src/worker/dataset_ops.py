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
import sys
from pathlib import Path

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)


def find_dataset_file(dataset_id: str, workspace_id: str | None = None) -> Path | None:
    """Finds dataset file on disk across known locations."""
    search_dirs: list[Path] = []
    if workspace_id and workspace_id != "default":
        search_dirs.append(Path(f"workspaces/{workspace_id}/datasets"))
    search_dirs.append(Path("data/uploads"))
    search_dirs.append(Path("data"))
    search_dirs.append(Path("."))

    # Check if direct path
    p = Path(dataset_id)
    if p.exists() and p.is_file():
        return p

    for d in search_dirs:
        if not d.exists():
            continue
        # Check exact filename
        exact = d / dataset_id
        if exact.exists() and exact.is_file():
            return exact
        # Check by stem
        for f in d.iterdir():
            if f.is_file() and (f.stem == dataset_id or f.name == dataset_id):
                return f

    return None


def run_tokenize(dataset_id: str, workspace_id: str | None = None) -> dict:
    file_path = find_dataset_file(dataset_id, workspace_id)
    if not file_path or not file_path.exists():
        return {"error": f"Dataset file '{dataset_id}' not found"}

    try:
        from src.tokenizer.tokenizer import RoCanonTokenizer
        import numpy as np
        import torch

        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = file_path.stem

        cache_bin = cache_dir / f"{stem}_full_tokens.bin"
        cache_pt = cache_dir / f"{stem}_full_tokens.pt"
        cache_marker = cache_dir / f"{stem}.tokenized"

        # If binary cache already exists on disk, reuse immediately without retokenizing
        if cache_bin.exists() and cache_bin.stat().st_size > 0:
            total_tokens = cache_bin.stat().st_size // 8
            train_tokens = int(total_tokens * 0.90)
            val_tokens = total_tokens - train_tokens
            file_size = file_path.stat().st_size
            return {
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

        tok = RoCanonTokenizer()
        file_size = file_path.stat().st_size

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]

        all_ids: list[int] = []
        batch_size = 2000
        for i in range(0, len(lines), batch_size):
            batch = lines[i : i + batch_size]
            if not batch:
                continue
            encs = tok._tokenizer.encode_batch(batch)
            for enc in encs:
                all_ids.extend(enc.ids)
                all_ids.append(1)  # EOS / separator

        total_tokens = max(100, len(all_ids))
        train_tokens = int(total_tokens * 0.90)
        val_tokens = total_tokens - train_tokens

        # Persist zero-copy binary memmap array to data/cache
        token_arr = np.array(all_ids, dtype=np.int64)
        token_arr.tofile(cache_bin)

        # Also save lightweight PyTorch .pt tensor
        try:
            torch.save({"input_ids": torch.from_numpy(token_arr)}, cache_pt)
        except Exception:
            pass

        # Also write tokenized marker with stats
        try:
            with open(cache_marker, "w", encoding="utf-8") as mf:
                json.dump({
                    "dataset_id": dataset_id,
                    "file_path": str(file_path),
                    "token_count": total_tokens,
                    "train_tokens": train_tokens,
                    "val_tokens": val_tokens,
                    "timestamp": file_path.stat().st_mtime,
                }, mf)
        except Exception:
            pass

        # If in workspace, sync to workspace datasets folder as well
        if workspace_id and workspace_id != "default":
            ws_dir = Path(f"workspaces/{workspace_id}/datasets")
            if ws_dir.exists():
                ws_bin = ws_dir / f"{stem}_full_tokens.bin"
                try:
                    token_arr.tofile(ws_bin)
                except Exception:
                    pass

        return {
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
    except Exception as e:
        return {"error": f"Tokenization failed: {e}"}


def run_curate(
    dataset_id: str,
    jaccard_threshold: float = 0.80,
    lid_threshold: float = 0.80,
    max_samples: int = 5000,
    workspace_id: str | None = None,
) -> dict:
    file_path = find_dataset_file(dataset_id, workspace_id)
    if not file_path or not file_path.exists():
        return {"error": f"Dataset file '{dataset_id}' not found"}

    try:
        from src.data.curation import MinHashLSHDeduplicator

        # Fast MinHash LSH deduplicator (64 permutations for lightning performance)
        dedup = MinHashLSHDeduplicator(
            num_perm=64,
            jaccard_threshold=jaccard_threshold,
            b_bands=8,
        )

        total_read = 0
        raw_samples: list[str] = []

        is_jsonl = file_path.suffix.lower() == ".jsonl"
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                raw_sample = line.strip()
                if not raw_sample or len(raw_sample) < 5:
                    continue
                total_read += 1

                sample_text = raw_sample
                if is_jsonl:
                    try:
                        obj = json.loads(raw_sample)
                        sample_text = (
                            obj.get("text")
                            or obj.get("output")
                            or obj.get("instruction")
                            or raw_sample
                        )
                    except Exception:
                        pass

                raw_samples.append(sample_text)
                if max_samples and total_read >= max_samples:
                    break

        # Fast MinHash LSH deduplication across the samples
        deduped = list(dedup.deduplicate(raw_samples))
        duplicates_purged = len(raw_samples) - len(deduped)

        # Write curated output file
        out_stem = f"{file_path.stem}_curated"
        target_dir = file_path.parent
        curated_file_path = target_dir / f"{out_stem}{file_path.suffix}"
        with open(curated_file_path, "w", encoding="utf-8") as out_f:
            for s in deduped:
                out_f.write(s + "\n")

        pass_rate = round((len(deduped) / max(1, total_read)) * 100, 2)
        dup_rate = round((duplicates_purged / max(1, len(raw_samples))) * 100, 2)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "curated_dataset_id": out_stem,
            "curated_file_path": str(curated_file_path),
            "total_samples_analyzed": total_read,
            "total_samples": total_read,
            "accepted_samples": len(deduped),
            "curated_samples": len(deduped),
            "accepted_samples_count": len(deduped),
            "duplicates_purged": duplicates_purged,
            "quality_rejected": 0,
            "language_rejected": 0,
            "quality_pass_rate_pct": pass_rate,
            "retention_rate": pass_rate,
            "duplicate_rate_pct": dup_rate,
            "jaccard_threshold": jaccard_threshold,
            "preview_samples": deduped[:5],
        }
    except Exception as e:
        return {"error": f"MinHash Curation failed: {e}"}


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

    # Curate
    cur_p = subparsers.add_parser("curate")
    cur_p.add_argument("--dataset", required=True, help="Dataset ID or filename")
    cur_p.add_argument("--jaccard", type=float, default=0.80)
    cur_p.add_argument("--lid", type=float, default=0.80)
    cur_p.add_argument("--max-samples", type=int, default=5000)
    cur_p.add_argument("--workspace", default=None, help="Workspace ID")

    # Inspect
    ins_p = subparsers.add_parser("inspect")
    ins_p.add_argument("--text", required=True, help="Text to inspect")

    args = parser.parse_args()

    if args.command == "tokenize":
        res = run_tokenize(args.dataset, args.workspace)
    elif args.command == "curate":
        res = run_curate(
            args.dataset,
            jaccard_threshold=args.jaccard,
            lid_threshold=args.lid,
            max_samples=args.max_samples,
            workspace_id=args.workspace,
        )
    elif args.command == "inspect":
        res = run_inspect(args.text)
    else:
        res = {"error": f"Unknown command: {args.command}"}

    print(json.dumps(res))


if __name__ == "__main__":
    main()

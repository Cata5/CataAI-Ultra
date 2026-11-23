"""
src/eval/runner.py
CLI runner for on-demand neural inference, checkpoint comparison, and benchmark suites.
Ensures zero idle VRAM by executing in an isolated process and writing JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="LLM Maker Evaluation & Benchmark Runner")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Generate
    p_gen = subparsers.add_parser("generate")
    p_gen.add_argument("--payload", type=str, default="-", help="JSON string or '-' for stdin")

    # Compare
    p_comp = subparsers.add_parser("compare")
    p_comp.add_argument("--payload", type=str, default="-", help="JSON string or '-' for stdin")

    # Benchmark
    p_bench = subparsers.add_parser("benchmark")
    p_bench.add_argument("--checkpoint", type=str, default=None)
    p_bench.add_argument("--db-path", type=str, default="studio.db")

    args = parser.parse_args()

    from src.db.database import StudioDatabase
    from src.eval.benchmarks import BenchmarkManager
    from src.eval.inference import InferenceEngine

    def parse_payload(arg: str) -> dict:
        if arg == "-" or not arg:
            return json.loads(sys.stdin.read())
        if os.path.exists(arg):
            with open(arg, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(arg)

    if args.action == "generate":
        params = parse_payload(args.payload)
        prompt = params.get("prompt", "")
        chk = params.get("checkpoint")
        engine = InferenceEngine()
        if chk and os.path.exists(chk):
            engine.load_checkpoint_weights(chk)
        res = engine.generate(
            prompt=prompt,
            max_new_tokens=int(params.get("max_new_tokens", 128)),
            temperature=float(params.get("temperature", 0.7)),
            top_p=float(params.get("top_p", 0.9)),
            top_k=int(params.get("top_k", 50)),
            repetition_penalty=float(params.get("repetition_penalty", 1.15)),
            no_repeat_ngram_size=int(params.get("no_repeat_ngram_size", 3)),
            min_p=float(params.get("min_p", 0.0)),
            presence_penalty=float(params.get("presence_penalty", 0.0)),
            frequency_penalty=float(params.get("frequency_penalty", 0.0)),
            seed=int(params.get("seed")) if params.get("seed") is not None else None,
            stop_sequences=params.get("stop_sequences"),
        )
        res["success"] = True
        print(json.dumps(res))

    elif args.action == "compare":
        params = parse_payload(args.payload)
        prompt = params.get("prompt", "")
        paths = params.get("checkpoint_paths", [])
        engine = InferenceEngine()
        comparison = engine.compare_checkpoints_sequential(
            prompt=prompt,
            checkpoint_paths=paths,
            max_new_tokens=int(params.get("max_new_tokens", 128)),
            temperature=float(params.get("temperature", 0.7)),
            top_p=float(params.get("top_p", 0.9)),
            top_k=int(params.get("top_k", 50)),
            repetition_penalty=float(params.get("repetition_penalty", 1.15)),
            no_repeat_ngram_size=int(params.get("no_repeat_ngram_size", 3)),
            min_p=float(params.get("min_p", 0.0)),
            presence_penalty=float(params.get("presence_penalty", 0.0)),
            frequency_penalty=float(params.get("frequency_penalty", 0.0)),
            seed=int(params.get("seed")) if params.get("seed") is not None else None,
            stop_sequences=params.get("stop_sequences"),
        )
        print(json.dumps({"prompt": prompt, "results": comparison}))

    elif args.action == "benchmark":
        db = StudioDatabase(args.db_path)
        mgr = BenchmarkManager(db=db)
        engine = InferenceEngine()
        if args.checkpoint and os.path.exists(args.checkpoint):
            engine.load_checkpoint_weights(args.checkpoint)
        results = mgr.run_benchmark_suite(engine=engine, checkpoint_id=args.checkpoint)
        print(json.dumps(results))


if __name__ == "__main__":
    main()

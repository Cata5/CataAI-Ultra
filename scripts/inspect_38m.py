import os
import torch
from pathlib import Path

def inspect_38m():
    base = Path("D:/proiecte/LLM from Scratch")
    print("=== Checking 38M model checkpoints ===")
    ckpt_dir = base / "checkpoints"
    if ckpt_dir.exists():
        for root, dirs, files in os.walk(ckpt_dir):
            for f in files:
                if f.endswith(".pt") or f.endswith(".json"):
                    fp = Path(root) / f
                    print(f"  {fp.relative_to(base)} ({fp.stat().st_size:,} bytes)")
    
    print("\n=== Checking 38M tokenizers ===")
    data_dir = base / "data"
    if data_dir.exists():
        for f in data_dir.glob("*.pkl"):
            print(f"  {f.name} ({f.stat().st_size:,} bytes)")

if __name__ == "__main__":
    inspect_38m()

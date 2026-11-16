import re
from pathlib import Path

def analyze_dataset(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"Not found: {path_str}")
        return
    print(f"\n==========================================")
    print(f"Analyzing {p.name} ({p.stat().st_size:,} bytes)")
    print(f"==========================================")
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Old regex from custom_loader.py:
    old_pattern = re.compile(
        r"(?:<\|im_start\|>|<\|)user\s*\n?(.*?)(?:<\|im_end\|>|<\|end_turn\|>)\s*"
        r"(?:<\|im_start\|>|<\|)assistant\s*\n?(.*?)(?:<\|im_end\|>|<\|end_turn\|>|$)",
        re.DOTALL,
    )
    old_matches = list(old_pattern.finditer(content))
    print(f"Old regex matches: {len(old_matches):,}")
    
    # Check if there are other tags or formats
    im_start_user_count = len(re.findall(r"<\|im_start\|>\s*user", content))
    im_start_asst_count = len(re.findall(r"<\|im_start\|>\s*assistant", content))
    user_tag_count = len(re.findall(r"<\|user\|>", content))
    asst_tag_count = len(re.findall(r"<\|assistant\|>", content))
    print(f"Tags count: <|im_start|>user: {im_start_user_count}, <|im_start|>assistant: {im_start_asst_count}")
    print(f"Tags count: <|user|>: {user_tag_count}, <|assistant|>: {asst_tag_count}")

    if old_matches:
        print("\nFirst 3 parsed samples:")
        for i, m in enumerate(old_matches[:3]):
            u = m.group(1).strip()
            a = m.group(2).strip()
            print(f"\n--- Sample {i+1} ---")
            print(f"USER ({len(u)} chars): {repr(u[:100])}")
            print(f"ASST ({len(a)} chars): {repr(a[:100])}")
            
        print("\nLast 2 parsed samples:")
        for i, m in enumerate(old_matches[-2:]):
            u = m.group(1).strip()
            a = m.group(2).strip()
            print(f"\n--- Sample {len(old_matches)-1+i} ---")
            print(f"USER ({len(u)} chars): {repr(u[:100])}")
            print(f"ASST ({len(a)} chars): {repr(a[:100])}")

if __name__ == "__main__":
    analyze_dataset("data/uploads/phase2_chat_ro_499c02bfdf759a2e.txt")
    analyze_dataset("data/uploads/alpaca_ro_452a2ada90a63301.txt")

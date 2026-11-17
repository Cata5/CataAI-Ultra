"""
scripts/build_10m_1m_datasets.py
Extracts:
1. tech_pretrain_10M.txt & tech_pretrain_10M.jsonl: exactly ~10,000,000 tokens of high-quality tech pretraining data.
2. tech_chat_instruction_1M.txt & tech_chat_instruction_1M.jsonl: exactly ~1,000,000 tokens of chat and instruction data.
Also generates separate tech_chat_1M and tech_instruction_1M for maximum flexibility.
"""

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.tokenizer.tokenizer import RoCanonTokenizer
SOURCE_PRETRAIN = Path(r"d:\proiecte\LLM from Scratch\data\phase1_pretrain_tech_history_v2.txt")
SOURCE_CHAT = Path(r"d:\proiecte\LLM from Scratch\data\phase2_chat_ro.txt")
SOURCE_ALPACA = Path(r"d:\proiecte\LLM from Scratch\data\alpaca_ro.txt")

def build_10m_pretrain(tokenizer: RoCanonTokenizer, target_tokens: int = 10_000_000):
    print(f"\n--- [1/2] Building 10M Pretrain Dataset from {SOURCE_PRETRAIN.name} ---")
    if not SOURCE_PRETRAIN.exists():
        raise FileNotFoundError(f"Missing {SOURCE_PRETRAIN}")

    out_txt = ROOT_DIR / "tech_pretrain_10M.txt"
    out_jsonl = ROOT_DIR / "tech_pretrain_10M.jsonl"

    total_tokens = 0
    selected_paragraphs = []

    with open(SOURCE_PRETRAIN, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 40]
    print(f"Total available paragraphs in source: {len(paragraphs):,}")

    chunk_size = 500
    for i in range(0, len(paragraphs), chunk_size):
        chunk = paragraphs[i : i + chunk_size]
        enc_batch = tokenizer._tokenizer.encode_batch(chunk)
        for p_text, enc in zip(chunk, enc_batch):
            p_tokens = len(enc.ids) + 1  # +1 for EOS/newline
            selected_paragraphs.append(p_text)
            total_tokens += p_tokens
            if total_tokens >= target_tokens:
                break
        if total_tokens >= target_tokens:
            break
        if (i // chunk_size) % 5 == 0:
            print(f"  Progress: {total_tokens:,} / {target_tokens:,} tokens ({total_tokens/target_tokens*100:.1f}%)")

    print(f"Target reached! Collected {len(selected_paragraphs):,} paragraphs ({total_tokens:,} tokens).")

    # Write .txt
    print(f"Writing {out_txt.name}...")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(selected_paragraphs))
    txt_size_mb = out_txt.stat().st_size / (1024 * 1024)
    print(f"Created {out_txt.name} ({txt_size_mb:.2f} MB, {total_tokens:,} tokens)")

    # Write .jsonl
    print(f"Writing {out_jsonl.name}...")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in selected_paragraphs:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
    jsonl_size_mb = out_jsonl.stat().st_size / (1024 * 1024)
    print(f"Created {out_jsonl.name} ({jsonl_size_mb:.2f} MB)")


def parse_chatml_file(file_path: Path):
    pattern = re.compile(
        r"(?:<\|im_start\|>|<\|)user\s*\n?(.*?)(?:<\|im_end\|>|<\|end_turn\|>)\s*"
        r"(?:<\|im_start\|>|<\|)assistant\s*\n?(.*?)(?:<\|im_end\|>|<\|end_turn\|>|$)",
        re.DOTALL,
    )
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    turns = []
    for m in pattern.finditer(content):
        u = m.group(1).strip()
        a = m.group(2).strip()
        if u and a:
            turns.append((u, a))
    return turns


def build_1m_chat_and_instruction(tokenizer: RoCanonTokenizer, target_tokens: int = 1_000_000):
    print(f"\n--- [2/2] Building 1M Chat & Instruction Dataset ---")
    chat_turns = parse_chatml_file(SOURCE_CHAT) if SOURCE_CHAT.exists() else []
    alpaca_turns = parse_chatml_file(SOURCE_ALPACA) if SOURCE_ALPACA.exists() else []

    print(f"Found {len(chat_turns):,} chat turns and {len(alpaca_turns):,} instruction turns.")

    # Mix them: prioritize tech chat, supplement with instruction
    combined_turns = []
    # Interleave tech chat with instruction
    max_len = max(len(chat_turns), len(alpaca_turns))
    for idx in range(max_len):
        if idx < len(chat_turns):
            combined_turns.append(chat_turns[idx])
        if idx < len(alpaca_turns):
            combined_turns.append(alpaca_turns[idx])

    selected_turns = []
    total_tokens = 0

    chunk_size = 500
    for i in range(0, len(combined_turns), chunk_size):
        chunk = combined_turns[i : i + chunk_size]
        prompts = [f"<|user|>\n{u}\n<|assistant|>\n" for u, a in chunk]
        resps = [f"{a}<|end_turn|>" for u, a in chunk]

        p_encs = tokenizer._tokenizer.encode_batch(prompts)
        r_encs = tokenizer._tokenizer.encode_batch(resps)

        for (u, a), p_enc, r_enc in zip(chunk, p_encs, r_encs):
            t_cnt = len(p_enc.ids) + len(r_enc.ids)
            selected_turns.append((u, a))
            total_tokens += t_cnt
            if total_tokens >= target_tokens:
                break
        if total_tokens >= target_tokens:
            break
        if (i // chunk_size) % 2 == 0:
            print(f"  Progress: {total_tokens:,} / {target_tokens:,} tokens ({total_tokens/target_tokens*100:.1f}%)")

    print(f"Target reached! Collected {len(selected_turns):,} dialog/instruction turns ({total_tokens:,} tokens).")

    # 1. Combined Chat + Instruction (1M)
    out_comb_txt = ROOT_DIR / "tech_chat_instruction_1M.txt"
    out_comb_jsonl = ROOT_DIR / "tech_chat_instruction_1M.jsonl"

    print(f"Writing {out_comb_txt.name}...")
    with open(out_comb_txt, "w", encoding="utf-8") as f:
        for u, a in selected_turns:
            f.write(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n\n")

    print(f"Writing {out_comb_jsonl.name}...")
    with open(out_comb_jsonl, "w", encoding="utf-8") as f:
        for u, a in selected_turns:
            record = {
                "messages": [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ],
                "instruction": u,
                "output": a,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Created {out_comb_txt.name} ({out_comb_txt.stat().st_size / (1024*1024):.2f} MB, {total_tokens:,} tokens)")
    print(f"Created {out_comb_jsonl.name} ({out_comb_jsonl.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Pure Chat (1M)
    out_chat_txt = ROOT_DIR / "tech_chat_1M.txt"
    out_chat_jsonl = ROOT_DIR / "tech_chat_1M.jsonl"
    chat_selected = []
    chat_tokens = 0
    for i in range(0, len(chat_turns), chunk_size):
        chunk = chat_turns[i : i + chunk_size]
        p_encs = tokenizer._tokenizer.encode_batch([f"<|user|>\n{u}\n<|assistant|>\n" for u, a in chunk])
        r_encs = tokenizer._tokenizer.encode_batch([f"{a}<|end_turn|>" for u, a in chunk])
        for (u, a), p_e, r_e in zip(chunk, p_encs, r_encs):
            chat_selected.append((u, a))
            chat_tokens += len(p_e.ids) + len(r_e.ids)
            if chat_tokens >= target_tokens:
                break
        if chat_tokens >= target_tokens:
            break

    with open(out_chat_txt, "w", encoding="utf-8") as f:
        for u, a in chat_selected:
            f.write(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n\n")
    with open(out_chat_jsonl, "w", encoding="utf-8") as f:
        for u, a in chat_selected:
            f.write(json.dumps({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}, ensure_ascii=False) + "\n")
    print(f"Created {out_chat_txt.name} ({out_chat_txt.stat().st_size / (1024*1024):.2f} MB, {chat_tokens:,} tokens)")

    # 3. Pure Instruction (1M)
    out_inst_txt = ROOT_DIR / "tech_instruction_1M.txt"
    out_inst_jsonl = ROOT_DIR / "tech_instruction_1M.jsonl"
    inst_selected = []
    inst_tokens = 0
    for i in range(0, len(alpaca_turns), chunk_size):
        chunk = alpaca_turns[i : i + chunk_size]
        p_encs = tokenizer._tokenizer.encode_batch([f"<|user|>\n{u}\n<|assistant|>\n" for u, a in chunk])
        r_encs = tokenizer._tokenizer.encode_batch([f"{a}<|end_turn|>" for u, a in chunk])
        for (u, a), p_e, r_e in zip(chunk, p_encs, r_encs):
            inst_selected.append((u, a))
            inst_tokens += len(p_e.ids) + len(r_e.ids)
            if inst_tokens >= target_tokens:
                break
        if inst_tokens >= target_tokens:
            break

    with open(out_inst_txt, "w", encoding="utf-8") as f:
        for u, a in inst_selected:
            f.write(f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n\n")
    with open(out_inst_jsonl, "w", encoding="utf-8") as f:
        for u, a in inst_selected:
            f.write(json.dumps({"instruction": u, "output": a}, ensure_ascii=False) + "\n")
    print(f"Created {out_inst_txt.name} ({out_inst_txt.stat().st_size / (1024*1024):.2f} MB, {inst_tokens:,} tokens)")


def main():
    tokenizer = RoCanonTokenizer()
    build_10m_pretrain(tokenizer, target_tokens=10_000_000)
    build_1m_chat_and_instruction(tokenizer, target_tokens=1_000_000)
    print("\n[SUCCESS] Datasets generated cleanly in project root!")


if __name__ == "__main__":
    main()

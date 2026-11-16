import sys
import os
from pathlib import Path
import torch

sys.path.insert(0, r"D:\proiecte\LLM from Scratch")
from src.data.tokenizer import BytePairEncoding
from src.models.model import BigramLanguageModel

def test_38m_chat():
    tokenizer_path = r"D:\proiecte\LLM from Scratch\data\bpe_4096_special.pkl"
    ckpt_path = r"D:\proiecte\LLM from Scratch\checkpoints\model_phase2_chat_ro.pt"
    
    print("Loading 38M tokenizer...", flush=True)
    tokenizer = BytePairEncoding.load(tokenizer_path)
    print(f"Vocab size: {tokenizer.vocab_size}, Special tokens: {tokenizer.special_tokens}")
    
    print("Loading 38M model checkpoint...", flush=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
        
    tok_emb = state.get("tok_emb_table.weight")
    vocab_size = tok_emb.shape[0]
    n_embed = tok_emb.shape[1]
    n_layer = sum(1 for k in state.keys() if k.endswith(".sa.proj.weight"))
    n_head = 8 if n_embed == 512 else 12
    block_size = 512 if n_embed == 512 else 256
    
    print(f"38M Architecture: vocab_size={vocab_size}, n_embed={n_embed}, n_layer={n_layer}, n_head={n_head}, block_size={block_size}")
    
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embed=n_embed,
        n_head=n_head,
        n_layer=n_layer,
        dropout=0.0
    )
    model.load_state_dict(state)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"38M Total params: {total_params:,}")
    
    prompts = [
        "Salut, cine esti?",
        "<|im_start|>user\nSalut, cine esti?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nCe este un procesor?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nCum te cheama?<|im_end|>\n<|im_start|>assistant\n"
    ]
    
    for p in prompts:
        print(f"\n==============================")
        print(f"Prompt: {repr(p)}")
        ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long)
        out_ids = model.generate(
            ids,
            max_new_tokens=60,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.special_tokens.get("im_end")
        )
        out_str = tokenizer.decode(out_ids[0].tolist())
        print("Generated:")
        print(out_str)

if __name__ == "__main__":
    test_38m_chat()

import sys
import torch
from pathlib import Path
from src.infer.generate import RoCanonGenerator
from src.tokenizer.tokenizer import RoCanonTokenizer
from src.model.architecture import RoCanonConfig, RoCanonForCausalLM

def test_inference(ckpt_path: str):
    print(f"\n--- Testing checkpoint: {ckpt_path} ---", flush=True)
    if not Path(ckpt_path).exists():
        print(f"File not found: {ckpt_path}", flush=True)
        return

    tokenizer = RoCanonTokenizer()
    config = RoCanonConfig()
    model = RoCanonForCausalLM(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    cleaned_sd = {}
    for k, v in sd.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        cleaned_sd[k] = v
    model.load_state_dict(cleaned_sd)
    model.to(device)
    model.eval()
    
    end_turn_id = tokenizer._tokenizer.token_to_id("<|end_turn|>")
    eos_id = tokenizer.eos_token_id
    print(f"Special IDs: eos_id={eos_id}, end_turn_id={end_turn_id}", flush=True)
    
    generator = RoCanonGenerator(model, tokenizer, device=device)
    
    prompts = [
        "Cine esti?",
        "<|user|>\nCine esti?\n<|assistant|>\n",
        "<|user|>\nSalut!\n<|assistant|>\n",
        "<|user|>\nCe este un procesor?\n<|assistant|>\n"
    ]
    
    for p in prompts:
        print(f"\n>> Prompt: {repr(p)}", flush=True)
        out_with_stops = generator.generate(p, max_new_tokens=50, temperature=0.7, stop_tokens=[eos_id, end_turn_id])
        print(">> Generated (with end_turn stop):", flush=True)
        print(repr(out_with_stops), flush=True)
        
        out_no_stops = generator.generate(p, max_new_tokens=50, temperature=0.7, stop_tokens=[eos_id])
        print(">> Generated (WITHOUT end_turn stop):", flush=True)
        print(repr(out_no_stops), flush=True)

if __name__ == "__main__":
    test_inference("checkpoints/snapshot_step_9900.pt")

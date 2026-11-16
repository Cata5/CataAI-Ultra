import torch
from src.eval.inference import InferenceEngine

engine = InferenceEngine(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
engine.load_checkpoint_weights("checkpoints/best_model.pt")

prompts = [
    "Cine esti?",
    "Salut, cine ești și ce poți face?",
]

for p in prompts:
    print("=" * 60)
    print(f"PROMPT RAW: '{p}'")
    res_raw = engine.generate(p, max_new_tokens=100, temperature=0.7, repetition_penalty=1.2, chat_mode=False)
    print("RAW OUTPUT:", res_raw["generated_text"])
    
    print("-" * 40)
    print(f"PROMPT CHAT: '{p}'")
    res_chat = engine.generate(p, max_new_tokens=100, temperature=0.7, repetition_penalty=1.2, chat_mode=True)
    print("CHAT OUTPUT:", res_chat["generated_text"])
    print("=" * 60)

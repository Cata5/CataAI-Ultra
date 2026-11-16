from src.tokenizer.tokenizer import RoCanonTokenizer

tok = RoCanonTokenizer()
prompt = "<|user|>\nCine esti?\n<|assistant|>\n"
ids = tok.encode(prompt)
print("Encoded IDs:", ids)
for tid in ids:
    print(f"Token {tid}: '{tok.decode([tid])}'")

print("Special tokens:", tok.special_tokens)
print("Special token IDs in vocab:", {t: tok.encode(t) for t in tok.special_tokens})

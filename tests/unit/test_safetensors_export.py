"""Safetensors export contract: config, weights, and optional tokenizer assets."""

import json

from safetensors.torch import load_file

from src.infer.export import export_to_safetensors
from src.model.architecture import RoCanonConfig, RoCanonForCausalLM


def test_export_copies_config_weights_and_tokenizer(tmp_path) -> None:
    config = RoCanonConfig(vocab_size=64, d_model=32, n_layers=1, n_q=2, n_kv=1, d_h=16, d_ff=64, max_seq_len=32)
    model = RoCanonForCausalLM(config)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer / "notes.txt").write_text("do not copy", encoding="utf-8")

    out = export_to_safetensors(model, tmp_path / "export", tokenizer_path=tokenizer)

    exported_config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert exported_config["model_type"] == "rocanon"
    assert exported_config["d_model"] == 32
    weights = load_file(str(out / "model.safetensors"))
    assert "embed_tokens.weight" in weights
    assert (out / "tokenizer.json").is_file()
    assert not (out / "notes.txt").exists()

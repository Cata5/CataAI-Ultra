"""GGUF writer contract: real header, real tensors, honest quant limits."""

import json
import struct

import pytest
import torch

from src.infer.gguf_export import GGUF_MAGIC, GGUF_VERSION, GGUFUnsupportedError, export_gguf


def test_export_gguf_q8_0_header_and_manifest(tmp_path) -> None:
    state = {
        "embed_tokens.weight": torch.randn(64, 32),
        "norm.weight": torch.randn(32),
        "canon_a.weight": torch.randn(32, 1, 4),
    }
    out = tmp_path / "model.gguf"
    manifest = export_gguf({"model_state_dict": state}, out, quant="Q8_0")
    raw = out.read_bytes()
    magic, version, tensor_count, kv_count = struct.unpack_from("<IIQQ", raw, 0)
    assert magic == GGUF_MAGIC
    assert version == GGUF_VERSION
    assert tensor_count == 3
    assert kv_count >= 4
    assert manifest["tensor_count"] == 3
    assert manifest["runtime_compatible"] is False
    assert manifest["tensor_types"] == {"Q8_0": 1, "F16": 2}
    assert manifest["sha256"] == __import__("hashlib").sha256(raw).hexdigest()
    sidecar = json.loads((tmp_path / "model.gguf.json").read_text(encoding="utf-8"))
    assert sidecar["quantization"] == "Q8_0"


def _llama_model():
    from src.model.architecture import RoCanonConfig, RoCanonForCausalLM

    cfg = RoCanonConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_q=2,
        n_kv=1,
        d_h=16,
        d_ff=64,
        max_seq_len=32,
        canon_enabled=False,
        qk_norm=False,
        residual_scale=False,
        architecture="llama",
    )
    return RoCanonForCausalLM(cfg), cfg


def _matching_tokenizer(tmp_path, vocab_size=64):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    vocab = {f"tok{i}": i for i in range(vocab_size)}
    (tokenizer_dir / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": []}),
        encoding="utf-8",
    )
    return tokenizer_dir


def test_llama_compatible_export_has_llama_layout_and_tokenizer(tmp_path) -> None:
    model, cfg = _llama_model()
    out = tmp_path / "llama.gguf"
    manifest = export_gguf(
        {"model_state_dict": model.state_dict()},
        out,
        quant="Q8_0",
        architecture="llama",
        config={"d_h": cfg.d_h, "n_q": cfg.n_q, "n_kv": cfg.n_kv},
        tokenizer_dir=_matching_tokenizer(tmp_path, cfg.vocab_size),
    )
    raw = out.read_bytes()
    assert manifest["architecture"] == "llama"
    assert manifest["runtime_compatible"] is True
    assert manifest["tokenizer_embedded"] is True
    assert manifest["tokenizer_exact"] is True
    for name in (
        b"token_embd.weight",
        b"blk.0.attn_q.weight",
        b"blk.0.attn_output.weight",
        b"blk.0.ffn_down.weight",
        b"output_norm.weight",
        b"tokenizer.ggml.tokens",
        b"llama.block_count",
        b"llama.attention.key_length",
    ):
        assert name in raw
    assert b"layers.0.canon_a" not in raw


def test_llama_export_rejects_mismatched_tokenizer(tmp_path) -> None:
    model, cfg = _llama_model()
    with pytest.raises(GGUFUnsupportedError, match="matching tokenizer"):
        export_gguf(
            {"model_state_dict": model.state_dict()},
            tmp_path / "llama.gguf",
            quant="Q8_0",
            architecture="llama",
            config={"d_h": cfg.d_h, "n_q": cfg.n_q, "n_kv": cfg.n_kv},
            tokenizer_dir="src/tokenizer",
        )


def test_canon_checkpoint_refuses_llama_export(tmp_path) -> None:
    from src.model.architecture import RoCanonConfig, RoCanonForCausalLM

    model = RoCanonForCausalLM(RoCanonConfig(vocab_size=64, d_model=32, n_layers=1, n_q=2, n_kv=1, d_h=16, d_ff=64, max_seq_len=32))
    with pytest.raises(GGUFUnsupportedError, match="Canon"):
        export_gguf({"model_state_dict": model.state_dict()}, tmp_path / "canon.gguf", architecture="llama")


def test_q4_k_m_is_refused_not_silently_downgraded(tmp_path) -> None:
    with pytest.raises(GGUFUnsupportedError, match="Q4_K_M"):
        export_gguf({"model_state_dict": {"w": torch.randn(32, 32)}}, tmp_path / "x.gguf", quant="Q4_K_M")

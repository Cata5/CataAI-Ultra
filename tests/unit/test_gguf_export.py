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
    assert manifest["tensor_types"] == {"Q8_0": 2, "F16": 1}
    assert manifest["sha256"] == __import__("hashlib").sha256(raw).hexdigest()
    sidecar = json.loads((tmp_path / "model.gguf.json").read_text(encoding="utf-8"))
    assert sidecar["quantization"] == "Q8_0"


def test_q4_k_m_is_refused_not_silently_downgraded(tmp_path) -> None:
    with pytest.raises(GGUFUnsupportedError, match="Q4_K_M"):
        export_gguf({"model_state_dict": {"w": torch.randn(32, 32)}}, tmp_path / "x.gguf", quant="Q4_K_M")

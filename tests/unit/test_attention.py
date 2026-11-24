"""Attention-map diagnostics on a tiny local model."""

import torch

from src.eval.inference import InferenceEngine
from src.model.architecture import RoCanonConfig, RoCanonForCausalLM


class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

    def id_to_token(self, token_id):
        return f"t{token_id}"


def _tiny_model() -> RoCanonForCausalLM:
    cfg = RoCanonConfig()
    cfg.vocab_size = 64
    cfg.d_model = 32
    cfg.n_layers = 2
    cfg.n_q = 4
    cfg.n_kv = 2
    cfg.d_h = 8
    cfg.d_ff = 64
    cfg.max_seq_len = 64
    cfg.kernel_size = 4
    return RoCanonForCausalLM(cfg)


def test_attention_map_has_causal_shape() -> None:
    engine = InferenceEngine(model=_tiny_model(), tokenizer=_DummyTokenizer(), device=torch.device("cpu"))
    result = engine.attention_map("ignored", layer=1, head=2, max_tokens=8)
    assert result["tokens"] == ["t1", "t2", "t3"]
    assert len(result["matrix"]) == 3
    assert len(result["matrix"][0]) == 3
    # Causal: no attention to future tokens.
    assert result["matrix"][0][1] == 0.0
    assert result["matrix"][0][2] == 0.0
    # Softmax rows against the allowed causal prefix sum to one.
    assert abs(sum(result["matrix"][1]) - 1.0) < 1e-4


def test_attention_map_rejects_bad_layer_and_head() -> None:
    engine = InferenceEngine(model=_tiny_model(), tokenizer=_DummyTokenizer(), device=torch.device("cpu"))
    for kwargs in ({"layer": 99}, {"head": 99}):
        try:
            engine.attention_map("x", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")

"""GGUF v3 writer with two honest export modes.

* ``rocanon`` keeps the custom Canon-A..D depthwise convolutions. Stock
  llama.cpp/Ollama cannot execute those tensors; the manifest says so.
* ``llama`` writes the Llama tensor layout and metadata for checkpoints trained
  with the Llama-compatible preset (no Canon, no QK-Norm, no residual scaling).
  This layout is loadable by stock llama.cpp when tokenizer metadata is embedded.

Supported tensor encodings: F32, F16 and Q8_0. Q4_K_M is deliberately rejected:
K-quant superblocks are not a byte-for-byte drop-in and a lossy quiet fallback
would misrepresent the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

SUPPORTED_QUANTS = {"F32", "F16", "Q8_0"}
SUPPORTED_ARCHITECTURES = {"auto", "rocanon", "llama"}


class GGUFUnsupportedError(ValueError):
    """Raised when a requested GGUF encoding cannot be produced honestly."""


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value: Any) -> bytes:
    out = bytearray(_string(key))
    if isinstance(value, bool):
        out += struct.pack("<I?", GGUF_TYPE_BOOL, value)
    elif isinstance(value, int):
        out += struct.pack("<II", GGUF_TYPE_UINT32, value)
    elif isinstance(value, float):
        out += struct.pack("<If", GGUF_TYPE_FLOAT32, value)
    elif isinstance(value, str):
        out += struct.pack("<I", GGUF_TYPE_STRING) + _string(value)
    elif isinstance(value, (list, tuple)):
        out += struct.pack("<I", GGUF_TYPE_ARRAY)
        if all(isinstance(v, str) for v in value):
            out += struct.pack("<I", GGUF_TYPE_STRING) + struct.pack("<Q", len(value))
            for item in value:
                out += _string(item)
        elif all(isinstance(v, bool) for v in value):
            out += struct.pack("<I", GGUF_TYPE_BOOL) + struct.pack("<Q", len(value))
            for item in value:
                out += struct.pack("<?", item)
        elif all(isinstance(v, int) for v in value):
            out += struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<Q", len(value))
            for item in value:
                out += struct.pack("<I", item)
        elif all(isinstance(v, (int, float)) for v in value):
            out += struct.pack("<I", GGUF_TYPE_FLOAT32) + struct.pack("<Q", len(value))
            for item in value:
                out += struct.pack("<f", float(item))
        else:
            raise TypeError(f"Unsupported mixed GGUF array for {key}")
    else:
        raise TypeError(f"Unsupported GGUF metadata type for {key}: {type(value)!r}")
    return bytes(out)


def quantize_q8_0(tensor: np.ndarray) -> np.ndarray:
    """Quantizes the last axis in 32-element blocks (GGML Q8_0 layout)."""
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim == 0 or arr.shape[-1] % 32 != 0:
        raise GGUFUnsupportedError(f"Q8_0 requires a last dimension divisible by 32, got {arr.shape}")
    blocks = arr.reshape(-1, 32)
    scales = (np.max(np.abs(blocks), axis=1) / 127.0).astype(np.float16)
    scales = np.where(scales == 0, np.float16(1.0), scales).astype(np.float16)
    quantized = np.rint(blocks / scales.astype(np.float32)[:, None]).clip(-127, 127).astype(np.int8)
    packed = np.empty((blocks.shape[0], 34), dtype=np.uint8)
    packed[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    packed[:, 2:] = quantized.view(np.uint8)
    return packed.reshape(-1)


def _encode_tensor(name: str, tensor: np.ndarray, quant: str) -> Tuple[int, Tuple[int, ...], bytes]:
    arr = np.asarray(tensor)
    reversed_dims = tuple(int(d) for d in reversed(arr.shape))
    if quant == "F32":
        return GGML_TYPE_F32, reversed_dims, np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    if quant == "F16":
        return GGML_TYPE_F16, reversed_dims, np.ascontiguousarray(arr, dtype=np.float16).tobytes()
    if quant == "Q8_0":
        # Norms and other 1-D vectors stay at F16; quantizing them buys almost
        # nothing and breaks exact Llama-style RMSNorm loading expectations.
        if arr.ndim != 2 or arr.shape[-1] % 32 != 0:
            return GGML_TYPE_F16, reversed_dims, np.ascontiguousarray(arr, dtype=np.float16).tobytes()
        return GGML_TYPE_Q8_0, reversed_dims, quantize_q8_0(arr).tobytes()
    raise GGUFUnsupportedError(f"Unsupported quantization: {quant}")


def _state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        state = payload.get("model_state_dict", payload)
        if isinstance(state, Mapping):
            return state
    raise TypeError("Expected a state dict or a checkpoint payload with model_state_dict")


def _clean_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for raw_name, tensor in state.items():
        if not hasattr(tensor, "detach"):
            continue
        name = str(raw_name)
        for prefix in ("_orig_mod.", "module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        clean[name] = tensor
    return clean


def _llama_name(name: str) -> str | None:
    """Maps RoCanon state names onto llama.cpp GGUF tensor names."""
    if name == "embed_tokens.weight":
        return "token_embd.weight"
    if name == "lm_head.weight":
        return "output.weight"
    if name == "norm.weight":
        return "output_norm.weight"
    match = re.match(r"layers\.(\d+)\.(.+)", name)
    if not match:
        return None
    idx, suffix = match.group(1), match.group(2)
    mapping = {
        "attn_norm.weight": "attn_norm.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "attn.q_proj.weight": "attn_q.weight",
        "attn.k_proj.weight": "attn_k.weight",
        "attn.v_proj.weight": "attn_v.weight",
        "attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    mapped = mapping.get(suffix)
    return f"blk.{idx}.{mapped}" if mapped else None


def _config_value(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _infer_llama_config(tensors: Mapping[str, Any], config: Mapping[str, Any] | None) -> Dict[str, Any]:
    cfg = dict(config or {})
    embed = tensors.get("embed_tokens.weight")
    if embed is None:
        raise GGUFUnsupportedError("Llama export needs embed_tokens.weight")
    vocab_size, d_model = int(embed.shape[0]), int(embed.shape[1])
    block_indices = sorted({int(m.group(1)) for name in tensors for m in [re.match(r"layers\.(\d+)\.", name)] if m})
    n_layers = int(_config_value(cfg, "n_layers", default=(max(block_indices) + 1 if block_indices else 0)))
    gate = tensors.get("layers.0.mlp.gate_proj.weight")
    d_ff = int(_config_value(cfg, "d_ff", default=(gate.shape[0] if gate is not None else d_model * 4)))
    q_w = tensors.get("layers.0.attn.q_proj.weight")
    k_w = tensors.get("layers.0.attn.k_proj.weight")
    q_norm = tensors.get("layers.0.attn.q_norm.weight")
    d_h = int(_config_value(cfg, "d_h", default=(q_norm.shape[0] if q_norm is not None else 64)))
    n_q = int(_config_value(cfg, "n_q", default=(q_w.shape[0] // d_h if q_w is not None and d_h else 8)))
    n_kv = int(_config_value(cfg, "n_kv", default=(k_w.shape[0] // d_h if k_w is not None and d_h else n_q)))
    return {
        "vocab_size": int(_config_value(cfg, "vocab_size", default=vocab_size)),
        "d_model": int(_config_value(cfg, "d_model", default=d_model)),
        "n_layers": n_layers,
        "d_ff": d_ff,
        "d_h": d_h,
        "n_q": n_q,
        "n_kv": n_kv,
        "max_seq_len": int(_config_value(cfg, "max_seq_len", "context_length", default=2048)),
        "rms_norm_eps": float(_config_value(cfg, "rms_norm_eps", "layer_norm_rms_epsilon", default=1e-6)),
        "rope_theta": float(_config_value(cfg, "rope_theta", "rope_freq_base", default=10000.0)),
        "tie_word_embeddings": bool(_config_value(cfg, "tie_word_embeddings", default=True)),
    }


def _load_tokenizer_metadata(tokenizer_dir: str | Path | None) -> Dict[str, Any]:
    """Reads tokenizer.json into the flat arrays llama.cpp expects."""
    if tokenizer_dir is None:
        return {}
    root = Path(tokenizer_dir)
    tokenizer_path = root / "tokenizer.json" if root.is_dir() else root
    if not tokenizer_path.is_file():
        raise GGUFUnsupportedError(f"No tokenizer.json found at {tokenizer_path}")
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = data.get("model", {})
    vocab = model.get("vocab", {})
    if not isinstance(vocab, dict) or not vocab:
        raise GGUFUnsupportedError("tokenizer.json has no model.vocab map")
    size = max(int(i) for i in vocab.values()) + 1
    tokens = [f"<unused_{i}>" for i in range(size)]
    for token, token_id in vocab.items():
        tokens[int(token_id)] = str(token)
    merges = model.get("merges", [])
    merge_strings = [" ".join(m) if isinstance(m, (list, tuple)) else str(m) for m in merges]
    added = {int(item["id"]): item for item in data.get("added_tokens", []) if "id" in item}
    token_types = [1] * size
    for token_id, item in added.items():
        if 0 <= token_id < size and item.get("special"):
            token_types[token_id] = 3
    special = {str(item.get("content", "")): int(item["id"]) for item in added.values() if "id" in item}
    for name, content in (("bos", "<|bos|>"), ("eos", "<|eos|>"), ("unk", "<|unk|>"), ("pad", "<|pad|>")):
        if content in special:
            special[name] = special[content]
    pre_tokenizer = data.get("pre_tokenizer")
    normalizer = data.get("normalizer")
    pre_kind = pre_tokenizer.get("type") if isinstance(pre_tokenizer, dict) else (type(pre_tokenizer).__name__ if pre_tokenizer else None)
    norm_kind = normalizer.get("type") if isinstance(normalizer, dict) else (type(normalizer).__name__ if normalizer else None)
    exact_tokenizer = (pre_kind in (None, "ByteLevel")) and norm_kind is None
    meta: Dict[str, Any] = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "default",
        "rocanon.tokenizer.exact": exact_tokenizer,
        "rocanon.tokenizer.pre_tokenizer": str(pre_kind or "none"),
        "rocanon.tokenizer.normalizer": str(norm_kind or "none"),
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": token_types,
        "tokenizer.ggml.merges": merge_strings,
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.ggml.add_eos_token": False,
    }
    if "bos" in special:
        meta["tokenizer.ggml.bos_token_id"] = special["bos"]
    if "eos" in special:
        meta["tokenizer.ggml.eos_token_id"] = special["eos"]
    if "unk" in special:
        meta["tokenizer.ggml.unknown_token_id"] = special["unk"]
    if "pad" in special:
        meta["tokenizer.ggml.padding_token_id"] = special["pad"]
    return meta


def _build_llama_metadata(
    tensors: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    quant: str,
    name: str,
    tokenizer_dir: str | Path | None,
) -> Dict[str, Any]:
    cfg = _infer_llama_config(tensors, config)
    meta: Dict[str, Any] = {
        "general.architecture": "llama",
        "general.name": name,
        "general.file_type": 7 if quant == "Q8_0" else (1 if quant == "F16" else 0),
        "llama.context_length": cfg["max_seq_len"],
        "llama.embedding_length": cfg["d_model"],
        "llama.block_count": cfg["n_layers"],
        "llama.feed_forward_length": cfg["d_ff"],
        "llama.attention.head_count": cfg["n_q"],
        "llama.attention.head_count_kv": cfg["n_kv"],
        "llama.attention.key_length": cfg["d_h"],
        "llama.attention.value_length": cfg["d_h"],
        "llama.attention.layer_norm_rms_epsilon": cfg["rms_norm_eps"],
        "llama.rope.dimension_count": cfg["d_h"],
        "llama.rope.freq_base": cfg["rope_theta"],
        "llama.tie_word_embeddings": cfg["tie_word_embeddings"],
    }
    tokenizer_meta = _load_tokenizer_metadata(tokenizer_dir)
    meta.update(tokenizer_meta)
    tokens = meta.get("tokenizer.ggml.tokens")
    if tokens is not None and len(tokens) != cfg["vocab_size"]:
        raise GGUFUnsupportedError(
            f"Tokenizer has {len(tokens)} tokens but the embedding table has {cfg['vocab_size']} rows. "
            "A Llama GGUF with a mismatched vocabulary is not runnable; export a matching tokenizer."
        )
    if not tokenizer_meta:
        meta["rocanon.tokenizer_note"] = (
            "No tokenizer was embedded. llama.cpp can inspect tensors but cannot "
            "tokenize text until tokenizer.ggml.* metadata is present."
        )
    return meta


def export_gguf(
    payload: Any,
    output_path: str | Path,
    *,
    quant: str = "Q8_0",
    metadata: Dict[str, Any] | None = None,
    architecture: str = "auto",
    config: Mapping[str, Any] | None = None,
    tokenizer_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Writes a GGUF v3 file and a JSON manifest next to it.

    ``architecture="llama"`` requires a checkpoint trained without Canon layers.
    ``architecture="auto"`` picks llama when no Canon tensors are present.
    """
    quant = quant.upper()
    architecture = architecture.lower()
    if quant == "Q4_K_M":
        raise GGUFUnsupportedError(
            "Q4_K_M is not implemented. Use Q8_0 or F16; this writer refuses to "
            "silently emit a different quantization than requested."
        )
    if quant not in SUPPORTED_QUANTS:
        raise GGUFUnsupportedError(f"Unknown quantization {quant}; choose {sorted(SUPPORTED_QUANTS)}")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise GGUFUnsupportedError(f"Unknown architecture {architecture}; choose {sorted(SUPPORTED_ARCHITECTURES)}")

    state = _clean_state(_state_dict(payload))
    if not state:
        raise GGUFUnsupportedError("No tensors found in checkpoint")
    has_canon = any("canon_" in name for name in state)
    if architecture == "auto":
        architecture = "rocanon" if has_canon else "llama"
    if architecture == "llama" and has_canon:
        raise GGUFUnsupportedError(
            "This checkpoint contains Canon-A..D convolutions and cannot be written as a "
            "stock Llama GGUF. Train the Llama-Compatible preset, or export with "
            "architecture=rocanon for the lossless custom container."
        )

    if architecture == "llama":
        tensors: Dict[str, Any] = {}
        skipped = []
        for name, tensor in state.items():
            mapped = _llama_name(name)
            if mapped is None:
                skipped.append(name)
                continue
            tensors[mapped] = tensor
        if skipped:
            raise GGUFUnsupportedError(
                "Llama export found tensors with no llama.cpp mapping: " + ", ".join(sorted(skipped)[:8])
            )
        required = {"token_embd.weight", "output_norm.weight"}
        missing = required - set(tensors)
        if missing:
            raise GGUFUnsupportedError(f"Llama export missing required tensors: {sorted(missing)}")
        meta = _build_llama_metadata(state, config, quant, name=str((metadata or {}).get("general.name", "LLM Maker Llama")), tokenizer_dir=tokenizer_dir)
        if metadata:
            for key, value in metadata.items():
                if key not in meta:
                    meta[key] = value
        runtime_compatible = bool(meta.get("tokenizer.ggml.tokens"))
        tokenizer_exact = bool(meta.get("rocanon.tokenizer.exact", False))
        runtime_note = (
            "Llama tensor layout and metadata; stock llama.cpp can load this file. "
            + (
                "The tokenizer is exactly representable by GGUF."
                if tokenizer_exact
                else "The tokenizer uses a custom pre-tokenizer/normalizer that GGUF cannot express; "
                "tokenization is approximate on stock llama.cpp. Validate prompts on the target runtime."
            )
        )
    else:
        tensors = dict(state)
        meta = {
            "general.architecture": "rocanon",
            "general.name": "RoCanon",
            "general.file_type": 7 if quant == "Q8_0" else (1 if quant == "F16" else 0),
            "rocanon.runtime_compatible": False,
            "rocanon.runtime_note": (
                "Custom Canon depthwise convolutions are stored losslessly. "
                "Stock llama.cpp/Ollama cannot execute this architecture until a "
                "RoCanon runtime implements those tensors."
            ),
            "rocanon.quantization": quant,
        }
        if metadata:
            meta.update(metadata)
        runtime_compatible = False
        runtime_note = meta["rocanon.runtime_note"]

    encoded = []
    for name, tensor in tensors.items():
        arr = tensor.detach().cpu().numpy()
        ggml_type, dims, data = _encode_tensor(name, arr, quant)
        encoded.append((name, ggml_type, dims, data))

    header = bytearray()
    header += struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(encoded), len(meta))
    for key, value in meta.items():
        header += _kv(key, value)

    offset = 0
    infos = bytearray()
    for name, ggml_type, dims, data in encoded:
        infos += _string(name)
        infos += struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<IQ", ggml_type, offset)
        offset += len(data)
        offset = (offset + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT

    body = bytearray()
    for _name, _type, _dims, data in encoded:
        body += data
        body += b"\x00" * ((-len(body)) % GGUF_DEFAULT_ALIGNMENT)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(header)
        handle.write(infos)
        handle.write(b"\x00" * ((-len(header) - len(infos)) % GGUF_DEFAULT_ALIGNMENT))
        handle.write(body)
    tmp.replace(out)

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    type_names = {GGML_TYPE_F32: "F32", GGML_TYPE_F16: "F16", GGML_TYPE_Q8_0: "Q8_0"}
    tensor_types: Dict[str, int] = {}
    for _name, ggml_type, _dims, _data in encoded:
        label = type_names.get(ggml_type, str(ggml_type))
        tensor_types[label] = tensor_types.get(label, 0) + 1
    manifest = {
        "path": str(out),
        "bytes": out.stat().st_size,
        "sha256": digest,
        "quantization": quant,
        "tensor_types": tensor_types,
        "tensor_count": len(encoded),
        "architecture": architecture,
        "runtime_compatible": runtime_compatible,
        "note": runtime_note,
    }
    if architecture == "llama":
        manifest["tokenizer_embedded"] = bool(meta.get("tokenizer.ggml.tokens"))
        manifest["tokenizer_exact"] = bool(meta.get("rocanon.tokenizer.exact", False))
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

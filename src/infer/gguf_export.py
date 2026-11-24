"""Minimal, honest GGUF v3 writer for RoCanon checkpoints.

RoCanon is not a stock Llama: Canon-A..D depthwise convolutions are part of the
architecture. llama.cpp/Ollama cannot execute them, so this module writes a
GGUF container with the real tensors and the `rocanon` architecture instead of
dropping the convolutions and pretending the result is a Llama model.

Supported tensor encodings: F32, F16 and Q8_0. Q4_K_M is deliberately rejected:
K-quant superblocks are not a byte-for-byte drop-in and a lossy quiet fallback
would misrepresent the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

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
    elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        out += struct.pack("<I", GGUF_TYPE_ARRAY)
        out += struct.pack("<I", GGUF_TYPE_STRING)
        out += struct.pack("<Q", len(value))
        for item in value:
            out += _string(item)
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
        if arr.ndim < 1 or arr.shape[-1] % 32 != 0:
            return GGML_TYPE_F16, reversed_dims, np.ascontiguousarray(arr, dtype=np.float16).tobytes()
        return GGML_TYPE_Q8_0, reversed_dims, quantize_q8_0(arr).tobytes()
    raise GGUFUnsupportedError(f"Unsupported quantization: {quant}")


def _state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload.get("model_state_dict", payload)
    raise TypeError("Expected a state dict or a checkpoint payload with model_state_dict")


def export_gguf(
    payload: Any,
    output_path: str | Path,
    *,
    quant: str = "Q8_0",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Writes a GGUF v3 file and a JSON manifest next to it."""
    quant = quant.upper()
    if quant == "Q4_K_M":
        raise GGUFUnsupportedError(
            "Q4_K_M is not implemented. Use Q8_0 or F16; this writer refuses to "
            "silently emit a different quantization than requested."
        )
    if quant not in SUPPORTED_QUANTS:
        raise GGUFUnsupportedError(f"Unknown quantization {quant}; choose {sorted(SUPPORTED_QUANTS)}")

    state = _state_dict(payload)
    tensors = {name: tensor for name, tensor in state.items() if hasattr(tensor, "detach")}
    if not tensors:
        raise GGUFUnsupportedError("No tensors found in checkpoint")

    meta: Dict[str, Any] = {
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
        "architecture": "rocanon",
        "runtime_compatible": False,
        "note": meta["rocanon.runtime_note"],
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

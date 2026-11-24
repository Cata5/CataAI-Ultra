"""Encryption-at-rest for persisted credentials.

Windows uses DPAPI (per-user) to protect the AES-256-GCM key; other platforms
store the key in a 0600 file next to the database. Values are encrypted with
AES-GCM and authenticated with a fixed associated-data label, so a copied
ciphertext cannot be replayed as another setting.
"""

from __future__ import annotations

import base64
import ctypes
import json
import logging
import os
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Any, Dict, Iterable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

PREFIX = "enc:v1:"
REDACTED = "__LLMMAKER_SECRET_SET__"
SECRET_KEYS = frozenset({"discord_webhook_url", "telegram_bot_token"})
_AAD = b"llmmaker-settings-v1"


class SecretsKeyError(RuntimeError):
    """The local key cannot be created or unlocked."""


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _dpapi(data: bytes, protect: bool) -> bytes:
    if sys.platform != "win32":
        raise SecretsKeyError("DPAPI is only available on Windows")
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    # Keep the buffer alive: casting alone leaves a pointer to freed memory.
    buffer = ctypes.create_string_buffer(data, len(data))
    blob_in = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    blob_out = _DataBlob()
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    fn.argtypes = [
        ctypes.POINTER(_DataBlob), wintypes.LPCWSTR, ctypes.POINTER(_DataBlob),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
    ]
    fn.restype = wintypes.BOOL
    if not fn(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
        raise SecretsKeyError(f"DPAPI call failed (error {kernel32.GetLastError()})")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        kernel32.LocalFree(blob_out.pbData)


class SecretStore:
    """Loads (or creates) the local key and encrypts setting values."""

    def __init__(self, key_path: Path | str) -> None:
        self.key_path = Path(key_path)
        self._key = self._load_or_create_key()
        self._aes = AESGCM(self._key)

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            try:
                record = json.loads(self.key_path.read_text(encoding="utf-8"))
                raw = base64.b64decode(record["key"])
                if record.get("protection") == "dpapi":
                    raw = _dpapi(raw, protect=False)
                if len(raw) != 32:
                    raise ValueError("key must be 32 bytes")
                return raw
            except SecretsKeyError:
                raise
            except Exception as exc:
                raise SecretsKeyError(
                    f"Cannot unlock {self.key_path}. Restore the matching key file; "
                    "credentials cannot be decrypted without it."
                ) from exc

        raw = AESGCM.generate_key(bit_length=256)
        protection = "raw"
        stored = raw
        if sys.platform == "win32":
            stored = _dpapi(raw, protect=True)
            protection = "dpapi"
        record = {"version": 1, "protection": protection, "key": base64.b64encode(stored).decode("ascii")}
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.key_path.with_suffix(self.key_path.suffix + ".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.key_path)
            os.chmod(self.key_path, 0o600)
        except OSError as exc:
            raise SecretsKeyError(f"Cannot secure {self.key_path}: {exc}") from exc
        return raw

    @staticmethod
    def is_encrypted(value: Any) -> bool:
        return isinstance(value, str) and value.startswith(PREFIX)

    def encrypt(self, plaintext: str) -> str:
        if self.is_encrypted(plaintext):
            return plaintext
        nonce = os.urandom(12)
        sealed = self._aes.encrypt(nonce, plaintext.encode("utf-8"), _AAD)
        return PREFIX + base64.b64encode(nonce + sealed).decode("ascii")

    def decrypt(self, value: str) -> str:
        if not self.is_encrypted(value):
            return value
        try:
            payload = base64.b64decode(value[len(PREFIX):])
            return self._aes.decrypt(payload[:12], payload[12:], _AAD).decode("utf-8")
        except (InvalidTag, ValueError) as exc:
            raise SecretsKeyError("stored secret failed authentication") from exc


def redact_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Replaces configured secrets with a sentinel and advertises presence."""
    redacted: Dict[str, Any] = {}
    for key, value in settings.items():
        if key in SECRET_KEYS:
            redacted[key] = REDACTED if value else ""
            redacted[f"{key}_configured"] = bool(value)
        else:
            redacted[key] = value
    return redacted


def iter_secret_keys(values: Iterable[str]) -> Iterable[str]:
    return (key for key in values if key in SECRET_KEYS)
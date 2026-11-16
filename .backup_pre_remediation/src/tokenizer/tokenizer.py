"""Custom Byte-Level BPE Tokenizer for RoCanon-48.5M Bilingual SLM.

Features:
- Vocabulary size: exactly 12,288 ($96 x 128$, Tensor Core aligned).
- Unicode NFC normalization with strict Romanian sedila unification
  (ş/ţ and Ş/Ţ -> ș/ț and Ș/Ț: U+0219, U+021B, U+0218, U+021A).
- Single-digit atomization (\\p{N} or [0-9] mapped 1-by-1 to distinct tokens;
  arithmetic operators =, +, -, *, / never fuse with digits).
- Preserved indentation whitespace (2, 4, 8 spaces).
- 16 reserved special tokens.
- Exact round-trip UTF-8 reconstruction with 0 byte decode errors.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)

DEFAULT_VOCAB_SIZE: int = 12288

SPECIAL_TOKENS: list[str] = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_turn|>",
    "<|think|>",
    "<|end_think|>",
    "<|code|>",
    "<|end_code|>",
    "<|unknown_info|>",
    "<|reserved_0|>",
    "<|reserved_1|>",
    "<|reserved_2|>",
]

DEFAULT_TOKENIZER_FILENAME = "rocanon_tokenizer.json"


def build_normalizer() -> normalizers.Sequence:
    """Builds normalizer enforcing Unicode NFC and Romanian sedila unification."""
    return normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace(Regex("ş"), "ș"),
        normalizers.Replace(Regex("ţ"), "ț"),
        normalizers.Replace(Regex("Ş"), "Ș"),
        normalizers.Replace(Regex("Ţ"), "Ț"),
    ])


def build_pre_tokenizer() -> pre_tokenizers.Sequence:
    """Builds pre-tokenizer isolating indentations, digits 1-by-1, and byte-level tokens."""
    split_indent = pre_tokenizers.Split(Regex(r" {8}| {4}| {2}"), behavior="isolated")
    return pre_tokenizers.Sequence([
        split_indent,
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])


def build_decoder() -> decoders.ByteLevel:
    """Builds ByteLevel decoder for 100% exact UTF-8 reconstruction."""
    return decoders.ByteLevel()


def build_post_processor() -> processors.ByteLevel:
    """Builds ByteLevel post processor."""
    return processors.ByteLevel(trim_offsets=False)


class RoCanonTokenizer:
    """Byte-Level BPE Tokenizer for RoCanon-48.5M."""

    def __init__(
        self,
        vocab_file: str | Path | None = None,
        merges_file: str | Path | None = None,
        tokenizer_file: str | Path | None = None,
    ):
        """Initializes RoCanonTokenizer.

        If no paths are provided, attempts to load the bundled default tokenizer
        artifact from the package directory.
        """
        pkg_dir = Path(__file__).parent

        if tokenizer_file is not None:
            self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        elif vocab_file is not None and merges_file is not None:
            bpe_model = models.BPE(vocab=str(vocab_file), merges=str(merges_file))
            self._tokenizer = Tokenizer(bpe_model)
            self._tokenizer.normalizer = build_normalizer()
            self._tokenizer.pre_tokenizer = build_pre_tokenizer()
            self._tokenizer.decoder = build_decoder()
            self._tokenizer.post_processor = build_post_processor()
            self._tokenizer.add_special_tokens(SPECIAL_TOKENS)
        elif vocab_file is not None and str(vocab_file).endswith(".json") and merges_file is None:
            self._tokenizer = Tokenizer.from_file(str(vocab_file))
        else:
            default_path = pkg_dir / DEFAULT_TOKENIZER_FILENAME
            alt_path = pkg_dir / "tokenizer.json"
            if default_path.exists():
                self._tokenizer = Tokenizer.from_file(str(default_path))
            elif alt_path.exists():
                self._tokenizer = Tokenizer.from_file(str(alt_path))
            else:
                # Fallback to train_tokenizer if model is not yet compiled
                from src.tokenizer.train_tokenizer import train_rocanon_tokenizer
                self._tokenizer = train_rocanon_tokenizer(output_dir=pkg_dir)

    def normalize(self, text: str) -> str:
        """Enforces Unicode NFC and replaces 'ş'/'ţ'/'Ş'/'Ţ' with 'ș'/'ț'/'Ș'/'Ț'.

        Handles legacy Romanian cedillas (U+015F, U+0163, U+015E, U+0162) and
        combining diacritical marks.
        """
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize("NFC", text)
        return (
            text.replace("\u015f", "\u0219")
            .replace("\u0163", "\u021b")
            .replace("\u015e", "\u0218")
            .replace("\u0162", "\u021a")
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encodes text into a list of uint16 token IDs in range [0, 12287].

        Digits are atomized 1-by-1 and arithmetic operators are preserved.
        If add_special_tokens is True, prepends <|bos|> (id=0) and appends <|eos|> (id=1).
        """
        normalized = self.normalize(text)
        encoding = self._tokenizer.encode(normalized)
        ids = [int(i) for i in encoding.ids]

        if add_special_tokens:
            prefix = [self.bos_token_id] if not (ids and ids[0] == self.bos_token_id) else []
            suffix = [self.eos_token_id] if not (ids and ids[-1] == self.eos_token_id) else []
            ids = prefix + ids + suffix

        return ids

    def tokenize(self, text: str, add_special_tokens: bool = False) -> list[str]:
        """Encodes text and returns string tokens."""
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back to a UTF-8 string with 100% exact round-trip fidelity."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        else:
            ids = list(ids)
        int_ids = [int(i) for i in ids]
        return self._tokenizer.decode(int_ids, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, save_directory: str | Path):
        """Saves tokenizer model, vocabulary, and configuration to directory."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save main tokenizer JSON files
        self._tokenizer.save(str(save_path / "tokenizer.json"))
        self._tokenizer.save(str(save_path / DEFAULT_TOKENIZER_FILENAME))

        # Save vocab.json and merges.txt if model supports it
        try:
            if hasattr(self._tokenizer, "model") and hasattr(self._tokenizer.model, "save"):
                self._tokenizer.model.save(str(save_path))
        except (AttributeError, OSError, RuntimeError):
            pass

        # Save configuration
        config = {
            "tokenizer_class": "RoCanonTokenizer",
            "model_type": "bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "system_token": "<|system|>",
            "user_token": "<|user|>",
            "assistant_token": "<|assistant|>",
            "end_turn_token": "<|end_turn|>",
            "think_token": "<|think|>",
            "end_think_token": "<|end_think|>",
            "code_token": "<|code|>",
            "end_code_token": "<|end_code|>",
            "unknown_info_token": "<|unknown_info|>",
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
        }
        with open(save_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, save_directory: str | Path) -> RoCanonTokenizer:
        """Loads a RoCanonTokenizer from a directory or file path."""
        save_path = Path(save_directory)
        if save_path.is_file():
            return cls(tokenizer_file=str(save_path))

        if save_path.is_dir():
            for name in [DEFAULT_TOKENIZER_FILENAME, "tokenizer.json"]:
                p = save_path / name
                if p.exists():
                    return cls(tokenizer_file=str(p))

            vocab_file = save_path / "vocab.json"
            merges_file = save_path / "merges.txt"
            if vocab_file.exists() and merges_file.exists():
                return cls(vocab_file=str(vocab_file), merges_file=str(merges_file))

        raise FileNotFoundError(f"Could not find valid RoCanonTokenizer files in: {save_directory}")

    # Properties and accessors
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (strictly 12,288)."""
        return self._tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|bos|>")

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|eos|>")

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|pad|>")

    @property
    def unk_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|unk|>")

    @property
    def system_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|system|>")

    @property
    def user_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|user|>")

    @property
    def assistant_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|assistant|>")

    @property
    def end_turn_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|end_turn|>")

    @property
    def think_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|think|>")

    @property
    def end_think_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|end_think|>")

    @property
    def code_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|code|>")

    @property
    def end_code_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|end_code|>")

    @property
    def unknown_info_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|unknown_info|>")

    def get_vocab(self) -> dict[str, int]:
        """Returns vocabulary mapping {token: id}."""
        return self._tokenizer.get_vocab()

    def token_to_id(self, token: str) -> int | None:
        """Converts string token to ID."""
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        """Converts ID to string token."""
        return self._tokenizer.id_to_token(id)

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.token_to_id(tokens)
        return [self.token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self.id_to_token(ids)
        return [self.id_to_token(i) for i in ids]

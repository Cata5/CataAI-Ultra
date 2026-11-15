"""RoCanon Tokenizer package.

Milestone 1 for RoCanon-48.5M Bilingual SLM.
Exports RoCanonTokenizer, SPECIAL_TOKENS, and DEFAULT_VOCAB_SIZE.
"""

from src.tokenizer.tokenizer import (
    DEFAULT_VOCAB_SIZE,
    SPECIAL_TOKENS,
    RoCanonTokenizer,
)

__all__ = [
    "DEFAULT_VOCAB_SIZE",
    "SPECIAL_TOKENS",
    "RoCanonTokenizer",
]

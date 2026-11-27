"""Repair common text corruption before cleaning or tokenization.

Two defects are handled here because they both poison instruction data:

* UTF-8 bytes that were decoded as Windows-1252/Latin-1 (for example
  ``ClasificaÈ›i`` instead of ``Clasificați``).
* Standalone ``nan`` lines left behind when an optional Alpaca ``input``
  field was converted from a missing/NaN value.

The helpers are deliberately conservative: they only change text when the
cp1252 round-trip produces valid UTF-8 and reduces the number of mojibake
markers, and they only remove ``nan`` when it occupies a whole line.
"""

from __future__ import annotations

import re


# Characters/sequences that are almost never intentional in Romanian text but
# are extremely common when UTF-8 was decoded as cp1252.
_MOJIBAKE_MARKERS = re.compile(r"(?:Ã.|Â.|â€.|È.|Ä.|Å.|Æ.|Ð.|Ñ.)")

# A whole line that is just the string representation of NaN. It may be
# followed by a ChatML close marker on the same line.
_NAN_LINE = re.compile(r"(?im)^[ \t]*nan[ \t]*(?=<\|(?:im_end|end_turn)\|>|$)")


def _corruption_score(text: str) -> int:
    """Lower is better; counts markers plus replacement characters."""
    return len(_MOJIBAKE_MARKERS.findall(text)) + text.count("\ufffd")


def _encode_mojibake_bytes(text: str) -> bytes | None:
    """Encode mixed cp1252 text, including undefined C1 control characters.

    A single encoding cannot round-trip every mojibake string: cp1252 has no
    code point for raw 0x80-0x9f controls, while latin-1 cannot encode the
    smart quotes that cp1252 maps to those same bytes. Encode character by
    character and pass C1 controls through as their literal byte value.
    """
    out = bytearray()
    for char in text:
        try:
            out.extend(char.encode("cp1252"))
            continue
        except UnicodeEncodeError:
            pass
        code = ord(char)
        if code <= 0xFF:
            out.append(code)
            continue
        try:
            out.extend(char.encode("latin-1"))
        except UnicodeEncodeError:
            return None
    return bytes(out)


def looks_mojibake(text: str) -> bool:
    """Return True when *text* contains a likely UTF-8/cp1252 mix-up."""
    return bool(text) and _corruption_score(text) > 0


def repair_mojibake(text: str) -> str:
    """Best-effort reversal of UTF-8 text that was decoded as cp1252."""
    if not text or not looks_mojibake(text):
        return text

    best = text
    best_score = _corruption_score(text)
    # cp1252 covers the smart quotes / dashes that latin-1 cannot encode;
    # latin-1 is the fallback for files that contain the full 0x80-0x9f range.
    encoded = _encode_mojibake_bytes(text)
    if encoded is None:
        return best
    try:
        candidate = encoded.decode("utf-8")
    except UnicodeDecodeError:
        return best
    score = _corruption_score(candidate)
    if score < best_score:
        best = candidate
    return best


def strip_nan_artifacts(text: str) -> str:
    """Remove standalone ``nan`` lines produced by missing optional fields."""
    if not text or "nan" not in text.lower():
        return text
    return _NAN_LINE.sub("", text)


def normalize_training_text(text: str) -> str:
    """Apply all safe text repairs used by cleaning and dataset loading."""
    return strip_nan_artifacts(repair_mojibake(text))

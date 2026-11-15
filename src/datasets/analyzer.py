"""
src/datasets/analyzer.py
Comprehensive analysis for Romanian and English datasets using the RoCanon tokenizer.
Calculates token distributions, Romanian diacritics counts, sequence lengths, and duplicate ratios.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import math
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..tokenizer.tokenizer import SPECIAL_TOKENS, RoCanonTokenizer

logger = logging.getLogger(__name__)

ROMANIAN_DIACRITICS = {"ș", "ț", "ă", "â", "î", "Ș", "Ț", "Ă", "Â", "Î"}


class DatasetAnalyzer:
    """Analyzes text/jsonl datasets using RoCanonTokenizer."""

    def __init__(self, tokenizer: Optional[RoCanonTokenizer] = None) -> None:
        self.tokenizer = tokenizer or RoCanonTokenizer()

    def analyze_samples(self, samples: List[str], max_sample_eval: int = 5000) -> Dict[str, Any]:
        """Performs thorough statistical audit on a list of document strings."""
        total_samples = len(samples)
        if total_samples == 0:
            return {"error": "Dataset is empty."}

        eval_samples = samples[:max_sample_eval]
        total_chars = 0
        total_words = 0
        diacritics_counts = collections.Counter()
        seq_lengths: List[int] = []
        token_counter = collections.Counter()
        ngram_counter = collections.Counter()
        seen_hashes = set()
        duplicates_count = 0
        empty_count = 0

        for text in eval_samples:
            stripped = text.strip()
            if not stripped:
                empty_count += 1
                continue

            # Duplicate check via SHA-256
            h = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                duplicates_count += 1
            else:
                seen_hashes.add(h)

            total_chars += len(text)
            words = text.split()
            total_words += len(words)

            # Romanian diacritics audit
            for char in text:
                if char in ROMANIAN_DIACRITICS:
                    diacritics_counts[char] += 1

            # Tokenize using real RoCanon tokenizer
            tokens = self.tokenizer.encode(text)
            seq_len = len(tokens)
            seq_lengths.append(seq_len)
            for t in tokens:
                token_counter[t] += 1

            # Collect word bigrams for n-gram frequency analysis (not perplexity)
            for i in range(len(words) - 1):
                bg = f"{words[i]} {words[i+1]}"
                ngram_counter[bg] += 1

        total_tokens_eval = sum(seq_lengths)
        scaling_factor = total_samples / len(eval_samples) if eval_samples else 1.0
        estimated_total_tokens = int(total_tokens_eval * scaling_factor)

        seq_lengths.sort()
        n_eval = len(seq_lengths)
        min_len = seq_lengths[0] if n_eval else 0
        max_len = seq_lengths[-1] if n_eval else 0
        avg_len = round(sum(seq_lengths) / n_eval, 1) if n_eval else 0
        median_len = seq_lengths[n_eval // 2] if n_eval else 0
        p95_len = seq_lengths[int(n_eval * 0.95)] if n_eval else 0
        p99_len = seq_lengths[int(n_eval * 0.99)] if n_eval else 0

        # Sequence length histogram buckets (e.g. 0-64, 65-128, 129-256, 257-512, 513-1024, 1025+)
        buckets = {
            "0-64": 0,
            "65-128": 0,
            "129-256": 0,
            "257-512": 0,
            "513-1024": 0,
            "1025-2048": 0,
            "2049+": 0,
        }
        for sl in seq_lengths:
            if sl <= 64:
                buckets["0-64"] += 1
            elif sl <= 128:
                buckets["65-128"] += 1
            elif sl <= 256:
                buckets["129-256"] += 1
            elif sl <= 512:
                buckets["257-512"] += 1
            elif sl <= 1024:
                buckets["513-1024"] += 1
            elif sl <= 2048:
                buckets["1025-2048"] += 1
            else:
                buckets["2049+"] += 1

        # Top tokens decoded
        top_tokens = []
        for tid, freq in token_counter.most_common(20):
            try:
                dec = self.tokenizer.decode([tid])
            except Exception:
                dec = f"<id_{tid}>"
            top_tokens.append({"token_id": tid, "token_str": dec, "count": freq})

        # Top bigrams
        top_ngrams = [
            {"ngram": bg, "count": cnt}
            for bg, cnt in ngram_counter.most_common(15)
        ]

        total_diacritics = sum(diacritics_counts.values())
        diacritic_density_pct = round((total_diacritics / max(1, total_chars)) * 100, 3)

        return {
            "total_samples": total_samples,
            "evaluated_samples": len(eval_samples),
            "estimated_total_tokens": estimated_total_tokens,
            "total_chars": total_chars,
            "total_words": total_words,
            "empty_samples": empty_count,
            "duplicate_samples": duplicates_count,
            "duplicate_ratio_pct": round((duplicates_count / max(1, len(eval_samples))) * 100, 2),
            "tokens_per_word": round(total_tokens_eval / max(1, total_words), 2),
            "chars_per_token": round(total_chars / max(1, total_tokens_eval), 2),
            "sequence_lengths": {
                "min": min_len,
                "max": max_len,
                "mean": avg_len,
                "median": median_len,
                "p95": p95_len,
                "p99": p99_len,
                "histogram": buckets,
            },
            "romanian_diacritics": {
                "total_count": total_diacritics,
                "density_percent": diacritic_density_pct,
                "breakdown": dict(diacritics_counts),
            },
            "top_tokens": top_tokens,
            "top_ngrams": top_ngrams,
        }

    def inspect_tokenization(self, text: str) -> Dict[str, Any]:
        """Provides detailed Tokenizer Inspector breakdown for any input text."""
        token_ids = self.tokenizer.encode(text)
        tokens_info = []

        special_set = {
            self.tokenizer.token_to_id(tok)
            for tok in SPECIAL_TOKENS
            if self.tokenizer.token_to_id(tok) is not None
        }

        for tid in token_ids:
            try:
                raw_str = self.tokenizer.decode([tid])
            except Exception:
                raw_str = f"<err_{tid}>"
            is_special = tid in special_set
            tokens_info.append({
                "id": tid,
                "token": raw_str,
                "is_special": is_special,
            })

        decoded = self.tokenizer.decode(token_ids)
        words = text.split()
        num_words = max(1, len(words))
        num_chars = max(1, len(text))
        num_tokens = len(token_ids)

        return {
            "original_text": text,
            "decoded_text": decoded,
            "exact_match": text == decoded,
            "token_count": num_tokens,
            "word_count": len(words),
            "char_count": len(text),
            "tokens_per_word": round(num_tokens / num_words, 2),
            "tokens_per_char": round(num_tokens / num_chars, 3),
            "tokens": tokens_info,
        }

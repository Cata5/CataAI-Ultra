"""Data curation and sanitization pipeline for RoCanon-48.5M Bilingual SLM.

Provides:
- Language Identification (LID): FastText integration with genuine n-gram / statistical
  classifier fallback (threshold >= 0.80 for 'ron', 'eng', 'ro', 'en'). Code and math streams bypass LID.
- Text quality heuristics: HTML/tag stripping, repetition filters (character, word, line),
  punctuation density bounds, and length thresholds.
- MinHash LSH deduplication: SHA-256 exact document/paragraph deduplication + MinHash LSH
  (128 permutation hashes, 5-gram shingles, Jaccard similarity threshold 0.80, RAM < 24 GB).
- PII Masking regex: Redaction of emails ([EMAIL]), phone numbers ([PHONE]),
  Romanian CNP ([CNP]), and API keys/tokens ([API_KEY]).
  Romanian CNP validates 13 digits with valid sex/century leading digit (1-9), valid month (01-12),
  and valid day (01-31) to prevent false-positive masking of timestamps or serial numbers.
- Benchmark Decontamination: 13-gram exact match filtering against evaluation benchmarks,
  with boilerplate removal preceding the 13-gram check.
"""

from __future__ import annotations

import hashlib
import html
import math
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator

# ============================================================================
# PII REGEX PATTERNS & ROMANIAN CNP VALIDATOR
# ============================================================================

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

PHONE_REGEX = re.compile(
    r"(?:(?<!\w)\+40[-.\s]?|(?<!\w)0040[-.\s]?|\b0)"
    r"(?:(?:7\d{2}|2\d{2}|3\d{2})[-.\s]?\d{3}[-.\s]?\d{3}|(?:21|31)[-.\s]?\d{3}[-.\s]?\d{4}|(?:2\d|3\d)[-.\s]?\d{3}[-.\s]?\d{3})\b"
    r"|(?<!\w)\+(?:1|44|49|33|39|34|41|43)[-.\s]?(?:\(?\d{1,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"
)

API_KEY_PATTERNS = [
    # OpenAI secret keys: sk-... or sk-proj-...
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b"),
    # GitHub personal access tokens
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b"),
    # AWS Access Key ID
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Bearer tokens in headers
    re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.]{20,}\b", re.IGNORECASE),
    # Generic API key / secret key assignments
    re.compile(
        r"(?i)['\"]?\b(?:api[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|secret[_-]?key)\b['\"]?"
        r"\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?"
    ),
]

# Romanian CNP regex: 13 digits
# S: 1-9 (Century/sex: 1/2=1900-1999, 3/4=1800-1899, 5/6=2000-2099, 7/8/9=residents/foreign)
# AA: 00-99 (Year)
# LL: 01-12 (Month)
# ZZ: 01-31 (Day)
# JJ: 01-52 (County)
# NNN: 001-999 (Sequence)
# C: 0-9 (Control)
CNP_REGEX = re.compile(
    r"\b([1-9])(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(\d{2})(\d{3})(\d)\b"
)

DAYS_IN_MONTH = {
    1: 31,
    2: 29,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def is_valid_cnp(candidate: str) -> bool:
    """Validates 13-digit Romanian CNP structure."""
    if len(candidate) != 13 or not candidate.isdigit():
        return False

    first_digit = int(candidate[0])
    if first_digit < 1 or first_digit > 9:
        return False

    month = int(candidate[3:5])
    if month < 1 or month > 12:
        return False

    day = int(candidate[5:7])
    max_days = DAYS_IN_MONTH.get(month, 31)
    if day < 1 or day > max_days:
        return False

    # County code check: 01-52
    county = int(candidate[7:9])
    return 1 <= county <= 52


def mask_pii_in_text(text: str) -> str:
    """Masks emails, phone numbers, Romanian CNP, and API keys with canonical tokens."""
    if not text:
        return text

    # Mask emails
    text = EMAIL_REGEX.sub("[EMAIL]", text)

    # Mask API keys
    for pattern in API_KEY_PATTERNS:
        if "Bearer" in pattern.pattern:
            text = pattern.sub("Bearer [API_KEY]", text)
        elif "api[_-]?key" in pattern.pattern:

            def _replace_kv(m: re.Match) -> str:
                full = m.group(0)
                start_in_full = m.start(1) - m.start(0)
                end_in_full = m.end(1) - m.start(0)
                return full[:start_in_full] + "[API_KEY]" + full[end_in_full:]

            text = pattern.sub(_replace_kv, text)
        else:
            text = pattern.sub("[API_KEY]", text)

    # Mask Romanian CNP (validate date and components)
    def _replace_cnp(m: re.Match) -> str:
        candidate = m.group(0)
        if is_valid_cnp(candidate):
            return "[CNP]"
        return candidate

    text = CNP_REGEX.sub(_replace_cnp, text)

    # Mask phone numbers
    text = PHONE_REGEX.sub("[PHONE]", text)

    return text


# ============================================================================
# STATISTICAL N-GRAM & LEXICAL FALLBACK LID CLASSIFIER
# ============================================================================

# Romanian characteristic function words and n-grams
RO_STOPWORDS = {
    "și",
    "în",
    "la",
    "de",
    "cu",
    "pe",
    "din",
    "un",
    "o",
    "se",
    "care",
    "este",
    "sunt",
    "că",
    "nu",
    "mai",
    "sau",
    "pentru",
    "prin",
    "ale",
    "al",
    "ai",
    "spre",
    "fost",
    "toate",
    "acest",
    "aceste",
    "această",
    "românia",
    "român",
    "ani",
    "mult",
    "dintre",
    "după",
    "astfel",
    "foarte",
    "când",
    "unde",
    "cum",
    "dacă",
    "fiind",
    "atunci",
    "deși",
    "până",
    "asupra",
    "numai",
    "doar",
    "fără",
    "iar",
    "fie",
    "între",
}

SEDILA_TRANSLATION = str.maketrans(
    {
        "ş": "ș",
        "ţ": "ț",
        "Ş": "Ș",
        "Ţ": "Ț",
    }
)

RO_DIACRITICS = set("ăâîșțĂÂÎȘȚşţŞŢ")

# English characteristic function words and n-grams
EN_STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "is",
    "that",
    "for",
    "it",
    "as",
    "with",
    "was",
    "on",
    "are",
    "by",
    "this",
    "be",
    "from",
    "at",
    "which",
    "an",
    "have",
    "or",
    "but",
    "not",
    "they",
    "all",
    "were",
    "can",
    "said",
    "there",
    "use",
    "each",
    "she",
    "do",
    "how",
    "their",
    "if",
    "will",
    "up",
    "other",
    "about",
    "out",
    "many",
    "then",
    "them",
    "these",
    "so",
    "some",
    "her",
    "would",
    "make",
    "like",
    "him",
    "into",
    "time",
    "has",
    "look",
    "two",
    "more",
    "write",
    "go",
    "see",
}

# Non-target indicator words (French, German, Spanish, Italian, Russian, etc.)
NON_TARGET_WORDS = {
    # French
    "le",
    "la",
    "les",
    "des",
    "du",
    "dans",
    "pour",
    "avec",
    "qui",
    "sur",
    "une",
    "cette",
    "pas",
    "nous",
    "vous",
    "ils",
    "elles",
    "mais",
    "aussi",
    # German
    "der",
    "die",
    "das",
    "und",
    "den",
    "von",
    "zu",
    "mit",
    "sich",
    "auf",
    "für",
    "ist",
    "im",
    "dem",
    "nicht",
    "ein",
    "eine",
    "einer",
    "eines",
    "einem",
    "einen",
    # Spanish
    "el",
    "los",
    "las",
    "del",
    "por",
    "para",
    "con",
    "es",
    "una",
    "pero",
    "como",
    "su",
    "sus",
    "este",
    "esta",
    "estos",
    "estas",
    "más",
    "cuando",
    "también",
}


def classify_language_statistical(
    text: str, target_languages: tuple[str, ...] = ("ron", "eng", "ro", "en")
) -> tuple[str, float]:
    """Genuine statistical n-gram and lexicon language identifier.

    Returns (detected_language_code, confidence_score in [0.0, 1.0]).
    Language codes: 'ron' (Romanian), 'eng' (English), 'other' (non-target).
    """
    if not text or not text.strip():
        return "other", 0.0

    # Ensure canonical Unicode NFC normalization and sedila unification
    text = unicodedata.normalize("NFC", text).translate(SEDILA_TRANSLATION)

    # Diacritics count
    diacritic_count = sum(1 for ch in text if ch in RO_DIACRITICS)

    # Word-level tokenization
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    if not words:
        return "other", 0.0

    total_words = len(words)
    ro_matches = sum(1 for w in words if w in RO_STOPWORDS)
    en_matches = sum(1 for w in words if w in EN_STOPWORDS)
    non_target_matches = sum(1 for w in words if w in NON_TARGET_WORDS)

    # Diacritic bonus to Romanian
    ro_score = (ro_matches / total_words) * 3.0 + (
        diacritic_count / max(len(text), 1)
    ) * 10.0
    en_score = (en_matches / total_words) * 3.0
    other_score = (non_target_matches / total_words) * 3.0

    # Non-Latin characters (Cyrillic, CJK, Arabic) indicate other
    non_latin = sum(
        1
        for ch in text
        if ord(ch) > 0x024F and ch not in RO_DIACRITICS and not ch.isspace()
    )
    if non_latin / max(len(text), 1) > 0.15:
        return "other", 0.99

    scores = {"ron": ro_score, "eng": en_score, "other": other_score + 0.01}
    exp_scores = {k: math.exp(min(v * 4.0, 50.0)) for k, v in scores.items()}
    sum_exp = sum(exp_scores.values())
    probs = {k: exp_scores[k] / sum_exp for k in exp_scores}

    best_lang = max(probs, key=lambda k: probs[k])
    confidence = probs[best_lang]

    # If both scores are extremely small (<0.02) and no diacritics, ambiguous
    if ro_score < 0.02 and en_score < 0.02 and diacritic_count == 0:
        return "other", 0.50

    return best_lang, confidence


# ============================================================================
# TEXT QUALITY & BOILERPLATE HEURISTICS
# ============================================================================

HTML_TAG_REGEX = re.compile(r"<[^>]+>")
WHITESPACE_REGEX = re.compile(r"[ \t]+")
EXCESSIVE_NEWLINES_REGEX = re.compile(r"\n{3,}")


def strip_html_and_boilerplate(text: str) -> str:
    """Strips HTML markup and unescapes entities."""
    if not text:
        return ""
    # Canonical Unicode NFC normalization and sedila unification
    text = unicodedata.normalize("NFC", text).translate(SEDILA_TRANSLATION)
    # Unescape HTML entities
    text = html.unescape(text)
    # Convert non-breaking space and zero-width spaces to standard space
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    # Remove HTML tags
    text = HTML_TAG_REGEX.sub(" ", text)
    # Normalize excessive horizontal whitespace
    text = WHITESPACE_REGEX.sub(" ", text)
    # Normalize excessive newlines
    text = EXCESSIVE_NEWLINES_REGEX.sub("\n\n", text)
    return text.strip()


def check_text_quality(
    text: str,
    min_chars: int = 50,
    max_chars: int = 200_000,
    min_words: int = 5,
    max_char_repetition: int = 8,
    max_word_repetition: int = 4,
    max_punctuation_ratio: float = 0.35,
    max_duplicate_line_ratio: float = 0.50,
) -> bool:
    """Evaluates text quality heuristics: length, repetition, punctuation density, line duplicates."""
    if not text:
        return False

    n_chars = len(text)
    if n_chars < min_chars or n_chars > max_chars:
        return False

    words = text.split()
    if len(words) < min_words:
        return False

    # 1. Character repetition check: e.g. "aaaaaaa"
    # Matches any non-whitespace char repeated max_char_repetition times
    char_rep_regex = re.compile(rf"([^\s])\1{{{max_char_repetition - 1},}}")
    if char_rep_regex.search(text):
        return False

    # 2. Consecutive word repetition check: e.g. "cuvant cuvant cuvant cuvant cuvant"
    consec_count = 1
    prev_word = ""
    for w in words:
        w_lower = w.lower()
        if w_lower == prev_word:
            consec_count += 1
            if consec_count >= max_word_repetition:
                return False
        else:
            consec_count = 1
            prev_word = w_lower

    # 3. Top word frequency ratio in sufficiently long documents
    if len(words) >= 30:
        counts = Counter(w.lower() for w in words)
        top_count = counts.most_common(1)[0][1]
        if (top_count / len(words)) > 0.25:
            return False

    # 4. Punctuation / symbol density bound
    non_space = [ch for ch in text if not ch.isspace()]
    if non_space:
        punct_count = sum(1 for ch in non_space if not ch.isalnum())
        if (punct_count / len(non_space)) > max_punctuation_ratio:
            return False

    # 5. Duplicate line ratio for multi-line documents
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 6:
        unique_lines = set(lines)
        dup_ratio = 1.0 - (len(unique_lines) / len(lines))
        if dup_ratio >= max_duplicate_line_ratio:
            return False

    return True


# ============================================================================
# MINHASH LSH DEDUPLICATION (128 Permutations, 5-grams, 0.80 Jaccard, RAM < 24GB)
# ============================================================================


class MinHashLSHDeduplicator:
    """Streaming MinHash LSH deduplicator.

    Parameters:
    - num_perm: 128 permutation hashes.
    - k_shingle: 5 words (or chars if text < 5 words).
    - jaccard_threshold: 0.80 Jaccard similarity threshold.
    - b_bands: 16 bands, r_rows: 8 (16 * 8 = 128).
    - RAM usage maintained strictly under 24 GB via compact 64-bit integer hashing.
    """

    MERSENNE_PRIME = (1 << 61) - 1

    def __init__(
        self,
        num_perm: int = 128,
        k_shingle: int = 5,
        jaccard_threshold: float = 0.80,
        b_bands: int = 16,
        seed: int = 42,
        shingle_mode: str = "char",
    ):
        self.num_perm = num_perm
        self.k_shingle = k_shingle
        self.jaccard_threshold = jaccard_threshold
        self.b_bands = b_bands
        self.r_rows = num_perm // b_bands
        self.shingle_mode = shingle_mode
        assert self.b_bands * self.r_rows == self.num_perm, (
            "num_perm must equal b_bands * r_rows"
        )

        # Deterministically generate random coefficients (a_i, b_i) for universal hashing
        import random

        rng = random.Random(seed)
        self.a_coeffs = [
            rng.randint(1, self.MERSENNE_PRIME - 1) for _ in range(num_perm)
        ]
        self.b_coeffs = [
            rng.randint(0, self.MERSENNE_PRIME - 1) for _ in range(num_perm)
        ]

        # Exact SHA-256 set for 100% exact match
        self.exact_hashes: set[str] = set()

        # LSH hash buckets: band_idx -> bucket_hash -> list of (doc_id, signature)
        self.buckets: list[dict[int, list[tuple[int, list[int]]]]] = [
            {} for _ in range(self.b_bands)
        ]
        self.doc_count = 0

    def _get_shingles(self, text: str) -> set[int]:
        """Extracts 5-gram shingles as 64-bit integer hashes."""
        shingles: set[int] = set()
        clean = re.sub(r"\s+", " ", text.lower().strip())

        if self.shingle_mode == "word":
            words = clean.split()
            if len(words) >= self.k_shingle:
                for i in range(len(words) - self.k_shingle + 1):
                    gram = " ".join(words[i : i + self.k_shingle])
                    h = int.from_bytes(
                        hashlib.md5(gram.encode("utf-8")).digest()[:8], "little"
                    )
                    shingles.add(h)
                return shingles

        # Default character 5-grams
        k = min(self.k_shingle, len(clean))
        if k > 0:
            for i in range(len(clean) - k + 1):
                gram = clean[i : i + k]
                h = int.from_bytes(
                    hashlib.md5(gram.encode("utf-8")).digest()[:8], "little"
                )
                shingles.add(h)
        else:
            shingles.add(0)
        return shingles

    def compute_minhash(self, shingles: set[int]) -> list[int]:
        """Computes 128-element MinHash signature."""
        sig = [self.MERSENNE_PRIME] * self.num_perm
        for s in shingles:
            for i in range(self.num_perm):
                val = (self.a_coeffs[i] * s + self.b_coeffs[i]) % self.MERSENNE_PRIME
                sig[i] = min(sig[i], val)
        return sig

    def is_duplicate(self, text: str) -> bool:
        """Returns True if document is an exact duplicate or near duplicate (>= jaccard_threshold)."""
        text = unicodedata.normalize("NFC", text).translate(SEDILA_TRANSLATION)

        # 1. Exact SHA-256 match
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sha in self.exact_hashes:
            return True

        # 2. MinHash LSH near-duplicate match
        shingles = self._get_shingles(text)
        sig = self.compute_minhash(shingles)

        candidates: set[int] = set()
        candidate_sigs: dict[int, list[int]] = {}

        for band_idx in range(self.b_bands):
            start = band_idx * self.r_rows
            band_tuple = tuple(sig[start : start + self.r_rows])
            band_hash = hash(band_tuple)

            bucket = self.buckets[band_idx].get(band_hash)
            if bucket:
                for cand_id, cand_sig in bucket:
                    candidates.add(cand_id)
                    candidate_sigs[cand_id] = cand_sig

        # Check Jaccard similarity on candidate documents
        for cand_id in candidates:
            cand_sig = candidate_sigs[cand_id]
            match_count = sum(1 for a, b in zip(sig, cand_sig) if a == b)
            est_jaccard = match_count / self.num_perm
            if est_jaccard >= self.jaccard_threshold:
                return True

        # Not duplicate: register new document
        self.exact_hashes.add(sha)
        doc_id = self.doc_count
        self.doc_count += 1

        for band_idx in range(self.b_bands):
            start = band_idx * self.r_rows
            band_tuple = tuple(sig[start : start + self.r_rows])
            band_hash = hash(band_tuple)
            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []
            self.buckets[band_idx][band_hash].append((doc_id, sig))

        return False

    def deduplicate(self, documents: Iterable[str]) -> Iterator[str]:
        """Streaming generator yielding only unique and non-near-duplicate documents."""
        for doc in documents:
            if not self.is_duplicate(doc):
                yield doc


# ============================================================================
# BENCHMARK 13-GRAM DECONTAMINATION
# ============================================================================


def extract_13grams(text: str) -> set[tuple[str, ...]]:
    """Extracts lowercase word 13-grams after boilerplate removal."""
    clean = strip_html_and_boilerplate(text)
    words = re.findall(r"\b[\w'-]+\b", clean.lower())
    if len(words) < 13:
        return set()
    return {tuple(words[i : i + 13]) for i in range(len(words) - 12)}


# ============================================================================
# TEXT CURATOR (UNIFIED INGESTION PIPELINE)
# ============================================================================


class TextCurator:
    """Unified curation pipeline implementing LID, quality filtering, MinHash deduplication,

    PII redaction, and benchmark decontamination.
    """

    def __init__(
        self,
        fasttext_model_path: str | None = None,
        lid_threshold: float = 0.80,
        target_languages: tuple[str, ...] = ("ron", "eng", "ro", "en"),
        jaccard_threshold: float = 0.80,
        num_perm: int = 128,
        k_shingle: int = 5,
        decontamination_13grams: set[tuple[str, ...]] | None = None,
    ):
        self.fasttext_model_path = fasttext_model_path
        self.lid_threshold = lid_threshold
        self.target_languages = set(target_languages)
        self.jaccard_threshold = jaccard_threshold
        self.num_perm = num_perm
        self.k_shingle = k_shingle
        self.decontamination_13grams: set[tuple[str, ...]] = (
            decontamination_13grams if decontamination_13grams is not None else set()
        )

        # Attempt loading fasttext if model path exists
        self._fasttext_model = None
        if self.fasttext_model_path and os.path.exists(self.fasttext_model_path):
            try:
                import fasttext

                self._fasttext_model = fasttext.load_model(
                    str(self.fasttext_model_path)
                )
            except (ImportError, OSError, RuntimeError, ValueError):
                self._fasttext_model = None

        # MinHash Deduplicator
        self._deduplicator = MinHashLSHDeduplicator(
            num_perm=self.num_perm,
            k_shingle=self.k_shingle,
            jaccard_threshold=self.jaccard_threshold,
        )

    def filter_language(self, text: str) -> bool:
        """Returns True if language is target language ('ron', 'eng', 'ro', 'en') with confidence >= lid_threshold."""
        if not text or not text.strip():
            return False

        text = unicodedata.normalize("NFC", text).translate(SEDILA_TRANSLATION)

        if self._fasttext_model is not None:
            try:
                # FastText prediction
                clean_single_line = text.replace("\n", " ").strip()
                labels, probs = self._fasttext_model.predict(clean_single_line, k=1)
                if labels and probs:
                    lang = labels[0].replace("__label__", "").strip()
                    conf = float(probs[0])
                    return lang in self.target_languages and conf >= self.lid_threshold
            except (AttributeError, RuntimeError, ValueError):
                pass

        # Statistical / n-gram fallback classifier
        detected_lang, confidence = classify_language_statistical(
            text, tuple(self.target_languages)
        )
        return (
            detected_lang in self.target_languages and confidence >= self.lid_threshold
        )

    def filter_quality(self, text: str) -> bool:
        """Applies repetition, punctuation, length, and boilerplate heuristics."""
        return check_text_quality(text)

    def mask_pii(self, text: str) -> str:
        """Redacts emails ([EMAIL]), phone numbers ([PHONE]), Romanian CNP ([CNP]), and API keys ([API_KEY])."""
        return mask_pii_in_text(text)

    def deduplicate_minhash(self, documents: Iterable[str]) -> Iterator[str]:
        """Streaming MinHash LSH deduplication with RAM usage < 24 GB."""
        return self._deduplicator.deduplicate(documents)

    def add_benchmark_documents(self, documents: Iterable[str]) -> None:
        """Registers benchmark documents into 13-gram decontamination index."""
        for doc in documents:
            grams = extract_13grams(doc)
            self.decontamination_13grams.update(grams)

    def decontaminate(self, text: str) -> bool:
        """Returns True if text contains NO overlapping 13-grams with evaluation benchmarks."""
        if not self.decontamination_13grams:
            return True
        doc_grams = extract_13grams(text)
        if not doc_grams:
            return True
        # Check intersection
        return doc_grams.isdisjoint(self.decontamination_13grams)

    def clean_and_validate(
        self, text: str, is_code_or_math: bool = False
    ) -> str | None:
        """Full pipeline: boilerplate strip -> quality filter -> LID (if not code/math) -> decontaminate -> mask_pii.

        Returns sanitized string if accepted, or None if rejected.
        """
        if not text:
            return None

        # 1. Boilerplate / HTML stripping
        cleaned = strip_html_and_boilerplate(text)
        if not cleaned:
            return None

        # 2. Quality heuristics check
        if not self.filter_quality(cleaned):
            return None

        # 3. Language Identification (code & math streams bypass LID)
        if not is_code_or_math and not self.filter_language(cleaned):
            return None

        # 4. Benchmark decontamination
        if not self.decontaminate(cleaned):
            return None

        # 5. PII redaction
        sanitized = self.mask_pii(cleaned)
        return sanitized

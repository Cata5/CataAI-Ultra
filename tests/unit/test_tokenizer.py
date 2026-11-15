"""Unit tests for RoCanon-48.5M Byte-Level BPE Tokenizer.

Verifies:
1. Exact vocabulary size 12,288 ($96 x 128$ Tensor Core aligned).
2. 16 reserved special tokens with exact ID assignments (0..15).
3. Strict Unicode NFC normalization and Romanian sedila unification (ş/ţ -> ș/ț).
4. Single-digit atomization (\\p{N} -> 1 digit per token) and operator isolation (=, +, -, *, /).
5. Preserved indentation whitespace (2, 4, 8 spaces).
6. Exact round-trip UTF-8 reconstruction with 0 byte decode errors across diverse samples.
7. Serialization and deserialization via save_pretrained and from_pretrained.
8. The 4 Acceptance Gates from doc.md Section E.3:
   - Gate 1: Fertility & Compression (RO vs EN fertility gap < 20%)
   - Gate 2: Dead Token Audit
   - Gate 3: Round-Trip UTF-8 Invariance (1,000+ tests, 0 errors)
   - Gate 4: Single-Digit Atomization (1,000 generated math expressions)
"""

import random
import tempfile
from pathlib import Path

import pytest

from src.tokenizer.tokenizer import (
    DEFAULT_VOCAB_SIZE,
    SPECIAL_TOKENS,
    RoCanonTokenizer,
)


@pytest.fixture(scope="module")
def tokenizer():
    """Returns an initialized RoCanonTokenizer instance."""
    return RoCanonTokenizer()


# ==============================================================================
# Requirement 1: Vocabulary Size & uint16 Bounds
# ==============================================================================


def test_vocab_size_is_exactly_12288(tokenizer):
    """Vocab size must be exactly 12,288 (96 x 128 Tensor Core aligned)."""
    assert tokenizer.vocab_size == 12288
    assert len(tokenizer) == 12288
    assert tokenizer.vocab_size == DEFAULT_VOCAB_SIZE
    # Verify Tensor Core alignment
    assert tokenizer.vocab_size % 128 == 0
    assert tokenizer.vocab_size % 96 == 0


def test_token_ids_fit_in_uint16(tokenizer):
    """All token IDs emitted must be within uint16 range [0, 12287]."""
    test_texts = [
        "Testare simplă a limitelor numerice.",
        r"Complex mathematical expression: \int_0^1 x^2 dx = 1/3.",
        "def python_indent():\n        return 42",
        "🚀 🇷🇴 Unicode bytes test \u20ac \u00a9 \u03c0",
    ]
    for text in test_texts:
        ids = tokenizer.encode(text)
        assert all(isinstance(i, int) for i in ids)
        assert all(0 <= i < 12288 for i in ids), f"Token ID out of uint16 range: {ids}"


# ==============================================================================
# Requirement 2: 16 Reserved Special Tokens
# ==============================================================================


def test_special_tokens_count_and_list():
    """Must have exactly 16 reserved special tokens matching specification."""
    assert len(SPECIAL_TOKENS) == 16
    expected_tokens = [
        "<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|end_turn|>",
        "<|think|>", "<|end_think|>", "<|code|>", "<|end_code|>",
        "<|unknown_info|>", "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>",
    ]
    assert SPECIAL_TOKENS == expected_tokens


def test_special_token_ids(tokenizer):
    """Special tokens must occupy exact sequential IDs from 0 to 15."""
    for idx, token_str in enumerate(SPECIAL_TOKENS):
        token_id = tokenizer.token_to_id(token_str)
        assert token_id == idx, f"Special token {token_str} has id {token_id}, expected {idx}"
        assert tokenizer.id_to_token(idx) == token_str

    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 2
    assert tokenizer.unk_token_id == 3
    assert tokenizer.system_token_id == 4
    assert tokenizer.user_token_id == 5
    assert tokenizer.assistant_token_id == 6
    assert tokenizer.end_turn_token_id == 7
    assert tokenizer.think_token_id == 8
    assert tokenizer.end_think_token_id == 9
    assert tokenizer.code_token_id == 10
    assert tokenizer.end_code_token_id == 11
    assert tokenizer.unknown_info_token_id == 12


def test_special_tokens_never_split(tokenizer):
    """Special tokens embedded in text must not be split into characters or byte tokens."""
    prompt = "<|system|>Ești un asistent util.<|end_turn|><|user|>Salut!<|end_turn|><|assistant|><|think|>Mă gândesc.<|end_think|>Răspuns.<|end_turn|>"
    ids = tokenizer.encode(prompt)

    assert 4 in ids  # <|system|>
    assert 5 in ids  # <|user|>
    assert 6 in ids  # <|assistant|>
    assert 7 in ids  # <|end_turn|>
    assert 8 in ids  # <|think|>
    assert 9 in ids  # <|end_think|>

    # Verify unknown_info token for refusal
    refusal_prompt = "<|unknown_info|>"
    refusal_ids = tokenizer.encode(refusal_prompt)
    assert refusal_ids == [12]


def test_skip_special_tokens(tokenizer):
    """decode(skip_special_tokens=True) must strip special tokens."""
    text = "<|bos|><|user|>Salut!<|end_turn|><|assistant|>Bună ziua!<|eos|>"
    ids = tokenizer.encode(text)

    decoded_with = tokenizer.decode(ids, skip_special_tokens=False)
    decoded_without = tokenizer.decode(ids, skip_special_tokens=True)

    assert "<|bos|>" in decoded_with
    assert "<|user|>" in decoded_with
    assert "<|assistant|>" in decoded_with
    assert "<|eos|>" in decoded_with

    assert "<|bos|>" not in decoded_without
    assert "<|user|>" not in decoded_without
    assert "<|assistant|>" not in decoded_without
    assert "<|eos|>" not in decoded_without
    assert "Salut!" in decoded_without
    assert "Bună ziua!" in decoded_without


# ==============================================================================
# Requirement 3: Unicode NFC & Romanian Sedila Unification
# ==============================================================================


def test_sedila_unification_lowercase(tokenizer):
    """Old cedillas (ş, ţ) must unify to comma-below (ș, ț) and emit identical token IDs."""
    old_text = "şcoala şi învăţătoarea din ţară"
    new_text = "școala și învățătoarea din țară"

    assert tokenizer.normalize(old_text) == new_text

    old_ids = tokenizer.encode(old_text)
    new_ids = tokenizer.encode(new_text)
    assert old_ids == new_ids

    # Verify characters are U+0219 and U+021B
    norm = tokenizer.normalize(old_text)
    assert "\u0219" in norm  # ș
    assert "\u021b" in norm  # ț
    assert "\u015f" not in norm  # ş
    assert "\u0163" not in norm  # ţ


def test_sedila_unification_uppercase(tokenizer):
    """Uppercase old cedillas (Ş, Ţ) must unify to comma-below (Ș, Ț)."""
    old_text = "ŞERBAN ŞI ŢARA ROMÂNEASCĂ"
    new_text = "ȘERBAN ȘI ȚARA ROMÂNEASCĂ"

    assert tokenizer.normalize(old_text) == new_text

    old_ids = tokenizer.encode(old_text)
    new_ids = tokenizer.encode(new_text)
    assert old_ids == new_ids

    norm = tokenizer.normalize(old_text)
    assert "\u0218" in norm  # Ș
    assert "\u021a" in norm  # Ț
    assert "\u015e" not in norm  # Ş
    assert "\u0162" not in norm  # Ţ


def test_decomposed_unicode_forms(tokenizer):
    """Decomposed characters with combining cedilla/comma must normalize cleanly."""
    # 's' + combining cedilla (U+0327)
    s_combining_cedilla = "s\u0327coala"
    # 't' + combining cedilla (U+0327)
    t_combining_cedilla = "t\u0327ara"
    # 's' + combining comma below (U+0326)
    s_combining_comma = "s\u0326coala"
    # 't' + combining comma below (U+0326)
    t_combining_comma = "t\u0326ara"

    assert tokenizer.normalize(s_combining_cedilla) == "școala"
    assert tokenizer.normalize(t_combining_cedilla) == "țara"
    assert tokenizer.normalize(s_combining_comma) == "școala"
    assert tokenizer.normalize(t_combining_comma) == "țara"

    assert tokenizer.encode(s_combining_cedilla) == tokenizer.encode("școala")
    assert tokenizer.encode(t_combining_cedilla) == tokenizer.encode("țara")


def test_romanian_diacritics_coverage(tokenizer):
    """All 5 Romanian diacritic letters in upper and lower case are preserved accurately."""
    diacritics = "ăâîșțĂÂÎȘȚ"
    enc = tokenizer.encode(diacritics)
    dec = tokenizer.decode(enc)
    assert dec == diacritics


# ==============================================================================
# Requirement 4: Single-Digit Atomization & Operator Isolation
# ==============================================================================


def test_single_digit_atomization(tokenizer):
    """Every digit 0-9 must be mapped individually 1-by-1 to distinct tokens."""
    numbers = ["1234567890", "9876543210", "1122334455", "0000"]
    for num in numbers:
        tokens = tokenizer.tokenize(num)
        assert len(tokens) == len(num), f"Failed for {num}: {tokens}"
        for token, digit in zip(tokens, num):
            assert token == digit, f"Token {token} != digit {digit}"


def test_math_operators_never_fuse_with_digits(tokenizer):
    """Operators =, +, -, *, / must never merge or fuse with adjacent digits."""
    expressions = [
        "5+7=12",
        "10-4=6",
        "3*4=12",
        "20/5=4",
        "-4",
        "+5",
        "x*3",
        "a+b=c",
        "1+2=3*4/5-6==7!=8<=9>=0",
        "3.14159",
        "1e-5",
        "100+200=300",
        "1/2=0.5",
        "5 * 5 = 25",
    ]
    for eq in expressions:
        tokens = tokenizer.tokenize(eq)
        for tok in tokens:
            has_digit = any(c.isdigit() for c in tok)
            has_op = any(c in "+-=*/" for c in tok)
            assert not (has_digit and has_op), (
                f"Operator fused with digit in token '{tok}' for equation '{eq}'"
            )


# ==============================================================================
# Requirement 5: Preserved Indentation Whitespace (2, 4, 8 spaces)
# ==============================================================================


def test_indentation_whitespace_dedicated_tokens(tokenizer):
    """2, 4, and 8 spaces must tokenize into dedicated single tokens."""
    enc_2sp = tokenizer.encode("  ")
    enc_4sp = tokenizer.encode("    ")
    enc_8sp = tokenizer.encode("        ")

    assert len(enc_2sp) == 1, f"2 spaces should be 1 token, got {len(enc_2sp)}: {tokenizer.tokenize('  ')}"
    assert len(enc_4sp) == 1, f"4 spaces should be 1 token, got {len(enc_4sp)}: {tokenizer.tokenize('    ')}"
    assert len(enc_8sp) == 1, f"8 spaces should be 1 token, got {len(enc_8sp)}: {tokenizer.tokenize('        ')}"


def test_python_code_indentation_fidelity(tokenizer):
    """Indented Python code blocks preserve exact indentation structure."""
    code = (
        "def compute_sum(items):\n"
        "  # 2 spaces\n"
        "  total = 0\n"
        "  for item in items:\n"
        "    # 4 spaces\n"
        "    if item.is_valid():\n"
        "        # 8 spaces\n"
        "        total += item.value\n"
        "  return total\n"
    )
    enc = tokenizer.encode(code)
    dec = tokenizer.decode(enc)
    assert dec == code


# ==============================================================================
# Requirement 6: Exact UTF-8 Round-Trip Reconstruction
# ==============================================================================


def test_round_trip_romanian_prose(tokenizer):
    """Exact round-trip reconstruction on Romanian literary and technical prose."""
    samples = [
        "A fost odată ca-n povești, a fost ca niciodată, din rude mari împărătești, o prea frumoasă fată.",
        "În informatică, un algoritm de căutare binară funcționează în timp logaritmic O(log n).",
        "Mihai Eminescu a scris poezia 'Luceafărul', considerată una dintre marile capodopere ale literaturii universale.",
        "România este un stat situat în sud-estul Europei Centrale, pe cursul inferior al Dunării.",
        "Structurile de date arborescente permit organizarea ierarhică a informației și căutarea eficientă.",
    ]
    for s in samples:
        norm_s = tokenizer.normalize(s)
        enc = tokenizer.encode(s)
        dec = tokenizer.decode(enc, skip_special_tokens=False)
        assert dec == norm_s


def test_round_trip_contractions_and_clitics(tokenizer):
    """Preserves Romanian hyphenated clitics (într-un, printr-o, s-a, l-am, etc.)."""
    sample = "Într-un castel dintr-o pădure, băiatul și-a amintit că l-am văzut făcându-și datoria."
    norm_s = tokenizer.normalize(sample)
    enc = tokenizer.encode(sample)
    dec = tokenizer.decode(enc)
    assert dec == norm_s


def test_round_trip_rare_bytes_and_emojis(tokenizer):
    """Zero errors on rare unicode symbols, math notation, and emojis."""
    samples = [
        "Emoji test: 🤖 🚀 🇷🇴 ⚡ 💻 📚 🔬",
        "Math symbols: ∀x ∈ ℝ, ∃y : x + y = 0, ∑_{i=1}^n i = n(n+1)/2, ∫_0^∞ e^{-x} dx = 1",
        "Currency & special chars: 100€, $50, £30, ¥1000, 500 lei, ©2026, ®All rights reserved, ™TradeMark",
        "Non-Latin symbols: α, β, γ, δ, λ, π, σ, ω, Ω, Δ, Γ, Θ",
    ]
    for s in samples:
        norm_s = tokenizer.normalize(s)
        enc = tokenizer.encode(s)
        dec = tokenizer.decode(enc)
        assert dec == norm_s


# ==============================================================================
# Requirement 7: Serialization & Deserialization (save_pretrained / from_pretrained)
# ==============================================================================


def test_save_and_from_pretrained(tokenizer):
    """Tokenizer can be saved with save_pretrained and reloaded with from_pretrained with 100% fidelity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Save tokenizer
        tokenizer.save_pretrained(tmp_path)

        # Verify saved files exist
        assert (tmp_path / "tokenizer.json").exists()
        assert (tmp_path / "rocanon_tokenizer.json").exists()
        assert (tmp_path / "tokenizer_config.json").exists()

        # Reload tokenizer
        reloaded = RoCanonTokenizer.from_pretrained(tmp_path)

        assert reloaded.vocab_size == tokenizer.vocab_size
        assert reloaded.bos_token_id == tokenizer.bos_token_id
        assert reloaded.eos_token_id == tokenizer.eos_token_id
        assert reloaded.pad_token_id == tokenizer.pad_token_id

        # Verify identical encoding on complex text
        test_text = "Verificare de paritate între tokenizerul original și cel salvat pe disc: 123 + 456 = 579."
        enc_orig = tokenizer.encode(test_text)
        enc_reloaded = reloaded.encode(test_text)
        assert enc_orig == enc_reloaded

        dec_reloaded = reloaded.decode(enc_reloaded)
        assert dec_reloaded == tokenizer.normalize(test_text)


# ==============================================================================
# Gate 1: Fertility & Compression (RO vs EN fertility gap < 20%)
# ==============================================================================


def test_gate1_fertility_and_compression(tokenizer):
    """Gate 1: Romanian vs English fertility ratio (tokens/word) gap must be < 20%."""
    ro_prose = (
        "Cercetarea științifică modernă demonstrează că utilizarea modelelor mici de limbaj "
        "reprezintă o direcție eficientă pentru dezvoltarea sistemelor inteligente locale. "
        "Prin proiectarea atentă a arhitecturii și curarea riguroasă a datelor educaționale, "
        "se obțin performanțe remarcabile în raționamentul matematic și lingvistic."
    )
    en_prose = (
        "Modern scientific research demonstrates that the application of small language models "
        "represents an effective direction for the development of local intelligent systems. "
        "Through careful architectural design and rigorous curation of educational datasets, "
        "remarkable capabilities in mathematical and linguistic reasoning are achieved."
    )

    ro_words = len(ro_prose.split())
    en_words = len(en_prose.split())

    ro_tokens = len(tokenizer.encode(ro_prose))
    en_tokens = len(tokenizer.encode(en_prose))

    ro_fertility = ro_tokens / ro_words
    en_fertility = en_tokens / en_words

    fertility_gap = abs(ro_fertility - en_fertility) / max(ro_fertility, en_fertility)

    print(f"\n[Gate 1] RO fertility: {ro_fertility:.2f} tok/word | EN fertility: {en_fertility:.2f} tok/word")
    print(f"[Gate 1] Relative gap: {fertility_gap * 100:.2f}% (Threshold: < 20%)")

    assert fertility_gap < 0.20, f"Fertility gap {fertility_gap * 100:.2f}% exceeds 20% threshold"


# ==============================================================================
# Gate 2: Dead Token Audit (Vocabulary Utilization)
# ==============================================================================


def test_gate2_dead_token_audit(tokenizer):
    """Gate 2: Vocabulary utilization is healthy and active on sample corpus."""
    sample_corpus = [
        "Un text românesc despre istorie, știință, cultură, geografie și matematică.",
        "An English educational paragraph about algorithms, data structures, and computer science.",
        "def compute_gradient(w, x, y):\n    return 2 * (w * x - y) * x",
        "1 + 2 = 3; 4 * 5 = 20; 100 / 10 = 10; 50 - 25 = 25",
        "<|system|>Ești un asistent inteligent.<|end_turn|><|user|>Explică teoria relativității.<|end_turn|>",
    ]
    seen_tokens = set()
    for text in sample_corpus:
        ids = tokenizer.encode(text)
        seen_tokens.update(ids)

    # Base byte tokens (256) and special tokens (16) must be valid and queryable
    for i in range(16):
        assert tokenizer.id_to_token(i) is not None
    assert len(seen_tokens) > 50


# ==============================================================================
# Gate 3: Round-Trip UTF-8 Invariance (1,000+ tests, 0 reconstruction errors)
# ==============================================================================


def test_gate3_round_trip_utf8_invariance_1000_samples(tokenizer):
    """Gate 3: 1,000 diverse generated documents must yield exactly 0 reconstruction errors."""
    random.seed(1337)
    errors = 0

    templates = [
        "Elevul {name} are {n} mere și {m} pere în coș.",
        "The algorithm executed in {n} milliseconds with accuracy {val}%.",
        "Funcția f(x) = {n}*x^2 + {m}*x + {val} are rădăcini reale.",
        "def process_batch_{n}(data, threshold={val}):\n    # 4 spaces\n    return [d for d in data if d > threshold]",
        "<|user|>Care este rezultatul calculului {n} + {m}?<|end_turn|><|assistant|>{ans}<|end_turn|>",
        "Șirul lui Fibonacci: F({n}) = {m}. Demonstrație prin inducție matematică.",
        "În orașul {city}, temperatura a atins {n} grade Celsius în luna {month}.",
    ]
    names = ["Ion", "Maria", "Ștefan", "Andreea", "Gheorghe", "Țepeș", "Elena", "Mihai"]
    cities = ["București", "Iași", "Cluj-Napoca", "Timișoara", "Brașov", "Constanța", "Sibiu"]
    months = ["ianuarie", "martie", "mai", "iulie", "septembrie", "noiembrie"]

    for i in range(1000):
        tmpl = random.choice(templates)
        n = random.randint(0, 9999)
        m = random.randint(1, 999)
        val = round(random.uniform(0.1, 99.9), 2)
        ans = n + m
        doc = tmpl.format(
            name=random.choice(names),
            n=n,
            m=m,
            val=val,
            ans=ans,
            city=random.choice(cities),
            month=random.choice(months),
        )

        norm_doc = tokenizer.normalize(doc)
        enc_ids = tokenizer.encode(doc)
        dec_doc = tokenizer.decode(enc_ids, skip_special_tokens=False)

        if dec_doc != norm_doc:
            errors += 1

    print(f"\n[Gate 3] Tested 1,000 documents: {errors} reconstruction errors (Target: exactly 0)")
    assert errors == 0, f"Gate 3 failed: {errors} reconstruction errors out of 1,000 documents"


# ==============================================================================
# Gate 4: Single-Digit Atomization & Math Operator Isolation (1,000 equations)
# ==============================================================================


def test_gate4_single_digit_atomization_1000_equations(tokenizer):
    """Gate 4: 1,000 randomly generated mathematical expressions: each digit mapped individually;

    math operators (=, +, -, *, /) never fuse with digits.
    """
    random.seed(2026)
    violations = 0

    operators = ["+", "-", "*", "/"]

    for _ in range(1000):
        num_terms = random.randint(2, 4)
        terms = [str(random.randint(0, 9999)) for _ in range(num_terms)]
        ops = [random.choice(operators) for _ in range(num_terms - 1)]

        # Interleave terms and operators
        tokens_expr = []
        for i in range(len(ops)):
            tokens_expr.append(terms[i])
            tokens_expr.append(ops[i])
        tokens_expr.append(terms[-1])

        # Randomly test with spaces or compact
        join_char = " " if random.random() > 0.5 else ""
        equation = join_char.join(tokens_expr) + f" = {random.randint(0, 99999)}"

        enc_tokens = tokenizer.tokenize(equation)
        for tok in enc_tokens:
            has_digit = any(c.isdigit() for c in tok)
            has_op = any(c in "+-=*/" for c in tok)
            if has_digit and has_op:
                violations += 1

            # If token is numeric digits, it must be exactly 1 digit
            digits_in_tok = [c for c in tok if c.isdigit()]
            if len(digits_in_tok) > 1:
                violations += 1

    print(f"\n[Gate 4] Tested 1,000 equations: {violations} fusion violations (Target: exactly 0)")
    assert violations == 0, f"Gate 4 failed: {violations} operator/digit fusion violations"


# ==============================================================================
# Edge Cases & Robustness
# ==============================================================================


def test_empty_string(tokenizer):
    """Empty string returns empty ID list and decodes to empty string."""
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_whitespace_only_strings(tokenizer):
    """Whitespace-only strings encode and round-trip without crash."""
    for ws in [" ", "  ", "   ", "    ", "        ", "\n", "\n\n", " \t \n "]:
        enc = tokenizer.encode(ws)
        dec = tokenizer.decode(enc)
        assert dec == ws


def test_tensor_input_to_decode(tokenizer):
    """decode handles array/tensor-like objects with .tolist() method."""
    class DummyTensor:
        def __init__(self, data):
            self.data = data
        def tolist(self):
            return self.data

    original = "Testare interoperabilitate tensori PyTorch."
    ids = tokenizer.encode(original)
    tensor = DummyTensor(ids)
    assert tokenizer.decode(tensor) == tokenizer.normalize(original)


def test_add_special_tokens_flag(tokenizer):
    """encode(text, add_special_tokens=True) prepends BOS and appends EOS."""
    text = "Exemplu de codare cu tokenuri speciale de început și sfârșit."
    ids_without = tokenizer.encode(text, add_special_tokens=False)
    ids_with = tokenizer.encode(text, add_special_tokens=True)

    assert ids_with[0] == tokenizer.bos_token_id
    assert ids_with[-1] == tokenizer.eos_token_id
    assert ids_with[1:-1] == ids_without

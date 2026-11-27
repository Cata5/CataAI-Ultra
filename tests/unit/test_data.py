"""Unit test suite for Milestone 3 Data Pipeline (RoCanon-48.5M).

Tests cover:
- Language Identification (LID): Romanian, English, non-target languages, code/math bypass.
- Text quality heuristics: HTML/boilerplate stripping, character/word/line repetition,
  punctuation noise, length bounds.
- PII Masking: Emails, phone numbers, Romanian CNP (century, month, day validation & negative controls),
  API keys (OpenAI, GitHub, AWS, Bearer tokens).
- MinHash LSH deduplication: SHA-256 exact match, MinHash near-duplicates (0.80 Jaccard), distinct docs.
- Benchmark decontamination: 13-gram exact match filtering and boilerplate precedence.
- Dynamic data mixing: Stage 1 and Stage 2 ratios, deterministic sampling, Romanian epoch capping.
- Flat uint16 serialization & PackedMemmapDataset: zero-copy memmap, padding-free packing,
  strictly monotonic cu_seqlens, out-of-bounds and corrupted file protection.
- Token-mean loss normalization.
"""

from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from src.data.curation import (
    MinHashLSHDeduplicator,
    TextCurator,
    check_text_quality,
    classify_language_statistical,
    is_valid_cnp,
    mask_pii_and_count,
    mask_pii_in_text,
    strip_html_and_boilerplate,
)
from src.data.dataset import (
    PackedMemmapDataset,
    build_cu_seqlens,
    compute_token_mean_loss,
    pack_sequences,
    serialize_tokens_to_shard,
)
from src.data.mixer import (
    DataMixer,
    MixingRatios,
)
from src.data.text_repair import normalize_training_text, repair_mojibake
from src.tokenizer.chat_template import ChatTemplateRenderer

# ============================================================================
# 1. LANGUAGE IDENTIFICATION (LID) & STATISTICAL CLASSIFIER TESTS
# ============================================================================


class TestLanguageIdentification:
    """Verifies language identification for target languages ('ron', 'eng') and rejection of non-target languages."""

    @pytest.fixture
    def curator(self) -> TextCurator:
        return TextCurator(lid_threshold=0.80)

    def test_lid_romanian_prose(self, curator: TextCurator):
        ro_sample = (
            "Modelul RoCanon-48.5M reprezintă o inovație în domeniul modelelor de limbaj compacte "
            "pentru limba română și limba engleză. Acest sistem utilizează rețele neuronale profunde "
            "pentru a asigura o înțelegere avansată a textelor educaționale și tehnice."
        )
        assert curator.filter_language(ro_sample) is True
        lang, conf = classify_language_statistical(ro_sample)
        assert lang in ("ron", "ro")
        assert conf >= 0.80

    def test_lid_english_prose(self, curator: TextCurator):
        en_sample = (
            "The transformer architecture relies on grouped-query attention and rotary position embeddings "
            "to process educational text sequences efficiently on consumer hardware. This model is trained "
            "from scratch using a curated bilingual dataset."
        )
        assert curator.filter_language(en_sample) is True
        lang, conf = classify_language_statistical(en_sample)
        assert lang in ("eng", "en")
        assert conf >= 0.80

    def test_lid_non_target_french(self, curator: TextCurator):
        fr_sample = (
            "Bonjour tout le monde. Ceci est un texte d'exemple rédigé en langue française "
            "pour vérifier que notre filtre d'identification linguistique rejette correctement "
            "les documents qui ne font pas partie des langues cibles du projet."
        )
        assert curator.filter_language(fr_sample) is False

    def test_lid_non_target_german(self, curator: TextCurator):
        de_sample = (
            "Guten Tag allerseits. Dies ist ein deutscher Beispielsatz für die Überprüfung "
            "unseres Sprachidentifikationsmodells. Diese Sprache gehört nicht zu den Zielsprachen "
            "des RoCanon Projekts und muss daher abgelehnt werden."
        )
        assert curator.filter_language(de_sample) is False

    def test_lid_non_target_cyrillic(self, curator: TextCurator):
        ru_sample = (
            "Привет мир. Это пример предложения на русском языке с использованием кириллических "
            "символов для тестирования фильтра языковой идентификации."
        )
        assert curator.filter_language(ru_sample) is False

    def test_lid_non_target_chinese(self, curator: TextCurator):
        zh_sample = "这是一个用于测试语言识别过滤器的中文测试句子。本项目仅支持罗马尼亚语和英语。"
        assert curator.filter_language(zh_sample) is False

    def test_lid_empty_and_whitespace(self, curator: TextCurator):
        assert curator.filter_language("") is False
        assert curator.filter_language("   \n\t  ") is False

    def test_code_and_math_bypass_lid(self, curator: TextCurator):
        code_sample = (
            "def calculate_fibonacci(n: int) -> int:\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        )
        # Without bypass flag, code might fail LID
        validated_code = curator.clean_and_validate(code_sample, is_code_or_math=True)
        assert validated_code is not None
        assert "def calculate_fibonacci" in validated_code

    def test_lid_romanian_nfd_decomposed_unicode(self, curator: TextCurator):
        """Verifies that Romanian text decomposed to Unicode NFD is correctly normalized to NFC and passes LID >= 0.80."""
        nfc_romanian = "Școala și învățământul din România oferă oportunități de dezvoltare pentru tineri în știință și tehnologie."
        # Decompose into NFD (combining characters)
        nfd_romanian = unicodedata.normalize("NFD", nfc_romanian)
        assert nfd_romanian != nfc_romanian  # Confirm it is actually decomposed

        # Direct statistical classifier test
        lang, conf = classify_language_statistical(nfd_romanian)
        assert lang in ("ron", "ro")
        assert conf >= 0.80

        # Filter language test
        assert curator.filter_language(nfd_romanian) is True

        # Full clean_and_validate test
        cleaned = curator.clean_and_validate(nfd_romanian)
        assert cleaned is not None
        # Assert result is canonical NFC
        assert unicodedata.is_normalized("NFC", cleaned)
        assert "Școala și învățământul din România" in cleaned


# ============================================================================
# 2. TEXT QUALITY & BOILERPLATE HEURISTICS TESTS
# ============================================================================


class TestQualityHeuristics:
    """Verifies boilerplate stripping, repetition detection, length checks, and punctuation density bounds."""

    def test_html_tag_and_entity_stripping(self):
        raw_html = (
            "<div class='content'><h1>Titlu Educațional</h1><p>Acesta este un paragraf &amp; un exemplu "
            "cu spații&nbsp;suplimentare și <a href='https://example.com'>linkuri</a> utile.</p></div>"
        )
        cleaned = strip_html_and_boilerplate(raw_html)
        assert "<div" not in cleaned
        assert "<p>" not in cleaned
        assert "&amp;" not in cleaned
        assert "&nbsp;" not in cleaned
        assert "Titlu Educațional" in cleaned
        assert (
            "Acesta este un paragraf & un exemplu cu spații suplimentare și linkuri utile."
            in cleaned
        )

    def test_character_repetition_rejection(self):
        # Repeated characters (e.g. 15 'a's)
        corrupt_text = (
            "Acest text conține zgomot grav "
            + ("a" * 20)
            + " care trebuie eliminat din dataset."
        )
        assert check_text_quality(corrupt_text) is False

    def test_consecutive_word_repetition_rejection(self):
        # 5 consecutive identical words
        repetitive_text = (
            "Acest paragraf conține o repetiție nepermisă de cuvinte: test test test test test "
            "care indică o problemă de calitate în textul prelucrat."
        )
        assert check_text_quality(repetitive_text) is False

    def test_top_ngram_concentration_rejection(self):
        # 30+ words where 'spam' is repeated over 30% of the time
        words = ["spam"] * 15 + [
            "cuvânt",
            "educațional",
            "românesc",
            "text",
            "informație",
        ] * 4
        spam_text = " ".join(words)
        assert len(words) >= 30
        assert check_text_quality(spam_text) is False

    def test_excessive_punctuation_density_rejection(self):
        noisy_text = "!!!!????....####$$$$%%%%^^^^&&&&**** Text cu prea multe simboluri @@@@%%%%^^^^"
        assert check_text_quality(noisy_text) is False

    def test_length_threshold_rejection(self):
        # Too short (< 50 chars)
        short_text = "Text prea scurt."
        assert check_text_quality(short_text) is False

        # Valid length natural text (>= 50 chars and >= 5 words)
        valid_text = (
            "Acesta este un text complet valid din punct de vedere al lungimii "
            "și al structurii propozițiilor pentru pre-antrenare."
        )
        assert check_text_quality(valid_text) is True

    def test_duplicate_line_ratio_rejection(self):
        lines = [
            "Această linie este reprodusă identic.",
            "Această linie este reprodusă identic.",
            "Această linie este reprodusă identic.",
            "Această linie este reprodusă identic.",
            "Linie unică unu.",
            "Linie unică doi.",
        ]
        multi_line_text = "\n".join(lines)
        # 4 out of 6 lines are duplicate -> dup ratio > 50%
        assert check_text_quality(multi_line_text) is False


# ============================================================================
# 3. PII MASKING & ROMANIAN CNP VALIDATION TESTS
# ============================================================================


class TestPIIMasking:
    """Verifies redaction of emails, phone numbers, valid Romanian CNP, and API keys,

    with negative controls ensuring timestamps and random numbers are not masked.
    """

    def test_email_masking(self):
        text = "Contactați-ne la contact@universitate.ro sau la admin.office+help@domain.co.uk pentru detalii."
        masked = mask_pii_in_text(text)
        assert masked == "Contactați-ne la [EMAIL] sau la [EMAIL] pentru detalii."

    def test_mask_and_count_single_pass(self):
        text = (
            "Contact: ana@example.ro, telefon 0722123456, "
            "CNP 1850514410015, cheie sk-abc12345def67890ghij12345klmn67890."
        )
        masked, counts = mask_pii_and_count(text)
        assert masked == mask_pii_in_text(text)
        assert counts == {"email": 1, "phone": 1, "cnp": 1, "api_key": 1}
        assert "ana@example.ro" not in masked
        assert "1850514410015" not in masked

    def test_romanian_phone_masking(self):
        text = "Sunați la numărul de mobil 0722123456 sau la fix +40 21 234 5678 pentru asistență tehnică."
        masked = mask_pii_in_text(text)
        assert "[PHONE]" in masked
        assert "0722123456" not in masked
        assert "+40 21 234 5678" not in masked

    def test_international_phone_masking(self):
        text = "Office support: +1-555-234-5678 or +44 20 7946 0958 for international queries."
        masked = mask_pii_in_text(text)
        assert "[PHONE]" in masked
        assert "+1-555-234-5678" not in masked

    def test_romanian_cnp_valid_masking(self):
        # Male born 1985 May 14 in Bucharest (sector 1): 1850514410015
        male_cnp = (
            "Date personale: Popescu Ion, CNP 1850514410015, născut la București."
        )
        masked_male = mask_pii_in_text(male_cnp)
        assert "CNP [CNP]" in masked_male
        assert "1850514410015" not in masked_male

        # Female born 2002 Nov 28 in Cluj: 6021128120024
        female_cnp = "Identificator: Ionescu Maria, CNP 6021128120024, studentă."
        masked_female = mask_pii_in_text(female_cnp)
        assert "CNP [CNP]" in masked_female
        assert "6021128120024" not in masked_female

        # Male resident / foreign citizen born 1990: 7900315400031
        resident_cnp = "Rezident: John Doe, CNP 7900315400031."
        masked_resident = mask_pii_in_text(resident_cnp)
        assert "CNP [CNP]" in masked_resident
        assert "7900315400031" not in masked_resident

    def test_romanian_cnp_negative_controls(self):
        # 13-digit Unix millisecond timestamp: e.g. 1609459200000 (year 60, month 94 -> invalid month)
        timestamp_text = "Timestamp tranzacție: 1609459200000 milisecunde de la Epoch."
        masked_ts = mask_pii_in_text(timestamp_text)
        assert "1609459200000" in masked_ts
        assert "[CNP]" not in masked_ts

        # Leading 0 (e.g. 0123456789012) -> invalid century/sex digit
        zero_leading = "Cod de bare: 0123456789012 verificat la casă."
        masked_zero = mask_pii_in_text(zero_leading)
        assert "0123456789012" in masked_zero
        assert "[CNP]" not in masked_zero

        # Invalid month 13
        invalid_month = "Număr eronat 1851314120015 cu luna treisprezece."
        assert is_valid_cnp("1851314120015") is False
        assert "1851314120015" in mask_pii_in_text(invalid_month)

        # Invalid day 32
        invalid_day = "Număr eronat 1850532120015 cu ziua treizeci și doi."
        assert is_valid_cnp("1850532120015") is False
        assert "1850532120015" in mask_pii_in_text(invalid_day)

    def test_api_keys_masking(self):
        openai_key = "Cheie: sk-abc12345def67890ghij12345klmn67890"
        assert mask_pii_in_text(openai_key) == "Cheie: [API_KEY]"

        github_pat = "Token personal: ghp_1234567890abcdef1234567890abcdef1234"
        assert mask_pii_in_text(github_pat) == "Token personal: [API_KEY]"

        aws_key = "AWS Access Key: AKIAIOSFODNN7EXAMPLE configurat în credentials."
        assert (
            mask_pii_in_text(aws_key)
            == "AWS Access Key: [API_KEY] configurat în credentials."
        )

        bearer_header = (
            "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token"
        )
        assert mask_pii_in_text(bearer_header) == "Authorization: Bearer [API_KEY]"

    def test_api_keys_dict_and_json_structures_masking(self):
        """Verifies masking of API keys in quoted JSON and Python dictionary formats."""
        # Single-quoted dictionary format
        dict_text = "config = {'api_key': 'super_secret_token_1234567890_abcdef'}"
        masked_dict = mask_pii_in_text(dict_text)
        assert masked_dict == "config = {'api_key': '[API_KEY]'}"
        assert "super_secret_token_1234567890_abcdef" not in masked_dict

        # Double-quoted JSON format
        json_text = '{"client_secret": "my_secret_token_value_9876543210"}'
        masked_json = mask_pii_in_text(json_text)
        assert masked_json == '{"client_secret": "[API_KEY]"}'
        assert "my_secret_token_value_9876543210" not in masked_json

        # access_token and auth_token formats
        access_tok = 'payload = {"access_token": "token_value_1234567890_abcdefghij"}'
        assert mask_pii_in_text(access_tok) == 'payload = {"access_token": "[API_KEY]"}'

        auth_tok = "settings = {'auth_token': 'auth_tok_12345678901234567890'}"
        assert mask_pii_in_text(auth_tok) == "settings = {'auth_token': '[API_KEY]'}"

        secret_key = 'export SECRET_KEY="sec_key_12345678901234567890"'
        assert mask_pii_in_text(secret_key) == 'export SECRET_KEY="[API_KEY]"'

        # Unquoted YAML/assignment format
        yaml_text = "api_key: super_secret_token_1234567890_abcdef"
        assert mask_pii_in_text(yaml_text) == "api_key: [API_KEY]"

        # Multi-key dictionary with mixed secrets and public fields
        mixed_dict = (
            "credentials = {\n"
            "    'api_key': 'key_12345678901234567890',\n"
            '    "client_secret": "sec_12345678901234567890",\n'
            '    "public_name": "public_dataset_v1",\n'
            "}"
        )
        masked_mixed = mask_pii_in_text(mixed_dict)
        assert "'api_key': '[API_KEY]'" in masked_mixed
        assert '"client_secret": "[API_KEY]"' in masked_mixed
        assert '"public_name": "public_dataset_v1"' in masked_mixed
        assert "key_12345678901234567890" not in masked_mixed
        assert "sec_12345678901234567890" not in masked_mixed


# ============================================================================
# 4. MINHASH LSH DEDUPLICATION TESTS
# ============================================================================


class TestMinHashDeduplication:
    """Verifies SHA-256 exact deduplication and MinHash LSH near-duplicate removal with low RAM consumption."""

    def test_exact_duplicate_elimination(self):
        dedup = MinHashLSHDeduplicator(jaccard_threshold=0.80)
        doc1 = "Document educațional complet identic pentru testarea deduplicării exacte SHA-256."
        doc2 = "Document educațional complet identic pentru testarea deduplicării exacte SHA-256."

        assert dedup.is_duplicate(doc1) is False  # First seen -> unique
        assert dedup.is_duplicate(doc2) is True  # Second seen -> duplicate

    def test_near_duplicate_elimination(self):
        dedup = MinHashLSHDeduplicator(
            num_perm=128, k_shingle=5, jaccard_threshold=0.80
        )
        # Original article
        doc1 = (
            "București este capitala României și cel mai important centru economic, cultural "
            "și universitar din țară. Orașul găzduiește numeroase instituții de învățământ superior "
            "și monumente istorice remarcabile din patrimoniul național."
        )
        # Near duplicate with only two words modified
        doc2 = (
            "București este capitala României și cel mai mare centru economic, cultural "
            "și universitar din țară. Orașul găzduiește multe instituții de învățământ superior "
            "și monumente istorice remarcabile din patrimoniul național."
        )

        assert dedup.is_duplicate(doc1) is False
        assert dedup.is_duplicate(doc2) is True

    def test_distinct_documents_preserved(self):
        dedup = MinHashLSHDeduplicator(
            num_perm=128, k_shingle=5, jaccard_threshold=0.80
        )
        doc1 = "Fizica cuantică studiază comportamentul materiei și energiei la nivel atomic și subatomic."
        doc2 = "Algoritmii de învățare automată optimizează parametrii ponderilor folosind gradient descent."

        assert dedup.is_duplicate(doc1) is False
        assert dedup.is_duplicate(doc2) is False

    def test_streaming_deduplication_generator(self):
        dedup = MinHashLSHDeduplicator(jaccard_threshold=0.80)
        docs = [
            "Text unic numărul unu despre matematică și algebră liniară aplicată.",
            "Text unic numărul doi despre istoria modernă a Europei centrale.",
            "Text unic numărul unu despre matematică și algebră liniară aplicată.",  # Duplicate of 0
            "Text unic numărul trei despre chimie organică și structuri moleculare.",
        ]
        unique_docs = list(dedup.deduplicate(docs))
        assert len(unique_docs) == 3
        assert unique_docs[0] == docs[0]
        assert unique_docs[1] == docs[1]
        assert unique_docs[2] == docs[3]


# ============================================================================
# 5. BENCHMARK DECONTAMINATION TESTS
# ============================================================================


class TestBenchmarkDecontamination:
    """Verifies 13-gram exact match filtering against evaluation benchmarks."""

    def test_13gram_contamination_detection(self):
        curator = TextCurator()
        benchmark_doc = (
            "Calculați derivata de ordinul doi a funcției exponențiale definite pe mulțimea numerelor reale "
            "și determinați punctele de inflexiune corespunzătoare."
        )
        curator.add_benchmark_documents([benchmark_doc])

        # Candidate text containing the exact 13-word benchmark sequence
        contaminated_doc = (
            "Iată o problemă din examen: calculați derivata de ordinul doi a funcției exponențiale "
            "definite pe mulțimea numerelor reale și determinați punctele de inflexiune corespunzătoare rezolvată."
        )
        assert curator.decontaminate(contaminated_doc) is False

        # Clean candidate text with no overlap
        clean_doc = (
            "Acest text descrie istoria calculului diferențial și integral dezvoltat de Isaac Newton "
            "și Gottfried Wilhelm Leibniz în secolul al șaptesprezecelea."
        )
        assert curator.decontaminate(clean_doc) is True

    def test_boilerplate_stripping_precedes_13gram_check(self):
        curator = TextCurator()
        benchmark_doc = (
            "Aceasta este o secvență de exact treisprezece cuvinte utilizată ca reper pentru testarea "
            "sistemului nostru de decontaminare."
        )
        curator.add_benchmark_documents([benchmark_doc])

        # Contaminated text wrapped in HTML tags
        html_contaminated = (
            "<p><b>Aceasta</b> este o <i>secvență</i> de exact treisprezece cuvinte utilizată "
            "ca reper pentru testarea sistemului nostru de decontaminare.</p>"
        )
        assert curator.clean_and_validate(html_contaminated) is None


# ============================================================================
# 6. DYNAMIC DATA MIXER & ROMANIAN EPOCH CAPPING TESTS
# ============================================================================


class TestDataMixer:
    """Verifies multi-stream mixing ratios, reproducible deterministic sampling,

    and strict Romanian epoch capping (<= 3-4 epochs).
    """

    def test_mixing_ratios_stage1_and_stage2(self):
        r1 = MixingRatios.stage1()
        assert sum(r1.to_dict().values()) == pytest.approx(1.0, abs=1e-5)
        assert r1.ro_edu == 0.35
        assert r1.en_edu == 0.30
        assert r1.textbooks == 0.15
        assert r1.math == 0.10
        assert r1.code == 0.10

        r2 = MixingRatios.stage2()
        assert sum(r2.to_dict().values()) == pytest.approx(1.0, abs=1e-5)
        assert r2.ro_edu == 0.25
        assert r2.en_edu == 0.15
        assert r2.textbooks == 0.15
        assert r2.math == 0.25
        assert r2.code == 0.20

    def test_deterministic_sampling_reproducibility(self):
        streams_a = {
            "ro_edu": [f"ro_{i}" for i in range(10)],
            "en_edu": [f"en_{i}" for i in range(10)],
            "textbooks": [f"tb_{i}" for i in range(10)],
            "math": [f"m_{i}" for i in range(10)],
            "code": [f"c_{i}" for i in range(10)],
        }
        streams_b = {k: list(v) for k, v in streams_a.items()}

        mixer1 = DataMixer(streams_a, seed=12345)
        mixer2 = DataMixer(streams_b, seed=12345)

        docs1 = [next(mixer1.stream_documents()) for _ in range(50)]
        docs2 = [next(mixer2.stream_documents()) for _ in range(50)]

        assert docs1 == docs2

        # Different seed produces different sequence
        mixer3 = DataMixer(streams_b, seed=99999)
        docs3 = [next(mixer3.stream_documents()) for _ in range(50)]
        assert docs1 != docs3

    def test_romanian_epoch_capping(self):
        # ro_edu has 3 items. max_ro_epochs=3 -> ro_edu can yield at most 3 * 3 = 9 items.
        streams = {
            "ro_edu": ["ro_1", "ro_2", "ro_3"],
            "en_edu": ["en_1", "en_2"],
            "math": ["m_1", "m_2"],
        }
        mixer = DataMixer(streams, max_ro_epochs=3, seed=42)

        yielded_items = []
        gen = mixer.stream_documents()
        for _ in range(100):
            try:
                item = next(gen)
                yielded_items.append(item)
            except StopIteration:
                break

        ro_count = sum(1 for dom, _ in yielded_items if dom == "ro_edu")
        assert ro_count <= 9, (
            f"Romanian stream exceeded max epoch limit! Got {ro_count} items."
        )
        assert mixer.trackers["ro_edu"].is_exhausted is True
        # Other streams continued to yield items
        assert len(yielded_items) == 100


# ============================================================================
# 7. BINARY SHARDS, PACKING & ZERO-COPY MEMMAP DATASET TESTS
# ============================================================================


class TestBinarySerializationAndDataset:
    """Verifies uint16 flat serialization, sequence packing without padding,

    monotonic cu_seqlens, and zero-copy np.memmap reading.
    """

    @pytest.fixture
    def sample_shards(self, tmp_path: Path) -> tuple[str, str, int]:
        bin_path = str(tmp_path / "test_shard.bin")
        idx_path = str(tmp_path / "test_shard.idx")

        # 4 documents with varying lengths
        token_docs = [
            list(range(100, 120)),  # 20 tokens
            list(range(200, 230)),  # 30 tokens
            list(range(300, 345)),  # 45 tokens
            list(range(400, 433)),  # 33 tokens
        ]
        # Total tokens = 128 (exact multiple of seq_len=64)
        total = serialize_tokens_to_shard(token_docs, bin_path, idx_path)
        return bin_path, idx_path, total

    def test_uint16_binary_serialization(self, sample_shards: tuple[str, str, int]):
        bin_path, idx_path, total_tokens = sample_shards
        assert total_tokens == 128

        # Verify uint16 binary format on disk (2 bytes per token)
        size_bytes = os.path.getsize(bin_path)
        assert size_bytes == 128 * 2

        # Verify index file (int64 document offsets)
        offsets = np.fromfile(idx_path, dtype=np.int64)
        assert len(offsets) == 5
        assert offsets.tolist() == [0, 20, 50, 95, 128]

        # Read back uint16 data
        data = np.fromfile(bin_path, dtype=np.uint16)
        assert data.dtype == np.uint16
        assert data[0] == 100
        assert data[19] == 119
        assert data[20] == 200

    def test_pack_sequences_chunk_length_and_boundaries(self):
        docs = [
            list(range(1, 21)),  # 20 tokens
            list(range(21, 51)),  # 30 tokens
            list(range(51, 81)),  # 30 tokens
        ]
        # Pack into seq_len=32 with eos_token_id=1
        packed_chunks = list(pack_sequences(docs, seq_len=32, eos_token_id=1))
        assert len(packed_chunks) > 0

        for chunk_tokens, boundaries in packed_chunks:
            assert len(chunk_tokens) == 32
            assert boundaries[0] == 0
            assert boundaries[-1] == 32
            # Strictly monotonic
            diffs = [
                boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)
            ]
            assert all(d > 0 for d in diffs)

    def test_build_cu_seqlens_utility(self):
        doc_lengths = [25, 35, 40]
        cu = build_cu_seqlens(doc_lengths)
        assert cu.dtype == torch.int32
        assert cu.dim() == 1
        assert cu.tolist() == [0, 25, 60, 100]

    def test_packed_memmap_dataset_reading(self, sample_shards: tuple[str, str, int]):
        bin_path, idx_path, total_tokens = sample_shards
        seq_len = 64

        with PackedMemmapDataset(bin_path, idx_path, seq_len=seq_len) as dataset:
            assert len(dataset) == total_tokens // seq_len  # 128 // 64 = 2

            # Test dict access
            item0 = dataset[0]
            assert isinstance(item0, dict)
            assert "input_ids" in item0
            assert "labels" in item0
            assert "cu_seqlens" in item0

            assert item0["input_ids"].shape == (seq_len,)
            assert item0["labels"].shape == (seq_len,)
            assert item0["input_ids"].dtype == torch.long
            assert item0["labels"].dtype == torch.long
            assert item0["cu_seqlens"].dtype == torch.int32

            # Test 3-tuple unpacking
            input_ids, labels, cu_seqlens = dataset[0]
            assert input_ids.shape == (seq_len,)
            assert labels.shape == (seq_len,)
            assert cu_seqlens.dim() == 1
            assert cu_seqlens[0].item() == 0
            assert cu_seqlens[-1].item() == seq_len
            assert torch.all(torch.diff(cu_seqlens) > 0), (
                "cu_seqlens must be strictly monotonic"
            )

    def test_packed_memmap_dataset_out_of_bounds_errors(
        self, sample_shards: tuple[str, str, int]
    ):
        bin_path, idx_path, _ = sample_shards
        with PackedMemmapDataset(bin_path, idx_path, seq_len=64) as dataset:
            with pytest.raises(IndexError):
                _ = dataset[2]
            with pytest.raises(IndexError):
                _ = dataset[-1]

    def test_packed_memmap_dataset_invalid_file_size_error(self, tmp_path: Path):
        corrupt_bin = str(tmp_path / "corrupt.bin")
        # 15 bytes is not a multiple of 2 * 64 = 128
        with open(corrupt_bin, "wb") as f:
            f.write(b"x" * 15)

        with pytest.raises(ValueError):
            _ = PackedMemmapDataset(corrupt_bin, seq_len=64)

    def test_token_mean_loss_calculation(self):
        # Per-token loss tensor of 4 elements
        per_token_loss = torch.tensor([1.5, 2.5, 3.0, 1.0])
        valid_tokens = 4
        norm_loss = compute_token_mean_loss(per_token_loss, valid_tokens=valid_tokens)
        assert norm_loss.item() == pytest.approx(2.0, abs=1e-5)

        # Ignore index filtering with labels
        labels = torch.tensor([10, -100, 20, 30])
        loss_with_labels = compute_token_mean_loss(
            per_token_loss, labels=labels, ignore_index=-100
        )
        # Sum = 8.0, valid count = 3 -> 8.0 / 3 = 2.6666...
        assert loss_with_labels.item() == pytest.approx(8.0 / 3.0, abs=1e-5)


class TestTrainingTextRepair:
    """Regression tests for corrupted instruction data before tokenization."""

    def test_repairs_utf8_decoded_as_cp1252(self):
        broken = "ClasificaÈ›i urmÄƒtoarele fotografii. Bun venit la jocul â€žPatru la rÃ¢ndâ€\x9d."
        assert repair_mojibake(broken) == (
            "Clasificați următoarele fotografii. Bun venit la jocul „Patru la rând”."
        )

    def test_strips_nan_but_keeps_words_containing_nan(self):
        assert normalize_training_text("Salut\nnan\n<|end_turn|>") == "Salut\n\n<|end_turn|>"
        assert "Banană" in normalize_training_text("Banană")

    def test_clean_text_is_unchanged(self):
        clean = "Clasificați următoarele fotografii. Bun venit la jocul „Patru la rând”."
        assert repair_mojibake(clean) == clean

    def test_cleaning_pipeline_removes_nan_and_repairs_mojibake(self):
        cleaned = strip_html_and_boilerplate(
            "ClasificaÈ›i urmÄƒtoarele fotografii.\nnan\nRezultatul este corect."
        )
        assert "Clasificați următoarele fotografii." in cleaned
        assert "\nnan\n" not in cleaned


class TestRoCanonChatTemplate:
    """The API and evaluation path must use the markers seen during SFT."""

    def test_generation_prompt_matches_sft_layout(self):
        renderer = ChatTemplateRenderer("rocanon")
        assert renderer.render(
            [{"role": "user", "content": "Cine esti?"}], add_generation_prompt=True
        ) == "<|user|>\nCine esti?\n<|assistant|>\n"

    def test_assistant_history_ends_with_native_turn_token(self):
        renderer = ChatTemplateRenderer("rocanon")
        rendered = renderer.render(
            [
                {"role": "user", "content": "Salut"},
                {"role": "assistant", "content": "Salut!"},
            ]
        )
        assert rendered == "<|user|>\nSalut\n<|assistant|>\nSalut!<|end_turn|>\n"

"""
src/eval/benchmarks.py
Benchmark evaluation suite for RoCanon-48.5M.
Includes standard benchmark categories: Romanian Language, Creator Persona (grounded in
configured facts, never fabricated), Hardware Knowledge (RTX 3060 & PC architecture),
Programming, Mathematics, and Instruction Following.
Calculates fact coverage, repetition loops, and instruction compliance metrics.
"""

from __future__ import annotations

import collections
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..db.database import StudioDatabase
from .inference import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkItem:
    id: str
    category: str
    prompt: str
    expected_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None
    compliance_rule: Optional[str] = None  # e.g., "single_word", "numbered_list", "starts_with"


# Configured Ground Truth Reference Facts for Creator Persona (Manu Cătălin)
# Grounded strictly in project metadata, zero hallucination
DEFAULT_CREATOR_PERSONA_BENCHMARKS: List[BenchmarkItem] = [
    BenchmarkItem(
        id="persona_1",
        category="creator_persona",
        prompt="Cine este creatorul modelului RoCanon-48.5M?",
        expected_keywords=["Manu Cătălin", "Cătălin"],
        reference_answer="Creatorul modelului RoCanon-48.5M este Manu Cătălin.",
    ),
    BenchmarkItem(
        id="persona_2",
        category="creator_persona",
        prompt="Care este viziunea lui Manu Cătălin pentru modelul RoCanon?",
        expected_keywords=["Small Language Model", "românesc", "eficient", "local"],
        reference_answer="Viziunea este dezvoltarea unui Small Language Model bilingv română-engleză de înaltă eficiență, capabil să ruleze local pe GPU-uri consumer de 6GB.",
    ),
    BenchmarkItem(
        id="persona_3",
        category="creator_persona",
        prompt="Pe ce hardware a proiectat Manu Cătălin arhitectura RoCanon-48.5M?",
        expected_keywords=["RTX 3060", "6GB", "laptop"],
        reference_answer="Arhitectura a fost concepută și optimizată special pentru hardware consumer cu buget restrâns de VRAM, în special NVIDIA RTX 3060 Laptop (6GB VRAM).",
    ),
    BenchmarkItem(
        id="persona_4",
        category="creator_persona",
        prompt="Ce limbaj și framework folosește proiectul lui Manu Cătălin?",
        expected_keywords=["Python", "PyTorch"],
        reference_answer="Proiectul folosește Python și PyTorch cu nuclee personalizate, WSD scheduling și optimizator hibrid Muon + AdamW.",
    ),
    BenchmarkItem(
        id="persona_5",
        category="creator_persona",
        prompt="Ce tip de tokenizer a fost creat pentru RoCanon?",
        expected_keywords=["Byte-Level BPE", "12288", "diacritice"],
        reference_answer="A fost conceput un tokenizer custom Byte-Level BPE cu vocabular de 12.288 tokeni și suport nativ pentru diacritice românești (ș, ț cu virgulă dedesubt).",
    ),
]

# Hardware Knowledge Benchmarks (RTX 3060 & PC Architecture)
DEFAULT_HARDWARE_BENCHMARKS: List[BenchmarkItem] = [
    BenchmarkItem(
        id="hw_1",
        category="hardware_knowledge",
        prompt="Câtă memorie VRAM are placa video NVIDIA GeForce RTX 3060 Laptop GPU?",
        expected_keywords=["6 GB", "6GB", "6 gigaocteți", "6 gigabytes"],
        reference_answer="NVIDIA GeForce RTX 3060 Laptop GPU are 6GB GDDR6 VRAM.",
    ),
    BenchmarkItem(
        id="hw_2",
        category="hardware_knowledge",
        prompt="Ce lățime de bandă de magistrală (bus width) are memoria plăcii RTX 3060 Laptop?",
        expected_keywords=["192-bit", "192 bit"],
        reference_answer="RTX 3060 Laptop GPU folosește o magistrală de memorie pe 192 de biți.",
    ),
    BenchmarkItem(
        id="hw_3",
        category="hardware_knowledge",
        prompt="De ce este important formatul bfloat16 pe arhitectura NVIDIA Ampere?",
        expected_keywords=["dynamic range", "FP32", "Tensor Core", "stabilitate", "overflow"],
        reference_answer="Bfloat16 are același dynamic range ca FP32 (8 biți de exponent), prevenind underflow/overflow și permițând antrenare stabilă pe Tensor Cores fără GradScaler.",
    ),
    BenchmarkItem(
        id="hw_4",
        category="hardware_knowledge",
        prompt="Ce rol au Tensor Cores în accelerarea rețelelor neuronale?",
        expected_keywords=["multiplicare", "matrice", "GEMM", "operații"],
        reference_answer="Tensor Cores sunt unități hardware specializate pentru multiplicări matriciale rapide (GEMM) în precizie mixtă.",
    ),
    BenchmarkItem(
        id="hw_5",
        category="hardware_knowledge",
        prompt="Care este diferența dintre VRAM alocat (allocated) și VRAM rezervat (reserved) în PyTorch?",
        expected_keywords=["caching allocator", "tensori", "memorie", "rezervat"],
        reference_answer="Memoria alocată este ocupată de tensori activi, în timp ce memoria rezervată include cache-ul gestionat de PyTorch Caching Allocator pentru a evita apeluri lente de sistem.",
    ),
]

# Romanian Language Benchmarks
DEFAULT_ROMANIAN_BENCHMARKS: List[BenchmarkItem] = [
    BenchmarkItem(
        id="ro_1",
        category="romanian_understanding",
        prompt="Scrie corect cu diacritice românești propoziția: In padure cresc ciuperci si copaci batrani.",
        expected_keywords=["În pădure", "și", "copaci bătrâni"],
        reference_answer="În pădure cresc ciuperci și copaci bătrâni.",
    ),
    BenchmarkItem(
        id="ro_2",
        category="romanian_understanding",
        prompt="Care sunt cele cinci litere cu diacritice specifice limbii române?",
        expected_keywords=["ă", "â", "î", "ș", "ț"],
        reference_answer="Cele cinci litere sunt: ă, â, î, ș, ț.",
    ),
]

# Programming & Math Benchmarks
DEFAULT_CODE_MATH_BENCHMARKS: List[BenchmarkItem] = [
    BenchmarkItem(
        id="code_1",
        category="programming",
        prompt="Scrie o funcție Python care calculează numărul de parametri dintr-un model PyTorch.",
        expected_keywords=["def", "sum", "parameters", "numel"],
        reference_answer="def count_params(model):\n    return sum(p.numel() for p in model.parameters() if p.requires_grad)",
    ),
    BenchmarkItem(
        id="math_1",
        category="mathematics",
        prompt="Dacă un model are 16 straturi și fiecare strat conține 2 matrice de proiecție, câte matrice sunt în total?",
        expected_keywords=["32"],
        reference_answer="16 * 2 = 32 matrice în total.",
    ),
]


class BenchmarkManager:
    """Manages benchmark collections, execution, and evaluation metric scoring."""

    def __init__(self, db: Optional[StudioDatabase] = None) -> None:
        self.db = db or StudioDatabase()
        self.items: Dict[str, BenchmarkItem] = {}
        # Register defaults
        for item in (
            DEFAULT_CREATOR_PERSONA_BENCHMARKS
            + DEFAULT_HARDWARE_BENCHMARKS
            + DEFAULT_ROMANIAN_BENCHMARKS
            + DEFAULT_CODE_MATH_BENCHMARKS
        ):
            self.items[item.id] = item

    def get_categories(self) -> List[str]:
        return sorted(list({item.category for item in self.items.values()}))

    def get_items_by_category(self, category: str) -> List[BenchmarkItem]:
        return [item for item in self.items.values() if item.category == category]

    def add_custom_item(self, item: BenchmarkItem) -> None:
        self.items[item.id] = item

    def evaluate_response(self, item: BenchmarkItem, response_text: str) -> Dict[str, Any]:
        """Calculates granular objective evaluation metrics for a generated response."""
        resp = response_text.strip()
        resp_lower = resp.lower()

        # 1. Expected fact / keyword coverage
        expected_hits = []
        for kw in item.expected_keywords:
            if kw.lower() in resp_lower:
                expected_hits.append(kw)

        coverage_score = (
            len(expected_hits) / len(item.expected_keywords)
            if item.expected_keywords
            else 1.0
        )

        # 2. Forbidden keywords check
        forbidden_hits = [
            fkw for fkw in item.forbidden_keywords
            if fkw.lower() in resp_lower
        ]

        # 3. Repetition loop detection (degenerate n-gram repetition)
        words = resp.split()
        repetition_detected = False
        repetition_ratio = 0.0
        if len(words) >= 6:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            counts = collections.Counter(bigrams)
            if counts:
                max_count = counts.most_common(1)[0][1]
                repetition_ratio = max_count / len(bigrams)
                if max_count >= 4 and repetition_ratio > 0.35:
                    repetition_detected = True

        # 4. Instruction compliance check
        compliance_score = 1.0
        if item.compliance_rule == "single_word":
            compliance_score = 1.0 if len(words) == 1 else 0.0
        elif item.compliance_rule == "numbered_list":
            compliance_score = 1.0 if bool(re.search(r"^\d+\.", resp, re.MULTILINE)) else 0.0

        # Overall composite score in [0.0, 1.0]
        penalty = 0.5 if repetition_detected else 0.0
        if forbidden_hits:
            penalty += 0.5
        overall_score = max(0.0, min(1.0, coverage_score * 0.7 + compliance_score * 0.3 - penalty))

        return {
            "overall_score": round(overall_score, 2),
            "fact_coverage_score": round(coverage_score, 2),
            "matched_keywords": expected_hits,
            "total_expected_keywords": len(item.expected_keywords),
            "forbidden_hits": forbidden_hits,
            "repetition_detected": repetition_detected,
            "repetition_ratio": round(repetition_ratio, 3),
            "instruction_compliance": round(compliance_score, 2),
            "word_count": len(words),
        }

    def run_benchmark_suite(
        self,
        engine: InferenceEngine,
        categories: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        step: int = 0,
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Runs benchmarks on the active model and logs results to SQLite."""
        cats = categories or self.get_categories()
        selected_items = [item for item in self.items.values() if item.category in cats]

        results_by_cat: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        overall_scores: List[float] = []

        for item in selected_items:
            gen_res = engine.generate(
                prompt=item.prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.3,  # lower temp for factual evaluation
                top_p=0.9,
            )
            eval_metrics = self.evaluate_response(item, gen_res["generated_text"])
            score = eval_metrics["overall_score"]
            overall_scores.append(score)

            record = {
                "id": item.id,
                "prompt": item.prompt,
                "response": gen_res["generated_text"],
                "generation_time_sec": gen_res["generation_time_sec"],
                "tokens_per_sec": gen_res["tokens_per_sec"],
                "metrics": eval_metrics,
            }
            results_by_cat[item.category].append(record)

        summary_by_cat: Dict[str, float] = {}
        for cat, recs in results_by_cat.items():
            avg_score = sum(r["metrics"]["overall_score"] for r in recs) / max(1, len(recs))
            summary_by_cat[cat] = round(avg_score, 3)

            # Persist category benchmark result in database
            self.db.log_benchmark_result(
                category=cat,
                score=avg_score,
                run_id=run_id,
                checkpoint_id=checkpoint_id,
                step=step,
                results={"summary": summary_by_cat[cat], "items": recs},
            )

        mean_overall = sum(overall_scores) / max(1, len(overall_scores)) if overall_scores else 0.0

        return {
            "run_id": run_id,
            "step": step,
            "checkpoint_id": checkpoint_id,
            "mean_score": round(mean_overall, 3),
            "category_scores": summary_by_cat,
            "detailed_results": dict(results_by_cat),
        }


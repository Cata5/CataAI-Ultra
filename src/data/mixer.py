"""Dynamic multi-stream data mixer for RoCanon-48.5M Bilingual SLM.

Provides:
- MixingRatios dataclass supporting Stage 1 (35% RO, 30% EN, 15% text, 10% math, 10% code)
  and Stage 2 (25% RO, 15% EN, 15% text, 25% math, 20% code).
- DataMixer multi-stream interleaving generator with deterministic reproducible sampling via seed.
- Strict Romanian data exposure capping at <= 3-4 epochs to prevent redundancy saturation.
  Re-normalizes sampling weights dynamically among active streams once Romanian cap is reached.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass
class MixingRatios:
    """Proportional sampling weights across domain streams."""

    ro_edu: float = 0.35
    en_edu: float = 0.30
    textbooks: float = 0.15
    math: float = 0.10
    code: float = 0.10

    @classmethod
    def stage1(cls) -> MixingRatios:
        """Stage 1: General Pretraining (5.0B tokens).

        35% Romanian, 30% English, 15% Textbooks, 10% Math, 10% Code.
        """
        return cls(
            ro_edu=0.35,
            en_edu=0.30,
            textbooks=0.15,
            math=0.10,
            code=0.10,
        )

    @classmethod
    def stage2(cls) -> MixingRatios:
        """Stage 2: Mid-Training / WSD Decay (1.8B tokens).

        25% Romanian, 15% English, 15% Textbooks, 25% Math, 20% Code.
        """
        return cls(
            ro_edu=0.25,
            en_edu=0.15,
            textbooks=0.15,
            math=0.25,
            code=0.20,
        )

    def to_dict(self) -> dict[str, float]:
        """Returns domain-to-weight dictionary."""
        return {
            "ro_edu": self.ro_edu,
            "en_edu": self.en_edu,
            "textbooks": self.textbooks,
            "math": self.math,
            "code": self.code,
        }


class _StreamTracker:
    """Tracks document iterations, epoch boundaries, and exhaustion for a single stream."""

    def __init__(
        self,
        domain: str,
        source: Iterable[str],
        max_epochs: int | None = None,
    ):
        self.domain = domain
        self.source = source
        self.max_epochs = max_epochs
        self.epoch = 0
        self.items_in_epoch = 0
        self.total_items = 0
        self.is_exhausted = False
        self._cached_items: list[str] | None = None
        self._iterator: Iterator[str] | None = None
        self._init_iterator()

    def _init_iterator(self) -> None:
        if self._cached_items is not None:
            self._iterator = iter(self._cached_items)
            return

        try:
            # Check if source is a restartable collection (list, tuple, etc.)
            if isinstance(self.source, (list, tuple)):
                self._cached_items = list(self.source)
                self._iterator = iter(self._cached_items)
            else:
                self._iterator = iter(self.source)
        except (TypeError, ValueError):
            self.is_exhausted = True

    def get_next(self) -> str | None:
        """Fetches the next document from this stream.

        Returns None if exhausted or capped.
        """
        if self.is_exhausted:
            return None

        if self.max_epochs is not None and self.epoch >= self.max_epochs:
            self.is_exhausted = True
            return None

        while not self.is_exhausted:
            try:
                assert self._iterator is not None
                doc = next(self._iterator)
                self.items_in_epoch += 1
                self.total_items += 1
                return doc
            except StopIteration:
                self.epoch += 1
                self.items_in_epoch = 0

                # Check if epoch limit hit
                if self.max_epochs is not None and self.epoch >= self.max_epochs:
                    self.is_exhausted = True
                    return None

                # Attempt to restart iterator
                if self._cached_items is not None and len(self._cached_items) > 0:
                    self._iterator = iter(self._cached_items)
                else:
                    try:
                        new_iter = iter(self.source)
                        # Check if restartable
                        test_doc = next(new_iter)
                        self._iterator = new_iter
                        self.items_in_epoch = 1
                        self.total_items += 1
                        return test_doc
                    except (TypeError, StopIteration, RuntimeError):
                        self.is_exhausted = True
                        return None

        return None


class DataMixer:
    """Interleaves multiple domain streams according to defined ratios and Romanian epoch caps."""

    def __init__(
        self,
        streams: dict[str, Iterable[str]],
        ratios: MixingRatios | None = None,
        max_ro_epochs: int = 4,
        seed: int = 42,
    ):
        self.streams = streams
        self.ratios = ratios if ratios is not None else MixingRatios()
        self.max_ro_epochs = max_ro_epochs
        self.seed = seed
        self.rng = random.Random(seed)

        ratio_dict = self.ratios.to_dict()
        self.trackers: dict[str, _StreamTracker] = {}

        for domain, iterable in streams.items():
            is_ro = domain == "ro_edu" or domain.startswith("ro")
            limit = self.max_ro_epochs if is_ro else None
            self.trackers[domain] = _StreamTracker(domain, iterable, max_epochs=limit)

        self.initial_weights: dict[str, float] = {}
        for domain in streams:
            self.initial_weights[domain] = ratio_dict.get(domain, 0.20)

    def get_active_weights(self) -> dict[str, float]:
        """Computes normalized weights among currently active (non-exhausted) streams."""
        active = {
            dom: self.initial_weights.get(dom, 0.0)
            for dom, trk in self.trackers.items()
            if not trk.is_exhausted
        }
        total = sum(active.values())
        if total <= 0:
            return {}
        return {dom: w / total for dom, w in active.items()}

    def stream_documents(self) -> Iterator[tuple[str, str]]:
        """Yields (domain_name, document_text) respecting mixing ratios and Romanian epoch capping."""
        while True:
            weights = self.get_active_weights()
            if not weights:
                # All streams exhausted
                break

            domains = list(weights.keys())
            probs = [weights[d] for d in domains]

            # Deterministic weighted selection
            chosen_domain = self.rng.choices(domains, weights=probs, k=1)[0]
            tracker = self.trackers[chosen_domain]

            doc = tracker.get_next()
            if doc is not None:
                yield (chosen_domain, doc)
            else:
                # If stream just became exhausted, re-sample in the next loop
                continue

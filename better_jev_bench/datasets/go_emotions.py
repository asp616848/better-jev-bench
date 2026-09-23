"""GoEmotions (`simplified`) -- 28 emotion labels over real Reddit comments.

License: Apache-2.0. Verified 2026-09-23 against the HF Hub API
(`cardData["license"] == ["apache-2.0"]`).

**The multi-label decision, stated rather than buried.** GoEmotions is natively
multi-label, and ekVachan's primitive contract has no multi-label primitive
(PRD §2.1 introduces no fourth primitive for text, deliberately). Two options
were available: invent a multi-label shape, or restrict to the single-label
subset. This loader restricts, because the alternative changes the serving
contract for one dataset, and a benchmark whose item format is the serving
format (§2.1's highest-leverage adoption decision) may not do that.

What that costs, measured rather than guessed: 83.0% of the first 5,000 training
rows carry exactly one label (checked 2026-09-23), so the restriction keeps the
large majority of the corpus and drops the genuinely ambiguous multi-emotion tail.
`multi_label = true` stays declared in the manifest because the *source* is
multi-label and a downstream user should know the single-label view is our
construction. A real multi-label extension is a §7.7 coverage gap, not a
pretend-it-away.

Skew: `neutral` dominates. The manifest declares `chance_mode = "majority"` so
Intelligence adjusts against the real majority-class floor rather than against
1/28, which would flatter any model that learned to say "neutral".
"""

from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item
from ._common import clean

HF_ID = "google-research-datasets/go_emotions"
HF_CONFIG = "simplified"
SPLITS = ("train", "validation", "test")

INSTRUCTIONS = (
    "The text below is a comment posted on Reddit. Choose which one of the "
    "following emotion options it expresses. Choose \"neutral\" if it expresses "
    "no particular emotion."
)


class GoEmotions(BenchmarkDataset):
    name = "go_emotions"

    def _names(self) -> tuple[str, ...]:
        ds = load_dataset(HF_ID, HF_CONFIG, split="train")
        return tuple(ds.features["labels"].feature.names)

    def schema(self) -> list[RequiredOptions]:
        return [
            RequiredOptions(
                task="emotion",
                question_key="emotion",
                primitive="choice",
                instructions=INSTRUCTIONS,
                options=self._names(),
                chance_mode="majority",
            )
        ]

    def load_items(self) -> Iterator[Item]:
        options = self.spec("emotion").options
        for split in SPLITS:
            ds = load_dataset(HF_ID, HF_CONFIG, split=split)
            names = ds.features["labels"].feature.names
            for row in ds:
                labels = row["labels"]
                if len(labels) != 1:  # single-label view -- see module docstring
                    continue
                text = clean(row["text"], max_chars=800)
                if not text:
                    continue
                yield self.item(
                    "emotion",
                    state=f"Reddit comment: {text}",
                    label=names[labels[0]],
                    options=options,
                    source=f"go_emotions_simplified_{split}",
                )

"""LEDGAR (via LexGLUE) -- 100-way contract provision classification.

License: CC-BY-4.0. Verified 2026-09-23 against the HF Hub API for
`coastalcph/lex_glue` (`cardData["license"] == ["cc-by-4.0"]`). The `nlpaueb/ledgar`
repo referenced in some write-ups 401s and is not used.

Sensitivity: `legal`. PRD §8.3 -- clause classification is not attorney review.
The manifest carries the responsible-use note and CI gate 5 refuses to merge it
empty.

This and CUAD are the corpus's two legal-domain entries and they are usefully
*different* rather than redundant: LEDGAR is a 100-way choice over a flat
provision taxonomy with short, clean provision text, while CUAD is a 41-way
choice plus a genuine binary `noul` over clause excerpts drawn from full
commercial contracts. Same domain, different width, different primitive -- which
is what a Generality axis needs from a domain rather than two near-copies.
"""

from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item
from ._common import clean

HF_ID = "coastalcph/lex_glue"
HF_CONFIG = "ledgar"
SPLITS = ("train", "validation", "test")

INSTRUCTIONS = (
    "The text below is a single provision taken from a commercial contract filed "
    "with the U.S. Securities and Exchange Commission. Choose which one of the "
    "following provision-type options best describes what this provision is."
)


class Ledgar(BenchmarkDataset):
    name = "ledgar"

    def _names(self) -> tuple[str, ...]:
        ds = load_dataset(HF_ID, HF_CONFIG, split="train")
        return tuple(ds.features["label"].names)

    def schema(self) -> list[RequiredOptions]:
        return [
            RequiredOptions(
                task="provision_type",
                question_key="provision_type",
                primitive="choice",
                instructions=INSTRUCTIONS,
                options=self._names(),
            )
        ]

    def load_items(self) -> Iterator[Item]:
        options = self.spec("provision_type").options
        for split in SPLITS:
            ds = load_dataset(HF_ID, HF_CONFIG, split=split)
            names = ds.features["label"].names
            for row in ds:
                text = clean(row["text"], max_chars=2000)
                if not text:
                    continue
                yield self.item(
                    "provision_type",
                    state=f"Contract provision: {text}",
                    label=names[row["label"]],
                    options=options,
                    source=f"ledgar_{split}",
                )

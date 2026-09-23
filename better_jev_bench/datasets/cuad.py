"""CUAD -- 41 contract-clause categories, as a 41-way `choice` and a binary `noul`.

License: CC-BY-4.0. Verified 2026-09-23 against the HF Hub API for
`theatticusproject/cuad` (`cardData["license"] == "cc-by-4.0"`).

**The reformulation, and why it is a real one rather than a wrapper.** CUAD ships
as SQuAD-style span extraction: 510 contracts, 41 questions each (20,910 QAs, of
which 6,702 have at least one answer span -- counted directly from
`CUAD_v1/CUAD_v1.json` on 2026-09-23). Two things make the raw form unusable as
typed decisions: the `context` is a whole contract averaging ~54,000 characters,
and span extraction is generation-shaped, which PRD §7.6 check 4 exists
specifically to reject.

So the loader inverts it. Each *answer span* is a real, human-annotated clause of
a known category, averaging 262 characters. That gives two honest decisions:

* **`clause_type`** (`choice`, 41 options) -- "which category is this clause?"
  Spans that are annotated under more than one category in the same contract are
  dropped rather than arbitrarily assigned, so every retained item has exactly
  one defensible answer.
* **`clause_present`** (`noul`, Yes/No) -- "is this excerpt a <category> clause?"
  Positives are a span paired with its own category. Negatives are the same span
  paired with a category that the span is *not* annotated under in that contract,
  chosen deterministically by content hash so the build is reproducible. This is
  the corpus's only naturally-balanced `noul` source: exactly one negative per
  positive, so the majority-class floor is 0.5 rather than the 90%+ skew the
  Civil Comments `noul` carries.

This is the reformulation PRD §7.6 check 4 asks a reviewer to confirm is real:
the decision a model makes here is a classification with ground truth, not a
generation task wearing a classification costume.

Sensitivity: `legal`. §8.3 -- flag risk-clause presence, never auto-approve or
auto-reject a contract.
"""

from __future__ import annotations

import collections
import json
from collections.abc import Iterator

from huggingface_hub import hf_hub_download

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item
from ._common import clean, stable_unit

HF_ID = "theatticusproject/cuad"
HF_FILE = "CUAD_v1/CUAD_v1.json"

MIN_SPAN_CHARS = 40
MAX_SPAN_CHARS = 1800

CHOICE_INSTRUCTIONS = (
    "The text below is an excerpt taken from a commercial contract. Choose which "
    "one of the following clause-category options best describes what this "
    "excerpt is about."
)
NOUL_TEMPLATE = (
    "The text below is an excerpt taken from a commercial contract. Is this "
    "excerpt a \"{category}\" clause? Answer Yes or No."
)
NOUL_OPTIONS = ("Yes", "No")


def _category(question: str, qa_id: str) -> str:
    """CUAD encodes the category inside the question text, in quotes."""
    if '"' in question:
        return question.split('"')[1].strip()
    return qa_id.rsplit("__", 1)[-1].strip()


class Cuad(BenchmarkDataset):
    name = "cuad"

    def _raw(self) -> dict:
        if not hasattr(self, "_raw_cache"):
            path = hf_hub_download(HF_ID, HF_FILE, repo_type="dataset")
            with open(path, encoding="utf-8") as fh:
                self._raw_cache = json.load(fh)
        return self._raw_cache

    def _categories(self) -> tuple[str, ...]:
        cats = set()
        for contract in self._raw()["data"]:
            for para in contract["paragraphs"]:
                for qa in para["qas"]:
                    cats.add(_category(qa["question"], qa["id"]))
        return tuple(sorted(cats))

    def schema(self) -> list[RequiredOptions]:
        cats = self._categories()
        return [
            RequiredOptions(
                task="clause_type",
                question_key="clause_type",
                primitive="choice",
                instructions=CHOICE_INSTRUCTIONS,
                options=cats,
            ),
            RequiredOptions(
                task="clause_present",
                question_key="clause_present",
                primitive="noul",
                instructions=NOUL_TEMPLATE.format(category="…"),
                options=NOUL_OPTIONS,
            ),
        ]

    def load_items(self) -> Iterator[Item]:
        categories = self.spec("clause_type").options
        for contract in self._raw()["data"]:
            title = contract["title"]
            # span text -> the set of categories it is annotated under, within
            # this contract. Multi-category spans are ambiguous by construction.
            spans: dict[str, set[str]] = collections.defaultdict(set)
            for para in contract["paragraphs"]:
                for qa in para["qas"]:
                    cat = _category(qa["question"], qa["id"])
                    for ans in qa.get("answers") or []:
                        text = clean(ans.get("text"), max_chars=MAX_SPAN_CHARS)
                        if len(text) >= MIN_SPAN_CHARS:
                            spans[text].add(cat)

            for text, cats in sorted(spans.items()):
                if len(cats) != 1:
                    continue
                (cat,) = cats
                state = f"Contract excerpt: {text}"

                yield self.item(
                    "clause_type", state=state, label=cat,
                    options=categories, source=f"cuad_{title[:40]}",
                )

                # One positive and one negative noul per span, so the task is
                # balanced by construction rather than by resampling.
                yield self.item(
                    "clause_present", state=state, label="Yes", options=NOUL_OPTIONS,
                    instructions=NOUL_TEMPLATE.format(category=cat),
                    source=f"cuad_{title[:40]}",
                )
                others = [c for c in categories if c not in cats]
                neg = others[int(stable_unit("cuad-neg", text) * len(others))]
                yield self.item(
                    "clause_present", state=state, label="No", options=NOUL_OPTIONS,
                    instructions=NOUL_TEMPLATE.format(category=neg),
                    source=f"cuad_{title[:40]}",
                )

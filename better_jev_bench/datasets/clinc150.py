"""CLINC150 (clinc_oos, `plus` config) -- 150 intents plus a native out-of-scope class.

License: CC-BY-3.0. Verified 2026-09-23 against the HF Hub API
(`cardData["license"] == ["cc-by-3.0"]`).

This is the single most diagnostically valuable entry in the first batch and the
`plus` config is chosen deliberately over `small`/`imbalanced`: it carries the
largest out-of-scope population. PRD §5.3 makes abstention a *named sub-metric
under Intelligence, never folded into the headline*, citing the sibling's KoBBQ
finding that forcing an answer on a deliberately unanswerable item pushed a model
to the dataset's stereotype 79% of the time. `oos` is presented here as a
first-class 151st option rather than as an absence, so a model that correctly
says "none of these" is scored as correct rather than as a coverage miss -- which
is the whole point of having the class.

Width note: 151 options is well past ekVachan's 26-letter decoder budget. That is
deliberate and is not the corpus's problem to solve -- `bjb export` narrows on
the way out (see `export.narrow_options`), and the benchmark keeps the true
schema so `width_wide` measures something real.
"""

from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item
from ._common import clean

HF_ID = "clinc/clinc_oos"
HF_CONFIG = "plus"
SPLITS = ("train", "validation", "test")

INSTRUCTIONS = (
    "A user has spoken a request to a virtual assistant, transcribed below. "
    "Choose which one of the following intent options the request expresses. "
    "Choose \"oos\" (out of scope) only if none of the other options match what "
    "the user asked for."
)


class ClincIntents(BenchmarkDataset):
    name = "clinc150"

    def _names(self) -> tuple[str, ...]:
        ds = load_dataset(HF_ID, HF_CONFIG, split="train")
        return tuple(ds.features["intent"].names)

    def schema(self) -> list[RequiredOptions]:
        return [
            RequiredOptions(
                task="intent",
                question_key="intent",
                primitive="choice",
                instructions=INSTRUCTIONS,
                options=self._names(),
            )
        ]

    def load_items(self) -> Iterator[Item]:
        options = self.spec("intent").options
        for split in SPLITS:
            ds = load_dataset(HF_ID, HF_CONFIG, split=split)
            names = ds.features["intent"].names
            for row in ds:
                text = clean(row["text"], max_chars=600)
                if not text:
                    continue
                yield self.item(
                    "intent",
                    state=f"User request: {text}",
                    label=names[row["intent"]],
                    options=options,
                    source=f"clinc150_plus_{split}",
                )

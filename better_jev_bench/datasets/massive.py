"""MASSIVE (en-US) -- 60 intents and 18 scenarios over the same utterances.

License: CC-BY-4.0. Verified 2026-09-23 against the HF Hub API
(`cardData["license"] == ["cc-by-4.0"]`).

**Why the parquet revision**: `AmazonScience/massive` is script-based and
`datasets >= 3` will not execute dataset scripts. Unlike BANKING77, this repo
*does* have a Hub-generated `refs/convert/parquet` branch (163 parquet files,
one set per locale, checked 2026-09-23), which is the Hub's own conversion of
the same repo -- so it is used directly. No third-party mirror is involved.

**Two tasks over one state, deliberately.** MASSIVE annotates each utterance
with both a fine intent (60-way) and a coarse scenario (18-way). Emitting both
gives the corpus a rare thing: two genuinely different-width decisions over
*identical* text, which is the cleanest available probe of whether a model's
accuracy tracks schema width rather than task difficulty. `split.py` keys on the
state hash, so an utterance's intent item and scenario item always land on the
same side of the public/held-out line.

v1 is en-US only. MASSIVE's 51 other locales are a strong multilingual extension
and are left as a follow-up rather than quietly inflating the item count with 52
copies of the same 11,514 decisions.
"""

from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item
from ._common import clean

HF_ID = "AmazonScience/massive"
HF_REVISION = "refs/convert/parquet"
LOCALE = "en-US"
SPLITS = ("train", "validation", "test")

INTENT_INSTRUCTIONS = (
    "A user has spoken a command to a voice assistant, transcribed below. Choose "
    "which one of the following fine-grained intent options the command expresses. "
    "Each option is written as scenario_action."
)
SCENARIO_INSTRUCTIONS = (
    "A user has spoken a command to a voice assistant, transcribed below. Choose "
    "which one of the following broad scenario options the command falls under."
)


class MassiveIntents(BenchmarkDataset):
    name = "massive"

    def _split(self, split: str):
        return load_dataset(
            HF_ID,
            data_files={split: f"{LOCALE}/{split}/0000.parquet"},
            revision=HF_REVISION,
            split=split,
        )

    def schema(self) -> list[RequiredOptions]:
        ds = self._split("train")
        return [
            RequiredOptions(
                task="intent",
                question_key="intent",
                primitive="choice",
                instructions=INTENT_INSTRUCTIONS,
                options=tuple(ds.features["intent"].names),
            ),
            RequiredOptions(
                task="scenario",
                question_key="scenario",
                primitive="choice",
                instructions=SCENARIO_INSTRUCTIONS,
                options=tuple(ds.features["scenario"].names),
            ),
        ]

    def load_items(self) -> Iterator[Item]:
        intent_opts = self.spec("intent").options
        scenario_opts = self.spec("scenario").options
        for split in SPLITS:
            ds = self._split(split)
            i_names = ds.features["intent"].names
            s_names = ds.features["scenario"].names
            for row in ds:
                text = clean(row["utt"], max_chars=600)
                if not text:
                    continue
                state = f"Voice assistant command: {text}"
                yield self.item(
                    "intent", state=state, label=i_names[row["intent"]],
                    options=intent_opts, source=f"massive_{LOCALE}_{split}",
                )
                yield self.item(
                    "scenario", state=state, label=s_names[row["scenario"]],
                    options=scenario_opts, source=f"massive_{LOCALE}_{split}",
                )

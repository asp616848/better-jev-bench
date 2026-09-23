"""BANKING77 -- 77 fine-grained banking intents (PRD §3.4, §3.7).

License: CC-BY-4.0. Verified 2026-09-23 against the Hugging Face Hub API
(`HfApi().dataset_info("PolyAI/banking77").cardData["license"] == ["cc-by-4.0"]`),
not from the dataset's name or from memory.

**Why this loader reads GitHub CSVs rather than `load_dataset("PolyAI/banking77")`**:
`PolyAI/banking77` is a script-based dataset, and `datasets >= 3` refuses to
execute dataset scripts ("Dataset scripts are no longer supported"). There is no
`refs/convert/parquet` branch on that repo either (checked 2026-09-23 -- the
revision 404s). Rather than route through a third-party re-upload of unverified
provenance, this loader fetches the exact two CSVs that PolyAI's own official
loading script fetches, from PolyAI's own repository. Same bytes, same
provenance, one less intermediary.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from collections.abc import Iterator

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item
from ._common import clean

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
SPLITS = ("train", "test")

INSTRUCTIONS = (
    "A customer has contacted an online bank's support channel. Choose which one "
    "of the following intent options best describes what the customer is asking "
    "about. Exactly one option is correct."
)


class BankingIntents(BenchmarkDataset):
    name = "banking77"

    def _fetch(self, split: str) -> list[dict[str, str]]:
        url = f"{BASE}/{split}.csv"
        with urllib.request.urlopen(url, timeout=120) as resp:
            body = resp.read().decode("utf-8")
        return list(csv.DictReader(io.StringIO(body)))

    def _labels(self) -> tuple[str, ...]:
        rows = self._fetch("train") + self._fetch("test")
        return tuple(sorted({r["category"] for r in rows}))

    def schema(self) -> list[RequiredOptions]:
        # The 77 intents are a fixed published taxonomy; resolved once from the
        # source so the declared width can never drift from the data.
        if not hasattr(self, "_cached_labels"):
            self._cached_labels = self._labels()
        return [
            RequiredOptions(
                task="intent",
                question_key="intent",
                primitive="choice",
                instructions=INSTRUCTIONS,
                options=self._cached_labels,
            )
        ]

    def load_items(self) -> Iterator[Item]:
        options = self.schema()[0].options
        for split in SPLITS:
            for row in self._fetch(split):
                text = clean(row["text"], max_chars=1200)
                if not text or row["category"] not in options:
                    continue
                yield self.item(
                    "intent",
                    state=f"Customer message: {text}",
                    label=row["category"],
                    options=options,
                    source=f"banking77_{split}",
                )

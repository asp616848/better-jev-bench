"""Civil Comments -- the corpus's native `score` source, plus a `noul`.

License: CC0-1.0. Verified 2026-09-23 against the HF Hub API for
`google/civil_comments` (`cardData["license"] == "cc0-1.0"`). CC0 means no
attribution obligation propagates, which is why this is the one large source in
the first batch that can sit in a commercial redistribution with no conditions.

PRD §1.2 calls this dataset "natively probability-shaped, near-ideal for the
`score` primitive", and §5.2 leans on it for the whole existence of a `prim_score`
stratum. It earns that: `toxicity` is the *fraction of human annotators who rated
the comment toxic*, so it is a real calibrated quantity rather than a 1-5 star
rating standing in for one.

**The discretisation is ours, and it is declared as ours.** `score` in ekVachan's
contract is an ordered 2-10 level scale, not a continuous float, so the
continuous toxicity fraction is cut into five bands at 0.2 boundaries. Those
boundaries are a construction of this loader; the underlying annotation is human
and is not heuristic, so `heuristic_label` stays false, but the manifest task
note says plainly where the levels came from. A future continuous-target
primitive would make this lossless; today's contract does not have one.

**Nothing is rebalanced** (see `datasets/__init__.py`). The natural distribution
is heavily skewed to non-toxic -- 82.6% of a 20,000-row sample falls in the
lowest band, measured 2026-09-23 -- so both tasks declare
`chance_mode = "majority"` and the Intelligence axis adjusts against the real
floor. A balanced resample would have produced a much better-looking number
about a distribution that does not exist.

Sensitivity: `moderation`. PRD §8.3 -- generic toxicity is not any specific
platform's policy, and auto-actioning a toxicity score without an appeal path is
a wrongful-action risk.
"""

from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item
from ._common import clean, stable_unit

HF_ID = "google/civil_comments"
SPLITS = ("train", "validation", "test")

#: Sampling budget per split. Civil Comments is 1.8M rows; the whole point of a
#: hash-rank sample (rather than a head slice) is that this stays a uniform
#: sample of the real distribution instead of a slice of one time period.
PER_SPLIT_BUDGET = {"train": 60_000, "validation": 8_000, "test": 8_000}

#: Ordered ascending. Order is load-bearing for an ordinal scale -- see
#: `types.Question.ordinal` and ekVachan's `build_primitives_slice.py`.
LEVELS = (
    "not toxic",
    "slightly toxic",
    "moderately toxic",
    "toxic",
    "severely toxic",
)
BOUNDARIES = (0.2, 0.4, 0.6, 0.8)

SCORE_INSTRUCTIONS = (
    "The text below is a comment posted on a news website. Human annotators rated "
    "how toxic it is -- rude, disrespectful, or likely to make someone leave the "
    "discussion. Choose the option that best describes its toxicity, on a scale "
    "running from least toxic to most toxic."
)
NOUL_INSTRUCTIONS = (
    "The text below is a comment posted on a news website. Would a majority of "
    "human annotators rate this comment as toxic -- rude, disrespectful, or "
    "likely to make someone leave the discussion? Answer Yes or No."
)
NOUL_OPTIONS = ("Yes", "No")


def level_of(toxicity: float) -> str:
    idx = sum(1 for b in BOUNDARIES if toxicity >= b)
    return LEVELS[idx]


class CivilComments(BenchmarkDataset):
    name = "civil_comments"

    def schema(self) -> list[RequiredOptions]:
        return [
            RequiredOptions(
                task="toxicity_level",
                question_key="toxicity",
                primitive="score",
                instructions=SCORE_INSTRUCTIONS,
                options=LEVELS,
                ordinal=True,
                chance_mode="majority",
            ),
            RequiredOptions(
                task="is_toxic",
                question_key="is_toxic",
                primitive="noul",
                instructions=NOUL_INSTRUCTIONS,
                options=NOUL_OPTIONS,
                chance_mode="majority",
            ),
        ]

    def load_items(self) -> Iterator[Item]:
        for split, budget in PER_SPLIT_BUDGET.items():
            ds = load_dataset(HF_ID, split=split)
            n = len(ds)
            keep = budget / n
            for i in range(n):
                if stable_unit("civil-comments", split, i) >= keep:
                    continue
                row = ds[i]
                text = clean(row["text"], max_chars=1500)
                if len(text) < 12:
                    continue
                tox = float(row["toxicity"])
                state = f"Comment: {text}"
                yield self.item(
                    "toxicity_level", state=state, label=level_of(tox),
                    options=LEVELS, source=f"civil_comments_{split}",
                )
                yield self.item(
                    "is_toxic", state=state, label="Yes" if tox >= 0.5 else "No",
                    options=NOUL_OPTIONS, source=f"civil_comments_{split}",
                )

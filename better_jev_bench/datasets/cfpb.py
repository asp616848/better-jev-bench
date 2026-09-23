"""CFPB consumer complaints -- real complaint narratives routed to real product lines.

PRD §3.7 names this the corpus's centre of gravity, and §3.4 calls it "the most
directly diagnostic for the 'zero items matched' JevBench/jabr-v2 failure":
genuine free-text written by a member of the public, routed into a production
taxonomy by the agency that operates it. It is the closest thing in the whole
99-entry catalogue to the support-routing shape that broke ekVachan.

## License

CC0 / U.S. Government work. Verified three ways on 2026-09-23, all primary:

* the CFPB's own public API returns `_meta.license = "CC0"` in every response
  (`https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`);
* `CFPB/consumer-finance-complaints` on the Hub declares `cc0-1.0`;
* the mirror actually used here, `BEE-spoke-data/consumer-finance-complaints`,
  declares `cc0-1.0` on its card.

## Why a Hub mirror rather than the CFPB bulk export -- a real finding, not a shortcut

This project's standing discipline is to pull from the primary source. It does
not work here, and the reason was established by doing it rather than assuming:

* `https://files.consumerfinance.gov/ccdb/complaints.csv.zip` was downloaded in
  full on 2026-09-23 (346,404,253 bytes, ETag
  `"f0d2ec2367bd2ae60f044b782ad85f43-42"`, 5,434,898,686 bytes uncompressed).
  Its header carries 15 columns and **`Consumer complaint narrative` is not one
  of them**. The bulk export no longer ships the narrative field at all.
* The public search API returns records whose `_source` has no
  `complaint_what_happened` key either, with or without `has_narrative=true`
  (checked against several parameter combinations the same day).

So the narrative column -- the entire reason this dataset is valuable to a
language-conditioned decision model -- is not obtainable from the primary
distribution today. The `has-text` config of the Hub mirror carries it
(1,689,573 rows, 1.23GB), under the same CC0 terms the primary source declares
for the underlying data. CC0 data stays CC0 through a mirror, so the license
posture is unchanged; the *provenance* is one hop longer, and that is recorded
here and in the manifest's `verified_how` rather than glossed.

## The product taxonomy is normalised, and that is a judgment call

CFPB has revised its product taxonomy several times and the archive carries all
versions: 21 distinct `Product` strings across the 1,689,573 narrative rows
(counted directly, 2026-09-23). Three of them are near-synonyms for credit
reporting, three more for payday/personal lending, two for deposit accounts.
Presenting all 21 as options would measure which taxonomy *version* a model
guesses, not which product line a complaint belongs to -- an item whose correct
answer depends on the year it was filed is ambiguous by construction rather than
hard.

`PRODUCT_MAP` below folds the legacy strings onto the current product lines,
producing a 10-option schema. Two source values are **dropped rather than
mapped**, because no defensible mapping exists: `Consumer Loan` (9,461 rows) spans
vehicle, personal and student lending in the current taxonomy, and
`Other financial service` (292 rows) is a catch-all. Dropping is the honest
handling; forcing them into a bucket would manufacture wrong labels.

Distribution is left as it is. Credit reporting is ~57% of the corpus, so the
manifest declares `chance_mode = "majority"` and Intelligence adjusts against
that floor (PRD §5.3).
"""

from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item
from ._common import clean, stable_unit

HF_ID = "BEE-spoke-data/consumer-finance-complaints"
HF_CONFIG = "has-text"

#: Hash-rank sample budget. The full `has-text` split is 1.69M rows; a uniform
#: sample of it beats a head slice, which would be one time period's complaints.
BUDGET = 60_000
MAX_NARRATIVE_CHARS = 2400

#: Current CFPB product lines, ascending nothing -- a flat taxonomy.
PRODUCTS = (
    "Checking or savings account",
    "Credit card or prepaid card",
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Debt or credit management",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Payday loan, title loan, personal loan, or advance loan",
    "Student loan",
    "Vehicle loan or lease",
)

#: Legacy source string -> current product line. See the module docstring for the
#: two values deliberately absent from this map.
PRODUCT_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports":
        "Credit reporting or other personal consumer reports",
    "Credit reporting": "Credit reporting or other personal consumer reports",
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Bank account or service": "Checking or savings account",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Payday loan, title loan, or personal loan":
        "Payday loan, title loan, personal loan, or advance loan",
    "Payday loan": "Payday loan, title loan, personal loan, or advance loan",
    **{p: p for p in PRODUCTS},
}

INSTRUCTIONS = (
    "A member of the public has submitted the complaint below to the U.S. Consumer "
    "Financial Protection Bureau. Choose which one of the following product-line "
    "options the complaint should be routed to. Personal details in the text have "
    "been redacted by the agency and appear as runs of X characters."
)


class CfpbComplaints(BenchmarkDataset):
    name = "cfpb_complaints"

    def schema(self) -> list[RequiredOptions]:
        return [
            RequiredOptions(
                task="product",
                question_key="product",
                primitive="choice",
                instructions=INSTRUCTIONS,
                options=PRODUCTS,
                chance_mode="majority",
            )
        ]

    def load_items(self) -> Iterator[Item]:
        ds = load_dataset(HF_ID, HF_CONFIG, split="train")
        n = len(ds)
        keep = BUDGET / n
        for i in range(n):
            if stable_unit("cfpb", i) >= keep:
                continue
            row = ds[i]
            product = PRODUCT_MAP.get((row["Product"] or "").strip())
            if product is None:
                continue
            text = clean(row["Consumer complaint narrative"], max_chars=MAX_NARRATIVE_CHARS)
            if len(text) < 40:
                continue
            yield self.item(
                "product",
                state=f"Consumer complaint: {text}",
                label=product,
                options=PRODUCTS,
                source="cfpb_complaints_has_text",
            )

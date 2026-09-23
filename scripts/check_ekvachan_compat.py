"""Verify a `bjb export` against ekVachan's OWN validator, not ours.

The body of `_check()` below is `_validate()` copied verbatim out of
`better-jev-for-all/training/build_primitives_slice.py` -- the script that built
the data behind the sibling project's real trained checkpoint. Copying it rather
than re-implementing it is the point: the question this answers is "would
ekVachan's own training pipeline accept these records", and only ekVachan's own
assertions can answer that.

    bjb export --out exports/compat --tiers A --max-per-task 500
    python scripts/check_ekvachan_compat.py exports/compat/public.jsonl

Run this whenever the export path or the record schema changes. It is cheap, it
is real evidence, and the alternative is discovering the mismatch inside a
two-hour GPU run.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

MAX_OPTIONS = 26  # build_primitives_slice.py's own ceiling
EXPECTED_KEYS = {
    "state", "question_key", "question_type", "instructions",
    "options", "label", "label_idx", "source",
}


def _check(records: list[dict], name: str) -> None:
    """Verbatim from build_primitives_slice.py::_validate, plus the key-set and
    fixed-scale-order checks that script enforces structurally."""
    score_orders: dict[str, set[tuple[str, ...]]] = collections.defaultdict(set)
    for r in records:
        n = len(r["options"])
        assert 2 <= n <= MAX_OPTIONS, f"{name}: option count {n} out of [2, {MAX_OPTIONS}]"
        assert 0 <= r["label_idx"] < n, f"{name}: label_idx {r['label_idx']} out of range for {n} options"
        assert r["options"][r["label_idx"]] == r["label"], f"{name}: label_idx doesn't point at label text"
        assert len(set(r["options"])) == n, f"{name}: duplicate option text in {r['options']}"
        assert r["question_type"] in ("choice", "noul", "score"), f"{name}: unexpected question_type {r['question_type']!r}"
        assert set(r) == EXPECTED_KEYS, f"{name}: record keys {sorted(r)} != ekVachan's eight"
        if r["question_type"] == "score":
            score_orders[r["source"]].add(tuple(r["options"]))
    for src, orders in score_orders.items():
        assert len(orders) == 1, (
            f"{name}: {src} emitted {len(orders)} distinct score scale orders; an ordinal scale's "
            "letter position IS its scale position and must be fixed"
        )
        print(f"  score scale order fixed for {src}: {list(orders)[0]}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _check(records, path.name)
    by_type = collections.Counter(r["question_type"] for r in records)
    widths = collections.Counter(len(r["options"]) for r in records)
    print(f"OK: {len(records):,} records pass ekVachan's own _validate() verbatim")
    print(f"  by_question_type: {dict(by_type)}")
    print(f"  option widths:    {dict(sorted(widths.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

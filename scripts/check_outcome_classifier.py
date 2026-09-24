"""Exhaustive check of `better_jev_bench/backend.py::classify()` against PRD
§14.3's four-outcome table (PRD §14.9 item 2). Same plain-assertion style as
the repo's other `scripts/check_*.py`:

    uv run python3 scripts/check_outcome_classifier.py

Specifically includes the two cases the PRD calls out by name: an answer
differing only by case is `out_of_schema`; one differing only by trailing
whitespace is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from better_jev_bench.backend import Outcome, classify  # noqa: E402

OPTIONS = ("Yes", "No")
n_checks = 0


def check(name: str, cond: bool) -> None:
    global n_checks
    n_checks += 1
    if not cond:
        raise SystemExit(f"FAIL: {name}")
    print(f"  ok  {name}")


# -- correct / incorrect ------------------------------------------------------

r = classify({"choice": "No"}, options=OPTIONS, expected="No")
check("exact match -> correct", r.outcome is Outcome.CORRECT)

r = classify({"choice": "Yes"}, options=OPTIONS, expected="No")
check("wrong-but-in-schema -> incorrect", r.outcome is Outcome.INCORRECT)

# -- the two cases the PRD calls out by name ----------------------------------

r = classify({"choice": "no"}, options=OPTIONS, expected="No")
check("PRD §14.3: case-only difference -> out_of_schema (no case folding)", r.outcome is Outcome.OUT_OF_SCHEMA)

r = classify({"choice": "No "}, options=OPTIONS, expected="No")
check("PRD §14.3: trailing-whitespace-only difference -> NOT out_of_schema (str.strip() applies)", r.outcome is Outcome.CORRECT)

r = classify({"choice": "  No"}, options=OPTIONS, expected="No")
check("leading whitespace also stripped", r.outcome is Outcome.CORRECT)

# -- out_of_schema: every named variant ---------------------------------------

check("empty string -> out_of_schema", classify({"choice": ""}, options=OPTIONS, expected="No").outcome is Outcome.OUT_OF_SCHEMA)
check("null choice -> out_of_schema", classify({"choice": None}, options=OPTIONS, expected="No").outcome is Outcome.OUT_OF_SCHEMA)
check("missing choice key -> out_of_schema", classify({}, options=OPTIONS, expected="No").outcome is Outcome.OUT_OF_SCHEMA)
check("a letter -> out_of_schema", classify({"choice": "A"}, options=OPTIONS, expected="No").outcome is Outcome.OUT_OF_SCHEMA)
check("an index (int) -> out_of_schema", classify({"choice": 1}, options=OPTIONS, expected="No").outcome is Outcome.OUT_OF_SCHEMA)
check("an index (numeric string not in options) -> out_of_schema", classify({"choice": "1"}, options=OPTIONS, expected="No").outcome is Outcome.OUT_OF_SCHEMA)
check(
    "paraphrase not matching any option verbatim -> out_of_schema",
    classify({"choice": "checking or savings"}, options=("Checking or savings account", "Other"), expected="Other").outcome
    is Outcome.OUT_OF_SCHEMA,
)

# -- declined ------------------------------------------------------------------

r = classify(None, options=OPTIONS, expected="No")
check("result is None (results missing this question_key) -> declined", r.outcome is Outcome.DECLINED)

# -- p: mass on the RETURNED option, never the max, never the gold's mass ----

r = classify(
    {"choice": "getting_virtual_card", "probabilities": {"getting_virtual_card": 0.83, "card_arrival": 0.06}},
    options=("getting_virtual_card", "card_arrival"),
    expected="getting_virtual_card",
)
check("p resolves to probabilities[returned] on a correct answer", r.p == 0.83)

# The single most common ECE bug the PRD calls out: on an INCORRECT answer, p
# must be the mass on the option the model actually returned -- not the max
# over all options, and not the mass on the gold option.
r = classify(
    {"choice": "card_arrival", "probabilities": {"getting_virtual_card": 0.83, "card_arrival": 0.06}},
    options=("getting_virtual_card", "card_arrival"),
    expected="getting_virtual_card",
)
check("p on an INCORRECT answer is the mass on the RETURNED option (0.06), not the max (0.83)", r.p == 0.06)
check("that item is classified incorrect", r.outcome is Outcome.INCORRECT)

r = classify({"choice": "No", "confidence": 0.71}, options=OPTIONS, expected="No")
check("falls back to confidence when no probabilities map is present", r.p == 0.71)

r = classify({"choice": "No"}, options=OPTIONS, expected="No")
check("p is None when neither probabilities nor confidence is present (uncalibrated, not miscalibrated)", r.p is None)

r = classify({"choice": "No", "probabilities": {"Yes": 0.9}}, options=OPTIONS, expected="No")
check("probabilities present but missing the RETURNED key -> falls through (no KeyError, no wrong fallback to max)", r.p is None)

print(f"\n{n_checks} checks run, 0 failed.")

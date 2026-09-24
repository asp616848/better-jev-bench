"""Pure scoring arithmetic (PRD §14.9 item 1). No I/O, no network, no corpus
reads -- every function here takes plain numbers or lists of numbers and
returns plain numbers. `evaluate.py` is the only module that calls these with
real run data; that split is deliberate so the arithmetic can be unit-tested
against hand-computed values with nothing else running (`scripts/
check_score_engine.py` does exactly that, including the §14.2 demonstration:
`I = 0` for an all-"No" model on `civil_comments/is_toxic` at the corpus's
real chance floor, 0.920729).

Every formula below is transcribed from PRD §14, not re-derived, and each
docstring cites the subsection it implements so a drift between this file and
the PRD is visible on inspection rather than only in a diff.

`SPEC_VERSION` is frozen the moment the first real `/v1/evaluate` result is
published (§14, epigraph); until then this module is code, not yet a
commitment.
"""

from __future__ import annotations

import collections
import math
from typing import Any

import numpy as np

from . import SPEC_VERSION  # re-exported for convenience; defined once in __init__.py

#: PRD §14.5 / §7.3 -- reuses `types.MIN_ITEMS_FOR_CALIBRATION` at the value
#: level rather than importing it, so this module stays free of any corpus
#: dependency; `evaluate.py` is responsible for keeping the two in sync (both
#: are 250, both cite PRD §5.3/§7.3).
MIN_ITEMS_FOR_STRATUM = 250
#: PRD §14.6 -- a task's ECE contribution is dropped below this many scored
#: (correct/incorrect) items.
MIN_ITEMS_FOR_ECE = 250


# ---------------------------------------------------------------------------
# Intelligence (PRD §14.1, §14.2)
# ---------------------------------------------------------------------------


def chance_adjusted(acc: float, chance: float) -> float:
    """`max(0, (acc - chance) / (1 - chance))` -- PRD §5.3's core formula,
    unamended. `chance` must be the task's *manifest*-declared floor (PRD
    §14.2); this function has no way to enforce that and trusts its caller."""
    if not (0.0 <= chance < 1.0):
        raise ValueError(f"chance must be in [0, 1), got {chance}")
    return max(0.0, (acc - chance) / (1.0 - chance))


def task_intelligence(n_correct: int, n_attempted: int, chance: float) -> dict[str, Any]:
    """One task's `I_t`, PRD §14.3: `acc_t`'s denominator is every item
    *attempted* in the task's selected slice, not just the covered ones --
    `out_of_schema` and `declined` items are already folded into
    `n_attempted - n_correct` by the caller, exactly as §14.3 requires
    ("a model cannot raise its accuracy by refusing the items it would have
    got wrong")."""
    if n_attempted <= 0:
        raise ValueError("task_intelligence: n_attempted must be > 0")
    acc = n_correct / n_attempted
    return {"n": n_attempted, "acc": acc, "chance": chance, "i": chance_adjusted(acc, chance)}


def aggregate_intelligence(
    task_i: dict[str, float],
    *,
    task_to_dataset: dict[str, str],
    dataset_to_domain: dict[str, str],
) -> dict[str, Any]:
    """PRD §14.1's three-level aggregation: `task -> dataset -> domain -> axis`,
    equal weight at every level (never item-weighted, PRD §2.5). `task_i` is
    `{"<dataset>/<task>": I_t}` for every task actually scored in this run;
    `domains_present` -- the PRD's phrase -- falls out naturally as
    `by_domain.keys()`, since a domain with no scored task simply never
    appears. Returns `{"by_dataset", "by_domain", "intelligence"}`; the caller
    merges in `by_task` itself, which this function doesn't need to touch."""
    by_dataset: dict[str, list[float]] = collections.defaultdict(list)
    for task_id, i_value in task_i.items():
        by_dataset[task_to_dataset[task_id]].append(i_value)
    dataset_scores = {ds: sum(vs) / len(vs) for ds, vs in by_dataset.items()}

    by_domain: dict[str, list[float]] = collections.defaultdict(list)
    for ds, i_value in dataset_scores.items():
        by_domain[dataset_to_domain[ds]].append(i_value)
    domain_scores = {dom: sum(vs) / len(vs) for dom, vs in by_domain.items()}

    intelligence = 100.0 * (sum(domain_scores.values()) / len(domain_scores)) if domain_scores else None
    return {"by_dataset": dataset_scores, "by_domain": domain_scores, "intelligence": intelligence}


# ---------------------------------------------------------------------------
# Breadth / Generality (PRD §14.5)
# ---------------------------------------------------------------------------


def geomean(values: list[float]) -> float:
    """Geometric mean by direct product, not `exp(mean(log(x)))` -- so a
    genuine zero in the list (a collapsed stratum or family) correctly zeroes
    the whole mean instead of raising on `log(0)`."""
    if not values:
        raise ValueError("geomean of an empty sequence")
    product = 1.0
    for v in values:
        if v < 0:
            raise ValueError(f"geomean requires non-negative values, got {v}")
        product *= v
    return product ** (1.0 / len(values))


def stratum_value(task_i_values: list[float]) -> float:
    """PRD §14.5 problem 1's resolution: a stratum's value is the **macro
    mean of per-task `I_t`** over the tasks that touch it -- never a pooled
    accuracy re-derived against some single chance floor, which is undefined
    when the stratum spans tasks with different floors (`prim_noul` spans
    0.9207 and 0.5). Never item-weighted (PRD §2.5)."""
    if not task_i_values:
        raise ValueError("stratum_value: no tasks in this stratum")
    return sum(task_i_values) / len(task_i_values)


def family_value(populated_stratum_values: list[float]) -> float | None:
    """Geometric mean over a family's *populated* strata (PRD §14.5 problem
    2). `None` if every stratum in the family was excluded -- the caller
    drops the family from `breadth()`'s input and records it in
    `excluded_strata`."""
    if not populated_stratum_values:
        return None
    return geomean(populated_stratum_values)


def breadth(family_values: list[float]) -> float | None:
    """Geometric mean over `{width, primitive, modality}` family values (PRD
    §14.5's nested resolution -- each family weighs exactly 1/3 of the
    log-score regardless of how many strata it happens to contain). `None`
    if every family was dropped."""
    if not family_values:
        return None
    return geomean(family_values)


def generality(breadth_value: float | None, coverage_fraction: float, *, n_families_dropped: int) -> float | None:
    """`100 * sqrt(Breadth * Coverage)` (PRD §14.5/§5.3). `None`, not 0, when
    two or more of the three families were dropped for underpopulation --
    §14.5's explicit point that "the benchmark could not measure this" must
    stay distinguishable from "the model has no generality"."""
    if breadth_value is None or n_families_dropped >= 2:
        return None
    return 100.0 * math.sqrt(max(0.0, breadth_value) * max(0.0, coverage_fraction))


# ---------------------------------------------------------------------------
# Calibration (PRD §14.6)
# ---------------------------------------------------------------------------


def pooled_ece(pairs: list[tuple[float, int]], *, n_bins: int = 15) -> float | None:
    """15-bin equal-mass ECE, ported **verbatim** from the sibling project's
    `eval/metrics.py::expected_calibration_error` (sort by confidence, bin
    edges at `np.linspace(0, n, n_bins + 1).astype(int)`, weight each bin by
    its population share) so the two projects' numbers are directly
    comparable, per PRD §14.6's explicit requirement. The sibling's version
    takes a full probability matrix and reads `probs.max(axis=-1)` /
    `probs.argmax(axis=-1) == labels` as its `(confidence, correct)` pair;
    this version takes that same pair directly, because here `p` is *the
    probability mass on whichever option the model actually returned* (PRD
    §5.3/§14.4a) -- for a correct answer that is also the max, but for an
    incorrect one it is not, and using the max there would silently change
    what is being measured. The binning and weighting arithmetic below is
    unchanged from the source.

    `pairs` is `(p, y)` for every `correct`/`incorrect` item in the pool;
    `out_of_schema` and `declined` items contribute nothing (PRD §14.3) and
    must already be excluded by the caller. `None` if the pool is empty."""
    if not pairs:
        return None
    confidences = np.array([p for p, _ in pairs], dtype=np.float64)
    correct = np.array([float(y) for _, y in pairs], dtype=np.float64)

    order = np.argsort(confidences)
    confidences, correct = confidences[order], correct[order]

    n = len(confidences)
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        bin_conf = confidences[lo:hi].mean()
        bin_acc = correct[lo:hi].mean()
        weight = (hi - lo) / n
        ece += weight * abs(bin_conf - bin_acc)
    return float(ece)


def hcs(acc: float, ece: float) -> float:
    """`HCS = 2*Acc*(1-ECE) / (Acc + (1-ECE))` at the frozen `β = 1` (PRD
    §5.3/§14.6). `acc` must be the **run-wide pooled raw accuracy over
    covered items** -- not chance-adjusted, not macro-averaged -- matching
    the exact pool the ECE was computed over."""
    reliability = 1.0 - ece
    denom = acc + reliability
    if denom <= 0:
        return 0.0
    return 2.0 * acc * reliability / denom


# ---------------------------------------------------------------------------
# Speed / Cost (PRD §14.6) -- frozen log-anchored reference scales
# ---------------------------------------------------------------------------


def speed_score(p50_latency_ms: float) -> float:
    """10ms -> 100, 100ms -> 50, 1000ms -> 0 (PRD §14.6, frozen for
    `bjb-score-1.0`)."""
    if p50_latency_ms <= 0:
        raise ValueError(f"p50_latency_ms must be positive, got {p50_latency_ms}")
    lo, hi = math.log10(10), math.log10(1000)
    x = (hi - math.log10(p50_latency_ms)) / (hi - lo)
    return 100.0 * min(1.0, max(0.0, x))


def cost_score(usd_per_1k_decisions: float) -> float:
    """$0.01/1k -> 100, $1/1k -> 50, $100/1k -> 0 (PRD §14.6, frozen for
    `bjb-score-1.0`)."""
    if usd_per_1k_decisions <= 0:
        raise ValueError(f"usd_per_1k_decisions must be positive, got {usd_per_1k_decisions}")
    lo, hi = math.log10(0.01), math.log10(100)
    x = (hi - math.log10(usd_per_1k_decisions)) / (hi - lo)
    return 100.0 * min(1.0, max(0.0, x))


# ---------------------------------------------------------------------------
# Top-level score (PRD §14.6, "the order of operations")
# ---------------------------------------------------------------------------


def combine_axes(
    intelligence: float | None,
    calibration: float | None,
    generality: float | None,
    speed: float | None,
    cost: float | None,
) -> tuple[float | None, float | None, bool]:
    """Geometric mean of the five axes, THEN the §5.4 floor penalty applied
    to Intelligence alone, in that order and no other (PRD §14.6). Returns
    `(score, score_no_cost, floor_penalty_applied)`. Any `None` axis
    propagates `score` to `None`; `score_no_cost` (the geomean of the other
    four, same floor-penalty rule) is populated whenever `cost is None`, per
    the response contract's "present iff `cost_model` was null"."""
    floor_penalty_applied = intelligence is not None and intelligence < 50.0

    def _combine(axes: list[float | None]) -> float | None:
        if any(a is None for a in axes):
            return None
        g = geomean(axes)  # type: ignore[arg-type]
        if floor_penalty_applied:
            g = g * (intelligence / 50.0) ** 2  # type: ignore[operator]
        return g

    score = _combine([intelligence, calibration, generality, speed, cost])
    score_no_cost = _combine([intelligence, calibration, generality, speed]) if cost is None else None
    return score, score_no_cost, floor_penalty_applied

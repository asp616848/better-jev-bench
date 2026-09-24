"""Unit-checks `better_jev_bench/score.py`'s pure functions against
hand-computed values (PRD §14.9 item 1). Same style as the repo's existing
`scripts/check_*.py` -- plain assertions, no pytest dependency, run directly:

    uv run python3 scripts/check_score_engine.py

Includes the exact demonstration PRD §14.2 calls out by name: `I = 0` for an
all-"No" model on `civil_comments/is_toxic` at the corpus's real chance floor
(0.920729), contrasted against the wrong pre-fix number (0.5) to show the
fix is not cosmetic -- it changes which side of "chance-level" a model lands
on.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from better_jev_bench import score as S  # noqa: E402

n_checks = 0


def check(name: str, cond: bool) -> None:
    global n_checks
    n_checks += 1
    if not cond:
        raise SystemExit(f"FAIL: {name}")
    print(f"  ok  {name}")


# -- chance_adjusted / task_intelligence -------------------------------------

check("chance_adjusted: at chance == 0 score", math.isclose(S.chance_adjusted(0.3, 0.3), 0.0))
check("chance_adjusted: below chance clamps to 0", S.chance_adjusted(0.1, 0.5) == 0.0)
check("chance_adjusted: perfect score is 1.0 regardless of chance", math.isclose(S.chance_adjusted(1.0, 0.9207), 1.0))

# PRD §14.2's headline demonstration: an all-"No" model on civil_comments/is_toxic.
# 92.07% raw accuracy (the corpus's real majority class fraction).
REAL_CHANCE_IS_TOXIC = 0.920729
WRONG_CHANCE_SHIPPED_BEFORE_FIX = 0.5
raw_acc = 0.920729
i_correct = S.chance_adjusted(raw_acc, REAL_CHANCE_IS_TOXIC)
i_wrong = S.chance_adjusted(raw_acc, WRONG_CHANCE_SHIPPED_BEFORE_FIX)
check("PRD §14.2 demonstration: I ~ 0 at the REAL chance floor", math.isclose(i_correct, 0.0, abs_tol=1e-9))
check(
    "PRD §14.2 demonstration: the pre-fix chance (0.5) would have scored this model ~0.841 (flattering, wrong)",
    math.isclose(i_wrong, 0.841, abs_tol=1e-3),
)
print(f"  (for the record: correct I={i_correct:.6f}, pre-fix-bug I would have been={i_wrong:.6f})")

# task_intelligence: acc's denominator is every ATTEMPTED item (PRD §14.3),
# out_of_schema/declined already folded into (n_attempted - n_correct).
ti = S.task_intelligence(n_correct=750, n_attempted=1000, chance=0.25)
check("task_intelligence: acc = n_correct / n_attempted", math.isclose(ti["acc"], 0.75))
check("task_intelligence: i uses chance_adjusted", math.isclose(ti["i"], S.chance_adjusted(0.75, 0.25)))

# aggregate_intelligence: task -> dataset -> domain -> axis, equal weight at every level.
task_i = {
    "civil_comments/is_toxic": 0.5,
    "civil_comments/toxicity_level": 0.3,
    "cfpb_complaints/product": 0.9,
    "banking77/intent": 0.7,
}
agg = S.aggregate_intelligence(
    task_i,
    task_to_dataset={
        "civil_comments/is_toxic": "civil_comments",
        "civil_comments/toxicity_level": "civil_comments",
        "cfpb_complaints/product": "cfpb_complaints",
        "banking77/intent": "banking77",
    },
    dataset_to_domain={"civil_comments": "nlp", "cfpb_complaints": "finance", "banking77": "nlp"},
)
check("aggregate_intelligence: dataset mean over its own tasks only", math.isclose(agg["by_dataset"]["civil_comments"], 0.4))
check("aggregate_intelligence: single-task dataset == that task", agg["by_dataset"]["cfpb_complaints"] == 0.9)
# nlp domain = mean(civil_comments=0.4, banking77=0.7) = 0.55; finance domain = 0.9
# Intelligence = 100 * mean(0.55, 0.9) = 72.5 -- equal weight per DOMAIN, not per dataset,
# which is exactly the point: CFPB (1 dataset) counts as much as nlp's 2 datasets combined.
check("aggregate_intelligence: nlp domain averages its 2 datasets equally", math.isclose(agg["by_domain"]["nlp"], 0.55))
check("aggregate_intelligence: Intelligence is 100 * mean over PRESENT domains", math.isclose(agg["intelligence"], 72.5))

# -- Breadth / Generality (PRD §14.5) ----------------------------------------

check("stratum_value: macro mean of task I values", math.isclose(S.stratum_value([0.2, 0.8]), 0.5))
check("family_value: geomean over populated strata", math.isclose(S.family_value([0.25, 1.0]), 0.5))
check("family_value: empty -> None (all strata excluded)", S.family_value([]) is None)
check("breadth: geomean over 3 family values", math.isclose(S.breadth([1.0, 1.0, 1.0]), 1.0))
check("breadth: a single zero family zeroes the whole geomean", S.breadth([0.0, 0.5, 0.9]) == 0.0)
check("breadth: no families -> None", S.breadth([]) is None)
check(
    "generality: 100*sqrt(breadth*coverage)",
    math.isclose(S.generality(0.25, 0.64, n_families_dropped=0), 100.0 * math.sqrt(0.25 * 0.64)),
)
check("generality: null (not 0) when >=2 families dropped", S.generality(0.5, 1.0, n_families_dropped=2) is None)
check("generality: still null with 3 dropped even if breadth were somehow given", S.generality(0.0, 1.0, n_families_dropped=3) is None)

# -- pooled_ece (ported from the sibling's eval/metrics.py) -------------------

check("pooled_ece: empty pool -> None", S.pooled_ece([]) is None)
# A perfectly calibrated 2-bin-worth of predictions: p=1.0 always correct, p=0.0 always wrong.
perfect = [(1.0, 1)] * 20 + [(0.0, 0)] * 20
check("pooled_ece: perfectly calibrated pool -> ~0", math.isclose(S.pooled_ece(perfect, n_bins=2), 0.0, abs_tol=1e-9))
# A systematically overconfident pool: always says p=0.9 but is only right half the time.
overconfident = [(0.9, 1)] * 50 + [(0.9, 0)] * 50
check("pooled_ece: overconfident pool -> ~0.4", math.isclose(S.pooled_ece(overconfident, n_bins=1), 0.4, abs_tol=1e-9))

# -- hcs / Calibration axis ----------------------------------------------------

check("hcs: perfect accuracy + perfect calibration (ece=0) -> 1.0", math.isclose(S.hcs(1.0, 0.0), 1.0))
check("hcs: zero accuracy -> 0", S.hcs(0.0, 0.0) == 0.0)

# -- Speed / Cost frozen reference scales (PRD §14.6) -------------------------

check("speed_score: 10ms -> 100", math.isclose(S.speed_score(10.0), 100.0, abs_tol=1e-9))
check("speed_score: 100ms -> 50", math.isclose(S.speed_score(100.0), 50.0, abs_tol=1e-9))
check("speed_score: 1000ms -> 0", math.isclose(S.speed_score(1000.0), 0.0, abs_tol=1e-9))
check("speed_score: clamps above 1000ms to 0, not negative", S.speed_score(50_000.0) == 0.0)
check("cost_score: $0.01/1k -> 100", math.isclose(S.cost_score(0.01), 100.0, abs_tol=1e-9))
check("cost_score: $1/1k -> 50", math.isclose(S.cost_score(1.0), 50.0, abs_tol=1e-9))
check("cost_score: $100/1k -> 0", math.isclose(S.cost_score(100.0), 0.0, abs_tol=1e-9))

# -- combine_axes: geomean, THEN floor penalty, THEN null propagation --------

score, score_no_cost, floor = S.combine_axes(80.0, 80.0, 80.0, 80.0, 80.0)
check("combine_axes: all-equal axes geomean == that value, no floor penalty", math.isclose(score, 80.0))
check("combine_axes: floor_penalty_applied is False when Intelligence >= 50", floor is False)
check("combine_axes: score_no_cost is None when cost IS given", score_no_cost is None)

score, score_no_cost, floor = S.combine_axes(40.0, 80.0, 80.0, 80.0, 80.0)
check("combine_axes: floor penalty applies below Intelligence=50", floor is True)
g = S.geomean([40.0, 80.0, 80.0, 80.0, 80.0])
check("combine_axes: floor penalty is g * (I/50)**2, in that order", math.isclose(score, g * (40.0 / 50.0) ** 2))

score, score_no_cost, floor = S.combine_axes(80.0, 80.0, 80.0, 80.0, None)
check("combine_axes: score is None when ANY axis is None (null propagates)", score is None)
check("combine_axes: score_no_cost IS populated when cost is None", score_no_cost is not None)
check(
    "combine_axes: score_no_cost is the geomean of the other four",
    math.isclose(score_no_cost, S.geomean([80.0, 80.0, 80.0, 80.0])),
)

score, _, _ = S.combine_axes(None, 80.0, 80.0, 80.0, 80.0)
check("combine_axes: score is None when Intelligence itself is None", score is None)

print(f"\n{n_checks} checks run, 0 failed.")

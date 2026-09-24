"""The run loop (PRD §14.9 item 3) and its request/response contract (§14.7).

`evaluate()` is the one function `bjb evaluate` (the CLI) and `POST
/v1/evaluate` (the FastAPI wrapper, `better_jev_bench/server.py`) both call --
PRD §14.7's explicit requirement that the library works with no server at
all, and that a hosted service is a thin wrapper rather than where the logic
lives. **No scoring arithmetic lives in this file** -- every number that ends
up in the response comes from `score.py`; this file's whole job is selecting
items, calling a `ModelBackend`, classifying each response, and accumulating
the counts and pools that `score.py`'s pure functions turn into axes.

Deterministic given `(slice, filters, shuffle_seed)`: item selection sorts by
`item_id` before any cap is applied, dict iteration is always over a sorted
key list, and the only randomness (option shuffling) is seeded per item from
`shuffle_seed` rather than drawn from a shared PRNG stream, so re-running the
identical request reproduces the identical permutation for every item
independent of concurrency or ordering.
"""

from __future__ import annotations

import base64
import collections
import gzip
import hashlib
import json
import platform
import time
import uuid
from pathlib import Path
from typing import Any

from . import CORPUS_VERSION, SPEC_VERSION, __version__
from .backend import ClassifyResult, Declined, ModelBackend, Outcome, classify
from .build import read_jsonl_gz
from .imagecache import image_cache_path
from .manifest import Manifest, discover
from .score import (
    MIN_ITEMS_FOR_ECE,
    MIN_ITEMS_FOR_STRATUM,
    aggregate_intelligence,
    breadth,
    combine_axes,
    cost_score,
    family_value,
    generality,
    hcs,
    pooled_ece,
    speed_score,
    stratum_value,
    task_intelligence,
)
from .types import Item, width_stratum

DEFAULT_SHUFFLE_SEED = 20260924
DEFAULT_POSITION_SENSITIVITY = {"enabled": True, "sample_per_task": 200}
_ERROR_SAMPLE_CAP = 20
#: PRD §14.7: the request field's default is the enabled dict above, and
#: passing `null` *explicitly* disables it -- two different things a plain
#: `= None` parameter default can't distinguish (was `None` the caller's
#: choice, or just "the argument wasn't given"?). This sentinel keeps them
#: apart: omit the keyword entirely to get the enabled default; pass
#: `position_sensitivity=None` to actually disable it. A prior version of
#: this function used `None` as the parameter default and treated it as "use
#: the default," which made disabling the diagnostic impossible from the
#: Python API -- caught while writing `bjb evaluate`'s
#: `--no-position-sensitivity` flag against this exact function.
_UNSET: Any = object()


class EvaluateError(ValueError):
    """Raised for a malformed request (PRD §14.7's `422`) -- the FastAPI
    wrapper turns this into the HTTP status; the CLI just prints it."""


# ---------------------------------------------------------------------------
# Item selection
# ---------------------------------------------------------------------------


def _slice_path(repo_root: Path, dataset: str, slice_name: str) -> Path:
    if slice_name == "heldout":
        return Path(repo_root) / "bench" / "heldout" / f"{dataset}.jsonl.gz"
    return Path(repo_root) / "data" / "public" / f"{dataset}.jsonl.gz"


def _task_id(dataset: str, task: str) -> str:
    return f"{dataset}/{task}"


def select_items(
    repo_root: Path,
    *,
    slice_name: str,
    datasets: list[str] | None,
    tasks: list[str] | None,
    license_tiers: list[str],
    modalities: list[str],
    primitives: list[str],
    max_items_per_task: int | None,
) -> tuple[dict[str, list[Item]], dict[str, Manifest], dict[str, str], dict[str, str], dict[str, float]]:
    """Returns `(items_by_task, manifest_by_dataset, task_to_dataset,
    dataset_to_domain, chance_by_task)`, all keyed by `"<dataset>/<task>"`
    where that's the natural key. Selection order (PRD §14.7's request table):
    slice -> license_tiers -> datasets -> modalities/primitives -> tasks ->
    max_items_per_task, first N by `item_id` ascending (already the on-disk
    sort order `build.py` writes)."""
    manifests = [m for m in discover(repo_root) if m.tier in license_tiers]
    if datasets:
        wanted = set(datasets)
        manifests = [m for m in manifests if m.name in wanted]
    if not manifests:
        raise EvaluateError(f"no datasets match license_tiers={license_tiers!r} datasets={datasets!r}")

    items_by_task: dict[str, list[Item]] = {}
    manifest_by_dataset: dict[str, Manifest] = {}
    task_to_dataset: dict[str, str] = {}
    dataset_to_domain: dict[str, str] = {}
    chance_by_task: dict[str, float] = {}

    for m in manifests:
        manifest_by_dataset[m.name] = m
        dataset_to_domain[m.name] = m.domain
        if m.modality not in modalities:
            continue
        path = _slice_path(repo_root, m.name, slice_name)
        if not path.exists():
            if slice_name == "heldout":
                raise EvaluateError(f"{m.name}: no frozen held-out slice at {path} -- run `bjb build` first")
            continue  # public slice is git-ignored/regenerated; absence is not an error (PRD §14.7)
        by_task: dict[str, list[Item]] = collections.defaultdict(list)
        for it in read_jsonl_gz(path):
            by_task[it.task].append(it)
        for te in m.tasks:
            if te.primitive not in primitives:
                continue
            task_id = _task_id(m.name, te.task)
            if tasks and task_id not in tasks:
                continue
            task_items = sorted(by_task.get(te.task, []), key=lambda it: it.item_id)
            if not task_items:
                continue
            if max_items_per_task is not None:
                task_items = task_items[:max_items_per_task]
            items_by_task[task_id] = task_items
            task_to_dataset[task_id] = m.name
            chance_by_task[task_id] = te.chance

    if not items_by_task:
        raise EvaluateError("request filters selected zero tasks")
    return items_by_task, manifest_by_dataset, task_to_dataset, dataset_to_domain, chance_by_task


# ---------------------------------------------------------------------------
# Option shuffling (PRD §14.6)
# ---------------------------------------------------------------------------


def shuffle_permutation(item_id: str, n_options: int, *, seed: int) -> list[int]:
    """A deterministic permutation of `range(n_options)`, seeded by
    `sha256(seed || item_id)` (PRD §14.6) -- replayable from the two values
    alone, with no shared PRNG stream to make ordering matter across items or
    concurrency. Fisher-Yates, drawing a fresh digest per swap position
    (keyed additionally by that position) rather than one shared digest, so
    the number of options never bounds the entropy available."""
    perm = list(range(n_options))
    for i in range(len(perm) - 1, 0, -1):
        digest = hashlib.sha256(f"{seed}\x1f{item_id}\x1f{i}".encode()).digest()
        j = int.from_bytes(digest[:8], "big") % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _permute_options(options: tuple[str, ...], perm: list[int]) -> list[str]:
    return [options[i] for i in perm]


# ---------------------------------------------------------------------------
# Request construction (PRD §6.1, §13.4/§13.5, §14.7 `images` modes)
# ---------------------------------------------------------------------------


def _build_request(
    repo_root: Path, item: Item, dataset: str, presented_options: list[str], *, images_mode: str
) -> tuple[dict[str, Any], bool]:
    """Returns `(request_body, is_leaderboard_eligible)`. The second value is
    only ever `False` here for the `images.mode == "omit"` case on an
    image-bearing item (PRD §14.7: "scoring a vision item with the picture
    removed is not a vision result")."""
    q = item.question
    question: dict[str, Any] = {"type": q.type, "instructions": q.instructions, "options": presented_options}
    if q.ordinal:
        question["ordinal"] = True
    body: dict[str, Any] = {"state": item.state, "questions": {q.key: question}}
    eligible = True
    if item.images:
        if images_mode == "omit":
            eligible = False
        else:
            imgs = []
            for img in item.images:
                ref = img.to_json()
                if images_mode == "inline_base64":
                    path = image_cache_path(repo_root, dataset, img.sha256, img.media_type)
                    if not path.exists():
                        raise EvaluateError(
                            f"images.mode=inline_base64 but {path} is not cached locally -- "
                            "run `bjb build` for this dataset first"
                        )
                    ref["data"] = base64.b64encode(path.read_bytes()).decode("ascii")
                imgs.append(ref)
            body["images"] = imgs
    return body, eligible


# ---------------------------------------------------------------------------
# The run loop
# ---------------------------------------------------------------------------


def evaluate(
    repo_root: Path,
    backend: ModelBackend,
    *,
    model: str,
    slice: str = "heldout",  # noqa: A002 - matches the PRD §14.7 field name
    datasets: list[str] | None = None,
    tasks: list[str] | None = None,
    license_tiers: list[str] | None = None,
    modalities: list[str] | None = None,
    primitives: list[str] | None = None,
    max_items_per_task: int | None = None,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    position_sensitivity: dict[str, Any] | None = _UNSET,
    images: dict[str, Any] | None = None,
    request: dict[str, Any] | None = None,
    cost_model: dict[str, Any] | None = None,
    spec_version: str = SPEC_VERSION,
) -> dict[str, Any]:
    """PRD §14.7's whole contract, minus the HTTP envelope. Every keyword
    argument and its default mirrors the request table exactly; the return
    value is exactly the `200` response body (`server.py` adds nothing but
    the envelope, per the same section).
    """
    if spec_version != SPEC_VERSION:
        raise EvaluateError(f"spec_version {spec_version!r} != engine's {SPEC_VERSION!r}")
    # PRD §14.7 / §5.5 rule 6: "verified, not merely logged" -- this engine has
    # no external revision registry to check a git SHA or weight hash against,
    # so the checkable half of "verified" it can enforce today is structural:
    # the `@<name>-part` must be present and non-empty. A registry-backed
    # check is future work, not silently downgraded to "trust the string."
    if "@" not in model or not all(part.strip() for part in model.split("@", 1)):
        raise EvaluateError(f"model {model!r} must be '<name>@<git-sha|weight-hash>' with both parts non-empty")

    repo_root = Path(repo_root)
    license_tiers = list(license_tiers) if license_tiers else ["A"]
    modalities = list(modalities) if modalities else ["text", "image"]
    primitives = list(primitives) if primitives else ["choice", "score", "noul"]
    images = images or {"mode": "reference"}
    request = request or {"timeout_s": 30.0, "concurrency": 1, "headers": {}}
    pos_sens_cfg = DEFAULT_POSITION_SENSITIVITY if position_sensitivity is _UNSET else position_sensitivity
    timeout_s = float(request.get("timeout_s", 30.0))
    concurrency = int(request.get("concurrency", 1))

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    run_id = f"run_{uuid.uuid4().hex[:20]}"

    items_by_task, manifest_by_dataset, task_to_dataset, dataset_to_domain, chance_by_task = select_items(
        repo_root,
        slice_name=slice,
        datasets=datasets,
        tasks=tasks,
        license_tiers=license_tiers,
        modalities=modalities,
        primitives=primitives,
        max_items_per_task=max_items_per_task,
    )

    raw_rows: list[dict[str, Any]] = []
    per_task_correct: dict[str, int] = collections.defaultdict(int)
    per_task_attempted: dict[str, int] = collections.defaultdict(int)
    per_task_ece_pairs: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    pooled_pairs: list[tuple[float, int]] = []
    pooled_correct_covered = 0
    n_covered = 0
    n_out_of_schema = 0
    n_declined = 0
    n_no_confidence = 0
    n_attempted_total = 0
    n_ineligible = 0
    chance_drift: list[dict[str, Any]] = []
    seen_drift_tasks: set[str] = set()
    latency_by_family: dict[str, list[float]] = {"text": [], "multimodal": []}
    latency_source_by_family: dict[str, set[str]] = {"text": set(), "multimodal": set()}
    http_status_counts: collections.Counter[str] = collections.Counter()
    error_samples: list[dict[str, Any]] = []
    ordinal_confusion: dict[str, dict[str, Any]] = {}

    def _run_one(task_id: str, item: Item, *, seed: int) -> dict[str, Any]:
        dataset = task_to_dataset[task_id]
        perm = shuffle_permutation(item.item_id, len(item.question.options), seed=seed) \
            if not item.question.ordinal else list(range(len(item.question.options)))
        presented = _permute_options(item.question.options, perm)
        req_body, eligible = _build_request(repo_root, item, dataset, presented, images_mode=images.get("mode", "reference"))

        t_call = time.perf_counter()
        try:
            resp = backend.answer(req_body, timeout_s=timeout_s)
            http_status_counts["200"] += 1
        except Declined as exc:
            http_status_counts[str(exc.status)] += 1
            if len(error_samples) < _ERROR_SAMPLE_CAP:
                error_samples.append({"item_id": item.item_id, "status": exc.status, "body": exc.detail[:2000]})
            return {
                "item_id": item.item_id, "task": task_id, "permutation": perm,
                "outcome": Outcome.DECLINED.value, "p": None, "latency_ms": None, "eligible": eligible,
                "http_status": exc.status,
            }
        engine_latency_ms = (time.perf_counter() - t_call) * 1000.0

        key = item.question.key
        result = resp.get("results", {}).get(key)
        cr: ClassifyResult = classify(result, options=item.question.options, expected=item.label)

        latency_ms = engine_latency_ms
        latency_source = "engine_wallclock"
        usage = resp.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("latency_ms"), (int, float)):
            latency_ms = float(usage["latency_ms"])
            latency_source = resp.get("_latency_source", "server_reported")

        return {
            "item_id": item.item_id, "task": task_id, "permutation": perm,
            "outcome": cr.outcome.value, "p": cr.p, "returned": cr.returned,
            "latency_ms": latency_ms, "latency_source": latency_source, "eligible": eligible,
            "http_status": 200,
        }

    for task_id in sorted(items_by_task):
        m = manifest_by_dataset[task_to_dataset[task_id]]
        te = next(t for t in m.tasks if _task_id(m.name, t.task) == task_id)
        chance = chance_by_task[task_id]
        modality_family = "text" if m.modality == "text" else "multimodal"

        for item in items_by_task[task_id]:
            if item.chance is not None and abs(item.chance - chance) > 1e-9 and task_id not in seen_drift_tasks:
                chance_drift.append(
                    {"task": task_id, "manifest": chance, "item_json": item.chance, "used": chance}
                )
                seen_drift_tasks.add(task_id)

            row = _run_one(task_id, item, seed=shuffle_seed)
            raw_rows.append(row)
            n_attempted_total += 1
            per_task_attempted[task_id] += 1
            if not row["eligible"]:
                n_ineligible += 1

            outcome = Outcome(row["outcome"])
            if outcome in (Outcome.CORRECT, Outcome.INCORRECT):
                n_covered += 1
                if outcome is Outcome.CORRECT:
                    per_task_correct[task_id] += 1
                    pooled_correct_covered += 1
                p = row["p"]
                if p is None:
                    n_no_confidence += 1
                else:
                    y = 1 if outcome is Outcome.CORRECT else 0
                    pooled_pairs.append((p, y))
                    per_task_ece_pairs[task_id].append((p, y))
                if te.primitive == "score" and row.get("returned") is not None:
                    _accumulate_ordinal(ordinal_confusion, task_id, item.question.options, item.label, row["returned"])
                if row["latency_ms"] is not None:
                    latency_by_family[modality_family].append(row["latency_ms"])
                    latency_source_by_family[modality_family].add(row.get("latency_source", "unknown"))
            elif outcome is Outcome.OUT_OF_SCHEMA:
                n_out_of_schema += 1
            else:
                n_declined += 1

    # -- Intelligence (PRD §14.1/§14.2/§14.3) ---------------------------------
    task_i: dict[str, float] = {}
    by_task_detail: dict[str, Any] = {}
    for task_id, n_attempted in per_task_attempted.items():
        chance = chance_by_task[task_id]
        detail = task_intelligence(per_task_correct[task_id], n_attempted, chance)
        n_scored = len(per_task_ece_pairs[task_id])
        detail["chance_source"] = "manifest"
        detail["calibration_bearing"] = n_scored >= MIN_ITEMS_FOR_ECE
        by_task_detail[task_id] = detail
        task_i[task_id] = detail["i"]

    agg = aggregate_intelligence(task_i, task_to_dataset=task_to_dataset, dataset_to_domain=dataset_to_domain)

    # -- Coverage (PRD §14.3) -------------------------------------------------
    coverage_fraction = n_covered / n_attempted_total if n_attempted_total else 0.0
    coverage = {
        "attempted": n_attempted_total, "answered": n_covered,
        "out_of_schema": n_out_of_schema, "declined": n_declined,
        "fraction": coverage_fraction,
    }

    # -- Breadth / Generality (PRD §14.5) -------------------------------------
    strata_out, families_out, excluded_strata = _strata_and_families(
        task_i, task_to_dataset, manifest_by_dataset, per_task_attempted
    )
    family_values = [v for v in families_out.values() if v is not None]
    n_families_dropped = sum(1 for v in families_out.values() if v is None)
    breadth_value = breadth(family_values)
    generality_value = generality(breadth_value, coverage_fraction, n_families_dropped=n_families_dropped)

    # -- Calibration (PRD §14.6) ----------------------------------------------
    pooled_ece_value = pooled_ece(pooled_pairs) if pooled_pairs else None
    pooled_accuracy = pooled_correct_covered / n_covered if n_covered else 0.0
    calibration_value = 100.0 * hcs(pooled_accuracy, pooled_ece_value) if pooled_ece_value is not None else None
    if pooled_ece_value is not None and n_covered and (n_no_confidence / n_covered) > 0.05:
        calibration_value = None

    per_dataset_ece = _grouped_ece(per_task_ece_pairs, task_to_dataset, by_dataset=True)
    per_stratum_ece = _stratum_ece(per_task_ece_pairs, task_to_dataset, manifest_by_dataset)

    # -- Speed / Cost (PRD §14.6) ---------------------------------------------
    text_lat = latency_by_family["text"]
    mm_lat = latency_by_family["multimodal"]
    speed_value = _speed_axis(text_lat) if text_lat else None
    speed_multimodal = _speed_axis(mm_lat) if mm_lat else None
    if concurrency > 1:
        speed_value = None

    cost_value = None
    if cost_model:
        hourly_rate = float(cost_model["hourly_rate"])
        n_decisions = n_attempted_total
        duration_hours = max((time.time() - t0) / 3600.0, 1e-9)
        throughput = n_decisions / duration_hours
        usd_per_1k = (hourly_rate / throughput) * 1000.0 if throughput > 0 else float("inf")
        cost_value = cost_score(usd_per_1k) if usd_per_1k > 0 else 0.0

    score, score_no_cost, floor_penalty_applied = combine_axes(
        agg["intelligence"], calibration_value, generality_value, speed_value, cost_value
    )

    # PRD §14.7 states two concrete triggers for `is_leaderboard_eligible`:
    # `slice != "heldout"` is *always* false, and `images.mode == "omit"`
    # stamps every image-bearing item false (we aggregate that per-item flag
    # up to the run level -- if any item was degraded, the run's headline
    # number was not a full vision result). It does NOT say `concurrency > 1`
    # or a missing `cost_model` disqualify a run -- those already have their
    # own explicit, narrower consequences (`speed: null`, `score_no_cost`
    # instead of `score`), and a self-hosted model with no list price is
    # exactly the case `score_no_cost` exists to keep publishable. Do not
    # add conditions beyond what §14.7 actually states.
    is_eligible = slice == "heldout" and n_ineligible == 0

    duration_s = round(time.time() - t0, 3)
    domains_present = sorted(agg["by_domain"])
    scored_datasets = sorted(set(task_to_dataset.values()))
    notes: dict[str, Any] = {"provisional": True}
    if "finance" in domains_present:
        finance_datasets = [d for d in scored_datasets if manifest_by_dataset[d].domain == "finance"]
        if len(finance_datasets) == 1:
            notes["domain_concentration"] = (
                f"finance is 1 dataset ({finance_datasets[0]}) of {len(scored_datasets)} scored and carries "
                f"{round(100 / len(domains_present))}% of Intelligence (PRD §14.1)"
            )
    declined_primitives = _declined_primitives(raw_rows, items_by_task, task_to_dataset, manifest_by_dataset)
    if declined_primitives:
        notes["primitive_unsupported_by_endpoint"] = declined_primitives

    response: dict[str, Any] = {
        "run_id": run_id,
        "spec_version": SPEC_VERSION,
        "corpus_version": CORPUS_VERSION,
        "model": model,
        "endpoint": getattr(backend, "endpoint", None),
        "started_at": started_at,
        "duration_s": duration_s,
        "score": score,
        "score_no_cost": score_no_cost,
        "floor_penalty_applied": floor_penalty_applied,
        "is_leaderboard_eligible": is_eligible,
        "axes": {
            "intelligence": agg["intelligence"],
            "calibration": calibration_value,
            "generality": generality_value,
            "speed": speed_value,
            "cost": cost_value,
            "speed_multimodal": speed_multimodal,
        },
        "coverage": coverage,
        "intelligence_detail": {
            "by_task": by_task_detail,
            "by_dataset": agg["by_dataset"],
            "by_domain": agg["by_domain"],
        },
        "strata": strata_out,
        "families": families_out,
        "breadth": breadth_value,
        "excluded_strata": excluded_strata,
        "diagnostics": {
            "pooled_ece": pooled_ece_value,
            "pooled_accuracy": pooled_accuracy,
            "ece_bins": 15,
            "ece_binning": "equal_mass",
            "per_dataset_ece": per_dataset_ece,
            "per_stratum_ece": per_stratum_ece,
            "ordinal_distance": _ordinal_summary(ordinal_confusion),
            "abstention": None,
            "position_sensitivity": _position_sensitivity(
                backend, items_by_task, task_to_dataset, manifest_by_dataset, chance_by_task,
                per_task_correct, per_task_attempted, shuffle_seed, timeout_s, images, repo_root, pos_sens_cfg,
            ),
            "n_no_confidence": n_no_confidence,
            "chance_source_drift": chance_drift,
            "latency_ms": {
                fam: {
                    "p50": _percentile(vals, 50), "p95": _percentile(vals, 95), "n": len(vals),
                    "source": "server_reported" if latency_source_by_family[fam] == {"server_reported"} else "mixed_or_engine_wallclock",
                }
                for fam, vals in latency_by_family.items() if vals
            },
            "http_status_counts": dict(http_status_counts),
            "error_samples": error_samples,
        },
        "notes": notes,
        "evidence_bundle": f"results/{run_id}/manifest.json",
    }
    # PRD §14.9 item 4 / dev-guidelines rule 10: the evidence bundle carries
    # more than the wire response -- `model_info` (whatever `backend.describe()`
    # says: weight hash, endpoint, checkpoint revision) and the exact frozen
    # slice hashes this run was scored against. Both are private-underscore
    # keys stripped before the HTTP response goes out (§14.7's contract is
    # exactly the fields documented there, no more), and consumed only by
    # `write_evidence_bundle()`.
    response["_raw_rows"] = raw_rows
    response["_model_info"] = backend.describe() if hasattr(backend, "describe") else {}
    response["_datasets_used"] = sorted(set(task_to_dataset.values()))
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _speed_axis(latencies_ms: list[float]) -> float:
    # `speed_score` is undefined at/below 0ms (log10); a same-process mock
    # backend can legitimately report ~0ms, so floor at 1 microsecond rather
    # than let a real (if absurdly fast) run crash the whole response.
    return speed_score(max(_percentile(latencies_ms, 50), 1e-3))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _strata_and_families(
    task_i: dict[str, float],
    task_to_dataset: dict[str, str],
    manifest_by_dataset: dict[str, Manifest],
    per_task_attempted: dict[str, int],
) -> tuple[dict[str, Any], dict[str, float | None], list[dict[str, Any]]]:
    """PRD §14.5: bucket every scored task into its width/primitive/modality
    stratum, macro-mean `I_t` within each populated stratum, geomean within
    each family, and report exclusions rather than silently dropping them."""
    family_of_stratum: dict[str, str] = {}
    stratum_tasks: dict[str, list[str]] = collections.defaultdict(list)
    stratum_items: dict[str, int] = collections.defaultdict(int)

    for task_id in task_i:
        m = manifest_by_dataset[task_to_dataset[task_id]]
        te = next(t for t in m.tasks if _task_id(m.name, t.task) == task_id)
        width = width_stratum(te.option_count)
        prim = f"prim_{te.primitive}"
        mod = "mod_text" if m.modality == "text" else "mod_multimodal"
        for s, fam in ((width, "width"), (prim, "primitive"), (mod, "modality")):
            family_of_stratum[s] = fam
            stratum_tasks[s].append(task_id)
            stratum_items[s] += per_task_attempted[task_id]

    strata_out: dict[str, Any] = {}
    excluded: list[dict[str, Any]] = []
    by_family: dict[str, list[float]] = collections.defaultdict(list)
    for s in sorted(stratum_tasks):
        fam = family_of_stratum[s]
        n_items = stratum_items[s]
        if n_items < MIN_ITEMS_FOR_STRATUM:
            strata_out[s] = {
                "family": fam, "items": n_items,
                "answered": n_items, "tasks": len(stratum_tasks[s]),
                "value": None, "status": "excluded",
            }
            excluded.append({"stratum": s, "reason": "benchmark_underpopulated", "items": n_items})
            continue
        value = stratum_value([task_i[t] for t in stratum_tasks[s]])
        status = "model_declined" if value == 0.0 else "scored"
        strata_out[s] = {
            "family": fam, "items": n_items, "answered": n_items,
            "tasks": len(stratum_tasks[s]), "value": value, "status": status,
        }
        by_family[fam].append(value)

    families_out: dict[str, float | None] = {}
    for fam in ("width", "primitive", "modality"):
        families_out[fam] = family_value(by_family.get(fam, []))
    return strata_out, families_out, excluded


def _grouped_ece(
    per_task_pairs: dict[str, list[tuple[float, int]]], task_to_dataset: dict[str, str], *, by_dataset: bool
) -> dict[str, float | None]:
    grouped: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    for task_id, pairs in per_task_pairs.items():
        key = task_to_dataset[task_id] if by_dataset else task_id
        grouped[key].extend(pairs)
    return {k: pooled_ece(v) for k, v in sorted(grouped.items())}


def _stratum_ece(
    per_task_pairs: dict[str, list[tuple[float, int]]],
    task_to_dataset: dict[str, str],
    manifest_by_dataset: dict[str, Manifest],
) -> dict[str, float | None]:
    grouped: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    for task_id, pairs in per_task_pairs.items():
        m = manifest_by_dataset[task_to_dataset[task_id]]
        te = next(t for t in m.tasks if _task_id(m.name, t.task) == task_id)
        for s in (width_stratum(te.option_count), f"prim_{te.primitive}", "mod_text" if m.modality == "text" else "mod_multimodal"):
            grouped[s].extend(pairs)
    return {k: pooled_ece(v) for k, v in sorted(grouped.items())}


def _accumulate_ordinal(
    store: dict[str, dict[str, Any]], task_id: str, options: tuple[str, ...], gold: str, returned: str
) -> None:
    entry = store.setdefault(task_id, {"options": options, "confusion": collections.Counter()})
    returned_stripped = returned.strip()
    if returned_stripped in options:
        entry["confusion"][(options.index(gold), options.index(returned_stripped))] += 1


def _ordinal_summary(store: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for task_id, entry in store.items():
        options = entry["options"]
        n = len(options)
        confusion = [[0] * n for _ in range(n)]
        total = 0
        abs_dist_sum = 0
        off_by_one = 0
        for (gi, ri), count in entry["confusion"].items():
            confusion[gi][ri] += count
            total += count
            abs_dist_sum += abs(gi - ri) * count
            if abs(gi - ri) == 1:
                off_by_one += count
        if total == 0:
            continue
        out[task_id] = {
            "mean_abs_distance": abs_dist_sum / total,
            "off_by_one_rate": off_by_one / total,
            "confusion": confusion,
        }
    return out if out else None


def _declined_primitives(
    raw_rows: list[dict[str, Any]],
    items_by_task: dict[str, list[Item]],
    task_to_dataset: dict[str, str],
    manifest_by_dataset: dict[str, Manifest],
) -> list[str]:
    """PRD §14.4b: name a primitive as endpoint-unsupported when *every* item
    of that primitive in the run came back `declined` with a 5xx -- the
    honest way to distinguish "the wire contract doesn't support this" from
    "the model got some of these wrong.\""""
    by_primitive_total: collections.Counter[str] = collections.Counter()
    by_primitive_declined_5xx: collections.Counter[str] = collections.Counter()
    task_primitive: dict[str, str] = {}
    for task_id in items_by_task:
        m = manifest_by_dataset[task_to_dataset[task_id]]
        te = next(t for t in m.tasks if _task_id(m.name, t.task) == task_id)
        task_primitive[task_id] = te.primitive
    for row in raw_rows:
        prim = task_primitive[row["task"]]
        by_primitive_total[prim] += 1
        if row["outcome"] == Outcome.DECLINED.value and isinstance(row.get("http_status"), int) and row["http_status"] >= 500:
            by_primitive_declined_5xx[prim] += 1
    return sorted(p for p in by_primitive_total if by_primitive_declined_5xx[p] == by_primitive_total[p])


def _position_sensitivity(
    backend: ModelBackend,
    items_by_task: dict[str, list[Item]],
    task_to_dataset: dict[str, str],
    manifest_by_dataset: dict[str, Manifest],
    chance_by_task: dict[str, float],
    per_task_correct: dict[str, int],
    per_task_attempted: dict[str, int],
    shuffle_seed: int,
    timeout_s: float,
    images: dict[str, Any],
    repo_root: Path,
    cfg: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """PRD §14.6: re-run a fixed subsample under a second seed and report the
    delta -- a *finding*, never averaged into the primary run. Restricted to
    non-ordinal `choice`/`noul` tasks, since an ordinal `score` question is
    never shuffled in the primary run either and re-running it would measure
    nothing."""
    if not cfg or not cfg.get("enabled", True):
        return None
    sample_per_task = int(cfg.get("sample_per_task", 200))
    reseed = int(cfg.get("reseed", shuffle_seed + 1))

    n = 0
    n_correct_primary = 0
    n_correct_reseed = 0
    for task_id, items in items_by_task.items():
        m = manifest_by_dataset[task_to_dataset[task_id]]
        te = next(t for t in m.tasks if _task_id(m.name, t.task) == task_id)
        if te.ordinal:
            continue
        dataset = task_to_dataset[task_id]
        subsample = items[:sample_per_task]
        for item in subsample:
            for seed, counter_name in ((shuffle_seed, "primary"), (reseed, "reseed")):
                perm = shuffle_permutation(item.item_id, len(item.question.options), seed=seed)
                presented = _permute_options(item.question.options, perm)
                req_body, _ = _build_request(repo_root, item, dataset, presented, images_mode=images.get("mode", "reference"))
                try:
                    resp = backend.answer(req_body, timeout_s=timeout_s)
                except Declined:
                    continue
                result = resp.get("results", {}).get(item.question.key)
                cr = classify(result, options=item.question.options, expected=item.label)
                if counter_name == "primary" and cr.outcome is Outcome.CORRECT:
                    n_correct_primary += 1
                elif counter_name == "reseed" and cr.outcome is Outcome.CORRECT:
                    n_correct_reseed += 1
            n += 1
    if n == 0:
        return {"n": 0, "delta_accuracy": 0.0, "flagged": False}
    delta = (n_correct_reseed - n_correct_primary) / n
    return {"n": n, "delta_accuracy": round(delta, 6), "flagged": abs(delta) > 0.02}


# ---------------------------------------------------------------------------
# Evidence bundle (dev-guidelines rule 10, PRD §8.2/§14.9 item 4)
# ---------------------------------------------------------------------------


def _corpus_receipt_hashes(repo_root: Path, dataset_names: list[str]) -> dict[str, str]:
    """The committed held-out slice SHA-256 for every dataset this run
    actually scored, read straight from `bench/receipts/<name>.build.json` --
    the concrete "corpus receipt hashes" PRD §14.9 item 4 asks the evidence
    bundle to carry, so a result stays checkable against the exact frozen
    bytes it was scored against even after the corpus grows."""
    out: dict[str, str] = {}
    for name in sorted(dataset_names):
        p = Path(repo_root) / "bench" / "receipts" / f"{name}.build.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        out[name] = rec.get("files", {}).get("heldout", {}).get("sha256", "unknown")
    return out


def write_evidence_bundle(
    response: dict[str, Any], request_echo: dict[str, Any], out_dir: Path, *, repo_root: Path | None = None
) -> dict[str, Path]:
    """`results/<run_id>/manifest.json` (the full §14.7 response plus the
    request echo, `model_info` from `backend.describe()`, the held-out slice
    SHA-256 of every dataset actually scored, and host/timestamp) and
    `results/<run_id>/raw.jsonl.gz` (one row per item: id, presented
    permutation, outcome, `p`, latency). A number that cannot be pointed at
    does not go in the PRD (dev-guidelines rule 10). `repo_root` is optional
    only so this function stays callable in a unit test with no corpus on
    disk; both real callers (`cli.py`, `server.py`) always pass it."""
    out_dir = Path(out_dir)
    run_dir = out_dir / response["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = response.pop("_raw_rows", [])
    model_info = response.pop("_model_info", {})
    datasets_used = response.pop("_datasets_used", [])
    manifest = {
        **response,
        "request": request_echo,
        "model_info": model_info,
        "corpus_receipts": _corpus_receipt_hashes(repo_root, datasets_used) if repo_root else {},
        "host": platform.node(),
        "bjb_version": __version__,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    raw_path = run_dir / "raw.jsonl.gz"
    with gzip.GzipFile(raw_path, "wb", compresslevel=9) as fh:
        for row in raw_rows:
            fh.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))

    # Restore the popped private keys for any other caller in-process (e.g.
    # the CLI prints `result` right after calling this).
    response["_raw_rows"] = raw_rows
    response["_model_info"] = model_info
    response["_datasets_used"] = datasets_used
    return {"manifest": manifest_path, "raw": raw_path}

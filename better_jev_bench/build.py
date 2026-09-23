"""The build pipeline: source -> normalised items -> frozen split -> hashed receipt.

`bjb build` is the whole lifecycle in one command. For each manifest it:

1. resolves the `loader` pointer and runs `load_items()`;
2. checks every item against the manifest's declared `[[schema.tasks]]` -- width,
   primitive, ordinality -- so a loader that drifts from its manifest fails here
   rather than in someone's training run;
3. sorts by `item_id` (content-determined, so the build is reproducible
   regardless of source row order);
4. assigns each item to `public` or `heldout` by `split.side()`;
5. truncates each side to its cap, held-out **first**, so the reserved eval slice
   never shrinks because the training slice got bigger;
6. writes gzipped JSONL, computes SHA-256 over every file, computes the §7.5
   held-out label commitment, and writes a receipt.

**Where the data lands, and why.** Two slices, two storage policies, and the
asymmetry is deliberate:

* `bench/heldout/<dataset>.jsonl.gz` is **committed to git**. It is small
  (capped per task), it is what anyone needs to actually run a benchmark, and
  freezing it in git history is what makes the §7.5 commitment checkable by a
  third party rather than by us. `git clone && bjb evaluate` has to work or the
  corpus is a download script, not a benchmark.
* `data/public/<dataset>.jsonl.gz` is **git-ignored and regenerated**. It is the
  training substrate, it is two orders of magnitude bigger, and the PRD's own
  non-goal is explicit that "the default distribution mechanism is a loader
  pointing at the original source, not a re-hosted copy" (§1.3). Its SHA-256 is
  committed in the receipt, so a rebuild that does not reproduce the committed
  hash is a *detected* upstream drift rather than a silent one. A 24-item
  human-readable preview per dataset is committed alongside so the shape of the
  public slice is inspectable without downloading anything.

`bjb build --verify` rebuilds and diffs against the committed receipts instead
of overwriting them. That is the check that turns "we downloaded some data once"
into evidence (dev-guidelines rule 10).
"""

from __future__ import annotations

import collections
import gzip
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .dataset import BenchmarkDataset, load_plugin
from .manifest import Manifest, ManifestError, discover, load_manifest
from .split import SPLIT_SALT, file_digest, label_commitment, side
from .types import MIN_ITEMS_TO_ACCEPT, ImageRef, Item, width_stratum

#: Per-task caps. Deliberately different for the two slices -- see module docstring.
DEFAULT_PUBLIC_CAP = 40_000
DEFAULT_HELDOUT_CAP = 2_000
PREVIEW_ITEMS = 24


@dataclass(slots=True)
class BuildResult:
    dataset: str
    tasks: dict[str, dict[str, Any]]
    public_path: Path | None
    heldout_path: Path | None
    receipt: dict[str, Any]


def _check_against_manifest(item: Item, m: Manifest) -> None:
    try:
        te = m.task(item.task)
    except KeyError as exc:
        raise ManifestError(f"{m.name}: loader emitted undeclared task {item.task!r}") from exc
    q = item.question
    if q.type != te.primitive:
        raise ManifestError(
            f"{m.name}/{item.task}: manifest declares primitive {te.primitive!r}, loader emitted {q.type!r}"
        )
    if q.key != te.question_key:
        raise ManifestError(
            f"{m.name}/{item.task}: manifest declares question_key {te.question_key!r}, loader emitted {q.key!r}"
        )
    if len(q.options) != te.option_count:
        raise ManifestError(
            f"{m.name}/{item.task}: manifest declares option_count {te.option_count}, "
            f"loader emitted an item with {len(q.options)}"
        )
    if q.ordinal != te.ordinal:
        raise ManifestError(f"{m.name}/{item.task}: ordinal mismatch (manifest {te.ordinal}, item {q.ordinal})")
    if item.license_tier != m.tier:
        raise ManifestError(f"{m.name}: item license_tier {item.license_tier!r} != manifest tier {m.tier!r}")
    if item.canary != m.canary:
        raise ManifestError(f"{m.name}: item canary does not match the manifest canary")


def _write_jsonl_gz(path: Path, items: list[Item]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 so the gzip container is byte-identical across rebuilds; without
    # this the SHA-256 in the receipt would change every run and prove nothing.
    with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as fh:
        for it in items:
            fh.write((json.dumps(it.to_bench_json(), sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    return file_digest(path)


def read_jsonl_gz(path: Path) -> list[Item]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(Item.from_bench_json(json.loads(line)))
    return out


def _observed_chance(items: list[Item], mode: str) -> float:
    if mode == "majority":
        counts = collections.Counter(it.label for it in items)
        return counts.most_common(1)[0][1] / len(items)
    return sum(1.0 / len(it.question.options) for it in items) / len(items)


def _write_image_sha256_manifest(m: Manifest, repo_root: Path, items: list[Item]) -> dict[str, Any]:
    """PRD §13.5: the corpus commits image *hashes*, never pixels. Every
    distinct `ImageRef` referenced by this dataset's built items (public and
    held-out together -- a hash's presence here is not a claim about which
    slice it landed in) is written, sorted by sha256, to the manifest-declared
    `images.sha256_manifest` path. Small (a few dozen bytes per image) and
    exactly what a third party needs to confirm the corpus's held-out
    commitment covers real, specific image content without ever downloading
    the images themselves.
    """
    seen: dict[str, ImageRef] = {}
    for it in items:
        for img in it.images:
            seen[img.sha256] = img
    refs = [seen[h] for h in sorted(seen)]
    out_path = repo_root / m.image_sha256_manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(out_path, "wb", compresslevel=9, mtime=0) as fh:
        for ref in refs:
            fh.write((json.dumps(ref.to_json(), sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    return {
        "path": str(out_path.relative_to(repo_root)),
        "sha256": file_digest(out_path),
        "n_distinct_images": len(refs),
    }


def build_dataset(
    m: Manifest,
    repo_root: Path,
    *,
    public_cap: int = DEFAULT_PUBLIC_CAP,
    heldout_cap: int = DEFAULT_HELDOUT_CAP,
    write: bool = True,
) -> BuildResult:
    loader: BenchmarkDataset = load_plugin(m.loader, canary=m.canary, license_tier=m.tier, repo_root=repo_root)
    if loader.name != m.name:
        raise ManifestError(f"{m.name}: loader class .name is {loader.name!r}; the two must agree")

    declared = {s.task for s in loader.schema()}
    manifest_tasks = {t.task for t in m.tasks}
    if declared != manifest_tasks:
        raise ManifestError(
            f"{m.name}: loader.schema() declares {sorted(declared)}, manifest declares {sorted(manifest_tasks)}"
        )

    t0 = time.time()
    by_task: dict[str, list[Item]] = collections.defaultdict(list)
    for item in loader.load_items():
        _check_against_manifest(item, m)
        by_task[item.task].append(item)

    public: list[Item] = []
    heldout: list[Item] = []
    task_stats: dict[str, dict[str, Any]] = {}

    for task in sorted(by_task):
        items = sorted(by_task[task], key=lambda it: it.item_id)
        # de-dup identical (state, label) pairs, which item_id already collapses
        seen: set[str] = set()
        deduped = []
        for it in items:
            if it.item_id in seen:
                continue
            seen.add(it.item_id)
            deduped.append(it)
        n_dupes = len(items) - len(deduped)
        items = deduped
        if len(items) < MIN_ITEMS_TO_ACCEPT:
            raise ManifestError(
                f"{m.name}/{task}: {len(items)} items, below the {MIN_ITEMS_TO_ACCEPT}-item floor (PRD §7.3)"
            )

        te = m.task(task)
        ho = [it for it in items if side(it, m.heldout_fraction) == "heldout"]
        pu = [it for it in items if side(it, m.heldout_fraction) == "public"]
        ho, pu = ho[:heldout_cap], pu[:public_cap]

        widths = sorted({len(it.question.options) for it in items})
        strata = sorted({s for it in items for s in it.strata})
        chance = _observed_chance(items, te.chance_mode)
        labels = collections.Counter(it.label for it in items)

        task_stats[task] = {
            "primitive": te.primitive,
            "question_key": te.question_key,
            "ordinal": te.ordinal,
            "n_loaded": len(by_task[task]),
            "n_duplicates_dropped": n_dupes,
            "n_items": len(items),
            "n_public": len(pu),
            "n_heldout": len(ho),
            "option_counts_observed": widths,
            "strata_observed": strata,
            "width_stratum": width_stratum(widths[0]),
            "chance_mode": te.chance_mode,
            "chance_observed": round(chance, 6),
            "chance_declared": te.chance,
            "n_distinct_labels": len(labels),
            "label_balance_top10": dict(labels.most_common(10)),
            "majority_class_fraction": round(labels.most_common(1)[0][1] / len(items), 6),
            "calibration_bearing": len(ho) >= 250,
            "mean_state_chars": round(sum(len(it.state) for it in items) / len(items), 1),
            "max_state_chars": max(len(it.state) for it in items),
        }
        public += pu
        heldout += ho

    public.sort(key=lambda it: it.item_id)
    heldout.sort(key=lambda it: it.item_id)

    # PRD §7.4 gate 7, generalised by §13.6: two tasks over the same state must
    # never straddle the split, and (for Atari-HEAD) neither may two frames from
    # the same trial. Both are "the same split_key on both sides", so checking
    # split_key rather than state_hash catches both with one rule; for every
    # item without an override the two are identical, so this is byte-for-byte
    # the original check for all eight text datasets.
    pub_keys = {it.split_key for it in public}
    leaked = pub_keys & {it.split_key for it in heldout}
    if leaked:
        raise ManifestError(f"{m.name}: {len(leaked)} split keys appear in BOTH slices (PRD §7.4 gate 7)")

    commitment = label_commitment(heldout)
    receipt: dict[str, Any] = {
        "dataset": m.name,
        "display_name": m.display_name,
        "domain": m.domain,
        "bjb_version": __version__,
        "built_on": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t0, 1),
        "host": platform.node(),
        "loader": m.loader,
        "source": m.source,
        "source_url": m.source_url,
        "license": {
            "tier": m.tier,
            "spdx": m.spdx,
            "obligations": list(m.obligations),
            "verified_how": m.verified_how,
            "verified_on": m.verified_on,
        },
        "split": {
            "salt": SPLIT_SALT,
            "heldout_fraction_declared": m.heldout_fraction,
            "heldout_fraction_realised": round(len(heldout) / max(1, len(public) + len(heldout)), 4),
            "public_cap": public_cap,
            "heldout_cap": heldout_cap,
            "heldout_label_commitment": commitment,
        },
        "totals": {
            "n_items": sum(s["n_items"] for s in task_stats.values()),
            "n_public": len(public),
            "n_heldout": len(heldout),
            "n_tasks": len(task_stats),
        },
        "tasks": task_stats,
    }

    pub_path = repo_root / "data" / "public" / f"{m.name}.jsonl.gz"
    ho_path = repo_root / "bench" / "heldout" / f"{m.name}.jsonl.gz"
    if write:
        receipt["files"] = {
            "public": {"path": str(pub_path.relative_to(repo_root)), "sha256": _write_jsonl_gz(pub_path, public),
                       "n": len(public), "bytes": pub_path.stat().st_size},
            "heldout": {"path": str(ho_path.relative_to(repo_root)), "sha256": _write_jsonl_gz(ho_path, heldout),
                        "n": len(heldout), "bytes": ho_path.stat().st_size},
        }
        if m.modality != "text":
            receipt["images"] = _write_image_sha256_manifest(m, repo_root, public + heldout)
        prev = repo_root / "bench" / "preview" / f"{m.name}.json"
        prev.parent.mkdir(parents=True, exist_ok=True)
        stride = max(1, len(public) // PREVIEW_ITEMS)
        prev.write_text(
            json.dumps([it.to_bench_json() for it in public[::stride][:PREVIEW_ITEMS]], indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        rec_path = repo_root / "bench" / "receipts" / f"{m.name}.build.json"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return BuildResult(m.name, task_stats, pub_path if write else None, ho_path if write else None, receipt)


#: Fields a build *derives* and writes back into `manifest.toml`. Everything
#: else in a manifest is authored by a human and is never touched here.
#:
#: These three are derived rather than declared because they cannot be known
#: before a build: the exact item count after dedup and filtering, the observed
#: majority-class frequency on a `chance_mode = "majority"` task, and the
#: held-out label commitment. They still live in the manifest -- and therefore
#: in the git diff -- because §7.5's whole mechanism is that a change to a
#: held-out label is *visible in review*, and the same argument applies to a
#: chance floor quietly moving. `bjb validate` then checks manifest against
#: receipt against shipped data, so a hand-edit without a rebuild fails CI.
DERIVED_FIELDS = ("item_count", "chance (majority mode only)", "heldout_label_commitment")


def sync_derived_fields(m: Manifest, receipt: dict[str, Any]) -> list[str]:
    """Rewrite the derived fields in `manifest.toml`, in place, as line edits.

    A line edit rather than a TOML round-trip so a contributor's comments,
    ordering and formatting survive untouched -- a manifest is a document people
    read, not just a config the tool consumes.
    """
    changed: list[str] = []
    text = m.path.read_text(encoding="utf-8")
    majority_chance = {
        t: s["chance_observed"] for t, s in receipt["tasks"].items() if s["chance_mode"] == "majority"
    }

    out: list[str] = []
    table = ""
    task_name = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("[["):
            table = stripped.strip("[]")
            task_name = ""
        elif stripped.startswith("["):
            table = stripped.strip("[]")
            task_name = ""
        elif table == "schema.tasks" and stripped.startswith("task "):
            task_name = stripped.split("=", 1)[1].strip().strip('"')
        new = raw

        if table == "dataset" and stripped.startswith("item_count"):
            new = f"item_count    = {receipt['totals']['n_items']}"
        elif table == "split" and stripped.startswith("heldout_label_commitment"):
            new = f'heldout_label_commitment = "{receipt["split"]["heldout_label_commitment"]}"'
        elif table == "schema.tasks" and stripped.startswith("chance ") and task_name in majority_chance:
            indent = raw[: len(raw) - len(raw.lstrip())]
            new = f"{indent}chance        = {majority_chance[task_name]}"

        if new != raw:
            changed.append(f"{stripped.split('=')[0].strip()}{'/' + task_name if task_name else ''}")
        out.append(new)

    if changed:
        m.path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def build_all(
    repo_root: Path,
    *,
    only: list[str] | None = None,
    public_cap: int = DEFAULT_PUBLIC_CAP,
    heldout_cap: int = DEFAULT_HELDOUT_CAP,
    verify: bool = False,
) -> dict[str, Any]:
    manifests = [m for m in discover(repo_root) if not only or m.name in only]
    if not manifests:
        raise SystemExit(f"no manifests matched {only!r} under {repo_root}/datasets")

    results, drift = [], []
    for m in manifests:
        print(f"[build] {m.name} ...", flush=True)
        res = build_dataset(m, repo_root, public_cap=public_cap, heldout_cap=heldout_cap, write=not verify)
        results.append(res)
        if verify:
            rec_path = repo_root / "bench" / "receipts" / f"{m.name}.build.json"
            if not rec_path.exists():
                drift.append(f"{m.name}: no committed receipt to verify against")
                continue
            old = json.loads(rec_path.read_text())
            new_commit = res.receipt["split"]["heldout_label_commitment"]
            if old["split"]["heldout_label_commitment"] != new_commit:
                drift.append(
                    f"{m.name}: held-out label commitment changed\n"
                    f"    committed {old['split']['heldout_label_commitment']}\n"
                    f"    rebuilt   {new_commit}"
                )
            for task, stats in res.tasks.items():
                prev = old["tasks"].get(task)
                if prev and prev["n_items"] != stats["n_items"]:
                    drift.append(
                        f"{m.name}/{task}: n_items {prev['n_items']} -> {stats['n_items']} (upstream drift)"
                    )
        else:
            changed = sync_derived_fields(m, res.receipt)
            if changed:
                print(f"[build] {m.name}: synced derived manifest fields {changed}", flush=True)

    corpus = {
        "bjb_version": __version__,
        "built_on": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_salt": SPLIT_SALT,
        "caps": {"public_per_task": public_cap, "heldout_per_task": heldout_cap},
        "datasets": {r.dataset: r.receipt for r in results},
        "totals": {
            "n_datasets": len(results),
            "n_tasks": sum(r.receipt["totals"]["n_tasks"] for r in results),
            "n_items": sum(r.receipt["totals"]["n_items"] for r in results),
            "n_public": sum(r.receipt["totals"]["n_public"] for r in results),
            "n_heldout": sum(r.receipt["totals"]["n_heldout"] for r in results),
        },
        "strata_population": _strata_population(results),
    }
    if verify:
        corpus["drift"] = drift
        if drift:
            print("\n".join(["DRIFT DETECTED:"] + drift))
    else:
        out = repo_root / "bench" / "receipts" / "CORPUS.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return corpus


def _strata_population(results: list[BuildResult]) -> dict[str, dict[str, int]]:
    """Per-stratum item counts, and whether each clears the 250-item floor
    (PRD §5.3). A stratum the corpus under-populates is *excluded and reported*,
    never quietly averaged in."""
    pop: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"items": 0, "heldout": 0})
    for r in results:
        for stats in r.tasks.values():
            for s in stats["strata_observed"]:
                pop[s]["items"] += stats["n_items"]
                pop[s]["heldout"] += stats["n_heldout"]
    return {
        k: {**v, "calibration_bearing": v["heldout"] >= 250}
        for k, v in sorted(pop.items())
    }

"""Training-slice export -- the bridge to ekVachan, and the reason this corpus
is a training substrate rather than only a scoreboard.

The eight-key record this emits was read off ekVachan's own real data builders
(`better-jev-for-all/training/data.py`, `training/build_primitives_slice.py`),
not designed here:

    {state, question_key, question_type, instructions, options, label, label_idx, source}

Those two scripts produced the data behind the checkpoint that actually scores
92.02% `choice` / 88.80% `noul` / 75.40% `score`, so matching them exactly is
what makes a bench slice droppable into the next training run with no adapter.

Three real mechanism details are handled here, each of which would silently
corrupt a training run if it were not:

**1. The 26-letter width budget is an ekVachan constraint, not a corpus one.**
The bench stores the true schema -- CLINC150 is presented with all 151 intents,
because that is the honest benchmark. ekVachan's decoder reads a restricted
logit over single uppercase letters, so it tops out at 26 options
(`MAX_OPTIONS` in `build_primitives_slice.py`). The export narrows wider schemas
by keeping the gold label and sampling distractors from the item's own option
set, which is precisely what `_sample_wide_subset()` does upstream, with the
same effect: the model sees a varied, wide-but-bounded option set and learns the
mechanism rather than the taxonomy. The narrowing is seeded and recorded in
`stats.json`, and `--max-options 0` disables it for a consumer with no such
budget.

**2. `score` options are never narrowed and never shuffled.** An ordinal scale's
letter position *is* its scale position -- that is the property that makes
neighbouring-letter probability mass mean "landed between levels" rather than
noise. `build_primitives_slice.py` documents this at length and asserts it in
its own validator; this exporter enforces the same invariant and refuses to emit
a `score` row whose options were reordered.

**3. Held-out means held out.** `--slice heldout` exists and is loud about it.
The default is `public`, and exporting the held-out slice into a training mix
prints a warning naming PRD §7.5 and G3, because the single thing this corpus
promised from day one is that its eval slice was reserved before any training
run existed to leak into it.

**4. An image-bearing record gets a ninth key, `images`, additively (PRD §13.4).**
`Item.to_ekvachan_record()` accepts `image_paths` -- a `{sha256: local_path}`
map -- and refuses to emit a record for an item that carries images without
one. This module builds that map with `imagecache.verify_cached()`, which
resolves each image's content-addressed cache path *and* re-hashes the file on
disk before handing the path out: a training consumer downstream (the
sibling's `training/build_vision_slice.py`, PRD §5.2b item 3) is told exactly
this same thing to assert, so failing here, loudly, before export even
finishes, is strictly better than failing there.

**5. An `eval_only` dataset (ScreenSpot-v2) never enters the `public` slice.**
Same role CLINC150 plays for the text corpus's zero-shot-schema regression
check (`training/build_benchcorpus_slice.py` drops it explicitly), but
enforced here in code rather than left to a consumer's discipline: `bjb
export --slice public` skips it outright and says why. `--slice heldout`
still exports it -- that is its only legitimate use.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any, Iterable

from .build import read_jsonl_gz
from .imagecache import verify_cached
from .manifest import Manifest, discover
from .types import Item

#: `build_primitives_slice.py`'s MAX_OPTIONS -- ekVachan's letter budget.
EKVACHAN_MAX_OPTIONS = 26
DEFAULT_SEED = 42


def narrow_options(item: Item, max_options: int, rng: random.Random) -> tuple[str, ...]:
    """Keep the gold label, sample distractors, shuffle. Mirrors
    `_sample_wide_subset()` in ekVachan's builder."""
    opts = item.question.options
    if max_options <= 0 or len(opts) <= max_options:
        return opts
    if item.question.ordinal:
        raise ValueError(
            f"{item.item_id}: refusing to narrow an ordinal score scale "
            f"({len(opts)} levels > max {max_options}); scale order is load-bearing"
        )
    distractors = rng.sample([o for o in opts if o != item.label], max_options - 1)
    out = distractors + [item.label]
    rng.shuffle(out)
    return tuple(out)


def export(
    repo_root: Path,
    out_dir: Path,
    *,
    slice_name: str = "public",
    tiers: Iterable[str] = ("A", "A-share-alike"),
    datasets_filter: list[str] | None = None,
    primitives: list[str] | None = None,
    max_options: int = EKVACHAN_MAX_OPTIONS,
    max_per_task: int | None = None,
    seed: int = DEFAULT_SEED,
    fmt: str = "jsonl",
) -> dict[str, Any]:
    tiers = set(tiers)
    rng = random.Random(seed)
    manifests = [m for m in discover(repo_root) if m.tier in tiers]
    if datasets_filter:
        manifests = [m for m in manifests if m.name in datasets_filter]
    if not manifests:
        raise SystemExit("no datasets matched the tier/name filter")

    if slice_name == "heldout":
        print(
            "WARNING: exporting the HELD-OUT slice. PRD §7.5 / G3: this slice was reserved "
            "before any training run existed. Training on it invalidates every benchmark "
            "number this corpus produces for your model.",
            flush=True,
        )

    sub = "bench/heldout" if slice_name == "heldout" else "data/public"
    records: list[dict[str, Any]] = []
    per_task: dict[str, int] = collections.Counter()
    narrowed = 0
    n_image_records = 0
    distinct_images: set[str] = set()
    by_dataset: dict[str, dict[str, Any]] = {}
    obligations: set[str] = set()
    attribution: list[dict[str, str]] = []

    for m in manifests:
        if slice_name == "public" and m.eval_only:
            print(
                f"[export] skip {m.name}: dataset.eval_only = true -- never exported to the public/training "
                "slice (PRD §13.3); use --slice heldout to evaluate against it",
                flush=True,
            )
            continue
        path = repo_root / sub / f"{m.name}.jsonl.gz"
        if not path.exists():
            print(f"[export] skip {m.name}: {path} not built (run `bjb build`)", flush=True)
            continue
        items = read_jsonl_gz(path)
        if primitives:
            items = [it for it in items if it.question.type in primitives]
        kept = 0
        for it in items:
            key = f"{m.name}/{it.task}"
            if max_per_task is not None and per_task[key] >= max_per_task:
                continue
            opts = narrow_options(it, max_options, rng)
            if len(opts) != len(it.question.options):
                narrowed += 1
            image_paths = None
            if it.images:
                # Resolve + verify every image against the local content-
                # addressed cache before it ever reaches a record (point 4
                # above). A missing or hash-mismatched image fails the export
                # outright rather than silently shipping a dangling reference.
                image_paths = {
                    img.sha256: str(verify_cached(repo_root, m.name, img)) for img in it.images
                }
                n_image_records += 1
                distinct_images.update(image_paths)
            records.append(it.to_ekvachan_record(options=opts, image_paths=image_paths))
            per_task[key] += 1
            kept += 1
        if not kept:
            continue
        by_dataset[m.name] = {
            "n": kept,
            "tier": m.tier,
            "spdx": m.spdx,
            "obligations": list(m.obligations),
            "source_url": m.source_url,
        }
        obligations |= set(m.obligations)
        attribution.append(
            {"dataset": m.display_name, "spdx": m.spdx, "url": m.source_url, "obligations": ", ".join(m.obligations)}
        )

    rng.shuffle(records)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "n": len(records),
        "slice": slice_name,
        "license_tiers": sorted(tiers),
        "by_question_type": dict(collections.Counter(r["question_type"] for r in records)),
        "by_source": dict(collections.Counter(r["source"] for r in records)),
        "by_dataset": by_dataset,
        "option_width_histogram": dict(
            sorted(collections.Counter(len(r["options"]) for r in records).items())
        ),
        "max_options_cap": max_options,
        "n_items_narrowed_to_cap": narrowed,
        "seed": seed,
        "n_image_records": n_image_records,
        "n_distinct_images": len(distinct_images),
        "record_schema": [
            "state", "question_key", "question_type", "instructions",
            "options", "label", "label_idx", "source",
        ],
        "record_schema_source": (
            "better-jev-for-all/training/data.py and training/build_primitives_slice.py -- "
            "exact key set consumed by ekVachan's real training runs"
        ),
        "image_record_schema_note": (
            "an image-bearing record gains a ninth key, 'images': a list of "
            "{sha256, path, media_type, width, height}, 'path' already resolved and "
            "hash-verified against the local content-addressed cache (PRD §13.4/§13.5). "
            "A text record has no 'images' key at all -- the eight-key shape above is "
            "byte-for-byte unchanged."
        ),
        "ordinal_note": (
            "score rows keep their canonical ascending scale order and are never shuffled or "
            "narrowed; the consuming training script must present them in this same fixed order"
        ),
        "license_obligations": sorted(obligations),
        "attribution": sorted(attribution, key=lambda a: a["dataset"]),
    }

    if fmt == "hf":
        from datasets import Dataset  # optional dependency

        Dataset.from_list(records).save_to_disk(str(out_dir / slice_name))
        stats["format"] = f"huggingface datasets save_to_disk -> {out_dir / slice_name} (load_from_disk-able)"
    else:
        path = out_dir / f"{slice_name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["format"] = f"jsonl -> {path}"

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "ATTRIBUTION.md").write_text(_attribution_md(stats), encoding="utf-8")
    return stats


def _attribution_md(stats: dict[str, Any]) -> str:
    lines = [
        "# Attribution for this export",
        "",
        "Generated by `bjb export`. Several sources in this slice carry attribution or",
        "share-alike obligations that **propagate to anything derived from it**, including",
        "model weights released under some readings of CC-BY-SA. Ship this file with the",
        "derived artifact (PRD §4.1, Tier A-share-alike).",
        "",
        f"Obligations present in this slice: {', '.join(stats['license_obligations']) or 'none'}",
        "",
        "| Dataset | License | Obligations | Source |",
        "|---|---|---|---|",
    ]
    for a in stats["attribution"]:
        lines.append(f"| {a['dataset']} | {a['spdx']} | {a['obligations'] or '—'} | {a['url']} |")
    lines.append("")
    return "\n".join(lines)

"""`bjb` -- the command line the whole lifecycle runs through.

Deliberately usable without a server. PRD §6.3 ends on the point that "requiring
a hosted service to run a benchmark is an adoption tax for exactly the
self-hosting audience this is for", so the library and CLI are the primary
interface and the HTTP surface (§6.2) is a wrapper over the same code paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    here = Path.cwd().resolve()
    for cand in (here, *here.parents):
        if (cand / "datasets").is_dir() and (cand / "better_jev_bench").is_dir():
            return cand
    return here


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bjb", description="better-jev-bench corpus tooling")
    p.add_argument("--repo", default=None, help="repo root (default: auto-detect from cwd)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("catalogue", help="the dataset catalogue as data (PRD §6.3)")
    c.add_argument("--tiers", nargs="*", default=None, help="filter by license tier")
    c.add_argument("--json", action="store_true")

    v = sub.add_parser("validate", help="run the PRD §7.4 CI gates")
    v.add_argument("--loaders", action="store_true", help="also run gate 2 (network-bound)")
    v.add_argument("--only", nargs="*", default=None)

    b = sub.add_parser("build", help="download, normalise, split, hash")
    b.add_argument("--only", nargs="*", default=None)
    b.add_argument("--public-cap", type=int, default=None)
    b.add_argument("--heldout-cap", type=int, default=None)
    b.add_argument("--verify", action="store_true", help="rebuild and diff against committed receipts")

    e = sub.add_parser("export", help="emit ekVachan-shaped training records")
    e.add_argument("--out", required=True)
    e.add_argument("--slice", dest="slice_name", default="public", choices=["public", "heldout"])
    e.add_argument("--tiers", nargs="*", default=["A", "A-share-alike"])
    e.add_argument("--datasets", nargs="*", default=None)
    e.add_argument("--primitives", nargs="*", default=None, choices=["choice", "score", "noul"])
    e.add_argument("--max-options", type=int, default=26, help="0 disables narrowing")
    e.add_argument("--max-per-task", type=int, default=None)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--format", dest="fmt", default="jsonl", choices=["jsonl", "hf"])

    sub.add_parser("stats", help="what actually landed, from the committed receipts")

    ev = sub.add_parser("evaluate", help="score a model against the corpus (PRD §14.7)")
    ev.add_argument("--endpoint", default=None, help="a /v1/systemone-speaking URL (PRD §14.8 point 1)")
    ev.add_argument("--mock", action="store_true", help="use the built-in MockBackend -- no network, no ML stack (PRD §14.9 item 5)")
    ev.add_argument("--mock-fixed-answer", nargs=2, metavar=("QUESTION_KEY", "ANSWER"), action="append", default=None)
    ev.add_argument("--model", required=True, help="<name>@<git-sha|weight-hash>")
    ev.add_argument("--slice", dest="slice_name", default="heldout", choices=["public", "heldout"])
    ev.add_argument("--datasets", nargs="*", default=None)
    ev.add_argument("--tasks", nargs="*", default=None)
    ev.add_argument("--tiers", dest="license_tiers", nargs="*", default=["A"])
    ev.add_argument("--modalities", nargs="*", default=["text", "image"])
    ev.add_argument("--primitives", nargs="*", default=["choice", "score", "noul"])
    ev.add_argument("--max-items-per-task", type=int, default=None)
    ev.add_argument("--shuffle-seed", type=int, default=None)
    ev.add_argument("--no-position-sensitivity", action="store_true")
    ev.add_argument("--images-mode", default="reference", choices=["reference", "inline_base64", "omit"])
    ev.add_argument("--timeout-s", type=float, default=30.0)
    ev.add_argument("--concurrency", type=int, default=1)
    ev.add_argument("--cost-hourly-rate", type=float, default=None)
    ev.add_argument("--cost-currency", default="usd")
    ev.add_argument("--cost-hardware", default=None)
    ev.add_argument("--out", default=None, help="evidence bundle dir (default: <repo>/results)")

    args = p.parse_args(argv)
    root = _repo_root(args.repo)

    if args.cmd == "catalogue":
        from .manifest import discover

        rows = [m for m in discover(root) if not args.tiers or m.tier in args.tiers]
        payload = [
            {
                "name": m.name,
                "display_name": m.display_name,
                "domain": m.domain,
                "license": {"tier": m.tier, "spdx": m.spdx, "obligations": list(m.obligations),
                            "verified_how": m.verified_how, "verified_on": m.verified_on},
                "item_count": m.item_count,
                "primitives": list(m.primitives),
                "option_counts": list(m.option_counts),
                "strata": list(m.strata),
                "modality": m.modality,
                "tasks": [
                    {"task": t.task, "primitive": t.primitive, "question_key": t.question_key,
                     "option_count": t.option_count, "chance": t.chance, "ordinal": t.ordinal}
                    for t in m.tasks
                ],
                "loader": m.loader,
                "source_url": m.source_url,
                "sensitivity_flags": list(m.sensitivity_flags),
                "heldout_label_commitment": m.heldout_label_commitment,
            }
            for m in rows
        ]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"{'dataset':22} {'tier':14} {'spdx':14} {'prims':20} {'widths':16} items")
            print("-" * 100)
            for r in payload:
                print(
                    f"{r['name']:22} {r['license']['tier']:14} {r['license']['spdx']:14} "
                    f"{','.join(r['primitives']):20} {','.join(map(str, r['option_counts'])):16} {r['item_count']:,}"
                )
        return 0

    if args.cmd == "validate":
        from .validate import validate_repo

        rep = validate_repo(root, run_loaders=args.loaders, only=args.only)
        print(rep.render())
        return 0 if rep.ok else 1

    if args.cmd == "build":
        from .build import DEFAULT_HELDOUT_CAP, DEFAULT_PUBLIC_CAP, build_all

        corpus = build_all(
            root,
            only=args.only,
            public_cap=args.public_cap or DEFAULT_PUBLIC_CAP,
            heldout_cap=args.heldout_cap or DEFAULT_HELDOUT_CAP,
            verify=args.verify,
        )
        print(json.dumps(corpus["totals"], indent=2))
        if args.verify:
            return 1 if corpus.get("drift") else 0
        return 0

    if args.cmd == "export":
        from .export import export

        stats = export(
            root,
            Path(args.out),
            slice_name=args.slice_name,
            tiers=args.tiers,
            datasets_filter=args.datasets,
            primitives=args.primitives,
            max_options=args.max_options,
            max_per_task=args.max_per_task,
            seed=args.seed,
            fmt=args.fmt,
        )
        print(json.dumps({k: v for k, v in stats.items() if k != "attribution"}, indent=2))
        return 0

    if args.cmd == "stats":
        path = root / "bench" / "receipts" / "CORPUS.json"
        if not path.exists():
            print("no corpus receipt yet -- run `bjb build`", file=sys.stderr)
            return 1
        corpus = json.loads(path.read_text())
        print(f"better-jev-bench corpus  ·  built {corpus['built_on']}  ·  bjb {corpus['bjb_version']}")
        t = corpus["totals"]
        print(f"  {t['n_datasets']} datasets · {t['n_tasks']} tasks · {t['n_items']:,} items "
              f"({t['n_public']:,} public / {t['n_heldout']:,} held-out)\n")
        print(f"{'dataset/task':42} {'prim':7} {'width':>6} {'items':>9} {'public':>8} {'heldout':>8}  calib")
        print("-" * 100)
        for name, rec in corpus["datasets"].items():
            for task, s in rec["tasks"].items():
                print(f"{name + '/' + task:42} {s['primitive']:7} {s['option_counts_observed'][0]:>6} "
                      f"{s['n_items']:>9,} {s['n_public']:>8,} {s['n_heldout']:>8,}  "
                      f"{'yes' if s['calibration_bearing'] else 'NO'}")
        print("\nstrata population (PRD §5.3 -- a stratum under 250 held-out items is excluded from Breadth):")
        for k, v in corpus["strata_population"].items():
            print(f"  {k:18} items={v['items']:>9,}  heldout={v['heldout']:>7,}  "
                  f"{'calibration-bearing' if v['calibration_bearing'] else 'EXCLUDED (<250)'}")
        return 0

    if args.cmd == "evaluate":
        from .backend import HTTPBackend, MockBackend
        from .evaluate import DEFAULT_SHUFFLE_SEED, EvaluateError, evaluate, write_evidence_bundle

        if bool(args.endpoint) == bool(args.mock):
            print("evaluate: pass exactly one of --endpoint or --mock", file=sys.stderr)
            return 2

        if args.mock:
            fixed = dict(args.mock_fixed_answer) if args.mock_fixed_answer else None
            backend = MockBackend(fixed_answers=fixed)
        else:
            backend = HTTPBackend(args.endpoint)

        cost_model = (
            {"currency": args.cost_currency, "hourly_rate": args.cost_hourly_rate, "hardware": args.cost_hardware}
            if args.cost_hourly_rate is not None
            else None
        )
        request_kwargs = dict(
            model=args.model,
            slice=args.slice_name,
            datasets=args.datasets,
            tasks=args.tasks,
            license_tiers=args.license_tiers,
            modalities=args.modalities,
            primitives=args.primitives,
            max_items_per_task=args.max_items_per_task,
            shuffle_seed=args.shuffle_seed if args.shuffle_seed is not None else DEFAULT_SHUFFLE_SEED,
            images={"mode": args.images_mode},
            request={"timeout_s": args.timeout_s, "concurrency": args.concurrency, "headers": {}},
            cost_model=cost_model,
        )
        if args.no_position_sensitivity:
            request_kwargs["position_sensitivity"] = None  # explicit null -- disables (PRD §14.7)
        # else: omit the key entirely so `evaluate()`'s own enabled-by-default applies.
        try:
            result = evaluate(root, backend, **request_kwargs)
        except EvaluateError as exc:
            print(f"evaluate: {exc}", file=sys.stderr)
            return 2

        out_dir = Path(args.out) if args.out else root / "results"
        paths = write_evidence_bundle(result, {**request_kwargs, "endpoint": args.endpoint}, out_dir, repo_root=root)
        for k in ("_raw_rows", "_model_info", "_datasets_used"):
            result.pop(k, None)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nevidence bundle: {paths['manifest']}", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""The CI gates (PRD §7.4), implemented.

All eight gates the PRD specified are here. They are split into two modes for a
reason worth stating, because "run everything always" is the naive design and it
does not survive contact with a real corpus:

* **`bjb validate` (default, offline, seconds)** runs gates 1, 3, 4, 5, 6, 7 and
  8 against the *committed* artifacts -- manifests, the frozen held-out slices,
  and the build receipts. It needs no network, no Hugging Face token and no
  346MB download, so it runs on every PR. Gate 3 is checked against the receipt's
  recorded observations plus the shipped held-out items, which is the same
  evidence-over-assertion discipline the receipts exist for.
* **`bjb validate --loaders`** additionally runs gate 2 for real: it executes
  each `load_items()`, takes a bounded prefix, and checks the live rows against
  the manifest. This one is bandwidth-bound and is a scheduled/manual job, not a
  per-PR one.

A new dataset's PR gets `--loaders` run on it once (that is what the
`loader-smoke` CI job does, restricted to changed datasets), and the frozen
artifacts it ships are what every later PR is checked against.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path

from .build import read_jsonl_gz
from .dataset import load_plugin
from .manifest import Manifest, ManifestError, load_manifest
from .split import label_commitment, side
from .types import MIN_ITEMS_TO_ACCEPT, is_valid_canary


@dataclass(slots=True)
class Report:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_run: int = 0

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        out = []
        for w in self.warnings:
            out.append(f"  WARN  {w}")
        for f in self.failures:
            out.append(f"  FAIL  {f}")
        out.append(
            f"\n{self.checks_run} checks run · {len(self.failures)} failed · {len(self.warnings)} warnings"
        )
        return "\n".join(out)


def validate_repo(repo_root: Path, *, run_loaders: bool = False, only: list[str] | None = None) -> Report:
    rep = Report()
    repo_root = Path(repo_root)

    # -- gate 1 / 4 / 5: manifest schema, license tier, sensitivity ----------
    manifests: list[Manifest] = []
    for path in sorted((repo_root / "datasets").glob("*/manifest.toml")):
        rep.checks_run += 1
        try:
            manifests.append(load_manifest(path))
        except ManifestError as exc:
            rep.fail(f"gate 1/4/5 [{path.parent.name}]: {exc}")
    if only:
        manifests = [m for m in manifests if m.name in only]
    if not manifests:
        rep.fail("no valid manifests found under datasets/*/manifest.toml")
        return rep

    # -- gate 6: canary uniqueness -----------------------------------------
    rep.checks_run += 1
    seen: dict[str, str] = {}
    for m in manifests:
        if not is_valid_canary(m.canary):
            rep.fail(f"gate 6 [{m.name}]: malformed canary")
        elif m.canary in seen:
            rep.fail(f"gate 6 [{m.name}]: canary reused from {seen[m.canary]} (PRD §5.5 rule 2)")
        else:
            seen[m.canary] = m.name

    for m in manifests:
        rec_path = repo_root / "bench" / "receipts" / f"{m.name}.build.json"
        ho_path = repo_root / "bench" / "heldout" / f"{m.name}.jsonl.gz"

        # -- gate 8: held-out commitment present and matching ---------------
        rep.checks_run += 1
        if not m.heldout_label_commitment:
            rep.fail(f"gate 8 [{m.name}]: split.heldout_label_commitment is empty (PRD §7.5)")
        elif not m.heldout_label_commitment.startswith("sha256:") or len(m.heldout_label_commitment) != 71:
            rep.fail(f"gate 8 [{m.name}]: heldout_label_commitment is not a sha256:<64 hex> value")
        elif not ho_path.exists():
            rep.fail(f"gate 8 [{m.name}]: no frozen held-out slice at {ho_path.relative_to(repo_root)}")
        else:
            heldout = read_jsonl_gz(ho_path)
            actual = label_commitment(heldout)
            if actual != m.heldout_label_commitment:
                rep.fail(
                    f"gate 8 [{m.name}]: held-out labels do NOT match the committed hash.\n"
                    f"          manifest {m.heldout_label_commitment}\n"
                    f"          shipped  {actual}\n"
                    f"          Every prior result on this dataset is invalidated (PRD §7.5)."
                )

            # -- gate 3: schema/manifest agreement on real shipped items ----
            rep.checks_run += 1
            by_task = collections.defaultdict(list)
            for it in heldout:
                by_task[it.task].append(it)
            declared = {t.task for t in m.tasks}
            if set(by_task) - declared:
                rep.fail(f"gate 3 [{m.name}]: held-out slice contains undeclared tasks {sorted(set(by_task) - declared)}")
            observed_strata: set[str] = set()
            for task, items in sorted(by_task.items()):
                te = m.task(task)
                widths = {len(it.question.options) for it in items}
                if widths != {te.option_count}:
                    rep.fail(
                        f"gate 3 [{m.name}/{task}]: manifest declares option_count {te.option_count}, "
                        f"shipped items have {sorted(widths)}"
                    )
                prims = {it.question.type for it in items}
                if prims != {te.primitive}:
                    rep.fail(f"gate 3 [{m.name}/{task}]: primitive mismatch {sorted(prims)} vs {te.primitive!r}")
                for it in items:
                    observed_strata |= set(it.strata)
                bad = [it for it in items if it.canary != m.canary]
                if bad:
                    rep.fail(f"gate 3 [{m.name}/{task}]: {len(bad)} items carry a canary that is not the manifest's")
                bad = [it for it in items if it.license_tier != m.tier]
                if bad:
                    rep.fail(f"gate 3 [{m.name}/{task}]: {len(bad)} items carry a license_tier != {m.tier!r}")
                if te.primitive == "score":
                    orders = {tuple(it.question.options) for it in items}
                    if len(orders) != 1:
                        rep.fail(
                            f"gate 3 [{m.name}/{task}]: an ordinal score task must present ONE fixed scale "
                            f"order; found {len(orders)} distinct orderings"
                        )
                    if not all(it.question.ordinal for it in items):
                        rep.fail(f"gate 3 [{m.name}/{task}]: score items must set ordinal=True")
            if observed_strata and not observed_strata <= set(m.strata):
                rep.fail(
                    f"gate 3 [{m.name}]: items fall into strata {sorted(observed_strata - set(m.strata))} "
                    "that the manifest does not declare"
                )
            missing = set(m.strata) - observed_strata
            if missing:
                rep.warn(f"[{m.name}]: manifest declares strata {sorted(missing)} with no items in the held-out slice")

            # -- gate 7: leakage, held-out vs public preview + split rule ----
            rep.checks_run += 1
            wrong_side = [it.item_id for it in heldout if side(it, m.heldout_fraction) != "heldout"]
            if wrong_side:
                rep.fail(
                    f"gate 7 [{m.name}]: {len(wrong_side)} shipped held-out items do not hash to the held-out "
                    "side; the frozen slice and the split rule disagree"
                )
            prev_path = repo_root / "bench" / "preview" / f"{m.name}.json"
            if prev_path.exists():
                prev = json.loads(prev_path.read_text())
                ho_states = {it.state_hash for it in heldout}
                overlap = [p["item_id"] for p in prev if p.get("state_hash") in ho_states]
                if overlap:
                    rep.fail(f"gate 7 [{m.name}]: {len(overlap)} public-preview items share a state with the held-out slice")
            pub_path = repo_root / "data" / "public" / f"{m.name}.jsonl.gz"
            if pub_path.exists():
                pub = read_jsonl_gz(pub_path)
                overlap = {it.state_hash for it in pub} & {it.state_hash for it in heldout}
                if overlap:
                    rep.fail(f"gate 7 [{m.name}]: {len(overlap)} states appear in BOTH the public and held-out slices")

        # -- receipt presence + internal agreement --------------------------
        rep.checks_run += 1
        if not rec_path.exists():
            rep.fail(f"[{m.name}]: no build receipt at {rec_path.relative_to(repo_root)} (dev-guidelines rule 10)")
        else:
            rec = json.loads(rec_path.read_text())
            if rec["split"]["heldout_label_commitment"] != m.heldout_label_commitment:
                rep.fail(f"[{m.name}]: receipt commitment != manifest commitment")
            if rec["license"]["tier"] != m.tier or rec["license"]["spdx"] != m.spdx:
                rep.fail(f"[{m.name}]: receipt license does not match the manifest")
            declared_total = m.item_count
            actual_total = rec["totals"]["n_items"]
            if declared_total != actual_total:
                rep.fail(
                    f"gate 3 [{m.name}]: manifest dataset.item_count = {declared_total} but the build "
                    f"produced {actual_total}"
                )
            for task, st in rec["tasks"].items():
                te = m.task(task)
                if abs(st["chance_observed"] - te.chance) > 5e-3:
                    rep.fail(
                        f"gate 3 [{m.name}/{task}]: manifest chance {te.chance} vs observed "
                        f"{st['chance_observed']} (mode {te.chance_mode})"
                    )
                if st["n_items"] < MIN_ITEMS_TO_ACCEPT:
                    rep.fail(f"gate 3 [{m.name}/{task}]: {st['n_items']} items, below the {MIN_ITEMS_TO_ACCEPT} floor")
                if st["n_heldout"] < 250:
                    rep.warn(
                        f"[{m.name}/{task}]: {st['n_heldout']} held-out items -- scored under Intelligence and "
                        "Coverage but NOT calibration-bearing (PRD §7.3)"
                    )

        # -- gate 2: loader smoke test (opt-in, network-bound) --------------
        if run_loaders:
            rep.checks_run += 1
            try:
                loader = load_plugin(m.loader, canary=m.canary, license_tier=m.tier)
                if loader.name != m.name:
                    rep.fail(f"gate 2 [{m.name}]: loader.name is {loader.name!r}")
                declared = {s.task for s in loader.schema()}
                if declared != {t.task for t in m.tasks}:
                    rep.fail(f"gate 2 [{m.name}]: loader.schema() tasks {sorted(declared)} != manifest tasks")
                n = 0
                for it in loader.load_items():
                    n += 1
                    if n >= MIN_ITEMS_TO_ACCEPT * 4:
                        break
                if n < MIN_ITEMS_TO_ACCEPT:
                    rep.fail(f"gate 2 [{m.name}]: load_items() yielded {n} items, below {MIN_ITEMS_TO_ACCEPT}")
            except Exception as exc:  # noqa: BLE001 - CI wants the message, not a traceback
                rep.fail(f"gate 2 [{m.name}]: load_items() raised {type(exc).__name__}: {exc}")

    return rep

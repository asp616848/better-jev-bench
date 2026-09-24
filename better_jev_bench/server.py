"""`POST /v1/evaluate` and its three supporting read endpoints (PRD §14.9 item
6, §6.3). A thin FastAPI wrapper over `evaluate.evaluate()` -- every field in
the response is exactly what `bjb evaluate` prints (PRD §14.7's explicit
requirement), because both call the same function. Nothing here computes a
score; this module's only job is HTTP: parse the request, call
`better_jev_bench.evaluate.evaluate()`, translate its exceptions into the
right status codes, write the evidence bundle, and serialise the result.

Requires the `serve` extra (`fastapi`, `pydantic`, `uvicorn`) -- deliberately
not a hard dependency of the corpus tooling (`bjb build`/`bjb export`/`bjb
evaluate` all work with nothing but the stdlib and `numpy`); see
`pyproject.toml`.

Run with:

    uvicorn better_jev_bench.server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import CORPUS_VERSION, SPEC_VERSION, __version__
from .backend import HTTPBackend
from .evaluate import EvaluateError, evaluate, write_evidence_bundle
from .manifest import discover

app = FastAPI(title="better-jev-bench", version=__version__)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / "datasets").is_dir() and (cand / "better_jev_bench").is_dir():
            return cand
    raise RuntimeError("could not locate the repo root from server.py's own path")


class EvaluateRequest(BaseModel):
    """PRD §14.7's request table, field for field. Defaults mirror the table
    exactly, not `evaluate()`'s own Python defaults, so the wire contract
    stays the normative copy even if the two ever drift."""

    endpoint: str
    model: str
    slice: Literal["public", "heldout"] = "heldout"
    datasets: list[str] | None = None
    tasks: list[str] | None = None
    license_tiers: list[str] = Field(default_factory=lambda: ["A"])
    modalities: list[str] = Field(default_factory=lambda: ["text", "image"])
    primitives: list[str] = Field(default_factory=lambda: ["choice", "score", "noul"])
    max_items_per_task: int | None = None
    shuffle_seed: int = 20260924
    position_sensitivity: dict[str, Any] | None = Field(
        default_factory=lambda: {"enabled": True, "sample_per_task": 200}
    )
    images: dict[str, Any] = Field(default_factory=lambda: {"mode": "reference"})
    request: dict[str, Any] = Field(default_factory=lambda: {"timeout_s": 30.0, "concurrency": 1, "headers": {}})
    cost_model: dict[str, Any] | None = None
    spec_version: str = SPEC_VERSION


@app.post("/v1/evaluate")
def post_evaluate(req: EvaluateRequest) -> dict[str, Any]:
    if req.spec_version != SPEC_VERSION:
        raise HTTPException(409, f"spec_version {req.spec_version!r} != engine's {SPEC_VERSION!r}")

    root = _repo_root()
    backend = HTTPBackend(req.endpoint, headers=req.request.get("headers") or {})
    try:
        result = evaluate(
            root,
            backend,
            model=req.model,
            slice=req.slice,
            datasets=req.datasets,
            tasks=req.tasks,
            license_tiers=req.license_tiers,
            modalities=req.modalities,
            primitives=req.primitives,
            max_items_per_task=req.max_items_per_task,
            shuffle_seed=req.shuffle_seed,
            position_sensitivity=req.position_sensitivity,
            images=req.images,
            request=req.request,
            cost_model=req.cost_model,
            spec_version=req.spec_version,
        )
    except EvaluateError as exc:
        msg = str(exc)
        # PRD §14.7 errors: 422 malformed/unverifiable, 404 unknown dataset/task.
        status = 404 if "no datasets match" in msg or "zero tasks" in msg else 422
        raise HTTPException(status, msg) from exc

    write_evidence_bundle(result, req.model_dump(), root / "results", repo_root=root)
    for k in ("_raw_rows", "_model_info", "_datasets_used"):
        result.pop(k, None)
    return result


@app.get("/v1/catalogue")
def get_catalogue(tiers: list[str] | None = None) -> list[dict[str, Any]]:
    """PRD §3/§4 as data -- the same payload `bjb catalogue --json` prints."""
    root = _repo_root()
    rows = [m for m in discover(root) if not tiers or m.tier in tiers]
    return [
        {
            "name": m.name,
            "display_name": m.display_name,
            "domain": m.domain,
            "license": {"tier": m.tier, "spdx": m.spdx, "obligations": list(m.obligations)},
            "item_count": m.item_count,
            "primitives": list(m.primitives),
            "option_counts": list(m.option_counts),
            "modality": m.modality,
            "tasks": [
                {"task": t.task, "primitive": t.primitive, "option_count": t.option_count, "chance": t.chance}
                for t in m.tasks
            ],
        }
        for m in rows
    ]


@app.get("/v1/items")
def get_items(dataset: str, task: str | None = None, limit: int = 24) -> list[dict[str, Any]]:
    """**Public slice only** -- serving a held-out item over this endpoint
    would defeat the entire point of a frozen eval slice (PRD §14.9 item 6
    explicitly requires this be asserted in code and tested, not just
    documented). Reads the committed preview file, never `bench/heldout/`.

    `dataset` is checked against the real catalogue (`manifest.discover()`)
    -- not interpolated into a path directly -- specifically so a crafted
    value like `dataset="../heldout/civil_comments"` can't path-traverse out
    of `bench/preview/` at all, into `bench/heldout/` or anywhere else. This
    was caught and fixed while writing `scripts/check_items_endpoint.py`,
    which asserts exactly this."""
    root = _repo_root()
    known = {m.name for m in discover(root)}
    if dataset not in known:
        raise HTTPException(404, f"unknown dataset {dataset!r}")
    preview_path = root / "bench" / "preview" / f"{dataset}.json"
    if not preview_path.exists():
        raise HTTPException(404, f"no public preview for dataset {dataset!r}")
    import json

    items = json.loads(preview_path.read_text())
    if task:
        items = [it for it in items if it.get("task") == task]
    return items[:limit]


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    root = _repo_root()
    manifest_path = root / "results" / run_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, f"no run {run_id!r}")
    import json

    return json.loads(manifest_path.read_text())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "spec_version": SPEC_VERSION, "corpus_version": CORPUS_VERSION}

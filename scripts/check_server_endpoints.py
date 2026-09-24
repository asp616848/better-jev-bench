"""Checks `better_jev_bench/server.py`'s read-only endpoints (PRD §14.9 item
6), most pointedly the one PRD §14.9 calls out by name: `GET /v1/items` must
serve the public slice only, "asserted in code, with a test that the
held-out path is unreachable." Calls the FastAPI route functions directly
(no ASGI server needed) -- same plain-assertion style as this repo's other
`scripts/check_*.py`:

    uv run python3 scripts/check_server_endpoints.py

Requires the `serve` extra installed (`uv sync --extra serve`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

from better_jev_bench import server  # noqa: E402

n_checks = 0


def check(name: str, cond: bool) -> None:
    global n_checks
    n_checks += 1
    if not cond:
        raise SystemExit(f"FAIL: {name}")
    print(f"  ok  {name}")


# -- GET /v1/catalogue --------------------------------------------------------

catalogue = server.get_catalogue()
check("catalogue: returns at least one dataset", len(catalogue) >= 1)
check("catalogue: every row carries a license tier", all("license" in row and "tier" in row["license"] for row in catalogue))

filtered = server.get_catalogue(tiers=["A"])
check("catalogue: tier filter narrows or keeps the set", len(filtered) <= len(catalogue))

# -- GET /v1/items: public slice only ------------------------------------------

real_dataset = catalogue[0]["name"]
items = server.get_items(dataset=real_dataset)
check("items: returns a non-empty preview for a real dataset", len(items) > 0)
check("items: respects the default cap (<=24, the preview's own size)", len(items) <= 24)

try:
    server.get_items(dataset="../heldout/civil_comments")
    raise SystemExit("FAIL: items: path-traversal dataset name was NOT rejected -- held-out path reachable!")
except HTTPException as exc:
    check("items: a path-traversal dataset name ('../heldout/civil_comments') is rejected, not path-joined", exc.status_code == 404)

try:
    server.get_items(dataset="not_a_real_dataset_xyz")
    raise SystemExit("FAIL: items: unknown dataset was not rejected")
except HTTPException as exc:
    check("items: an unknown (but non-traversal) dataset name is rejected the same way", exc.status_code == 404)

limited = server.get_items(dataset=real_dataset, limit=2)
check("items: limit is honoured", len(limited) <= 2)

# -- GET /v1/runs/{run_id}: unknown run -> 404 --------------------------------

try:
    server.get_run("run_does_not_exist")
    raise SystemExit("FAIL: runs: unknown run_id was not rejected")
except HTTPException as exc:
    check("runs: unknown run_id -> 404", exc.status_code == 404)

# -- GET /health ---------------------------------------------------------------

health = server.health()
check("health: reports the frozen spec_version", health["spec_version"] == server.SPEC_VERSION)

print(f"\n{n_checks} checks run, 0 failed.")

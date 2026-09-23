"""Public / held-out split, and the tamper-evident label commitment (PRD §7.5, G3).

Three properties this file is responsible for, in order of how badly they'd hurt
if they were wrong:

**1. The split is content-determined, not order-determined.** An item's side is
`sha256(salt || state_hash)`, so re-running `bjb build` on a machine where the
source rows arrive in a different order, or where the upstream gained rows,
produces the same side for every item that existed before. A split that depends
on iteration order silently reshuffles the held-out set on every rebuild, which
would make the §7.5 commitment meaningless.

**2. It keys on the *state* hash, not the item id.** Civil Comments emits a
`score` item and a `noul` item over the same comment text; CUAD emits several
tasks over the same excerpt. Keying on item id would put those on opposite sides
and leak the eval text into the training slice through the back door. This is
the real content of §7.4 gate 7, and keying on the state is what makes the gate
pass by construction rather than by luck.

**3. The commitment is over (item_id, label) pairs in sorted order**, so it is
invariant to everything except the actual answers. A contributor who quietly
changes a held-out label changes the hash, the diff shows it, and every prior
result on that dataset is invalidated automatically rather than by anyone's
judgment.

**What this is not, stated plainly**: a *secret* held-out set. The held-out
slices in `bench/heldout/` are committed to a public repo with their labels, so
nothing here prevents someone training on them -- only the canary marker (§5.5
rule 2) and this commitment's tamper-evidence do. A genuinely sealed slice needs
hosted infrastructure that does not exist yet and is on the roadmap. Freezing
first and sealing later is the right order: a split created after a training run
is a split someone has to be trusted about, and that is the failure §7.5 exists
to prevent.

**4. The bucket key is `Item.split_key`, not always `state_hash` (PRD §13.6).**
For every text item and for an ordinary one-image-per-state vision item,
`split_key` *is* `state_hash` -- content-determined, as above. Atari-HEAD is the
one real exception: adjacent frames within a human trial are near-duplicate
images with near-duplicate state text, so a frame-level split leaks trivially
between neighbours. Its loader sets `Item.split_key_override` to a per-trial key
so every frame from one trial buckets identically. This module stays ignorant of
*why* a key was overridden; it only ever needs `item.split_key`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .types import Item

#: Frozen at spec time. Changing it reshuffles every split in the corpus and is
#: a spec version bump, not a tuning knob (PRD §5.5 rule 8).
SPLIT_SALT = "better-jev-bench/split/v1"

_BUCKETS = 1_000_000


def bucket(split_key: str, *, salt: str = SPLIT_SALT) -> int:
    digest = hashlib.sha256(f"{salt}\x1f{split_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _BUCKETS


def side(item: Item, heldout_fraction: float, *, salt: str = SPLIT_SALT) -> str:
    """"heldout" or "public". Deterministic in `item.split_key` alone -- the
    item's own state+image content hash by default, or a loader's explicit
    override (PRD §13.6)."""
    return "heldout" if bucket(item.split_key, salt=salt) < heldout_fraction * _BUCKETS else "public"


def label_commitment(items: Iterable[Item]) -> str:
    """`sha256:<hex>` over sorted `item_id\\x1flabel` lines -- the §7.5 commitment."""
    lines = sorted(f"{it.item_id}\x1f{it.label}" for it in items)
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def file_digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"

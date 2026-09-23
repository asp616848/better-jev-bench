"""Real-behaviour check for an image-bearing `bjb export`.

Companion to `check_ekvachan_compat.py`, same discipline: run it against a real
export, not a mock. Where that script re-checks ekVachan's own eight-key
`_validate()`, this one checks the ninth key PRD §13.4 adds for an
image-bearing record: that every `images[].path` is a real file on this
machine and that its content actually hashes to `images[].sha256` -- exactly
what the sibling's `training/build_vision_slice.py` (PRD §5.2b item 3) is
specified to assert per-row before training on it. Failing here, before a
training run ever starts, is strictly better than failing there.

    bjb export --out exports/vision_compat --datasets screenspot_v2 --slice heldout
    python scripts/check_vision_export.py exports/vision_compat/heldout.jsonl

Run this whenever the export path, the image cache layout, or a vision loader
changes.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_TEXT_KEYS = {
    "state", "question_key", "question_type", "instructions",
    "options", "label", "label_idx", "source",
}
EXPECTED_IMAGE_KEYS = EXPECTED_TEXT_KEYS | {"images"}
EXPECTED_IMAGE_FIELDS = {"sha256", "path", "media_type", "width", "height"}


def _check(records: list[dict], name: str) -> None:
    n_text = n_image = n_image_refs = 0
    for r in records:
        keys = set(r)
        if "images" in r:
            assert keys == EXPECTED_IMAGE_KEYS, f"{name}: image record keys {sorted(keys)} != {sorted(EXPECTED_IMAGE_KEYS)}"
            assert isinstance(r["images"], list) and r["images"], f"{name}: 'images' must be a non-empty list"
            n_image += 1
            for img in r["images"]:
                assert set(img) == EXPECTED_IMAGE_FIELDS, f"{name}: image ref fields {sorted(img)} != {sorted(EXPECTED_IMAGE_FIELDS)}"
                path = Path(img["path"])
                assert path.is_file(), f"{name}: {path} does not exist -- image reference does not resolve"
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                assert actual == img["sha256"], (
                    f"{name}: {path} hashes to {actual}, expected {img['sha256']} -- cache content does not "
                    "match the record's own claim"
                )
                assert img["media_type"] in ("image/png", "image/jpeg"), f"{name}: unexpected media_type {img['media_type']!r}"
                assert img["width"] > 0 and img["height"] > 0, f"{name}: non-positive dimensions in {img}"
                n_image_refs += 1
        else:
            assert keys == EXPECTED_TEXT_KEYS, f"{name}: text record keys {sorted(keys)} != ekVachan's eight -- NOT byte-identical"
            n_text += 1
    print(f"OK: {name}: {n_text:,} text records (eight keys, byte-identical), {n_image:,} image records "
          f"(nine keys), {n_image_refs:,} image references all resolved + hash-verified")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print(f"{path}: no records -- nothing to check (is this dataset eval_only and you exported --slice public?)")
        return 1
    _check(records, path.name)
    widths = collections.Counter(len(r["options"]) for r in records)
    print(f"  option widths: {dict(sorted(widths.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

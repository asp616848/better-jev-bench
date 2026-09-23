"""ScreenSpot-v2 (`OS-Copilot/ScreenSpot-v2`) -- the corpus's third-party,
zero-training-exposure vision eval (PRD §13.3, §5.2b of the sibling PRD: "the
vision analogue of what JevBench/jabr-v2 are for text").

License: Apache-2.0, verified 2026-09-24 against the HF Hub API
(`cardData["license"] == "apache-2.0"`) -- re-confirmed independently of
`better-jev-bench_PRD.md` §13.3, not copied from it.

**This dataset is `eval_only` (manifest `dataset.eval_only = true`).** It is a
re-annotation of ScreenSpot that corrected 11.32% of the original's mislabelled
grounding targets (~1.2k items total across desktop/mobile/web) -- far too
small to train on and, more importantly, the single dataset in this corpus's
multimodal slate whose entire value is being a screenshot a model has *never*
seen a training gradient from. `bjb export`'s default public slice refuses it
outright (see `export.py`); this loader building both a public and held-out
slice at all is a manifest-schema formality -- `bjb build` requires it -- not
an invitation to train on the public 5% of it.

**The `choice` transform, and why most rows are dropped.** Each source row is
one `(image, instruction, target bbox)` triple; it does *not* ship a list of
"other elements in this screenshot" the way OS-Atlas's per-image `elements`
list does. The only real, same-screenshot distractor pool available is *other
rows in the same split that happen to reference the same image file* -- many
images carry exactly one annotated row, and those cannot honestly become a
`choice` item (PRD §13.6: "distractors drawn from the same screenshot's other
annotated elements ... never from another screenshot"). Rows whose image has
only one annotation are dropped rather than filled with a distractor from a
different screenshot, which would make the item trivially solvable from
instruction-image mismatch rather than genuine grounding. Verified on the real
download (2026-09-24): 900 of 1,272 rows across all three splits survive this
filter, comfortably above the §7.3 32-item floor.

**Option text is a real, per-row bbox descriptor, not the row's own
instruction.** Each candidate's own `instruction` field would leak the target
identity as plain text overlap with the state's instruction; the source
carries no separate element caption. What it does carry honestly is each
candidate's own bounding box and UI role (`data_type`), so an option reads
`"<data_type> at (x=..., y=...), size WxHpx"` -- real, per-element, same-
screenshot content that requires looking at the image to resolve, not a
caption invented here.

**Every item is a fixed-width binary choice, one correct element vs. one real
distractor.** `build.py._check_against_manifest` requires one `option_count`
per task across every item; a screenshot's real candidate count varies (2 to a
handful), so this loader does not present the whole candidate set at once. For
a group of *k* real, distinct-bbox candidates sorted into a deterministic
reading order (top-to-bottom, left-to-right), row *i*'s distractor is its
cyclic neighbour at index `(i + 1) mod k` -- always a genuine, same-screenshot
element, never invented, and every row in a multi-candidate group gets exactly
one binary item.
"""

from __future__ import annotations

import collections
import json
import zipfile
from collections.abc import Iterator

from huggingface_hub import hf_hub_download

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item, Provenance
from ._common import clean, stable_unit

HF_REPO = "OS-Copilot/ScreenSpot-v2"
HF_REVISION = "5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c"
SPLIT_FILES = {
    "desktop": "screenspot_desktop_v2.json",
    "mobile": "screenspot_mobile_v2.json",
    "web": "screenspot_web_v2.json",
}
IMAGE_ZIP = "screenspotv2_image.zip"
IMAGE_ZIP_PREFIX = "screenspotv2_image/"
MIN_CANDIDATES_PER_IMAGE = 2

INSTRUCTIONS = (
    "A screenshot of a graphical user interface is shown, along with a natural-language "
    "instruction describing an action a user wants to take. Choose which of the following "
    "screen regions is the element the instruction refers to."
)


def _option_text(row: dict) -> str:
    x, y, w, h = row["bbox"]
    return f"{row['data_type']} at (x={x}, y={y}), size {w}x{h}px"


class ScreenSpotV2(BenchmarkDataset):
    name = "screenspot_v2"

    def schema(self) -> list[RequiredOptions]:
        return [
            RequiredOptions(
                task="target_element",
                question_key="target_element",
                primitive="choice",
                instructions=INSTRUCTIONS,
                options=("A", "B"),  # placeholder text; every item supplies its own real pair via `item()`
            )
        ]

    def _rows(self) -> Iterator[tuple[str, dict]]:
        for split, filename in SPLIT_FILES.items():
            path = hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset", revision=HF_REVISION, filename=filename
            )
            with open(path, encoding="utf-8") as fh:
                for row in json.load(fh):
                    yield split, row

    def load_items(self) -> Iterator[Item]:
        zip_path = hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", revision=HF_REVISION, filename=IMAGE_ZIP
        )

        by_image: dict[str, list[dict]] = collections.defaultdict(list)
        for split, row in self._rows():
            row = {**row, "_split": split}
            by_image[row["img_filename"]].append(row)

        with zipfile.ZipFile(zip_path) as zf:
            image_bytes_cache: dict[str, bytes] = {}
            for img_filename in sorted(by_image):
                rows = by_image[img_filename]
                # De-dup identical bboxes: a same-image duplicate annotation
                # would otherwise produce two options with identical text,
                # which Question rejects (and which is genuinely ambiguous --
                # PRD §2.4, "no defensible mapping exists" is the same call
                # CFPB's dropped taxonomy values made).
                by_bbox: dict[tuple[int, int, int, int], dict] = {}
                for row in rows:
                    by_bbox[tuple(row["bbox"])] = row
                candidates = list(by_bbox.values())
                if len(candidates) < MIN_CANDIDATES_PER_IMAGE:
                    continue
                candidates.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))  # reading order, deterministic
                k = len(candidates)

                if img_filename not in image_bytes_cache:
                    image_bytes_cache[img_filename] = zf.read(IMAGE_ZIP_PREFIX + img_filename)
                image_ref = self.image(
                    image_bytes_cache[img_filename],
                    source_uri=f"hf://datasets/{HF_REPO}/{IMAGE_ZIP_PREFIX}{img_filename}",
                )

                for i, row in enumerate(candidates):
                    instruction = clean(row["instruction"], max_chars=300)
                    if not instruction:
                        continue
                    label = _option_text(row)
                    distractor = _option_text(candidates[(i + 1) % k])
                    if distractor == label:
                        continue  # identical descriptor text (rare edge case) -- ambiguous, drop rather than fake
                    state = f"Instruction: {instruction}"
                    # Deterministic, not fixed-position: a constant "label always
                    # first" order would let a model win this task by always
                    # answering the first option, without ever looking at the
                    # image. Order is a stable hash of the item's own content, so
                    # a rebuild reproduces it exactly without needing a shared RNG.
                    pair = (label, distractor) if stable_unit(self.name, img_filename, i) < 0.5 else (distractor, label)
                    yield self.item(
                        "target_element",
                        state=state,
                        label=label,
                        options=pair,
                        source=f"screenspot_v2_{row['_split']}_{row['data_source']}",
                        modality="image",
                        images=(image_ref,),
                        provenance=Provenance(natural_language_state=False),
                    )

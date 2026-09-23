"""OS-Atlas-data (`OS-Copilot/OS-Atlas-data`), `desktop_domain/linux` subset only.

License: Apache-2.0 declared on the repo card, verified 2026-09-24 against the
HF Hub API -- but the repo as a whole is an *aggregation* (AMEX, UIBert,
RICO/Widget-Captioning, SeeClick, FineWeb, plus this project's own desktop
captures), and each constituent keeps its own upstream terms
(better-jev-bench_PRD.md §13.3, the same "aggregate license != subset license"
caution §3.6 already puts on MedMNIST). **This loader pulls exactly one
subset: `desktop_domain/linux_{splited,images}`** -- Linux desktop screenshots
and annotations OS-Atlas's own authors collected and captioned themselves, not
sourced from any of the named third-party corpora. That is what makes Apache-
2.0 actually defensible here: there is no external dataset's license to
conflict with, unlike `mobile_domain` (RICO/AMEX/UIBert) or `web_domain`
(SeeClick/FineWeb), which this loader deliberately does not touch.

**The `choice` transform.** Each row is one screenshot plus a real, human-
captioned `elements` list (`{instruction, bbox, data_type}`, `bbox` normalised
`[left, top, right, bottom]` in `[0, 1]`) -- richer per-element ground truth
than ScreenSpot-v2 provides, since every element carries its own caption
("Save to Google Drive", "Forward", "More options menu"), not just a type and
a box. That caption is exactly why it must **not** appear in an item's
options: the *state* names one element's own caption as the thing to find, so
using captions as option text would let a model win by verbatim string match
against the state, never looking at the image. Options are therefore the same
bbox-position descriptor `screenspot_v2.py` uses, built from real per-element,
same-screenshot content -- just a different field of it. See that module's
docstring for the fixed-binary-width rationale (`build.py` requires one
`option_count` per task; real per-image candidate counts vary 1-5) and the
cyclic-neighbour distractor selection, both reused here unchanged.

Rows with only one annotated element (345 of 9,671, verified on the real pull
2026-09-24) are dropped -- no same-screenshot distractor exists for them, and
PRD §13.6 is explicit that a distractor must never come from another
screenshot.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator

from huggingface_hub import hf_hub_download

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item, Provenance
from ._common import clean, stable_unit

HF_REPO = "OS-Copilot/OS-Atlas-data"
HF_REVISION = "e3a4c90c5f6129c25efdaa0671b08a11f7cb8f3f"
ANNOTATIONS_FILE = "desktop_domain/linux_splited.json"
IMAGES_ZIP = "desktop_domain/linux_images.zip"
MIN_ELEMENTS_PER_ROW = 2

INSTRUCTIONS = (
    "A screenshot of a Linux desktop application is shown, along with a short description of "
    "one interface element on it. Choose which of the following screen regions is that element."
)


def _option_text(bbox: list[float], data_type: str) -> str:
    left, top, right, bottom = bbox
    cx, cy = round((left + right) / 2 * 100, 1), round((top + bottom) / 2 * 100, 1)
    return f"{data_type} centred at ({cx}%, {cy}%) of the screenshot"


class OSAtlasDesktopLinux(BenchmarkDataset):
    name = "os_atlas"

    def schema(self) -> list[RequiredOptions]:
        return [
            RequiredOptions(
                task="target_element",
                question_key="target_element",
                primitive="choice",
                instructions=INSTRUCTIONS,
                options=("A", "B"),  # placeholder; every item supplies its own real pair
            )
        ]

    def load_items(self) -> Iterator[Item]:
        ann_path = hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", revision=HF_REVISION, filename=ANNOTATIONS_FILE
        )
        zip_path = hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", revision=HF_REVISION, filename=IMAGES_ZIP
        )
        with open(ann_path, encoding="utf-8") as fh:
            rows = json.load(fh)

        with zipfile.ZipFile(zip_path) as zf:
            available = set(zf.namelist())
            image_bytes_cache: dict[str, bytes] = {}
            for row_idx, row in enumerate(rows):
                img_filename = row["img_filename"]
                if img_filename not in available:
                    # Real, minor upstream inconsistency: 6 of 1,186 referenced
                    # filenames in `linux_splited.json` are not actually present
                    # in `linux_images.zip` (verified 2026-09-24). Skip rather
                    # than fail the whole build over a handful of dangling refs.
                    continue
                # De-dup identical bboxes within this row -- an exact repeat
                # would otherwise produce two options with identical text,
                # which Question rejects (see screenspot_v2.py for the same
                # judgment call).
                by_bbox: dict[tuple[float, ...], dict] = {}
                for el in row["elements"]:
                    by_bbox[tuple(el["bbox"])] = el
                elements = list(by_bbox.values())
                if len(elements) < MIN_ELEMENTS_PER_ROW:
                    continue
                elements.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))  # reading order
                k = len(elements)

                if img_filename not in image_bytes_cache:
                    image_bytes_cache[img_filename] = zf.read(img_filename)
                image_ref = self.image(
                    image_bytes_cache[img_filename],
                    source_uri=f"hf://datasets/{HF_REPO}/{IMAGES_ZIP}#{img_filename}",
                )

                for i, el in enumerate(elements):
                    instruction = clean(el["instruction"], max_chars=300)
                    if not instruction:
                        continue
                    label = _option_text(el["bbox"], el["data_type"])
                    distractor_el = elements[(i + 1) % k]
                    distractor = _option_text(distractor_el["bbox"], distractor_el["data_type"])
                    if distractor == label:
                        continue
                    state = f"Element description: {instruction}"
                    pair = (
                        (label, distractor)
                        if stable_unit(self.name, img_filename, row_idx, i) < 0.5
                        else (distractor, label)
                    )
                    yield self.item(
                        "target_element",
                        state=state,
                        label=label,
                        options=pair,
                        source="os_atlas_desktop_linux",
                        modality="image",
                        images=(image_ref,),
                        provenance=Provenance(natural_language_state=False),
                    )

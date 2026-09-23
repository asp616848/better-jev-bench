"""Atari-HEAD (Zenodo record 3451322) -- PRD §13.3's "best single fit in the
whole survey": a real human playing Atari, frame-by-frame, with the actual
keystroke recorded for every frame.

License: CC BY 4.0, verified 2026-09-24 directly against the Zenodo REST API
(`GET /api/records/3451322` -> `metadata.license.id == "cc-by-4.0"`), not
copied from the record's HTML page or from `better-jev-bench_PRD.md` §13.3.

**Scale, and why this loader pulls one game rather than all twenty.** The full
record is ~8M frame-action pairs across 20 games, 12.1 GB. PRD §13.3 itself
names ~10,000 rows as the right first-pass size -- "far more than you need for
a first pass" is the PRD's own words for the full pull. This loader downloads
one game's archive (`breakout.zip`, 132 MB, 17 human trials) and samples a
bounded, deterministic stride of frames from each trial, which reaches that
target without downloading gigabytes this corpus will never present as items.
Widening to more games later is a config change (`GAME`), not a rewrite.

**The `choice` transform.** State: a one-line game context (the state text is
deliberately short and near-identical across every frame of a game -- exactly
the case PRD §13.4's `state_hash` fix exists for for; see that module).
Options: the fixed, global 18-action ALE joystick vocabulary from the record's
own `action_enums.txt`, well inside the 26-letter budget, so **no width
narrowing is applied or needed** (PRD §13.6). Label: the actual human
keystroke recorded for that exact frame -- a real action, not a heuristic one.

**Split key: per-trial, not per-frame (PRD §13.6).** Two adjacent frames in
one trial are near-duplicate images with near-duplicate state text; splitting
on `state_hash` (even the fixed, image-aware one) would still let one frame of
a near-static sequence land in training and its neighbour a moment later land
in held-out -- leakage in substance if not in exact hash collision. Every item
from one `(game, trial)` sets `Item.split_key_override` to that trial's own
key, so `bjb build`'s leak check (which now checks `split_key`, not
`state_hash`, precisely for this) guarantees whole trials, never frames,
straddle the public/held-out line.
"""

from __future__ import annotations

import csv
import io
import re
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import requests

from ..dataset import BenchmarkDataset, RequiredOptions
from ..types import Item, Provenance

ZENODO_RECORD = "3451322"
ZENODO_FILES_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
GAME = "breakout"
#: Per-trial frame sample target. 17 trials x ~600 ~= 10,000, matching the
#: PRD §13.3 first-pass estimate.
FRAMES_PER_TRIAL = 600

INSTRUCTIONS = (
    "A single frame from a human playing an Atari video game is shown. Choose the joystick "
    "action the human player took at that exact frame."
)


def _download(url: str, dest: Path) -> Path:
    """Cache under `raw_downloads/atari_head/` (gitignored, see `.gitignore`) --
    a real network fetch on first use, reused on every rebuild after."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    return dest


def _parse_action_enums(text: str) -> tuple[str, ...]:
    """`action_enums.txt` -> 18 human-readable action names, ordered by their
    integer id. `PLAYER_A_UPRIGHTFIRE = 14` -> `"UPRIGHTFIRE"` at index 14."""
    by_id: dict[int, str] = {}
    for line in text.splitlines():
        m = re.match(r"\s*PLAYER_A_(\w+)\s*=\s*(\d+)", line)
        if m:
            by_id[int(m.group(2))] = m.group(1)
    if not by_id:
        raise ValueError("action_enums.txt: no PLAYER_A_* entries parsed")
    return tuple(by_id[i] for i in sorted(by_id))


class AtariHead(BenchmarkDataset):
    name = "atari_head"

    def _raw_dir(self) -> Path:
        return self.repo_root / "raw_downloads" / "atari_head"

    def _action_names(self) -> tuple[str, ...]:
        path = _download(f"{ZENODO_FILES_BASE}/action_enums.txt?download=1", self._raw_dir() / "action_enums.txt")
        return _parse_action_enums(path.read_text(encoding="utf-8"))

    def schema(self) -> list[RequiredOptions]:
        return [
            RequiredOptions(
                task="action",
                question_key="action",
                primitive="choice",
                instructions=INSTRUCTIONS,
                options=self._action_names(),
            )
        ]

    def load_items(self) -> Iterator[Item]:
        options = self.spec("action").options
        zip_path = _download(f"{ZENODO_FILES_BASE}/{GAME}.zip?download=1", self._raw_dir() / f"{GAME}.zip")

        with zipfile.ZipFile(zip_path) as game_zip:
            trial_names = sorted(
                {n[: -len(".tar.bz2")] for n in game_zip.namelist() if n.endswith(".tar.bz2")}
            )
            for trial_path in trial_names:
                trial_name = trial_path.split("/")[-1]
                yield from self._trial_items(game_zip, trial_path, trial_name, options)

    def _trial_items(
        self, game_zip: zipfile.ZipFile, trial_path: str, trial_name: str, options: tuple[str, ...]
    ) -> Iterator[Item]:
        rows = list(csv.DictReader(io.StringIO(game_zip.read(f"{trial_path}.txt").decode("utf-8"))))
        n = len(rows)
        if n == 0:
            return
        stride = max(1, n // FRAMES_PER_TRIAL)
        sampled = rows[::stride]

        wanted_frame_ids = {r["frame_id"] for r in sampled if r["action"].strip().lstrip("-").isdigit()}
        if not wanted_frame_ids:
            return

        # One sequential pass over the (compressed) tar extracts exactly the
        # frames this trial needs -- cheaper than reopening the archive per
        # frame, and the only access pattern a bz2-compressed tar supports
        # without decompressing the whole thing up front regardless.
        frame_png: dict[str, bytes] = {}
        with game_zip.open(f"{trial_path}.tar.bz2") as fh, tarfile.open(fileobj=fh, mode="r|bz2") as tar:
            for member in tar:
                stem = Path(member.name).stem
                if stem in wanted_frame_ids:
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        frame_png[stem] = extracted.read()
                    if len(frame_png) == len(wanted_frame_ids):
                        break

        split_key = f"{GAME}/{trial_name}"
        for row in sampled:
            action_str = row["action"].strip()
            if not action_str.lstrip("-").isdigit():
                continue  # a handful of trials omit action on their last row
            action_id = int(action_str)
            if not (0 <= action_id < len(options)):
                continue
            frame_id = row["frame_id"]
            png_bytes = frame_png.get(frame_id)
            if png_bytes is None:
                continue  # tar pass ended early (all wanted frames already found)

            image_ref = self.image(
                png_bytes,
                source_uri=f"https://zenodo.org/records/{ZENODO_RECORD}/files/{GAME}.zip#{trial_path}.tar.bz2!{frame_id}.png",
            )
            state = f"Atari game: {GAME}. A single video game frame is shown; choose the action taken."
            yield self.item(
                "action",
                state=state,
                label=options[action_id],
                options=options,
                source=f"atari_head_{GAME}_{trial_name}",
                modality="image",
                images=(image_ref,),
                provenance=Provenance(natural_language_state=False, heuristic_label=False),
                split_key_override=split_key,
            )

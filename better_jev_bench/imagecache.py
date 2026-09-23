"""The content-addressed image cache (PRD §13.5).

The corpus does not redistribute pixels -- several sources permit *use* but not
*redistribution*, and one of them (OS-Atlas-data) is 816 GB uncompressed, which
would make `git clone && bjb evaluate` (the §8.1 promise the whole two-slice
storage policy in `build.py` already protects) impossible. So an image lives on
disk exactly once, named by its own SHA-256, under a directory `bjb build`
never commits to git:

    data/images/<dataset>/<sha256[:2]>/<sha256>.<ext>

The two-level `<sha256[:2]>` fan-out keeps any one directory from accumulating
hundreds of thousands of entries (Atari-HEAD alone samples ~10k frames; a real
pull of the full corpus would be in the millions). `store_image_bytes()` is the
only writer: it hashes first, checks whether that hash is already cached, and
if not, writes the bytes and re-reads them back to verify -- the same
"verify-what-you-wrote" discipline `build.py` already applies to the gzipped
JSONL slices via `file_digest()`.

No image-decoding dependency (Pillow, etc.) is added for this -- consistent
with `pyproject.toml`'s existing stance that `bjb build`/`bjb export` must work
without a web stack. PNG and baseline/progressive JPEG both carry their pixel
dimensions in a small, well-known header, so `_dimensions()` reads just enough
of the byte stream to answer `width, height` without decoding a single pixel.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from .types import ImageRef

#: Kept in sync with `types.IMAGE_MEDIA_TYPES`.
EXT_FOR_MEDIA_TYPE: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


def cache_root(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "images"


def image_cache_path(repo_root: Path, dataset: str, sha256: str, media_type: str) -> Path:
    """Where an image with this hash lives (or would live) for this dataset.
    Deterministic from its inputs alone -- no directory listing required."""
    ext = EXT_FOR_MEDIA_TYPE.get(media_type)
    if ext is None:
        raise ValueError(f"no cache extension registered for media_type {media_type!r}")
    return cache_root(repo_root) / dataset / sha256[:2] / f"{sha256}{ext}"


def sniff_media_type(data: bytes) -> str:
    """The container format, read from the byte stream itself -- never trust a
    filename extension. Verified necessary, not theoretical: 111 of 757 files
    in ScreenSpot-v2's `screenspotv2_image.zip` are JPEG bytes behind a
    `.png` filename (checked 2026-09-24)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    raise ValueError(f"unrecognised image format (first bytes {data[:8]!r})")


def _dimensions(data: bytes, media_type: str) -> tuple[int, int]:
    """Pixel width/height read directly from the container header."""
    if media_type == "image/png":
        return _png_dimensions(data)
    if media_type == "image/jpeg":
        return _jpeg_dimensions(data)
    raise ValueError(f"cannot determine dimensions for media_type {media_type!r}")


def _png_dimensions(data: bytes) -> tuple[int, int]:
    sig = b"\x89PNG\r\n\x1a\n"
    if data[:8] != sig:
        raise ValueError("not a PNG file (bad signature)")
    # IHDR is always the first chunk: 8-byte sig, 4-byte length, 4-byte "IHDR",
    # then width (u32be), height (u32be).
    if data[12:16] != b"IHDR":
        raise ValueError("malformed PNG: first chunk is not IHDR")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG file (bad SOI marker)")
    i = 2
    n = len(data)
    # SOFn markers (0xC0-0xC3, 0xC5-0xC7, 0xC9-0xCB, 0xCD-0xCF) carry the frame
    # dimensions; every other marker segment is skipped by its own length field.
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2  # markers with no payload
            continue
        if i + 4 > n:
            break
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if marker in sof_markers:
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        i += 2 + seg_len
    raise ValueError("malformed JPEG: no SOF segment found")


def store_image_bytes(repo_root: Path, dataset: str, data: bytes, *, source_uri: str) -> ImageRef:
    """Hash `data`, write it into the content-addressed cache if not already
    present, verify the write, and return the `ImageRef` an `Item` carries.

    `media_type` is never taken on trust from a caller -- it is sniffed from
    `data` itself via `sniff_media_type()`. A source's filename extension is
    not reliable evidence of its actual container format (ScreenSpot-v2's own
    archive proves this: 111 of 757 files are JPEG bytes behind a `.png` name),
    and a wrong media_type would silently corrupt both the cache's file
    extension and the dimensions read out of the header.

    Idempotent and safe to call twice with the same bytes (a rebuild does this
    routinely): if the path already exists, its content is re-hashed and
    checked against the hash we are about to store under, rather than
    trusted -- a stale or corrupted cache entry from an interrupted prior run
    must fail loudly here, not surface three steps later as a validate gate
    failure with a confusing message.
    """
    media_type = sniff_media_type(data)
    sha = hashlib.sha256(data).hexdigest()
    width, height = _dimensions(data, media_type)
    path = image_cache_path(repo_root, dataset, sha, media_type)
    if path.exists():
        existing = path.read_bytes()
        existing_sha = hashlib.sha256(existing).hexdigest()
        if existing_sha != sha:
            raise ValueError(
                f"cache collision at {path}: on-disk content hashes to {existing_sha}, "
                f"expected {sha} -- the cache is corrupt and must be cleared"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        written = path.read_bytes()
        if hashlib.sha256(written).hexdigest() != sha:
            raise ValueError(f"write verification failed for {path}: on-disk content does not hash to {sha}")
    return ImageRef(sha256=sha, source_uri=source_uri, media_type=media_type, width=width, height=height)


def verify_cached(repo_root: Path, dataset: str, image: ImageRef) -> Path:
    """Resolve `image` to its cache path and assert the file exists and its
    content hash matches -- exactly the check PRD §13.4/§5.2b says the
    sibling's `build_vision_slice.py` will perform on every row; `bjb export`
    performs the same check before handing out a path (see `export.py`)."""
    path = image_cache_path(repo_root, dataset, image.sha256, image.media_type)
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset}: image {image.sha256} not found in the local cache at {path}. "
            "Run `bjb build` for this dataset first."
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != image.sha256:
        raise ValueError(f"{dataset}: cached image at {path} hashes to {actual}, expected {image.sha256}")
    return path

"""Shared helpers for the first-party loaders."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

_WS = re.compile(r"[ \t ]+")


def clean(text: str | None, *, max_chars: int | None = None) -> str:
    """Normalise whitespace and (optionally) truncate to a stated budget.

    Truncation appends an explicit marker rather than cutting silently: a model
    reading a clipped contract should be able to tell that it was clipped.
    """
    if not text:
        return ""
    out = _WS.sub(" ", str(text).replace("\r\n", "\n").replace("\r", "\n")).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    if max_chars is not None and len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + " […truncated]"
    return out


def stable_unit(*parts: Any) -> float:
    """A deterministic float in [0, 1) from the given parts.

    Used for sampling decisions that must survive a rebuild on another machine.
    Python's `hash()` is salted per process and `random` depends on call order;
    neither is acceptable for something a build receipt commits a hash to.
    """
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def take_stable(rows: Iterable[Any], key, n: int, *, salt: str) -> list[Any]:
    """Deterministically keep about `n` of `rows` by hash rank.

    A hash-rank sample rather than a head-slice, because a head-slice of a
    date-ordered source (CFPB, Civil Comments) samples one time period.
    """
    scored = sorted(((stable_unit(salt, key(r)), r) for r in rows), key=lambda t: t[0])
    return [r for _, r in scored[:n]]

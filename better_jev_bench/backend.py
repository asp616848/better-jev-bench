"""The model integration surface (PRD §14.8, §14.9 item 2).

Two things live here, and the PRD is explicit about which is primary:

1. **`ModelBackend`** -- the `Protocol` an in-process model implements. This
   is the stub the whole engine is built against (§14.8's "the one piece of
   code worth writing before the engine itself").
2. **`HTTPBackend`** -- the reference implementation of that protocol, and
   the one that actually matters: it speaks `/v1/systemone` over the wire, so
   any endpoint that already serves that contract -- ekVachan, Von, Rizzo
   Flow, a third party's hosted Jev -- needs **zero integration code** (PRD
   §14.8 point 1). `HTTPBackend` is deliberately *just* an implementation of
   `ModelBackend`, which is what keeps the two paths honestly identical: PRD
   §14.8 states outright "there is no scoring path that only the reference
   model can take," and this module is that sentence enforced by the type
   system rather than merely asserted in prose.

`classify()` is PRD §14.3's whole per-item outcome table, translated into the
one function that must get it exactly right: exact string equality after
`str.strip()` and nothing else, no case folding, no nearest-option snapping.
`scripts/check_outcome_classifier.py` tests it exhaustively, including the
two cases the PRD calls out by name -- an answer differing only by case
(`out_of_schema`) and one differing only by trailing whitespace (not
`out_of_schema`).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Outcome(str, Enum):
    """PRD §14.3's four buckets. A scored item lands in exactly one."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    OUT_OF_SCHEMA = "out_of_schema"
    DECLINED = "declined"


class Declined(Exception):
    """Raised by a `ModelBackend.answer()` for anything that isn't a usable
    `/v1/systemone`-shaped 200: a non-200 status, a timeout, an unparseable
    body, or a transport failure. PRD §14.3: `declined` is *uncovered and
    wrong*, exactly like `out_of_schema` -- a backend cannot improve its score
    by raising this instead of answering badly (§5.5 rule 9)."""

    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"declined (status={status}): {detail}")


@runtime_checkable
class ModelBackend(Protocol):
    """PRD §14.8's stub, verbatim in shape. A ~15-line adapter is enough for
    a local checkpoint with no HTTP server of its own; `HTTPBackend` below
    is the zero-code path for anything that already speaks `/v1/systemone`."""

    name: str

    def describe(self) -> dict[str, Any]:
        """Everything that goes into the evidence bundle's `model_info`:
        weight hash, base model, revision, device, quantisation. Free-form
        beyond those keys."""
        ...

    def answer(self, request: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        """`request` is the item's verbatim `/v1/systemone` body (PRD §6.1).
        Return a `/v1/systemone` response body, or raise `Declined`.
        Returning an out-of-schema `choice` is legal -- it scores as
        `Outcome.OUT_OF_SCHEMA` -- and a backend must never repair its own
        answer (PRD §14.8's "no schema-repair hook")."""
        ...


class HTTPBackend:
    """Speaks `/v1/systemone` over HTTP with nothing but the stdlib, mirroring
    the sibling project's own `benchmarks/common/backends.py::HTTPBackend` in
    spirit: no new dependency for the one integration path that has to work
    everywhere. Every non-200, timeout, or unparseable body becomes a
    `Declined` rather than an exception the run loop has to know the shape
    of -- PRD §14.7's "a model endpoint that is unreachable is not an error;
    it is a run in which every item is declined.\""""

    name = "http"

    def __init__(self, endpoint: str, *, headers: dict[str, str] | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.headers = dict(headers or {})

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "endpoint": self.endpoint}

    def answer(self, request: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        data = json.dumps(request).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json", **self.headers},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            # PRD §14.3/§14.9 item 2: a `501` for an unimplemented primitive
            # (serve/server.py's real behaviour for `noul`/`score` today) is
            # first-class, not a crash -- it becomes a `Declined` and the run
            # keeps going, exactly as PRD §14.8 point 1's table promises.
            body = exc.read().decode("utf-8", "replace") if hasattr(exc, "read") else str(exc)
            raise Declined(exc.code, body) from exc
        except urllib.error.URLError as exc:
            raise Declined(0, f"connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise Declined(0, f"timeout after {timeout_s}s: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise Declined(200, f"unparseable JSON body: {exc}") from exc
        if not isinstance(payload, dict) or "results" not in payload:
            raise Declined(200, "response body has no 'results' key")

        usage = payload.setdefault("usage", {})
        if not isinstance(usage, dict) or not isinstance(usage.get("latency_ms"), (int, float)):
            # PRD §14.6: fall back to the engine's own wall clock, and record
            # which source was used -- the fallback includes network time the
            # server-reported figure would not.
            payload["usage"] = {**(usage if isinstance(usage, dict) else {}), "latency_ms": latency_ms}
            payload["_latency_source"] = "engine_wallclock"
        else:
            payload["_latency_source"] = "server_reported"
        return payload


class MockBackend:
    """A deterministic, explicitly non-trained backend, for exactly two jobs
    and no others (PRD §14.9 items 5/7's "must run to completion with a mock
    backend and no network" requirement, mirroring the sibling project's own
    `MockBackend`/`harness_selftest` discipline):

    1. **Harness wiring self-test.** With no `fixed_answers`, every question
       is answered with its own first *presented* option (post-shuffle) --
       generic across every task in the corpus, since it never needs to know
       which dataset a request came from (the wire contract doesn't say, by
       design -- PRD §6.1). Proves `bjb evaluate` runs item selection ->
       backend call -> classification -> aggregation -> evidence bundle end
       to end with zero ML stack and zero network.
    2. **A named, targeted demonstration.** `fixed_answers={"<question_key>":
       "<answer>"}` overrides the first-option default for that key, e.g.
       `{"is_toxic": "No"}` to build the exact all-"No" model PRD §12.2/§14.2
       uses to justify chance adjustment. This works because `is_toxic` is a
       `question_key` unique to `civil_comments/is_toxic` in the current
       corpus; a fixed answer is applied regardless of shuffle position, by
       *string*, never by presented index.

    `describe()` stamps `harness_selftest: True` so a result from this
    backend can never be mistaken for a real model's score, the same
    discipline the sibling project applies to its own `MockBackend`.
    """

    name = "mock"

    def __init__(self, *, fixed_answers: dict[str, str] | None = None, latency_ms: float = 1.0):
        self.fixed_answers = dict(fixed_answers or {})
        self.latency_ms = latency_ms

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "fixed_answers": self.fixed_answers, "harness_selftest": True}

    def answer(self, request: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        results = {}
        for key, q in request["questions"].items():
            options = q["options"]
            chosen = self.fixed_answers.get(key)
            if chosen not in options:
                chosen = options[0]
            results[key] = {"choice": chosen, "probabilities": {chosen: 1.0}, "confidence": 1.0}
        return {"model": self.name, "results": results, "usage": {"latency_ms": self.latency_ms}}


@dataclass(frozen=True, slots=True)
class ClassifyResult:
    outcome: Outcome
    p: float | None  # mass on the RETURNED option, never the max, never the gold's mass (PRD §14.4a)
    returned: str | None  # the raw (unstripped) `choice` the model sent, or None if absent/declined


def classify(result: dict[str, Any] | float | None, *, options: tuple[str, ...], expected: str,
             question_type: str = "choice") -> ClassifyResult:
    """PRD §14.3's outcome table for one question-key's `results[key]` object.

    `result` is `None` when the response was a 200 but `results` was missing
    this item's `question_key` entirely (PRD §14.3's fourth `declined`
    trigger) -- callers that already raised/caught `Declined` for a
    non-200/timeout/unparseable response never reach this function at all for
    that item.

    **Fixed 2026-09-24, found by the first real run against a live server**:
    this originally assumed every `result` was a `choice`-shaped dict
    (`{"choice": ..., "probabilities": {...}, "confidence": ...}`) and crashed
    with `AttributeError: 'float' object has no attribute 'get'` on the first
    real `noul` item -- ekVachan's `noul` returns a bare float (`P(true)`,
    better-jev-for-all PRD.md 13a.14), matching Jev's own real wire contract,
    not a dict. `score` returns `{"score": <continuous, probability-weighted
    position>, "probabilities": {...}, "confidence": ...}` -- no `"choice"`
    key at all. `question_type` now dispatches to the right shape; `choice`'s
    own path is byte-identical to before.

    Comparison is exact string equality after `str.strip()` on the *derived
    predicted label* (for `choice`, the raw returned string; for `noul`, the
    thresholded Yes/No; for `score`, the argmax-probability level name) --
    PRD §14.3. `options` and `expected` are assumed already-clean corpus
    strings and are never themselves stripped or folded.
    """
    if result is None:
        return ClassifyResult(Outcome.DECLINED, None, None)

    if question_type == "noul":
        if not isinstance(result, (int, float)) or isinstance(result, bool) or not (0.0 <= float(result) <= 1.0):
            return ClassifyResult(Outcome.OUT_OF_SCHEMA, None, None if result is None else str(result))
        p_yes = float(result)
        # ekvachan's noul is trained/served as an internal 2-way Yes/No choice
        # (better-jev-for-all PRD.md 13a.14) -- P(Yes) >= 0.5 predicts "Yes".
        predicted = "Yes" if p_yes >= 0.5 else "No"
        if predicted not in options:
            return ClassifyResult(Outcome.OUT_OF_SCHEMA, None, predicted)
        outcome = Outcome.CORRECT if predicted == expected else Outcome.INCORRECT
        p = p_yes if predicted == "Yes" else (1.0 - p_yes)
        return ClassifyResult(outcome, p, predicted)

    if question_type == "score":
        if not isinstance(result, dict):
            return ClassifyResult(Outcome.OUT_OF_SCHEMA, None, None if result is None else str(result))
        probs = result.get("probabilities")
        if not isinstance(probs, dict) or not probs:
            return ClassifyResult(Outcome.OUT_OF_SCHEMA, None, None)
        # The discrete predicted level is the argmax of `probabilities`, not a
        # round() of the continuous `score` -- this matches `confidence`,
        # which RoutingDecoderModel already defines as the argmax level's own
        # probability (see better-jev-for-all serve/inference.py predict_score).
        predicted = max(probs, key=lambda level: probs[level])
        if predicted not in options:
            return ClassifyResult(Outcome.OUT_OF_SCHEMA, None, predicted)
        outcome = Outcome.CORRECT if predicted == expected else Outcome.INCORRECT
        p_val = probs.get(predicted)
        p = float(p_val) if isinstance(p_val, (int, float)) else None
        return ClassifyResult(outcome, p, predicted)

    # question_type == "choice" -- unchanged from before the fix.
    if not isinstance(result, dict):
        return ClassifyResult(Outcome.OUT_OF_SCHEMA, None, None if result is None else str(result))
    choice = result.get("choice")
    if not isinstance(choice, str):
        return ClassifyResult(Outcome.OUT_OF_SCHEMA, None, choice if choice is None else str(choice))

    stripped = choice.strip()
    if stripped not in options:
        return ClassifyResult(Outcome.OUT_OF_SCHEMA, None, choice)

    outcome = Outcome.CORRECT if stripped == expected else Outcome.INCORRECT

    # PRD §14.6's resolution order for calibration mass: probabilities[choice],
    # else confidence, else uncalibrated (caller counts this into
    # `diagnostics.n_no_confidence`).
    p: float | None = None
    probs = result.get("probabilities")
    if isinstance(probs, dict) and isinstance(probs.get(stripped), (int, float)):
        p = float(probs[stripped])
    elif isinstance(result.get("confidence"), (int, float)):
        p = float(result["confidence"])

    return ClassifyResult(outcome, p, choice)

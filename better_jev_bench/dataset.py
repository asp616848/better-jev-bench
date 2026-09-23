"""The contributor-facing plugin interface (PRD §7.1).

The shape is the one the PRD specified -- a `BenchmarkDataset` base class with
`schema()` and `load_items()`, resolved from a `manifest.toml` `loader =
"module:Class"` pointer, HELM's `Scenario` pattern applied to datasets. Three
refinements were made while actually building the first eight loaders, and each
is called out here rather than silently diverging from §7.1:

1. **`load_items()` returns an iterable, not a `list`.** Civil Comments is 1.8M
   rows and CFPB is a 346MB zipped CSV; materialising either as a list of
   dataclasses before the sampler runs is gratuitous. Loaders `yield`.

2. **`schema()` returns `RequiredOptions` as a per-task mapping**, because a real
   source often carries several genuinely different decisions over the same
   state -- Civil Comments is natively a `score` (graded toxicity) *and* a
   `noul` (is this toxic at all), and pretending it is one task to fit a
   one-dataset-one-schema interface would have thrown away the corpus's only
   native `score` source. A dataset declares N tasks; CI checks all N.

3. **`RequiredOptions` may declare `option_count` instead of the literal option
   list** for a schema whose vocabulary is data-derived (CFPB's product lines
   come from the live download, not a constant). CI then checks the observed
   count, and `build.py` records the resolved vocabulary in the build receipt so
   it is still pinned -- just pinned by the receipt rather than by the source
   file.

Contributors subclass, point `manifest.toml` at the class, open a PR. No fork.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .types import Item, Provenance, Question


@dataclass(frozen=True, slots=True)
class RequiredOptions:
    """One task's declared answer space. Declared, not inferred, so CI can
    verify the manifest against reality rather than trusting it (PRD §7.1)."""

    task: str
    question_key: str
    primitive: str
    instructions: str
    options: tuple[str, ...] | None = None
    option_count: int | None = None
    ordinal: bool = False
    #: "uniform" -> 1/|options|; "majority" -> observed majority-class frequency,
    #: which is the only defensible floor on a severely imbalanced source
    #: (PRD §5.3 uses ULB fraud at 0.172% positive as the motivating case).
    chance_mode: str = "uniform"

    def __post_init__(self) -> None:
        if (self.options is None) == (self.option_count is None):
            raise ValueError(f"{self.task}: declare exactly one of options / option_count")
        if self.chance_mode not in ("uniform", "majority"):
            raise ValueError(f"{self.task}: chance_mode must be uniform|majority")

    @property
    def declared_width(self) -> int:
        return len(self.options) if self.options is not None else int(self.option_count)

    def question(self, options: Iterable[str] | None = None) -> Question:
        opts = tuple(options) if options is not None else self.options
        if opts is None:
            raise ValueError(f"{self.task}: options must be supplied at load time")
        return Question(
            key=self.question_key,
            type=self.primitive,
            instructions=self.instructions,
            options=opts,
            ordinal=self.ordinal,
        )


class BenchmarkDataset(ABC):
    """Subclass this to contribute a dataset.

    `name` must match the `[dataset].name` in the sibling `manifest.toml`, and
    must match the directory the manifest lives in (`datasets/<name>/`). CI
    checks all three agree -- a mismatch is the single most common way a plugin
    registry silently loads the wrong thing.
    """

    #: Machine name. Lowercase, `[a-z0-9_]`.
    name: str = ""
    #: Set by the loader framework before `load_items()` is called, from the
    #: manifest, so a loader never hard-codes its own canary or tier.
    canary: str = ""
    license_tier: str = ""
    #: Optional hard cap applied by `build.py`; a loader may also self-limit.
    max_items_per_task: int | None = None

    def __init__(self, canary: str = "", license_tier: str = "") -> None:
        if canary:
            self.canary = canary
        if license_tier:
            self.license_tier = license_tier

    # -- required ------------------------------------------------------

    @abstractmethod
    def schema(self) -> list[RequiredOptions]:
        """The option sets this dataset presents, one entry per task."""

    @abstractmethod
    def load_items(self) -> Iterator[Item]:
        """Yield `Item`s. MAY download from the original source at load time --
        raw data need NOT be redistributed into this repo (PRD §7.1). For a
        Tier C source, detect the user's credentialed copy and raise an
        actionable error if absent; never work around a gate."""

    # -- helpers for subclasses ----------------------------------------

    def spec(self, task: str) -> RequiredOptions:
        """Cached lookup. `schema()` may hit the network to resolve a taxonomy
        from its source (banking77 does), and `item()` calls this once per row,
        so caching here is load-bearing rather than an optimisation."""
        cache = getattr(self, "_spec_cache", None)
        if cache is None:
            cache = {s.task: s for s in self.schema()}
            self._spec_cache = cache
        try:
            return cache[task]
        except KeyError:
            raise KeyError(f"{self.name}: no task {task!r}") from None

    def item(
        self,
        task: str,
        state: str,
        label: str,
        *,
        options: Iterable[str] | None = None,
        source: str | None = None,
        provenance: Provenance | None = None,
        modality: str = "text",
        instructions: str | None = None,
    ) -> Item:
        """`instructions` may be overridden per item for a task whose question
        text is parameterised by the row -- CUAD's `noul` asks about a different
        named clause category each time, which is the question, not decoration."""
        spec = self.spec(task)
        question = spec.question(options)
        if instructions is not None:
            question = Question(
                key=question.key,
                type=question.type,
                instructions=instructions,
                options=question.options,
                ordinal=question.ordinal,
            )
        return Item(
            dataset=self.name,
            task=task,
            state=state,
            question=question,
            label=label,
            source=source or f"{self.name}/{task}",
            canary=self.canary,
            license_tier=self.license_tier,
            modality=modality,
            provenance=provenance or Provenance(),
        )


# -- registry ----------------------------------------------------------


def load_plugin(spec: str, *, canary: str = "", license_tier: str = "") -> BenchmarkDataset:
    """Resolve a `module:Class` loader pointer from a manifest into an instance."""
    if ":" not in spec:
        raise ValueError(f"loader spec {spec!r} must be 'module.path:ClassName'")
    mod_name, cls_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:  # pragma: no cover - surfaced as a CI failure
        raise ImportError(f"loader module {mod_name!r} is not importable: {exc}") from exc
    try:
        cls = getattr(mod, cls_name)
    except AttributeError as exc:
        raise ImportError(f"{mod_name} has no attribute {cls_name!r}") from exc
    if not (isinstance(cls, type) and issubclass(cls, BenchmarkDataset)):
        raise TypeError(f"{spec} is not a BenchmarkDataset subclass")
    return cls(canary=canary, license_tier=license_tier)

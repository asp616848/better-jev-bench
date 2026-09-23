"""better-jev-bench -- a wide, multi-domain, license-tiered corpus and benchmark
for typed decision models.

Contributor entry points (PRD §7.1):

    from better_jev_bench import BenchmarkDataset, Item, Question, RequiredOptions

Everything else is reachable through the `bjb` CLI:

    bjb catalogue                 # the corpus as data (PRD §6.3 GET /v1/catalogue)
    bjb validate                  # the §7.4 CI gates, offline
    bjb validate --loaders        # + gate 2, runs every loader against its source
    bjb build                     # source -> normalised -> split -> hashed receipts
    bjb build --verify            # rebuild and diff against the committed receipts
    bjb export --out <dir>        # ekVachan-shaped training records
    bjb stats                     # what actually landed
"""

from .dataset import BenchmarkDataset, RequiredOptions, load_plugin
from .manifest import Manifest, ManifestError, discover, load_manifest
from .types import (
    MIN_ITEMS_FOR_CALIBRATION,
    MIN_ITEMS_TO_ACCEPT,
    PRIMITIVES,
    WIDTH_STRATA,
    ImageRef,
    Item,
    Provenance,
    Question,
    width_stratum,
)

__version__ = "0.1.0"
#: Frozen scoring-spec version (PRD §5.5 rule 8). Changing any frozen constant
#: -- strata boundaries, beta, aggregation weights, the split salt -- is a bump
#: here plus a migration note, never a silent recalculation.
SPEC_VERSION = "bjb-score-1.0"

__all__ = [
    "BenchmarkDataset",
    "ImageRef",
    "Item",
    "Manifest",
    "ManifestError",
    "Provenance",
    "Question",
    "RequiredOptions",
    "PRIMITIVES",
    "WIDTH_STRATA",
    "MIN_ITEMS_TO_ACCEPT",
    "MIN_ITEMS_FOR_CALIBRATION",
    "SPEC_VERSION",
    "__version__",
    "discover",
    "load_manifest",
    "load_plugin",
    "width_stratum",
]

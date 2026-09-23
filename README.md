# better-jev-bench

**A wide, multi-domain, license-tiered corpus and benchmark for typed decision models.**

A *typed decision* is the thing a production system actually needs from a model: pick one of
these N options, place this on this scale, answer this yes/no proposition — with a calibrated
probability attached. Three primitives, `choice` / `score` / `noul`, borrowed unchanged from
[ekVachan](https://github.com/asp616848/better-jev-for-all)'s serving contract.

This repo exists because of one measured result. ekVachan trained a real model, built two real
benchmark harnesses (231 and 944 items), and ran the checkpoint against both. **Zero items in
either matched the model's schema** — not low accuracy, zero answerable items out of 1,175. The
model had been trained on 1.2M NLI examples: one option set, repeated 1.2 million times. Scale
on a single schema teaches the schema, not the skill.

So the bottleneck was never model quality. It was that no public corpus spans the actual
distribution of option-set shapes a deployed decision layer meets. That corpus is what this is.

Full reasoning, with sources: [`better-jev-bench_PRD.md`](better-jev-bench_PRD.md).
Current state, honestly: [`STATUS.md`](STATUS.md).

---

## What's here right now

**8 datasets · 11 tasks · 422,878 items** — 278,513 in the public (training) slice, 21,174 in
the frozen held-out (evaluation) slice. All four width strata (2 → 151 options) and all three
primitives populated. Every license verified against a primary source on 2026-09-23, none
assumed from a name; all eight are Tier A.

| Dataset | Domain | Task | Primitive | Options | Items | License |
|---|---|---|---|---|---|---|
| CFPB Consumer Complaints | finance | `product` | `choice` | 10 | 54,923 | CC0-1.0 |
| BANKING77 | nlp | `intent` | `choice` | 77 | 13,071 | CC-BY-4.0 |
| CLINC150 (`plus`) | nlp | `intent` | `choice` | 151 | 23,849 | CC-BY-3.0 |
| MASSIVE (en-US) | nlp | `intent` | `choice` | 60 | 16,439 | CC-BY-4.0 |
| MASSIVE (en-US) | nlp | `scenario` | `choice` | 18 | 16,438 | CC-BY-4.0 |
| GoEmotions | nlp | `emotion` | `choice` | 28 | 45,270 | Apache-2.0 |
| LEDGAR | operational | `provision_type` | `choice` | 100 | 78,497 | CC-BY-4.0 |
| CUAD | operational | `clause_type` | `choice` | 41 | 8,055 | CC-BY-4.0 |
| CUAD | operational | `clause_present` | `noul` | 2 | 16,106 | CC-BY-4.0 |
| Civil Comments | nlp | `toxicity_level` | `score` (ordinal) | 5 | 75,121 | CC0-1.0 |
| Civil Comments | nlp | `is_toxic` | `noul` | 2 | 75,109 | CC0-1.0 |

**Chance adjustment is not a refinement here, it is the difference between a meaningful number
and a meaningless one.** Four of the eleven tasks are severely imbalanced and declare a
majority-class chance floor: answering "No" to every `civil_comments/is_toxic` item scores
92.07% raw accuracy and **0** chance-adjusted.

Run `bjb stats` for the live counts — they come from the committed build receipts, not from
this table.

## Quick start

```bash
pip install -e .

bjb catalogue          # the corpus as data, filterable by license tier
bjb validate           # the PRD §7.4 CI gates, offline, seconds
bjb stats              # what actually landed
```

### Benchmark a model

The held-out slice is **committed to this repo**, so this works immediately after a clone:

```python
import gzip, json

for line in gzip.open("bench/heldout/banking77.jsonl.gz", "rt"):
    item = json.loads(line)
    # item["request"] is a verbatim /v1/systemone body — POST it as-is
    response = requests.post(endpoint, json=item["request"]).json()
    correct = response["questions"]["intent"]["choice"] == item["expected"]["intent"]
```

That is the highest-leverage design decision in the project: **an item *is* a request body**.
Anything speaking `/v1/systemone` — ekVachan, Von, Rizzo Flow, `open-alternative-jev`, hosted
Jev — is evaluable with zero adapter code. Not "exportable to". Evaluable.

Want to eyeball the data first? `bench/preview/*.json` is 24 items per dataset, pretty-printed,
committed.

### Pull a training slice

```bash
bjb build                                        # fetch + normalise + split + hash
bjb export --out exports/mix --tiers A --max-options 26

# and verify the export against ekVachan's own validator, copied verbatim:
python scripts/check_ekvachan_compat.py exports/mix/public.jsonl
```

`bjb export` emits the exact eight-key record shape ekVachan's real training scripts already
consume:

```json
{"state": "...", "question_key": "intent", "question_type": "choice",
 "instructions": "...", "options": ["..."], "label": "...", "label_idx": 3,
 "source": "bjb:banking77/intent"}
```

That schema was read off `training/data.py` and `training/build_primitives_slice.py` in the
sibling repo, not invented here — so a bench slice drops into a training run with no
translation layer. `--max-options 26` narrows wider schemas the way ekVachan's own
`_sample_wide_subset()` does, keeping the gold label and sampling distractors; `--max-options 0`
turns it off for a consumer with no such budget. Ordinal `score` scales are never narrowed and
never shuffled, because letter position *is* scale position for them.

Every export writes an `ATTRIBUTION.md` listing the license obligations that propagate to
whatever you build from it. Ship it.

## Add a dataset

Subclass, write a manifest, open a PR. No fork.

```python
from better_jev_bench import BenchmarkDataset, RequiredOptions

class MyDataset(BenchmarkDataset):
    name = "my_dataset"

    def schema(self) -> list[RequiredOptions]:
        return [RequiredOptions(task="triage", question_key="severity",
                                primitive="choice", instructions="...",
                                options=("low", "medium", "high"))]

    def load_items(self):
        for row in fetch_from_original_source():
            yield self.item("triage", state=row.text, label=row.severity)
```

`load_items()` may download from the original source at load time — **raw data need not be
redistributed into this repo**. That is what makes research-only and access-gated sources usable
at all: the loader is shared, the data never is.

Then `datasets/my_dataset/manifest.toml`, and CI runs the eight gates from PRD §7.4:
manifest schema, loader smoke test, schema/manifest agreement, **license tier (`tier = "D"`
fails the build)**, sensitivity gate, canary uniqueness, leakage check, held-out commitment.

See [`better-jev-bench_PRD.md` §7](better-jev-bench_PRD.md) for the full contribution
framework, including the two-reviewer minimum and the acceptance criteria.

## Licensing is enforced, not advisory

Roughly half the 99-entry catalogue in the PRD is **not** cleanly redistributable, and a
benchmark that quietly ignores that ships a legal problem to everyone who adopts it. So every
dataset declares a tier, `verified_how` is a required non-empty field naming the page the claim
was read from, and CI refuses to merge `tier = "D"`.

All eight datasets in the current batch are Tier A. `bjb export --tiers A` gives you a legally
coherent subset with a stated coverage number.

## Held-out slice: frozen and tamper-evident, not secret

Reserved before any training run existed to leak into it (PRD G3). Each dataset's held-out
labels are hashed and the hash is committed in its `manifest.toml`, so changing a held-out
answer changes the hash, shows up in the diff, and automatically invalidates every prior result
on that dataset.

Being straight about the limit: the held-out slices are committed to a public repo **with their
labels**, so nothing here physically prevents training on them — only the frozen canary markers
and this tamper-evidence. A genuinely sealed slice needs hosted infrastructure that does not
exist yet. Freezing first and sealing later is the right order; a split created *after* a
training run is a split someone has to be trusted about.

## Responsible use

The legal, medical, security and moderation entries carry sensitivity flags and a required
responsible-use note, and CI fails a flagged dataset with an empty note. Clause classification
is not attorney review; a toxicity score is not any platform's policy; none of this corpus is
bias-audited. PRD §8.3 and §8.4 state the constraints per domain, and they are constraints on
*this corpus*, because this corpus is what would put them into someone else's training run.

## Relationship to ekVachan

[ekVachan](https://github.com/asp616848/better-jev-for-all) is expected to be this corpus's
first consumer and is not its purpose. Separate repos, separate PRDs, separate histories — a
benchmark maintained inside the repo of the model it scores is not a benchmark other people will
trust. ekVachan gets no special treatment: same public path, same rules, and its current score
here would be approximately zero on Generality.

---

Apache-2.0. Individual datasets keep their own licenses — see each `manifest.toml` and the
`ATTRIBUTION.md` your export generates.

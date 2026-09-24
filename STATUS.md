# better-jev-bench — status

Living task list for `better-jev-bench` (the standalone dataset/benchmark corpus repo). Update this file in the same commit as the work it describes — mark items done only after verifying them for real, not after a prior claim says so (see `skills/dev-guidelines/SKILL.md`). Full reasoning lives in `better-jev-bench_PRD.md`; this file is the checklist.

Sibling project: **ekVachan / better-jev-for-all** (separate repo, separate PRD, own `STATUS.md`) — the model this corpus will eventually train/evaluate. Don't mix work across the two repos in one commit.

Server clone: `abhijeet-labgpu:~/ekvachan/bench-repo` — this did not exist until 2026-09-23; before that, this repo had no server-side clone at all, which is worth remembering as a category of risk (see dev-guidelines rule 2/11).

Last verified: 2026-09-24 (**scoring-engine design pass — PRD §14**. No code written; §5's paper
spec is now an implementation contract, and a real chance-floor defect was found in the shipped
corpus — see "Scoring engine" below, which is the next milestone and is now fully specified).
Prior: 2026-09-24 (multimodal build — PRD §12.6. `mod_multimodal` is populated for real: 3 new datasets, 51,561 items, 4,059 held-out, calibration-bearing). Prior: 2026-09-24 planning pass (PRD §13, no items built); 2026-09-23 second pass — first real corpus build.

## Done

- [x] Repo public, Apache-2.0 `LICENSE` committed
- [x] Founding PRD (`better-jev-bench_PRD.md`): scoring spec, API design, contribution framework, license tiering, roadmap. **v0.2 adds §11 (build pipeline design) and §12 (what actually landed).**
- [x] 6 research docs (`research/00`–`05`): methodology + 5 domain surveys — every license claim traced to a live-checked primary source
- [x] 99 dataset entries catalogued across 4 license tiers (~51 Tier A permissive, subset share-alike, ~12 Tier B research-only, 3 Tier C access-gated, ~31 Tier D unverified/blocked)

### New 2026-09-23 (second pass) — the corpus exists now

- [x] **Loader/plugin framework built** (`better_jev_bench/`) — `BenchmarkDataset` + `RequiredOptions` (PRD §7.1), `manifest.toml` parsing and static validation (§7.2), `module:Class` loader resolution, `Item` types whose JSON *is* a `/v1/systemone` request body (§6.1)
- [x] **All eight CI gates from §7.4 implemented** (`better_jev_bench/validate.py`) + a GitHub Actions workflow at `ci/github-workflow-ci.yml`. Offline gates are designed to run on every PR; the network-bound loader smoke test runs on a schedule. **The workflow is not active yet**: the server's `gh` token carries `repo` but not `workflow` scope, so GitHub rejects a push that creates `.github/workflows/`. The file is committed intact and `ci/README.md` has the three commands to activate it from an account with that scope. The gates themselves are code and run today via `bjb validate`. **Current tree: 41 checks, 0 failures, 0 warnings.**
- [x] **Build pipeline** (`better_jev_bench/build.py`) — source → normalise → check against manifest → content-determined split → gzipped JSONL → SHA-256 receipts. `bjb build --verify` rebuilds and diffs against the committed receipts instead of overwriting them.
- [x] **All 8 high-value Tier-A datasets pulled for real**: CFPB, BANKING77, CLINC150, MASSIVE, GoEmotions, LEDGAR, CUAD, Civil Comments. Every license re-verified against a primary source at build time via the HF Hub API on the server, not assumed. **All eight are Tier A; no swaps were needed.**
- [x] **422,878 normalised items across 11 tasks** — 278,513 public / 21,174 held-out. All four width strata (2 → 151 options) and all three primitives (`choice` / `score` / `noul`) populated and calibration-bearing.
- [x] **Held-out slice reserved and hash-committed** (G3, §7.5) — frozen *before* any training pull existed, labels hashed into each `manifest.toml`, committed at `bench/heldout/*.jsonl.gz` (4.03 MB total) so `git clone && evaluate` works with no download.
- [x] **Evidence committed** (dev-guidelines rule 10): `bench/receipts/<name>.build.json` per dataset + `bench/receipts/CORPUS.json`, each carrying both slices' SHA-256, the label commitment, per-task label balance, duplicate counts and the source pin.
- [x] **Training export to ekVachan's real record shape** (`bjb export`, PRD §11.4) — emits `{state, question_key, question_type, instructions, options, label, label_idx, source}`, the exact eight keys `training/data.py` and `training/build_primitives_slice.py` consume in the sibling repo. Width narrowing to the decoder's 26-letter budget mirrors that project's own `_sample_wide_subset()`; ordinal `score` scales are never narrowed or shuffled.
- [x] **Compatibility verified by running the sibling's own validator**: `scripts/check_ekvachan_compat.py` is `_validate()` copied verbatim out of `build_primitives_slice.py` and run against a real export — 5,500 records, 0 failures (choice 4,000 / noul 1,000 / score 500), and the `score` scale order confirmed fixed.
- [x] README rewritten as a real front door: quick start, benchmark-a-model snippet, add-a-dataset snippet, license posture, honest held-out caveat.

### New 2026-09-24 — `mod_multimodal` populated for real (PRD §12.6)

- [x] **`state_hash` correctness fix, shipped first** (PRD §13.4, `better_jev_bench/types.py`). It hashed `state` text alone; a vision item's text state is short boilerplate repeated across thousands of rows, so the split (which keys on it) would have swept a whole dataset onto one side. Now mixes in every image's own sha256 for an image-bearing item — verified with a real script, `scripts/check_state_hash.py` (200 identical-boilerplate items hash distinctly and split across both sides; a text item's hash stays byte-identical to `sha256(state)`; `bjb build --verify` against the original 8 datasets showed **zero drift**).
- [x] **`ImageRef` + `Item.images` + `Item.split_key`/`split_key_override`** added to `types.py`, additively — a text record's `to_ekvachan_record()` output is unchanged (verified: `scripts/check_ekvachan_compat.py` still passes 1,000/1,000 on a text-only export). A new content-addressed image cache (`better_jev_bench/imagecache.py`) stores images at `data/images/<dataset>/<sha256[:2]>/<sha256>.<ext>` (gitignored), sniffing the real container format from the byte stream rather than trusting a filename extension — real bug found this way: 111 of ScreenSpot-v2's 757 image files are JPEG bytes behind a `.png` name.
- [x] **Three new CI gates live in `validate.py`**: every shipped image's sha256 is in its dataset's committed `sha256_manifest` and vice versa (no orphaned entry either direction); no local cache file goes unreferenced; a hash claimed by two datasets must carry the same license. Corpus-wide: **59 checks, 0 failures, 0 warnings** (was 41 before this pass, on 8 datasets).
- [x] **Three loaders built and pulled for real**: **Atari-HEAD** (`breakout`, CC-BY-4.0, 10,355 items, split by *trial* not frame via `Item.split_key_override` — verified for real: all 17 trials land wholly on one side, zero overlap), **OS-Atlas-data** (`desktop_domain/linux` subset only, Apache-2.0, 40,308 items — the aggregate's other subsets, RICO/AMEX/SeeClick/FineWeb, were deliberately not pulled), **ScreenSpot-v2** (Apache-2.0, `eval_only = true`, 898 items, the vision analogue of CLINC150's zero-shot-schema role — `bjb export`'s public slice refuses it in code, not just in docs).
- [x] **GUIAct independently re-verified, then deliberately left unbuilt.** Fetched the HF card (`apache-2.0`) and the GUICourse GitHub README's Licensing Information section (`CC BY 4.0`) directly on 2026-09-24 — the discrepancy is real. OS-Atlas alone met the size target for this line item, so GUIAct stays unresolved rather than redistributed under an unreconciled license.
- [x] **`bjb export` emits image-bearing records**: an image item's training record gains a ninth key, `images` (`{sha256, path, media_type, width, height}`, path resolved + hash-verified against the local cache), additive to the eight-key text shape. Verified with a new script, `scripts/check_vision_export.py`, against a real combined export: 43,338 image records, every reference resolved and hash-matched.
- [x] **`prim_score` × `mod_multimodal` deliberately left at zero**, exactly as PRD §13.6 predicted — no vision `score` source in this pull is a genuine, non-invented ordinal scale. Reported as excluded, not filled.

## What the corpus holds (real numbers, from `bench/receipts/CORPUS.json`, built 2026-09-24)

| Dataset | License | Tasks | Widths | Items | Held-out |
|---|---|---|---|---|---|
| BANKING77 | CC-BY-4.0 | `intent` | 77 | 13,071 | 2,000 |
| CFPB Consumer Complaints | CC0-1.0 | `product` | 10 | 54,923 | 2,000 |
| Civil Comments | CC0-1.0 | `toxicity_level`, `is_toxic` | 5, 2 | 150,230 | 4,000 |
| CLINC150 (`plus`) | CC-BY-3.0 | `intent` | 151 | 23,849 | 2,000 |
| CUAD | CC-BY-4.0 | `clause_type`, `clause_present` | 41, 2 | 24,161 | 3,174 |
| GoEmotions | Apache-2.0 | `emotion` | 28 | 45,270 | 2,000 |
| LEDGAR | CC-BY-4.0 | `provision_type` | 100 | 78,497 | 2,000 |
| MASSIVE (en-US) | CC-BY-4.0 | `intent`, `scenario` | 60, 18 | 32,877 | 4,000 |
| Atari-HEAD (`breakout`) | CC-BY-4.0 | `action` | 18 | 10,355 | 1,201 |
| OS-Atlas-data (desktop/linux) | Apache-2.0 | `target_element` | 2 | 40,308 | 2,000 |
| ScreenSpot-v2 (`eval_only`) | Apache-2.0 | `target_element` | 2 | 898 | 858 |
| **Total** | 3 tiers, all A | **14 tasks** | 2 → 151 | **474,439** | **25,233** |

Run `bjb stats` for the live version of this table — it reads the receipts, not this file.

### New 2026-09-24 (second pass) — scoring engine designed, not built (PRD §14)

A design/planning pass only: **zero engine code was written, by intent.** What landed is PRD §14,
which turns §5's paper design into a contract a Sonnet implementer can build from without making a
design decision. Read §14 before writing any of it; the checklist below is §14.9 in checkbox form.

- [x] **Three-level aggregation resolved** (§14.1, *amends* §5.3). The scoring unit is a **task**,
      not a dataset — 11 datasets carry 14 tasks, and `civil_comments`/`cuad`/`massive` each mix
      primitives or widths, so a dataset has no single primitive, width or chance floor.
      `task → dataset → domain → axis`. Stated consequence: **CFPB alone carries 25% of
      Intelligence**, being the only `finance` dataset; an Intelligence number published before the
      rest of the Tier A finance slate is built is provisional and the result field says so.
- [x] **Four per-item outcomes frozen** (§14.3): `correct` / `incorrect` / `out_of_schema` /
      `declined`. Exact string equality after `strip()` and nothing else; no repair, no retry
      (§5.5 rule 3); Coverage's denominator is every item in the slice, so declining can never
      raise a score (§5.5 rule 9).
- [x] **Breadth restructured into three families** (§14.5, *amends* §5.3). A stratum's value is the
      macro mean of per-task `I_t` (a stratum has no chance floor of its own — `prim_noul` spans a
      0.9207 floor and a 0.5 floor), and the outer geometric mean runs over `{width, primitive,
      modality}`, not over nine flat buckets that triple-count every item and weight width 4/9 by
      accident.
- [x] **Every remaining axis mechanic pinned** (§14.6): pooled-ECE definition and the exact
      15-bin equal-mass algorithm (ported from the sibling's `eval/metrics.py`), the HCS accuracy
      term, frozen log-anchored Speed/Cost reference scales, the `cost_model: null` →
      `score_no_cost` path, ordinal-safe option shuffling, and the abstention sub-metric.
- [x] **`POST /v1/evaluate` specified to exact field names and types** (§14.7) — full request
      table, full response body, error codes, and the rule that `bjb evaluate` and the server are
      the same code path.
- [x] **Bring-your-own-model surface defined** (§14.8): speak `/v1/systemone` and there is no
      integration code at all; a `ModelBackend` protocol for local checkpoints; explicitly no
      schema-repair hook.

**Real defect found while doing this, and it is the reason item 0 exists (PRD §14.2).** All four
`chance_mode = "majority"` tasks **ship items whose `chance` field carries the uniform
`1/|options|` value, not the declared majority floor** — found by reading `bench/heldout/*.jsonl.gz`
directly rather than trusting the receipts:

| Task | manifest / receipt | **shipped item JSON** |
|---|---|---|
| `civil_comments/is_toxic` | 0.920729 | **0.5** |
| `civil_comments/toxicity_level` | 0.792774 | **0.2** |
| `cfpb_complaints/product` | 0.541100 | **0.1** |
| `go_emotions/emotion` | 0.353100 | **0.035714** |

Root cause, traced in code: `types.py`'s `Item.chance` docstring claims `build.py` resolves it from
the manifest; `build.py` never does. It writes `chance_observed` into the receipt and back into
`manifest.toml`, and CI gate 3 then checks those two derived copies against *each other* — never
against the field consumers actually read. §6.1 puts `chance` inside the item, so the obvious
implementation reads the wrong number: the all-"No" model on `civil_comments/is_toxic` would score
**0.841 chance-adjusted instead of 0.0**, i.e. the exact demonstration §12.2 uses to justify the
axis would silently come out flattering.

## Scoring engine — the next milestone, fully specified (PRD §14.9)

Ordered; each step verifiable before the next. **Nothing here is started.**

- [ ] **0. Fix the chance defect first.** Stamp `Item.chance` from the manifest in `build.py`;
      rebuild all 11 datasets; regenerate receipts. `bjb build --verify` must show every
      `heldout_label_commitment` **unchanged** (it hashes `(item_id, label)`; `chance` is in
      neither) while slice SHA-256s do change. Add CI gate 12: shipped item `chance` == manifest
      `chance`. Do not start step 1 until `bjb validate` is green.
- [ ] **1. `better_jev_bench/score.py`** — pure functions, no I/O. Unit-test against hand-computed
      values, including `I = 0` for the all-"No" model at chance 0.920729.
- [ ] **2. `better_jev_bench/backend.py`** — the `ModelBackend` protocol, `HTTPBackend`, `Declined`,
      and the §14.3 outcome classifier, tested exhaustively on all four outcomes.
- [ ] **3. `better_jev_bench/evaluate.py`** — the run loop. No scoring arithmetic here; it calls
      `score.py`. Deterministic given `(slice, filters, shuffle_seed)`.
- [ ] **4. Evidence bundle** (rule 10, §8.2): `results/<run_id>/manifest.json` + `raw.jsonl.gz`.
- [ ] **5. `bjb evaluate` CLI** — must run to completion against a mock backend with no network.
- [ ] **6. `POST /v1/evaluate`** + §6.3's `GET /v1/catalogue`, `GET /v1/items` (public slice only,
      asserted in code and tested), `GET /v1/runs/{run_id}`. `POST /v1/submit` stays unbuilt.
- [ ] **7. First real run** against the sibling's checkpoint. Expect `prim_noul`/`prim_score` to
      return `model_declined` against `serve/server.py` as it stands — it returns **501 for every
      non-`choice` primitive**, so that first run measures the *wire contract*, not the weights
      (the decoder behind it reports 95.48% `noul` on its own eval). The result's `notes` must say
      so. **The Generality = 0 line stays a prediction until this run's manifest is on disk.**
- [ ] **8. Optional after 7**: `abstention_options = ["oos"]` on `clinc150/intent`;
      `bjb score --from-raw <run_id>` so a spec bump can rescore an old run's raw rows.

Deliberately out of scope for this milestone: leaderboard, submission rate limiting (§5.5 rule 4),
gated held-out slice, Elo/win-rate secondary view. None blocks a first real score.

## Not started

- [ ] **No scoring implementation.** §5's five axes, the HCS formula, the pooled-ECE rule and the floor penalty are still spec. `POST /v1/evaluate` (§6.2) does not exist. This is the single thing between "a corpus with a frozen eval slice" and "a benchmark", and it is next. **It is now fully designed — follow PRD §14.9's ordered checklist; do not re-derive the design.** See the "Scoring engine" section below for the checklist in checkbox form.
- [ ] **Never run any ekVachan checkpoint against this corpus.** Now possible for the first time — the held-out slices are committed and every item is a request body — but not done. The Generality = 0 line below remains a *prediction*.
- [ ] ~40 remaining Tier A entries, the ~12 Tier B research-tier entries, the 3 Tier C pointer-only loaders
- [ ] **`request.images` not yet agreed with the sibling's `/v1/systemone` server contract** (PRD §13.4 point 2). §6.1's claim that an item *is* a request body needs the sibling's serving side to accept the same `images` extension; that is the sibling repo's own work, tracked there.
- [ ] **`prim_score` × `mod_multimodal` still unfilled** (PRD §13.6/§12.6) — deliberately. The only candidates (`deepghs/nsfw_detect`, MedMNIST ordinal subsets) still carry the same taxonomy/medical caveats §13.6 flagged; not revisited this pass.
- [ ] **OS-Atlas's `mobile_domain` and `web_domain` not pulled** — those subsets repackage RICO/AMEX/UIBert/SeeClick/FineWeb, each with its own license to verify separately; only the authors'-own `desktop_domain/linux` subset was pulled in this pass.
- [ ] No leaderboard, no submission path, no rate limiting (§5.5 rules 4 and 6 are written, unimplemented)
- [ ] The eight text datasets remain English-only

## Next up, with GPU time estimates

Anchor: sibling project's real Qwen3.5-4B LoRA runs took 65–75 min for 24k examples / 1 epoch, same server (abhijeet-labgpu, single GPU, ~46GB VRAM). Most of this project's remaining work is still data/software engineering, not GPU-bound.

| # | Task | GPU time | Notes |
|---|---|---|---|
| 1 | ~~Loader/plugin framework + manifest parsing + CI gates~~ | — | **done 2026-09-23** |
| 2 | ~~Manifests + pull the 8 high-value Tier-A datasets~~ | — | **done 2026-09-23** — 422,878 items |
| 3 | ~~Reserve + hash-commit held-out slice~~ | — | **done 2026-09-23** — 21,174 items, committed |
| 4 | Implement the scoring spec (§5) + `/v1/evaluate` (§6.2), validate against the sibling's decoder checkpoint | minutes–hours | latency per item is still unmeasured for the decoder; the ~30ms in sibling PRD 13a.3 is an *encoder* number and does not carry over |
| 5 | Build the remaining ~40 Tier A entries | none | bandwidth-bound; each needs its license **and its shape** re-verified at build time (see the CFPB finding below) |
| 6 | ~~Multimodal: on-distribution slice (Atari-HEAD, OS-Atlas, ScreenSpot-v2)~~ | ~15 min (downloads + build, no GPU) | **done 2026-09-24** — PRD §12.6. 51,561 items, 4,059 held-out, `mod_multimodal` calibration-bearing. §2.3's original breadth slate (EuroSAT/AI2D/etc.) and OS-Atlas's mobile/web subsets remain unpulled. Directly unblocks the sibling's vision tier (its PRD §5.2b) |
| 7 | *(belongs in the sibling repo)* Retrain ekVachan on the assembled corpus, including the new vision slice | ~7-8 hrs, per sibling PRD §5.2b's own estimate | depends on final pull size; its results belong in the sibling PRD's run log, not here |

**Note on task 4, still current.** Generality is expected to score **0** for the sibling's checkpoint, but the reason matters and validating *which* reason is the value of the task. §5.3's internal geometric mean collapses when any primitive stratum collapses. As of 2026-09-23 the sibling does now have a decoder trained on all three primitives (92.02% `choice`, 88.80% `noul`, 75.40% `score` on its own eval), so the old "no model for `score`/`noul` at all" framing is out of date. What is untested is whether those numbers survive contact with schemas and domains it has never seen — 151-way intent routing, 100-way contract provisions, a 5-level ordinal scale built from annotator-agreement fractions. **That is now an empirical question this repo can answer, and it could not answer it yesterday.** Predicting the answer in this file would be exactly the unverified-claim failure dev-guidelines rule 3 exists to prevent.

## Findings worth carrying forward

**The CFPB bulk export no longer contains complaint narratives.** Downloaded in full on 2026-09-23 (346,404,253 bytes, ETag `f0d2ec2367bd2ae60f044b782ad85f43-42`) — its header has 15 columns and `Consumer complaint narrative` is not among them, and the public search API returns no `complaint_what_happened` field either. `research/01` describes this source as "~8GB, real free-text complaint narrative"; the license claim held, the *shape* claim had decayed. The corpus uses a CC0 Hugging Face mirror's `has-text` config instead, with the deviation recorded in the manifest's `verified_how`. **Every remaining catalogue entry is owed the same check at build time, not at catalogue time** — see PRD §8.2's fifth bullet, which is the one v0.1 bullet that survived contact with reality.

**Three catalogued datasets are script-based and unloadable under `datasets >= 3`** (BANKING77, MASSIVE, the official CFPB repo). MASSIVE has a Hub-generated parquet branch; BANKING77 does not, so its loader reads the same GitHub CSVs PolyAI's own script reads. Expect this for other older catalogue entries.

**A dataset's own image files can lie about their format.** 111 of ScreenSpot-v2's 757 images in `screenspotv2_image.zip` are JPEG bytes behind a `.png` filename — found when the first build attempt failed a PNG-header parse, not by inspection. `imagecache.py` now sniffs the real container format from the file's own magic bytes and never trusts an extension; any future image loader inherits this for free. The same pull found 6 of OS-Atlas's 1,186 referenced filenames simply missing from its images zip (skipped, not fatal). Both are exactly the class of thing a catalogue-time license check cannot catch — they only show up at build time, against the real bytes.

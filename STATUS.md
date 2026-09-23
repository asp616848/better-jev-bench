# better-jev-bench — status

Living task list for `better-jev-bench` (the standalone dataset/benchmark corpus repo). Update this file in the same commit as the work it describes — mark items done only after verifying them for real, not after a prior claim says so (see `skills/dev-guidelines/SKILL.md`). Full reasoning lives in `better-jev-bench_PRD.md`; this file is the checklist.

Sibling project: **ekVachan / better-jev-for-all** (separate repo, separate PRD, own `STATUS.md`) — the model this corpus will eventually train/evaluate. Don't mix work across the two repos in one commit.

Server clone: `abhijeet-labgpu:~/ekvachan/bench-repo` — this did not exist until 2026-09-23; before that, this repo had no server-side clone at all, which is worth remembering as a category of risk (see dev-guidelines rule 2/11).

Last verified: 2026-09-24 (multimodal planning pass — PRD §13; no new items built, `mod_multimodal` is still empty and §12.3's table is still accurate). Prior: 2026-09-23 second pass — first real corpus build.

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

## What the corpus holds (real numbers, from `bench/receipts/CORPUS.json`)

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

Run `bjb stats` for the live version of this table — it reads the receipts, not this file.

## Not started

- [ ] **No scoring implementation.** §5's five axes, the HCS formula, the pooled-ECE rule and the floor penalty are still spec. `POST /v1/evaluate` (§6.2) does not exist. This is the single thing between "a corpus with a frozen eval slice" and "a benchmark", and it is next.
- [ ] **Never run any ekVachan checkpoint against this corpus.** Now possible for the first time — the held-out slices are committed and every item is a request body — but not done. The Generality = 0 line below remains a *prediction*.
- [ ] ~43 remaining Tier A entries, the ~12 Tier B research-tier entries, the 3 Tier C pointer-only loaders
- [ ] **Multimodal: still nothing built.** `mod_multimodal` is an empty stratum, correctly excluded from Breadth and reported as excluded. **But it now has a full build plan — PRD §13, added 2026-09-24** — and a decision: the static licensed multimodal data belongs *here*, not in the sibling model repo (§13.1, on §9's own "no special treatment" argument). Ordered checklist below; each item is specific enough to pick up without re-deriving §13.

#### `mod_multimodal` build plan (PRD §13) — ordered

- [x] **License pass extended, 2026-09-24** (PRD §13.3). `research/05`'s 16 entries are real but are *generic image classification* — none of it is a GUI screenshot, game frame or agent action choice. New verified entries: **Atari-HEAD (CC BY 4.0, Zenodo record — ~8M frame→human-keystroke demonstrations, the single best on-distribution fit)**, **ScreenSpot-v2 (Apache-2.0, eval-only, ~1.2k items)**, **OS-Atlas-data (apache-2.0 declared, aggregation — per-subset verification owed)**, **GUIAct (apache-2.0 on the card vs CC BY 4.0 in the paper — discrepancy recorded)**, **Mind2Web (CC BY 4.0, but HTML-only so `mod_text`)**. Reported as *not* usable at Tier A: **Multimodal-Mind2Web (`openrail`, use-restricted → Tier B)** and **AitW (no LICENSE file, no README terms → Tier D)**, plus **ShowUI-desktop (no license + GPT-4o-derived → Tier D)**.
- [ ] **Fix `state_hash` for image items — do this first, it is a latent correctness bug** (PRD §13.4). It hashes `state` text alone; a vision item's text state is short boilerplate repeated across thousands of rows, so the public/held-out split (which keys on it, §7.4 gate 7) would sweep a whole dataset onto one side. Mix the image content hashes in for items carrying images.
- [ ] **Extend `better_jev_bench/types.py` with `ImageRef` + `Item.images`** (PRD §13.4) — `{sha256, source_uri, media_type, width, height}`; validate `modality != "text"` iff `images` non-empty. `to_bench_json()` gains `request.images`; `to_ekvachan_record()` gains an `images` key. **Additive and defaulted** — a text record must serialise byte-identically to today or every committed receipt and the sibling's `scripts/check_ekvachan_compat.py` break.
- [ ] **Agree `request.images` with the sibling's `/v1/systemone` contract in the same pass** (PRD §13.4 point 2). §6.1's claim that an item *is* a request body stops being true the moment an item carries an image the server won't accept. Coordinate with the sibling PRD §6.2.
- [ ] **Add the `[images]` manifest block + content-addressed cache** (PRD §13.5) — `kind`/`urls`/`sha256_manifest` in `manifest.toml`; `bjb build` materialises to a gitignored `data/images/<dataset>/<sha256[:2]>/<sha256>.<ext>` verifying each hash on write. **The corpus never redistributes pixels**: licenses forbid it for several entries, and `git clone && evaluate` (§8.1) does not survive 12 GB of Atari frames. Held-out commits label hashes + image sha256s only.
- [ ] **Three new CI gates** (PRD §13.7): image-hash gate (every `ImageRef.sha256` appears in the manifest's `sha256_manifest`); modality/strata consistency gate; split-key gate (a `modality != "text"` dataset uses the image-aware hash or declares an explicit alternative — the Atari per-trial case).
- [ ] **Build the first three loaders**, in this order: **Atari-HEAD** (`choice` over the ≤18-way legal action set — **no width narrowing, and split by *trial*, never by frame**; adjacent frames are near-duplicates and a frame-level split leaks catastrophically), then **GUIAct**, then **ScreenSpot-v2** as eval-only (it is below §7.3's training-size floor, deliberately). OS-Atlas after its per-subset licenses are verified.
- [ ] **Do not fill `prim_score` × `mod_multimodal`** (PRD §13.6). The only candidates are `deepghs/nsfw_detect` (taxonomy unpinned) and MedMNIST's ordinal subsets (medical caveat). Report it as under-populated and excluded — the same discipline §12.3 already applies to `mod_multimodal` itself — rather than inventing an ordinal scale for GUI/game data.
- [ ] **Re-verify every §13.3 asterisk at build time, not catalogue time** — OS-Atlas per-subset, GUIAct's license discrepancy, `nsfw_detect`'s taxonomy. The CFPB finding at the bottom of this file is the standing reminder of what a decayed catalogue claim costs.
- [ ] No leaderboard, no submission path, no rate limiting (§5.5 rules 4 and 6 are written, unimplemented)
- [ ] Everything built so far is English and text-only

## Next up, with GPU time estimates

Anchor: sibling project's real Qwen3.5-4B LoRA runs took 65–75 min for 24k examples / 1 epoch, same server (abhijeet-labgpu, single GPU, ~46GB VRAM). Most of this project's remaining work is still data/software engineering, not GPU-bound.

| # | Task | GPU time | Notes |
|---|---|---|---|
| 1 | ~~Loader/plugin framework + manifest parsing + CI gates~~ | — | **done 2026-09-23** |
| 2 | ~~Manifests + pull the 8 high-value Tier-A datasets~~ | — | **done 2026-09-23** — 422,878 items |
| 3 | ~~Reserve + hash-commit held-out slice~~ | — | **done 2026-09-23** — 21,174 items, committed |
| 4 | Implement the scoring spec (§5) + `/v1/evaluate` (§6.2), validate against the sibling's decoder checkpoint | minutes–hours | latency per item is still unmeasured for the decoder; the ~30ms in sibling PRD 13a.3 is an *encoder* number and does not carry over |
| 5 | Build the remaining ~43 Tier A entries | none | bandwidth-bound; each needs its license **and its shape** re-verified at build time (see the CFPB finding below) |
| 6 | Multimodal: §2.3's breadth slate **plus §13.3's on-distribution slice** (Atari-HEAD, GUIAct, ScreenSpot-v2) | none | closes the one empty stratum. **Now planned in full — PRD §13**, with a schema change, a `state_hash` correctness fix and three new CI gates as prerequisites. See the ordered checklist above. Directly unblocks the sibling's vision tier (its PRD §5.2b) |
| 7 | *(belongs in the sibling repo)* Retrain ekVachan on the assembled corpus | ~2–6 hrs, rough | depends on final pull size; its results belong in the sibling PRD's run log, not here |

**Note on task 4, still current.** Generality is expected to score **0** for the sibling's checkpoint, but the reason matters and validating *which* reason is the value of the task. §5.3's internal geometric mean collapses when any primitive stratum collapses. As of 2026-09-23 the sibling does now have a decoder trained on all three primitives (92.02% `choice`, 88.80% `noul`, 75.40% `score` on its own eval), so the old "no model for `score`/`noul` at all" framing is out of date. What is untested is whether those numbers survive contact with schemas and domains it has never seen — 151-way intent routing, 100-way contract provisions, a 5-level ordinal scale built from annotator-agreement fractions. **That is now an empirical question this repo can answer, and it could not answer it yesterday.** Predicting the answer in this file would be exactly the unverified-claim failure dev-guidelines rule 3 exists to prevent.

## Findings worth carrying forward

**The CFPB bulk export no longer contains complaint narratives.** Downloaded in full on 2026-09-23 (346,404,253 bytes, ETag `f0d2ec2367bd2ae60f044b782ad85f43-42`) — its header has 15 columns and `Consumer complaint narrative` is not among them, and the public search API returns no `complaint_what_happened` field either. `research/01` describes this source as "~8GB, real free-text complaint narrative"; the license claim held, the *shape* claim had decayed. The corpus uses a CC0 Hugging Face mirror's `has-text` config instead, with the deviation recorded in the manifest's `verified_how`. **Every remaining catalogue entry is owed the same check at build time, not at catalogue time** — see PRD §8.2's fifth bullet, which is the one v0.1 bullet that survived contact with reality.

**Three catalogued datasets are script-based and unloadable under `datasets >= 3`** (BANKING77, MASSIVE, the official CFPB repo). MASSIVE has a Hub-generated parquet branch; BANKING77 does not, so its loader reads the same GitHub CSVs PolyAI's own script reads. Expect this for other older catalogue entries.

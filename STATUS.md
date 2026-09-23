# better-jev-bench — status

Living task list for `better-jev-bench` (the standalone dataset/benchmark corpus repo). Update this file in the same commit as the work it describes — mark items done only after verifying them for real, not after a prior claim says so (see `skills/dev-guidelines/SKILL.md`). Full reasoning lives in `better-jev-bench_PRD.md`; this file is the checklist.

Sibling project: **ekVachan / better-jev-for-all** (separate repo, separate PRD, own `STATUS.md`) — the model this corpus will eventually train/evaluate. Don't mix work across the two repos in one commit.

Server clone: `abhijeet-labgpu:~/ekvachan/bench-repo` — this did not exist until 2026-09-23; before that, this repo had no server-side clone at all, which is worth remembering as a category of risk (see dev-guidelines rule 2/11).

Last verified: 2026-09-23.

## Done

- [x] Repo public, Apache-2.0 `LICENSE` committed
- [x] Founding PRD (`better-jev-bench_PRD.md`, ~83KB): scoring spec, API design, contribution framework, license tiering, roadmap
- [x] 6 research docs (`research/00`–`05`): methodology + 5 domain surveys (finance, academic/STEM, NLP, operational/safety, multimodal) — every license claim traced to a live-checked primary source
- [x] 99 dataset entries catalogued across 4 license tiers (~51 Tier A permissive, subset share-alike, ~12 Tier B research-only, 3 Tier C access-gated, ~31 Tier D unverified/blocked)

## Not started

By the PRD's own honest self-audit: *"a catalogue, not a corpus."*

- [ ] No data downloaded — zero rows collected
- [ ] No loader/plugin framework (`BenchmarkDataset` interface, `manifest.toml` parsing)
- [ ] No CI validation gates
- [ ] No train/held-out split, nothing hash-committed
- [ ] No scoring implementation — the `/v1/evaluate` spec in the PRD is illustrative only, not code
- [ ] Never run any ekVachan checkpoint against this corpus (can't yet — no data exists)

## Next up, with GPU time estimates

Anchor: sibling project's real Qwen3.5-4B LoRA runs took 65–75 min for 24k examples / 1 epoch, same server (abhijeet-labgpu, single GPU, ~46GB VRAM). Most of this project's own remaining work is data/software engineering, not GPU-bound — flagged explicitly below.

| # | Task | GPU time | Notes |
|---|---|---|---|
| 1 | Build loader/plugin framework + manifest parsing + CI gates | none | pure engineering |
| 2 | Write manifests + pull the 8 high-value Tier-A datasets first (CFPB, BANKING77, CLINC150, MASSIVE, GoEmotions, LEDGAR, CUAD, Civil Comments) | none | bandwidth-bound, not compute-bound |
| 3 | Reserve + hash-commit held-out slice | none | — |
| 4 | Implement scoring spec, validate against `ekvachan-base` (expect it to correctly report Generality=0, since it can't answer arbitrary schemas) | minutes | single inference pass/item, ~30ms/item per sibling PRD 13a.3 |
| 5 | *(explicitly deferred by this repo's own roadmap)* Retrain ekVachan on the assembled corpus | ~2–6 hrs, rough | depends on final pull size (50–200k example range); belongs in the sibling repo, not here |

---
name: dev-guidelines
description: Fundamental operating rules for any AI agent working on better-jev-bench and its sibling ekVachan (better-jev-for-all). Read this before starting any non-trivial task. This is the bench-repo copy — the fuller version, with the incidents that produced each rule, lives at better-jev-for-all/skills/dev-guidelines/SKILL.md; keep both in sync when adding a new rule.
---

# better-jev-bench dev guidelines

Shared rules with the sibling ekVachan project (`better-jev-for-all`) — both repos are part of the same effort and should follow the same discipline. See that repo's `skills/dev-guidelines/SKILL.md` for the fuller writeup with incident detail; this copy exists so an agent working on this repo alone doesn't need to have that one open.

## 1. The PRD is the source of truth
`better-jev-bench_PRD.md` carries the sourced reasoning for the scoring spec, license tiering, API design, and roadmap. Read the relevant section before deciding something it already covers. Update it as part of the same work, not as a follow-up — an unwritten result didn't happen, as far as the next agent is concerned.

## 2. The server clone is the only place to edit and commit
`abhijeet-labgpu:~/ekvachan/bench-repo`. This repo had **no server-side clone at all** until 2026-09-23 — before that, any work risked the same trap the sibling project hit (real content existing only in a non-git-tracked local file, never pushed). Don't let that happen again: if you ever find yourself editing a file under a local Mac `Documents/GitHub/...` path for this project, stop and verify with `git status`/`git remote -v` on that exact path first — it may not be a git repo at all.

## 3. Verify before trusting — including your own prior summary
Before citing any "already built" claim, re-derive it from real state: `gh repo view`, `gh api` tree listing, or the actual server clone's files — not from what a prior turn or agent said. `git fetch` before trusting local git history, always.

## 4. GPU job launching (for step 5's eventual training work, and step 4's eval)
```
setsid nohup env PYTHONUNBUFFERED=1 uv run python3 -u -m <module> <args> \
  > <logfile> 2>&1 < /dev/null & disown
```
Missing `PYTHONUNBUFFERED=1`/`-u` makes a healthy job look stalled for 20+ minutes when output is piped to a file — this exact bug wasted real GPU time on the sibling project three separate times. Missing `setsid`/`nohup`/`disown` means an SSH drop kills the job.

## 5. No Claude co-author trailer on commits in this project
Explicit standing instruction, applies to both repos.

## 6. Git identity/auth can reset between sessions
Check `git log -1 --format='%an <%ae>'` and match it with repo-local `git config user.name`/`user.email` rather than inventing a new identity. If `gh auth status` shows logged in but push fails with "could not read Username," run `gh auth setup-git`.

## 7. No destructive git ops to resolve a divergence
`git reset --hard`, `git checkout -B`, force-push are blocked by the auto-mode classifier here for good reason. `git fetch`, look at what actually diverged, `git merge`, resolve conflicts explicitly file-by-file.

## 8. Domain hygiene between the two repos
This repo and `better-jev-for-all` are separate, with separate PRDs and separate `STATUS.md`s. Finish and push one repo's work before starting on the other — don't leave mixed uncommitted state spanning both.

## 9. Delegating to subagents/forks
Tell them explicitly to verify against real current state rather than trust anything already claimed in the conversation.

## 10. Evidence discipline
Any eval run against this corpus (once data exists) gets a result file committed, referenced by its real path — not just a number in prose.

## 11. Sync discipline: server and GitHub must never silently diverge
Before ending work on this repo, confirm `git status` is clean and `git log origin/main..HEAD` / `git log HEAD..origin/main` are both empty (`git fetch` first) — i.e. the server clone and GitHub agree exactly, in both directions. Never leave a commit sitting unpushed, and never assume local git history reflects the remote without fetching first. A local Mac path is not part of this sync loop at all — it is not a git remote, should not be treated as one, and any content found there should be assumed stale until proven otherwise against the server/GitHub state.

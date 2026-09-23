# CI

The gates themselves are code, not YAML: `bjb validate` runs the PRD §7.4 checks
and exits non-zero on failure. Anything that can run Python can run them —
GitHub Actions, a pre-commit hook, a cron job, a reviewer's laptop.

```bash
pip install -e .
bjb validate              # offline: gates 1, 3, 4, 5, 6, 7, 8 — seconds, no network
bjb validate --loaders    # + gate 2: executes every loader against its real source
bjb build --verify        # rebuild and diff against the committed receipts
```

## Installing the GitHub Actions workflow

`github-workflow-ci.yml` is the workflow, and it lives here rather than at
`.github/workflows/ci.yml` for one mundane reason: the token that pushed it has
`repo` scope but not `workflow`, and GitHub rejects a push that creates or
changes a workflow file without it. Rather than leave a half-configured
`.github/` directory behind, the file sits here intact and unmodified.

To activate it, from an account with `workflow` scope:

```bash
gh auth refresh -s workflow       # once
mkdir -p .github/workflows
git mv ci/github-workflow-ci.yml .github/workflows/ci.yml
git commit -m "Activate CI workflow" && git push
```

Nothing else changes — the workflow only calls `bjb validate`, `bjb catalogue`
and `bjb stats`, all of which are already committed and already work.

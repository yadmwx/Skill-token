# Git workflow

## Branches

- `main` contains the maintained FlowMLP + H24/DenseFiLM research line.
- Create short-lived branches named `experiment/<topic>` or `fix/<topic>`.
- Historical code under `archive/` is read-only unless an old result is being reproduced.

## Before launching a run

Record the exact source state:

```bash
git status --short
git rev-parse HEAD
```

Do not launch a paper-facing run from a dirty worktree. Put the commit SHA, command, seed, dataset, task IDs, dependency versions, and checkpoint source in the run's provenance JSON.

## Commits

- Keep source/configuration changes separate from generated results.
- Never commit checkpoints, optimizer state, datasets, logs, rollouts, credentials, or machine-specific caches.
- Commit an experiment configuration before launching it.
- Tag paper snapshots with an annotated tag, for example `paper/object-v1`.

## Review gate

Before merging to `main`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests
git diff --check
```

The main result table must follow `docs/EXPERIMENT_SCOPE.md`. Adding a new model family requires an explicit scope decision rather than another flag in the shared action head.

## Remote backup

This repository currently has local history only. Add a private remote before relying on it as the sole backup:

```bash
git remote add origin <private-repository-url>
git push -u origin main --tags
```

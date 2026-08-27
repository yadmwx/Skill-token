# Depth-Conditioned VLA

Research code for studying which frozen vision-language-model representations are useful for continuous robot control.

## Current research scope

The maintained mainline is deliberately narrow:

- a frozen DINOv2 + SigLIP + Qwen2.5-0.5B vision-language backbone;
- a FlowMLP continuous action head;
- fixed final-layer (`H24`) conditioning as the baseline;
- state-conditioned dense depth FiLM residuals as the proposed extension;
- `full`, `proprio`, `static`, and `shuffled` state modes as mechanism ablations;
- LIBERO Object for in-domain evaluation and LIBERO-10 as a separate generalization evaluation.

DiT action heads, prototype/query-key routing, and expected-risk routing are retained only for checkpoint compatibility and historical reference. The failed E04/E07 protocol files were removed from the current tree and remain recoverable through Git history. None of these tracks belongs in the main result table. See `docs/EXPERIMENT_SCOPE.md` and `archive/README.md`.

## Important entry points

- `vla-scripts/finetune.py`: training entry point.
- `experiments/robot/libero/run_libero_eval.py`: LIBERO evaluation entry point.
- `prismatic/models/action_heads.py`: continuous action heads and depth interfaces.
- `prismatic/models/dense_depth_film.py`: maintained dense depth FiLM module.
- `tests/`: tests for maintained behavior.

## Repository policy

Model weights, datasets, rollouts, logs, optimizer states, and generated results are intentionally excluded from Git. Every reported run should instead record its command, source commit, checkpoint path, dataset, task IDs, seed, and evaluation count in a small text or JSON manifest.

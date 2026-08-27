# Experiment scope

## Mainline

The paper-facing comparison is:

1. `H24`: FlowMLP conditioned on the fixed final VLM layer.
2. `DenseFiLM/full`: H24 plus dense residuals from intermediate layers, conditioned on task and proprioception.
3. `DenseFiLM/proprio`: the state-conditioning ablation that removes task-token state.
4. `DenseFiLM/static`: the state-conditioning ablation that zeros task and proprio state.
5. `DenseFiLM/shuffled`: the negative-control state intervention.

The 40-episode four-task screen and the 100-episode full LIBERO Object evaluation are two evaluation scopes, not distinct model variants. LIBERO-10 is a generalization study and must not be pooled with LIBERO Object.

## Historical tracks

The following tracks are archived and excluded from the main result table:

- DiT and pure-DiT flow-matching action heads;
- prototype-soft, query-key-soft, and expected-risk routing;
- routing curricula and skill-token experiments;
- early E04 fixed-layer and E07 context-control protocols;
- one-off recovery, smoke, remote-sync, and checkpoint-repair scripts.

The implementation branches may remain in the model code when required to load old checkpoints. New mainline work must not add dependencies on them.

## Minimum run provenance

Every new result must record:

- Git commit SHA and whether the worktree was clean;
- complete command/configuration;
- source and output checkpoint paths;
- dataset and normalization statistics;
- task suite and explicit task IDs;
- trial count and random seed;
- dependency versions;
- success count as well as success rate.

Only results satisfying this provenance contract should enter paper-facing tables.


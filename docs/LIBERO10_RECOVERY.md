# LIBERO-10 recovery

## Diagnosis

The old H24 series was not a permanent zero-success failure:

- 5,000 steps: 0/100;
- 10,000 steps: 3/50 (6%); all successes came from one task;
- 15,000 steps: the interrupted evaluation reached 10/30 (33.3%) over the first six tasks.

The 5k-to-10k continuation was repeatedly interrupted/relaunched and did not use the same
effective schedule as the successful 10k-to-15k segment. The 15k segment used a constant
`2e-4` learning rate. Dataset statistics and gripper post-processing were consistent with
`libero_10_no_noops`; they do not explain the zero at 5k.

## Recovery protocol

Train a clean H24 model for 15,000 uninterrupted optimizer steps with
`scripts/train_libero10_h24.sh`. It deliberately excludes DiT, routing, skill tokens,
continuous-context conditioning, and DenseFiLM. Only the latest checkpoint is retained.

After training, run `scripts/eval_libero10_h24.sh` with ten trials for each of all ten tasks.
The evaluation refuses to use normalization statistics from another LIBERO suite.

Do not promote the run into the paper-facing table unless the provenance file names a clean
Git commit and the complete 100-episode evaluation finishes.

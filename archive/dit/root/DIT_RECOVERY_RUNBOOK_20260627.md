# DIT Recovery Runbook 2026-06-27

## Goal

Restore the LIBERO DIT action-head success rate to the normal range. The working hypothesis is that the low success rate is caused by implementation/conditioning bugs, not only by seed choice or insufficient training steps.

## Reference: openpi / pi0.5

The openpi pi0.5 action path uses flow matching with image/language prefix tokens and noisy action suffix tokens inside the same action expert transformer. Its important structural property is prefix/suffix attention:

- prefix image/language tokens attend to valid prefix tokens;
- suffix action tokens attend to prefix tokens and action suffix tokens;
- conditioning is not only a compressed VLM vector used through external cross-attention.

This motivates the local `joint_prefix` ablation added to the current DIT implementation.

## Local Changes

- Gripper normalized-space fix in `prismatic/models/flow_matching_head.py`:
  - BCE targets use `target[..., -1] > 0.0`;
  - gripper-head inference override uses `torch.tanh(gripper_logits)`.
- DIT condition injection switch:
  - `--dit_condition_injection_mode cross_attn`
  - `--dit_condition_injection_mode joint_prefix`
- `joint_prefix` lives in `prismatic/models/ditx_vla_adapter.py` and builds a pi0-style prefix/suffix self-attention mask.
- Offline probe supports `--dit_condition_injection_mode`.
- Joint-prefix queue runs offline probe before online LIBERO eval for each checkpoint.

## Remote Recovery

When `100.106.143.20` is reachable again:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\tmp_sync_joint_prefix_dit_and_queue_20260627.ps1
```

That script:

1. syncs local DIT/gripper/joint-prefix code to the remote repo;
2. runs remote `py_compile`;
3. runs `scripts/tmp_smoke_dit_condition_injection_20260627.py` in the remote torch environment;
4. evaluates the existing 5500 cross-attn checkpoint with the gripper fix;
5. starts a new `joint_prefix` MLP-anchor-init DIT run;
6. runs offline probe and LIBERO online eval on produced checkpoints.

## Result Fetch

After the queue has had time to run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\tmp_fetch_dit_recovery_results_20260627.ps1
```

Read results in this order:

1. `gripper_fixed_existing5500`: if this becomes high, the gripper normalized-space bug was a main cause.
2. `offline_probe`: if teacher-flow reconstruction / velocity MAE improves under `joint_prefix`, conditioning was likely the main bug.
3. `mlp_anchorinit step=... successes=.../3`: online confirmation.

## Current Remote Blocker

As of 2026-06-27, `ubuntu` / `100.106.143.20` appears active in Tailscale status, but port 22 is unreachable:

- plain SSH times out;
- `tailscale ssh ubuntu@ubuntu` returns a dial timeout / 502;
- `192.168.166.139:22` also times out.

This points to a remote peer / SSH service / firewall / routing issue rather than a local command or authentication issue.

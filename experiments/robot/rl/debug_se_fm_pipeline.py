"""
Lightweight debugging harness for SE-FM data/loading pipeline.

Usage example:
    conda run -n vla-adapter python experiments/robot/rl/debug_se_fm_pipeline.py \
        --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
        --offline_dataset_path /home/xgx/code/VLA-Adapter/modified_libero_rlds \
        --offline_dataset_name libero_spatial_no_noops \
        --offline_batch_size 1 \
        --offline_shuffle_buffer_size 65536 \
        --num_offline_batches 2 \
        --num_online_rollouts 1

The script prints wall-clock time for each offline batch fetch and each online
rollout to help diagnose slow start-up behaviour.
"""


import argparse
import time
from pathlib import Path
from typing import Optional

import torch

from experiments.robot.rl.train_se_fm import (
    SEFMConfig,
    build_generate_config,
    build_offline_dataloader,
    collect_online_rollouts,
    prepare_policy_inputs,
)
from experiments.robot.libero.run_libero_eval import initialize_model, TaskSuite
from libero.libero import benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Debug SE-FM data pipeline")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--offline_dataset_path", type=str, default="")
    parser.add_argument("--offline_dataset_name", type=str, default=None)
    parser.add_argument("--offline_batch_size", type=int, default=1)
    parser.add_argument("--offline_shuffle_buffer_size", type=int, default=65_536)
    parser.add_argument("--num_offline_batches", type=int, default=1)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = SEFMConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        offline_dataset_path=args.offline_dataset_path or None,
        offline_dataset_name=args.offline_dataset_name,
        offline_batch_size=args.offline_batch_size,
        offline_shuffle_buffer_size=args.offline_shuffle_buffer_size,
        task_suite=args.task_suite,
    )

    print("[DEBUG] Building generate config...")
    gen_cfg = build_generate_config(cfg)
    print("[DEBUG] Loading pretrained model / action head...")
    model, action_head, proprio_projector, _, processor = initialize_model(gen_cfg)
    model.to(device)
    action_head.to(device)
    proprio_projector.to(device)
    model.eval()
    proprio_projector.eval()

    print("[DEBUG] Building offline dataloader...")
    offline_iter = None
    if cfg.offline_dataset_path:
        offline_loader = build_offline_dataloader(cfg, gen_cfg, model, proprio_projector, processor)
        offline_iter = iter(offline_loader)

    for offline_idx in range(args.num_offline_batches):
        if offline_iter is None:
            print("[DEBUG] Offline dataloader disabled; skipping offline batch timing.")
            break
        start = time.time()
        batch = next(offline_iter)
        elapsed = time.time() - start
        print(f"[DEBUG] Offline batch {offline_idx+1}/{args.num_offline_batches} fetched in {elapsed:.2f}s "
              f"(actions_hidden_states shape={batch.actions_hidden_states.shape})")

    if not args.num_online_rollouts:
        print("[DEBUG] Skipping online rollout debug.")
        return

    suite_cls = benchmark.get_benchmark(TaskSuite(cfg.task_suite).value)
    suite = suite_cls()
    tasks = suite.tasks

    for rollout_idx in range(args.num_online_rollouts):
        start = time.time()
        batches, stats = collect_online_rollouts(
            cfg=cfg,
            gen_cfg=gen_cfg,
            model=model,
            action_head=action_head,
            proprio_projector=proprio_projector,
            processor=processor,
            suite_tasks=tasks,
            iteration=rollout_idx,
            device=device,
        )
        elapsed = time.time() - start
        print(
            f"[DEBUG] Online rollout {rollout_idx+1}/{args.num_online_rollouts} "
            f"completed in {elapsed:.2f}s "
            f"(batches={len(batches)}, episode_success={stats['episode_success']:.2f}, "
            f"chunk_success_rate={stats['chunk_success_rate']:.2f}, episode_return={stats['episode_return']:.2f})"
        )


if __name__ == "__main__":
    main()


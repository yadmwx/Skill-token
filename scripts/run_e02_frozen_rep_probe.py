"""E02: frozen, layer-wise representation probe.

The VLM is frozen.  Identical linear probes are trained on layer 1..16 and
four predefined layer groups.  Splits are by trajectory id, never by sample.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.robot.openvla_utils import get_processor, get_proprio_projector, get_vla
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, STOP_INDEX
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cfg_for(checkpoint, seed):
    return SimpleNamespace(
        pretrained_checkpoint=checkpoint,
        load_in_8bit=False,
        load_in_4bit=False,
        use_film=False,
        num_images_in_input=2,
        seed=seed,
        use_proprio=True,
        use_minivlm=True,
        unnorm_key="libero_object_no_noops",
    )


@torch.no_grad()
def extract_features(model, batch, proprio_projector):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    pixels = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    proprio = batch.get("proprio")
    if proprio is not None:
        proprio = proprio.to(device, dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixels,
            labels=labels,
            output_hidden_states=True,
            output_projector_features=True,
            proprio=proprio,
            proprio_projector=proprio_projector,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=False,
        )

    projector_features = getattr(output, "projector_features", None)
    if projector_features is None:
        raise RuntimeError("E02 requires projector_features to locate action-token positions")
    num_patches = int(projector_features.shape[1])
    gt_token_ids = labels[:, 1:]
    all_actions_mask = get_current_action_mask(gt_token_ids) | get_next_actions_mask(gt_token_ids)
    action_indices = [torch.where(all_actions_mask[b])[0] for b in range(labels.shape[0])]
    expected = ACTION_DIM * NUM_ACTIONS_CHUNK
    if any(x.numel() != expected for x in action_indices):
        raise RuntimeError(f"E02 action-token count mismatch: {[x.numel() for x in action_indices]}")
    action_idx = torch.stack(action_indices, dim=0)
    mm_pos = action_idx + 1 + num_patches
    layer_features = []
    for layer_hidden in output.hidden_states:
        gather_index = mm_pos.unsqueeze(-1).expand(layer_hidden.shape[0], expected, layer_hidden.shape[-1])
        action_latents = layer_hidden.gather(dim=1, index=gather_index)
        layer_features.append(action_latents.mean(dim=1).float().cpu())
    return torch.stack(layer_features, dim=1), batch["actions"][:, 0, :].float().cpu()


def collect(args, model, processor, proprio_projector):
    tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(
        tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
        use_minivlm=True,
    )
    dataset = EpisodicRLDSDataset(
        args.data_root_dir,
        "libero_object_no_noops",
        transform,
        resize_resolution=tuple(model.config.image_sizes),
        shuffle_buffer_size=1,
        # The bundled LIBERO RLDS export contains only a `train` split.
        # We perform our own trajectory-level train/validation/test split below.
        train=True,
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    all_features, all_targets, metadata = [], [], []
    for trajectory_id, episode in enumerate(dataset):
        if trajectory_id >= args.num_episodes:
            break
        episode = episode[: args.max_steps_per_episode]
        for start in range(0, len(episode), args.batch_size):
            steps = episode[start : start + args.batch_size]
            batch = collator(steps)
            feats, targets = extract_features(model, batch, proprio_projector)
            all_features.append(feats)
            all_targets.append(targets)
            for j, item in enumerate(steps):
                metadata.append({
                    "sample_id": len(metadata),
                    "trajectory_id": trajectory_id,
                    "env_step": int(item.get("timestep", start + j)),
                    "task": item.get("task_language", "unknown"),
                })
    if not all_features:
        raise RuntimeError("E02 collected no samples")
    return torch.cat(all_features), torch.cat(all_targets), metadata


class ProbeBank(nn.Module):
    def __init__(self, num_probes, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_probes, dim, ACTION_DIM))
        self.bias = nn.Parameter(torch.zeros(num_probes, ACTION_DIM))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.einsum("bld,ldh->blh", x, self.weight) + self.bias.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root_dir", default="data/libero")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num_episodes", type=int, default=30)
    ap.add_argument("--max_steps_per_episode", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = cfg_for(args.checkpoint, args.seed)
    model = get_vla(cfg)
    model.eval()
    processor = get_processor(cfg)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    proprio_projector.eval()
    features, targets, metadata = collect(args, model, processor, proprio_projector)
    np.savez_compressed(
        out / "probe_source.npz",
        features=features.numpy(),
        targets=targets.numpy(),
        trajectory_id=np.asarray([m["trajectory_id"] for m in metadata]),
        env_step=np.asarray([m["env_step"] for m in metadata]),
    )
    (out / "probe_metadata.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in metadata) + "\n")

    layer_indices = list(range(1, min(16, features.shape[1] - 1) + 1))
    group_defs = {"L1-4": [1, 2, 3, 4], "L5-8": [5, 6, 7, 8], "L9-12": [9, 10, 11, 12], "L13-16": [13, 14, 15, 16]}
    variants = [(f"L{i}", [i]) for i in layer_indices] + list(group_defs.items())
    variant_features = torch.stack([features[:, idxs].mean(1) for _, idxs in variants], dim=1)

    trajectories = sorted(set(int(x) for x in features.new_tensor([m["trajectory_id"] for m in metadata]).tolist()))
    n_train = max(1, int(len(trajectories) * 0.70))
    n_val = max(1, int(len(trajectories) * 0.15))
    split = {t: "train" if i < n_train else "val" if i < n_train + n_val else "test" for i, t in enumerate(trajectories)}
    split_ids = torch.tensor([0 if split[m["trajectory_id"]] == "train" else 1 if split[m["trajectory_id"]] == "val" else 2 for m in metadata])
    train_mask, val_mask, test_mask = split_ids == 0, split_ids == 1, split_ids == 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = ProbeBank(len(variants), features.shape[-1]).to(device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=1e-3)
    x, y = variant_features.to(device), targets.to(device)
    for _ in range(args.epochs):
        bank.train()
        optimizer.zero_grad(set_to_none=True)
        pred = bank(x[train_mask.to(device)])
        loss = (pred - y[train_mask.to(device), None, :]).pow(2).mean()
        loss.backward()
        optimizer.step()

    bank.eval()
    with torch.no_grad():
        pred = bank(x).cpu().numpy()
    y_np = targets.numpy()
    rows = []
    for variant_idx, (variant, indices) in enumerate(variants):
        for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            m = mask.numpy()
            err = pred[m, variant_idx] - y_np[m]
            gripper_pred = pred[m, variant_idx, -1] >= 0
            gripper_true = y_np[m, -1] >= 0
            tp = np.logical_and(gripper_pred, gripper_true).sum()
            f1 = float(2 * tp / max(gripper_pred.sum() + gripper_true.sum(), 1))
            rows.append({"variant": variant, "layer_indices": ",".join(map(str, indices)), "split": split_name, "n": int(m.sum()), "mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()), "gripper_f1": f1, "seed": args.seed})
    with (out / "probe_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    torch.save({"state_dict": bank.state_dict(), "variants": variants, "seed": args.seed, "split": split}, out / "probe_checkpoints.pt")
    (out / "manifest.json").write_text(json.dumps(vars(args) | {"variants": variants, "split": split}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(metadata), "trajectories": len(trajectories), "variants": len(variants), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

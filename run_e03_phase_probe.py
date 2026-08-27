"""E03: phase-conditioned probe using the frozen E02 feature package.

The phase rule is predeclared and post hoc: normalized trajectory time quartiles
are labeled approach/interaction/transport/completion.  A permutation null is
reported; this is intentionally conservative because time-based labels can
overstate semantic phase evidence.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from prismatic.vla.constants import ACTION_DIM


PHASES = ("approach", "interaction", "transport", "completion")


def phase_for(step, max_step):
    q = float(step) / max(float(max_step), 1.0)
    return PHASES[min(int(q * 4.0), 3)]


def f1_score(pred, true):
    pred = pred >= 0
    true = true >= 0
    tp = np.logical_and(pred, true).sum()
    return float(2 * tp / max(pred.sum() + true.sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e02_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--num_shuffles", type=int, default=1000)
    args = ap.parse_args()
    src = Path(args.e02_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = np.load(src / "probe_source.npz")
    features = raw["features"][:, 1:17].astype(np.float32)
    targets = raw["targets"].astype(np.float32)
    traj = raw["trajectory_id"].astype(int)
    steps = raw["env_step"].astype(int)
    metadata = json.loads((src / "manifest.json").read_text())
    split_map = {int(k): v for k, v in metadata["split"].items()}
    max_steps = {t: int(steps[traj == t].max()) for t in np.unique(traj)}
    phases = np.asarray([phase_for(s, max_steps[t]) for s, t in zip(steps, traj)])

    np.savez_compressed(out / "phase_labels.npz", trajectory_id=traj, env_step=steps, phase=phases)
    (out / "label_rules.json").write_text(json.dumps({
        "source": "posthoc_time_quartile_v1",
        "phase_order": list(PHASES),
        "rule": "normalized env_step / trajectory_max_step split into four equal quartiles",
        "warning": "This rule is temporal, not geometric; semantic phase claims require caution.",
    }, indent=2) + "\n")

    rows = []
    probe_states = {}
    for phase in PHASES:
        phase_train = np.asarray([(split_map[t] == "train" and p == phase) for t, p in zip(traj, phases)])
        phase_test = np.asarray([(split_map[t] == "test" and p == phase) for t, p in zip(traj, phases)])
        if phase_train.sum() == 0 or phase_test.sum() == 0:
            continue
        x_train = torch.from_numpy(features[phase_train])
        y_train = torch.from_numpy(targets[phase_train])
        x_test = torch.from_numpy(features[phase_test])
        y_test = targets[phase_test]
        weights = []
        biases = []
        for layer in range(16):
            torch.manual_seed(1000 + layer)
            probe = nn.Linear(features.shape[-1], ACTION_DIM)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            for _ in range(args.epochs):
                opt.zero_grad(set_to_none=True)
                loss = (probe(x_train[:, layer]) - y_train).pow(2).mean()
                loss.backward(); opt.step()
            with torch.no_grad():
                pred = probe(x_test[:, layer]).numpy()
            err = pred - y_test
            rows.append({"phase": phase, "variant": f"L{layer+1}", "n_train": int(phase_train.sum()), "n_test": int(phase_test.sum()), "mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()), "gripper_f1": f1_score(pred[:, -1], y_test[:, -1])})
            weights.append(probe.weight.detach().numpy()); biases.append(probe.bias.detach().numpy())
        probe_states[phase] = {"weight": np.stack(weights), "bias": np.stack(biases)}

    # Fixed-probe permutation null: preserve each phase count and shuffle phase labels.
    null = np.zeros((args.num_shuffles, len(PHASES), 16), dtype=np.float32)
    rng = np.random.default_rng(20260713)
    test_mask = np.asarray([split_map[t] == "test" for t in traj])
    test_idx = np.where(test_mask)[0]
    for si in range(args.num_shuffles):
        shuffled = phases[test_idx].copy(); rng.shuffle(shuffled)
        for pi, phase in enumerate(PHASES):
            idx = test_idx[shuffled == phase]
            for layer in range(16):
                state = probe_states.get(phase)
                if state is None or len(idx) == 0:
                    null[si, pi, layer] = np.nan
                else:
                    pred = features[idx, layer] @ state["weight"][layer].T + state["bias"][layer]
                    null[si, pi, layer] = float(((pred - targets[idx]) ** 2).mean())
    np.save(out / "phase_shuffle_null.npy", null)
    with (out / "phase_probe.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (out / "manifest.json").write_text(json.dumps(vars(args) | {"phase_order": list(PHASES), "seed": 20260713}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "phases": {p: int((phases == p).sum()) for p in PHASES}, "output": str(out)}))


if __name__ == "__main__":
    main()

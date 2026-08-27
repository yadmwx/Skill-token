#!/usr/bin/env python3
"""Create the auditable Figure 5 phase-profile panel from pooled traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASES = ("approach", "interaction", "transport", "completion")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def vector(row, key):
    return np.asarray(json.loads(row[key]), dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    src, out = Path(args.derived_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profiles = read_csv(src / "phase_profile_ci.csv")
    null = read_csv(src / "phase_shuffle_null.csv")
    episode_phase = read_csv(src / "episode_phase_summary.csv")
    episode_outcomes = read_csv(src / "episode_outcomes.csv")
    success_n = sum(r["success"].lower() == "true" for r in episode_outcomes)
    failure_n = len(episode_outcomes) - success_n

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    x = np.arange(len(PHASES))
    colors = {"True": "#2166ac", "False": "#b2182b"}
    labels = {"True": f"success (n={success_n})", "False": f"failure (n={failure_n})"}

    for success in ("True", "False"):
        rows = {r["phase"]: r for r in profiles if r["success"] == success}
        if not all(p in rows for p in PHASES):
            continue
        y = np.asarray([float(rows[p]["mean_entropy"]) for p in PHASES])
        lo = np.asarray([float(rows[p]["entropy_ci95_low"]) for p in PHASES])
        hi = np.asarray([float(rows[p]["entropy_ci95_high"]) for p in PHASES])
        axes[0, 0].plot(x, y, marker="o", lw=2.2, color=colors[success], label=labels[success])
        axes[0, 0].fill_between(x, lo, hi, color=colors[success], alpha=0.15, linewidth=0)
        if success == "True":
            w = np.stack([vector(rows[p], "mean_layer_weights") for p in PHASES])
            im = axes[1, 0].imshow(w.T, aspect="auto", origin="lower", cmap="viridis", vmin=0.0)
            axes[1, 0].set_title("Mean layer weights — success")
            axes[1, 0].set_ylabel("VLM layer index")
            axes[1, 0].set_xticks(x, PHASES, rotation=20)
            fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, label="weight")

    observed_depth = np.asarray([
        np.mean([float(r["mean_expected_depth"]) for r in episode_phase if r["phase"] == p])
        for p in PHASES
    ])
    axes[0, 1].plot(x, observed_depth, marker="o", lw=2.2, color="#762a83", label="observed")
    null_depth = {p: np.asarray([float(r["mean_expected_depth"]) for r in null if r["phase"] == p]) for p in PHASES}
    lo = np.asarray([np.quantile(null_depth[p], 0.025) for p in PHASES])
    hi = np.asarray([np.quantile(null_depth[p], 0.975) for p in PHASES])
    axes[0, 1].fill_between(x, lo, hi, color="#762a83", alpha=0.15, label="phase shuffle 95% null")
    axes[0, 1].set_title("Expected depth vs. phase shuffle")
    axes[0, 1].set_ylabel("expected layer depth")
    axes[0, 1].legend(frameon=False)

    failure_rows = {r["phase"]: r for r in profiles if r["success"] == "False"}
    if all(p in failure_rows for p in PHASES):
        wf = np.stack([vector(failure_rows[p], "mean_layer_weights") for p in PHASES])
        axes[1, 1].imshow(wf.T, aspect="auto", origin="lower", cmap="magma", vmin=0.0)
        axes[1, 1].set_title("Mean layer weights — failure")
    else:
        axes[1, 1].text(0.5, 0.5, "No failure-phase rows", ha="center", va="center")
        axes[1, 1].set_title("Failure panel")
    axes[1, 1].set_ylabel("VLM layer index")
    axes[1, 1].set_xticks(x, PHASES, rotation=20)

    axes[0, 0].set_title(f"Routing entropy by phase (success n={success_n}; failure n={failure_n})")
    axes[0, 0].set_ylabel("routing entropy")
    axes[0, 0].set_xticks(x, PHASES, rotation=20)
    axes[0, 0].legend(frameon=False)
    for ax, label in zip(axes.flat, ("a", "b", "c", "d")):
        ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=12, va="top")
    fig.suptitle("Figure 5 — routing depth across execution phases (all 80 episodes; temporal-quartile proxy)", fontsize=14)
    fig.savefig(out / "figure5_phase_profiles.png", dpi=300)
    fig.savefig(out / "figure5_phase_profiles.svg")
    plt.close(fig)

    checksum_lines = []
    for path in sorted(out.iterdir()):
        if path.name == "checksums.sha256" or not path.is_file():
            continue
        checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print({"output": str(out), "episodes": len(episode_outcomes), "successes": success_n, "failures": failure_n})


if __name__ == "__main__":
    main()

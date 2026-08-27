#!/usr/bin/env python3
"""Create a full, non-cherry-picked Figure 5 source figure from pooled traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASES = ("approach", "interaction", "transport", "completion")


def rows_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rows_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean_matrix(rows, key, success=None):
    out = []
    for phase in PHASES:
        vals = [np.asarray(r[key][0], dtype=float) for r in rows if r.get("phase") == phase and (success is None or bool(r["success"]) == success)]
        out.append(np.mean(vals, axis=0) if vals else np.full(len(rows[0][key][0]), np.nan))
    return np.stack(out, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--derived_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    trace = rows_jsonl(args.trace)
    derived = Path(args.derived_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = rows_csv(derived / "phase_profile_ci.csv")
    null = rows_csv(derived / "phase_shuffle_null.csv")
    outcomes = rows_csv(derived / "episode_outcomes.csv")
    n_success = sum(r["success"].lower() == "true" for r in outcomes)
    n_failure = len(outcomes) - n_success

    skill = mean_matrix(trace, "skill_probs")
    weights_success = mean_matrix(trace, "layer_weights", True)
    weights_failure = mean_matrix(trace, "layer_weights", False)

    fig = plt.figure(figsize=(13.2, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, hspace=0.24, wspace=0.24)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_d, ax_e, ax_f = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(PHASES))

    im = ax_a.imshow(skill, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=max(0.15, float(np.nanmax(skill))))
    ax_a.set_title("Skill posterior by phase")
    ax_a.set_xlabel("phase (temporal quartile proxy)")
    ax_a.set_ylabel("skill ID")
    ax_a.set_xticks(x, PHASES, rotation=25)
    fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04, label="probability")

    colors = {"True": "#2166ac", "False": "#b2182b"}
    for success, label in (("True", f"success (n={n_success})"), ("False", f"failure (n={n_failure})")):
        rs = {r["phase"]: r for r in profile if r["success"] == success}
        y = np.asarray([float(rs[p]["mean_entropy"]) for p in PHASES])
        lo = np.asarray([float(rs[p]["entropy_ci95_low"]) for p in PHASES])
        hi = np.asarray([float(rs[p]["entropy_ci95_high"]) for p in PHASES])
        ax_b.plot(x, y, marker="o", lw=2.0, color=colors[success], label=label)
        ax_b.fill_between(x, lo, hi, color=colors[success], alpha=0.15, linewidth=0)
    ax_b.set_title("Routing entropy")
    ax_b.set_ylabel("entropy")
    ax_b.set_xticks(x, PHASES, rotation=25)
    ax_b.legend(frameon=False)

    episode_phase = rows_csv(derived / "episode_phase_summary.csv")
    observed = np.asarray([np.mean([float(r["mean_expected_depth"]) for r in episode_phase if r["phase"] == p]) for p in PHASES])
    null_by_phase = {p: np.asarray([float(r["mean_expected_depth"]) for r in null if r["phase"] == p]) for p in PHASES}
    lo = np.asarray([np.quantile(null_by_phase[p], 0.025) for p in PHASES])
    hi = np.asarray([np.quantile(null_by_phase[p], 0.975) for p in PHASES])
    ax_c.plot(x, observed, marker="o", lw=2.0, color="#762a83", label="observed")
    ax_c.fill_between(x, lo, hi, color="#762a83", alpha=0.15, label="phase shuffle 95% null")
    ax_c.set_title("Expected depth vs. shuffle")
    ax_c.set_ylabel("expected layer depth")
    ax_c.set_xticks(x, PHASES, rotation=25)
    ax_c.legend(frameon=False)

    for ax, mat, title, cmap in ((ax_d, weights_success, f"Layer weights — success (n={n_success})", "viridis"), (ax_e, weights_failure, f"Layer weights — failure (n={n_failure})", "magma")):
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=0.0)
        ax.set_title(title)
        ax.set_xlabel("phase")
        ax.set_ylabel("VLM layer index")
        ax.set_xticks(x, PHASES, rotation=25)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")

    success_rate = n_success / len(outcomes)
    ax_f.bar(["success", "failure"], [n_success, n_failure], color=["#2166ac", "#b2182b"], alpha=0.9)
    ax_f.set_title(f"All outcomes (N={len(outcomes)}; success={success_rate:.1%})")
    ax_f.set_ylabel("episodes")
    for i, v in enumerate((n_success, n_failure)):
        ax_f.text(i, v + 0.8, str(v), ha="center", fontweight="bold")
    for ax, label in zip((ax_a, ax_b, ax_c, ax_d, ax_e, ax_f), "abcdef"):
        ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    fig.suptitle("Figure 5 — skill-conditioned routing and depth across execution phases (all episodes; no case selection)", fontsize=14)
    fig.savefig(out / "figure5_full.png", dpi=300)
    fig.savefig(out / "figure5_full.svg")
    plt.close(fig)

    files = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "checksums.sha256":
            files.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out / "checksums.sha256").write_text("\n".join(files) + "\n", encoding="utf-8")
    print({"episodes": len(outcomes), "successes": n_success, "failures": n_failure, "queries": len(trace), "output": str(out)})


if __name__ == "__main__":
    main()

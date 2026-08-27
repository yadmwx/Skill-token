#!/usr/bin/env python3
"""Derive auditable Figure 5 tables from the complete routing trace.

The source trace is never filtered or overwritten.  Phase labels are a
predeclared temporal proxy (four equal query-time quartiles), not semantic or
geometric annotations.  Every success and failure episode is retained.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


PHASES = ("approach", "interaction", "transport", "completion")


def phase_for(query_step: int, max_query_step: int) -> str:
    frac = float(query_step) / max(float(max_query_step), 1.0)
    return PHASES[min(int(frac * len(PHASES)), len(PHASES) - 1)]


def mean_flat(values):
    if not values:
        return ""
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr))


def skill_index(record):
    value = record.get("skill_id")
    while isinstance(value, list) and value:
        value = value[0]
    return int(value) if value is not None else -1


def skill_entropy(record):
    probs = (record.get("skill_probs") or [[]])[0]
    return float(-sum(float(p) * np.log(max(float(p), 1e-12)) for p in probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    source = Path(args.trace)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_episode = defaultdict(list)
    for record in records:
        by_episode[record["episode_id"]].append(record)

    derived = []
    for episode_id, episode_records in by_episode.items():
        max_query = max(int(r["query_step"]) for r in episode_records)
        for record in episode_records:
            row = dict(record)
            row["phase"] = phase_for(int(record["query_step"]), max_query)
            row["label_source"] = "posthoc_time_quartile_v1"
            row["phase_rule_max_query_step"] = max_query
            derived.append(row)

    derived.sort(key=lambda r: (r["episode_id"], int(r["query_step"])))
    with (out / "trace_with_phase.jsonl").open("w", encoding="utf-8") as f:
        for row in derived:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    phase_rows = []
    grouped = defaultdict(list)
    for row in derived:
        grouped[(row["task"], row["phase"], bool(row["success"]))].append(row)
    for (task, phase, success), group in sorted(grouped.items()):
        phase_rows.append({
            "task": task,
            "phase": phase,
            "success": success,
            "queries": len(group),
            "episodes": len({r["episode_id"] for r in group}),
            "mean_expected_depth": mean_flat([r.get("expected_depth") for r in group]),
            "mean_gripper": mean_flat([r.get("gripper") for r in group]),
            "mean_skill_entropy": mean_flat([skill_entropy(r) for r in group]),
        })
    with (out / "phase_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=phase_rows[0].keys())
        writer.writeheader()
        writer.writerows(phase_rows)

    layer_rows = []
    layer_grouped = defaultdict(list)
    for row in derived:
        weights = (row.get("layer_weights") or [[]])[0]
        for layer_index, weight in enumerate(weights, start=1):
            layer_grouped[(row["task"], row["phase"], bool(row["success"]), layer_index)].append(float(weight))
    for (task, phase, success, layer_index), values in sorted(layer_grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        layer_rows.append({
            "task": task,
            "phase": phase,
            "success": success,
            "layer_index": layer_index,
            "queries": len(values),
            "mean_weight": float(np.mean(arr)),
            "std_weight": float(np.std(arr)),
        })
    with (out / "layer_weight_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=layer_rows[0].keys())
        writer.writeheader()
        writer.writerows(layer_rows)

    episode_rows = []
    for episode_id, group in sorted(by_episode.items()):
        episode_rows.append({
            "episode_id": episode_id,
            "task": group[0]["task"],
            "success": bool(group[-1]["success"]),
            "queries": len(group),
            "mean_expected_depth": mean_flat([r.get("expected_depth") for r in group]),
            "mean_skill_entropy": mean_flat([skill_entropy(r) for r in group]),
        })
    with (out / "episode_outcomes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)

    # Episode x phase table required by the Figure 5(d) contract.
    episode_phase_rows = []
    for episode_id, group in sorted(by_episode.items()):
        phase_group = defaultdict(list)
        for row in derived:
            if row["episode_id"] == episode_id:
                phase_group[row["phase"]].append(row)
        ordered = sorted(group, key=lambda r: int(r["query_step"]))
        transition_steps = [
            int(curr["query_step"])
            for prev, curr in zip(ordered, ordered[1:])
            if skill_index(prev) != skill_index(curr)
        ]
        for phase in PHASES:
            rows = phase_group.get(phase, [])
            if not rows:
                continue
            histogram = [0] * 16
            for row in rows:
                idx = skill_index(row)
                if 0 <= idx < len(histogram):
                    histogram[idx] += 1
            weights = np.asarray([(r.get("layer_weights") or [[]])[0] for r in rows], dtype=np.float64)
            episode_phase_rows.append({
                "episode_id": episode_id,
                "task": group[0]["task"],
                "phase": phase,
                "success": bool(group[-1]["success"]),
                "queries": len(rows),
                "skill_histogram": json.dumps(histogram, separators=(",", ":")),
                "mean_layer_weights": json.dumps(np.mean(weights, axis=0).round(8).tolist(), separators=(",", ":")),
                "entropy": mean_flat([skill_entropy(r) for r in rows]),
                "mean_expected_depth": mean_flat([r.get("expected_depth") for r in rows]),
                "transition_times": json.dumps([s for s in transition_steps if phase_for(s, max(int(r["query_step"]) for r in ordered)) == phase]),
            })
    with (out / "episode_phase_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_phase_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_phase_rows)

    # Episode-level means give a clustered 95% CI without treating queries as independent.
    profile_rows = []
    for (phase, success), rows in sorted({
        (phase, success): [r for r in episode_phase_rows if r["phase"] == phase and r["success"] == success]
        for phase in PHASES for success in (False, True)
    }.items()):
        if not rows:
            continue
        entropy = np.asarray([float(r["entropy"]) for r in rows])
        layer_values = np.asarray([json.loads(r["mean_layer_weights"]) for r in rows], dtype=np.float64)
        se = entropy.std(ddof=1) / np.sqrt(len(entropy)) if len(entropy) > 1 else 0.0
        profile_rows.append({
            "phase": phase,
            "success": success,
            "episodes": len(rows),
            "mean_entropy": float(entropy.mean()),
            "entropy_ci95_low": float(entropy.mean() - 1.96 * se),
            "entropy_ci95_high": float(entropy.mean() + 1.96 * se),
            "mean_layer_weights": json.dumps(layer_values.mean(axis=0).round(8).tolist(), separators=(",", ":")),
            "layer_weight_ci95_low": json.dumps((layer_values.mean(axis=0) - 1.96 * layer_values.std(axis=0, ddof=1) / np.sqrt(len(rows))).round(8).tolist(), separators=(",", ":")) if len(rows) > 1 else json.dumps(layer_values.mean(axis=0).round(8).tolist(), separators=(",", ":")),
            "layer_weight_ci95_high": json.dumps((layer_values.mean(axis=0) + 1.96 * layer_values.std(axis=0, ddof=1) / np.sqrt(len(rows))).round(8).tolist(), separators=(",", ":")) if len(rows) > 1 else json.dumps(layer_values.mean(axis=0).round(8).tolist(), separators=(",", ":")),
        })
    with (out / "phase_profile_ci.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=profile_rows[0].keys())
        writer.writeheader()
        writer.writerows(profile_rows)

    # Phase shuffle: preserve each episode's phase counts, randomize assignment.
    rng = random.Random(20260713)
    shuffle_rows = []
    for shuffle_id in range(1000):
        values = {phase: [] for phase in PHASES}
        for episode_id, group in by_episode.items():
            ordered = sorted(group, key=lambda r: int(r["query_step"]))
            labels = [r["phase"] for r in derived if r["episode_id"] == episode_id]
            rng.shuffle(labels)
            for row, label in zip(ordered, labels):
                values[label].append(row)
        for phase in PHASES:
            rows = values[phase]
            shuffle_rows.append({"shuffle_id": shuffle_id, "phase": phase, "queries": len(rows), "mean_expected_depth": mean_flat([r.get("expected_depth") for r in rows]), "mean_entropy": mean_flat([skill_entropy(r) for r in rows])})
    with (out / "phase_shuffle_null.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=shuffle_rows[0].keys())
        writer.writeheader()
        writer.writerows(shuffle_rows)

    (out / "phase_rules.json").write_text(json.dumps({
        "source_trace": str(source),
        "rule_id": "posthoc_time_quartile_v1",
        "rule": "For each episode, normalized query_step/max_query_step split into four equal quartiles.",
        "phase_order": list(PHASES),
        "warning": "Temporal proxy only; not a geometric or semantic phase annotation.",
        "records": len(derived),
        "episodes": len(by_episode),
        "successes": sum(bool(r["success"]) for r in episode_rows),
        "failures": sum(not bool(r["success"]) for r in episode_rows),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps({
        "source_trace": str(source),
        "source_trace_sha256": source_sha256,
        "output_files": [
            "trace_with_phase.jsonl",
            "phase_summary.csv",
            "layer_weight_summary.csv",
            "episode_outcomes.csv",
            "episode_phase_summary.csv",
            "phase_profile_ci.csv",
            "phase_shuffle_null.csv",
            "phase_rules.json",
            "checksums.sha256",
        ],
        "episodes": len(by_episode),
        "queries": len(derived),
        "successes": sum(bool(r["success"]) for r in episode_rows),
        "failures": sum(not bool(r["success"]) for r in episode_rows),
        "layer_count": max(len((r.get("layer_weights") or [[]])[0]) for r in derived),
        "phase_rule_id": "posthoc_time_quartile_v1",
        "no_skill_control": "not_collected_in_P0_scope; do not interpret this manifest as a completed no-skill comparison",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    checksums = []
    for path in sorted(out.iterdir()):
        if path.name == "checksums.sha256" or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {path.name}")
    (out / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(by_episode), "queries": len(derived), "successes": sum(bool(r["success"]) for r in episode_rows), "failures": sum(not bool(r["success"]) for r in episode_rows), "output": str(out)}))


if __name__ == "__main__":
    main()

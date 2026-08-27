#!/usr/bin/env python3
"""Audit the latest soft-prototype routing metrics without changing a run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PREFIX = "[train_metrics] "


def latest_metrics(log_path: Path) -> dict:
    latest = None
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            marker = line.find(PREFIX)
            if marker >= 0:
                latest = json.loads(line[marker + len(PREFIX) :])
    if latest is None:
        raise RuntimeError(f"no {PREFIX.strip()} record found in {log_path}")
    return latest


def metric(metrics: dict, name: str) -> float:
    candidates = (name, f"VLA Train/{name}")
    for candidate in candidates:
        if candidate in metrics:
            value = float(metrics[candidate])
            if not math.isfinite(value):
                raise RuntimeError(f"{candidate} is not finite: {value}")
            return value
    raise RuntimeError(f"required metric {name!r} is absent")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-step", type=int, default=1000)
    args = parser.parse_args()

    metrics = latest_metrics(args.log)
    step = int(metrics.get("global_step", -1))
    if step < args.min_step:
        raise RuntimeError(f"latest optimizer step {step} is below gate {args.min_step}")

    values = {
        name: metric(metrics, name)
        for name in (
            "skill_max_prob",
            "skill_effective_count",
            "skill_usage_ema_max",
            "skill_template_max_cosine",
            "routing_layer_entropy",
            "routing_expected_depth",
            "routing_early3_mass",
            "routing_layer_batch_variation",
        )
    }
    failures = []
    if values["skill_max_prob"] >= 0.995:
        failures.append("per-sample skill posterior is effectively one-hot")
    if values["skill_usage_ema_max"] >= 0.90:
        failures.append("one latent control mode dominates the usage EMA")
    if values["skill_effective_count"] <= 1.10:
        failures.append("effective latent-control count has collapsed to one")
    if values["routing_early3_mass"] >= 0.95:
        failures.append("at least 95% of routing mass is confined to the first three depths")
    if values["skill_template_max_cosine"] >= 0.9999:
        failures.append("two routing prototypes are numerically identical")

    report = {
        "status": "fail" if failures else "pass",
        "global_step": step,
        "metrics": values,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

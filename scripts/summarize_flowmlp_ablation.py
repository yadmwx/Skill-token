#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def last_match(pattern, text, default=""):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiment_results/flowmlp_ablation.csv"))
    args = parser.parse_args()
    rows = []
    for log_dir in args.log_dirs:
        for path in sorted(log_dir.glob("flowmlp_ablation_*.eval.log")):
            match = re.match(r"flowmlp_ablation_(no_skill|direct_only|routing_only|routing_direct)_seed(\d+)_(.+)\.eval\.log", path.name)
            if not match:
                continue
            text = path.read_text(errors="replace")
            variant, seed, device = match.groups()
            episodes = last_match(r"Total episodes:\s*(\d+)", text)
            successes = last_match(r"Total successes:\s*(\d+)", text)
            rate = last_match(r"Overall success rate:\s*([0-9.]+)%", text)
            task_rates = re.findall(r"Current task success rate:\s*([0-9.]+)", text)
            rows.append({
                "variant": variant,
                "seed": seed,
                "device": device,
                "episodes": episodes,
                "successes": successes,
                "success_rate_percent": rate,
                "task_rates": ";".join(task_rates[-4:]),
                "log": str(path),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["variant", "seed", "device"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

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
    parser.add_argument(
        "--output", type=Path, default=Path("experiment_results/dit_prefix.csv")
    )
    args = parser.parse_args()

    rows = []
    for log_dir in args.log_dirs:
        for eval_log in sorted(log_dir.glob("DIT-prefix-*.eval.log")):
            match = re.match(
                r"DIT-prefix-(baseline|skill)-from-MLP5000-(.+)\.eval\.log",
                eval_log.name,
            )
            if not match:
                continue
            variant, device = match.groups()
            text = eval_log.read_text(errors="replace")
            status_path = eval_log.with_suffix("").with_suffix(".status")
            status = status_path.read_text(errors="replace").strip() if status_path.exists() else ""
            rows.append(
                {
                    "family": "dit_prefix",
                    "variant": variant,
                    "device": device,
                    "seed": last_match(r"seed[= ](\d+)", text, "7"),
                    "initialization": "MLP-5000",
                    "condition_injection": "action_expert_prefix",
                    "latent_skill_token": "true" if variant == "skill" else "false",
                    "episodes": last_match(r"Total episodes:\s*(\d+)", text),
                    "successes": last_match(r"Total successes:\s*(\d+)", text),
                    "success_rate_percent": last_match(
                        r"Overall success rate:[\s\S]{0,160}?\(([0-9.]+)%\)", text
                    ),
                    "status": status,
                    "eval_log": str(eval_log),
                    "status_file": str(status_path) if status_path.exists() else "",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family", "variant", "device", "seed", "initialization",
        "condition_injection", "latent_skill_token", "episodes", "successes",
        "success_rate_percent", "status", "eval_log", "status_file",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

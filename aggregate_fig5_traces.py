#!/usr/bin/env python3
"""Combine complete Figure 5 traces from independent seeds without filtering."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dirs", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    for directory in args.trace_dirs:
        source = Path(directory) / "trace.jsonl"
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seed = int(row.get("seed", -1))
            original_id = row["episode_id"]
            row["source_seed"] = seed
            row["source_episode_id"] = original_id
            row["episode_id"] = f"seed{seed}_{original_id}"
            seen.add(row["episode_id"])
            records.append(row)
    records.sort(key=lambda r: (int(r["source_seed"]), r["source_episode_id"], int(r["query_step"])))
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source_directories": [str(Path(d)) for d in args.trace_dirs],
        "episodes": len(seen),
        "queries": len(records),
        "seeds": sorted({int(r["source_seed"]) for r in records}),
        "successes": sum(bool(r["success"]) for r in {r["episode_id"]: r for r in records}.values()),
    }
    manifest["failures"] = manifest["episodes"] - manifest["successes"]
    (out / "aggregate_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checksums = []
    for path in sorted(out.iterdir()):
        if path.name == "checksums.sha256" or not path.is_file():
            continue
        checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (out / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit and aggregate independent fixed-layer LIBERO runs for E04."""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # Local paper/audit environments may only provide Pillow.
    plt = None


TASKS = ("task3", "task4", "task7", "task9")
TRIALS_PER_TASK = 10
MIN_EPISODES = 40
SEEDS_EXPECTED = (7, 8, 9)
# These are independently trained single hidden-state indices, not four-layer groups.
GROUPS = {
    1: "fixed index 1",
    5: "fixed index 5",
    9: "fixed index 9",
    13: "fixed index 13",
    24: "final index 24",
}
LOG_RE = re.compile(
    r"^flowmlp_ablation_routing_only_seed(?P<seed>\d+)_"
    r"(?P<device>.+?)_fixed(?P<fixed>\d+)(?P<suffix>.*?)\.(?P<kind>train|eval)\.log$"
)


def _last_int(pattern: str, text: str, default: int = 0) -> int:
    values = re.findall(pattern, text)
    return int(values[-1]) if values else default


def _last_float(pattern: str, text: str) -> float:
    values = re.findall(pattern, text, flags=re.S)
    return float(values[-1]) if values else math.nan


def _identity(path: Path, device_tag: str):
    match = LOG_RE.match(path.name)
    if not match or match.group("device") != device_tag:
        return None
    suffix = match.group("suffix").lstrip("_")
    return int(match.group("fixed")), int(match.group("seed")), suffix or "initial", match.group("kind")


def parse_eval(path: Path) -> dict:
    text = path.read_text(errors="replace")
    episodes = _last_int(r"Total episodes:\s*(\d+)", text)
    successes = _last_int(r"Total successes:\s*(\d+)", text)
    rates = [
        float(value)
        for value in re.findall(r"Current task success rate:.*?\n\s*([0-9.]+)", text, flags=re.S)
    ]
    outcomes = [value == "True" for value in re.findall(r"Success:\s*(True|False)", text)]
    complete = episodes >= MIN_EPISODES and len(outcomes) >= MIN_EPISODES and len(rates) >= len(TASKS)
    return {
        "episodes": episodes,
        "successes": successes,
        "overall_success_rate": _last_float(r"Overall success rate:.*?\n\s*([0-9.]+)", text),
        "task_rates": rates[-len(TASKS) :],
        "outcomes": outcomes,
        "complete": complete,
        "status": "Complete" if complete else "Incomplete",
    }


def discover_attempts(log_dir: Path, device_tag: str) -> list[dict]:
    attempts: dict[tuple[int, int, str], dict] = {}
    for path in sorted(log_dir.glob("flowmlp_ablation_routing_only_seed*_fixed*.log")):
        identity = _identity(path, device_tag)
        if identity is None:
            continue
        fixed, seed, attempt, kind = identity
        key = fixed, seed, attempt
        record = attempts.setdefault(
            key,
            {
                "fixed_layer_index": fixed,
                "seed": seed,
                "attempt": attempt,
                "device_tag": device_tag,
                "train_log": "",
                "eval_log": "",
                "mtime_ns": 0,
            },
        )
        record[f"{kind}_log"] = str(path)
        record["mtime_ns"] = max(record["mtime_ns"], path.stat().st_mtime_ns)
    for record in attempts.values():
        if record["eval_log"]:
            record.update(parse_eval(Path(record["eval_log"])))
        else:
            record.update(
                episodes=0,
                successes=0,
                overall_success_rate=math.nan,
                task_rates=[],
                outcomes=[],
                complete=False,
                status="TrainOnly",
            )
        record["formal_eligible"] = bool(
            record["complete"] and record["fixed_layer_index"] in GROUPS
        )
        record["audit_only"] = bool(
            record["complete"] and record["fixed_layer_index"] not in GROUPS
        )
        record["canonical"] = False

    # Exactly one latest complete attempt is canonical for every fixed_layer/seed.
    # Planned formal depths are distinguished from retained audit-only depths below.
    by_config: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in attempts.values():
        if record["complete"]:
            by_config[(record["fixed_layer_index"], record["seed"])].append(record)
    for candidates in by_config.values():
        max(candidates, key=lambda r: (r["mtime_ns"], r["attempt"]))["canonical"] = True
    return sorted(attempts.values(), key=lambda r: (r["fixed_layer_index"], r["seed"], r["attempt"]))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _heat_color(value: float) -> tuple[int, int, int]:
    stops = [
        (0.00, (255, 255, 217)),
        (0.25, (199, 233, 180)),
        (0.50, (127, 205, 187)),
        (0.75, (44, 127, 184)),
        (1.00, (8, 29, 88)),
    ]
    value = min(1.0, max(0.0, float(value)))
    for (left_x, left), (right_x, right) in zip(stops, stops[1:]):
        if value <= right_x:
            ratio = (value - left_x) / (right_x - left_x)
            return tuple(round(a + ratio * (b - a)) for a, b in zip(left, right))
    return stops[-1][1]


def _render_heatmap_fallback(heat: np.ndarray, out: Path) -> None:
    from html import escape
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 1120
    left, top, cell_w, cell_h = 360, 170, 280, 150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    font_path = next((path for path in font_candidates if path.exists()), None)

    def font(size: int):
        return ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default()

    title_font, label_font, cell_font = font(46), font(34), font(34)
    title = "Figure 3(b) - representative fixed-index comparison"
    draw.text((width / 2, 55), title, fill="black", font=title_font, anchor="mm")

    for row, label in enumerate(GROUPS.values()):
        y = top + row * cell_h
        draw.text((left - 24, y + cell_h / 2), label, fill="black", font=label_font, anchor="rm")
        for col, task in enumerate(TASKS):
            x = left + col * cell_w
            value = float(heat[row, col])
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=_heat_color(value), outline="white", width=2)
            text_color = "white" if value >= 0.72 else "black"
            draw.text((x + cell_w / 2, y + cell_h / 2), f"{100 * value:.0f}%", fill=text_color, font=cell_font, anchor="mm")

    for col, task in enumerate(TASKS):
        x = left + col * cell_w + cell_w / 2
        draw.text((x, top + len(GROUPS) * cell_h + 30), task, fill="black", font=label_font, anchor="ma")
    draw.text((left + len(TASKS) * cell_w / 2, height - 85), "LIBERO Object task", fill="black", font=label_font, anchor="mm")
    image.save(out / "figure3b_fixed_layer_heatmap.png", dpi=(300, 300))

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="72" text-anchor="middle" font-family="Arial,sans-serif" font-size="46">{escape(title)}</text>',
    ]
    for row, label in enumerate(GROUPS.values()):
        y = top + row * cell_h
        svg_parts.append(
            f'<text x="{left-24}" y="{y+cell_h/2+12}" text-anchor="end" font-family="Arial,sans-serif" font-size="34">{escape(label)}</text>'
        )
        for col in range(len(TASKS)):
            x = left + col * cell_w
            value = float(heat[row, col])
            red, green, blue = _heat_color(value)
            text_color = "white" if value >= 0.72 else "black"
            svg_parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="rgb({red},{green},{blue})" stroke="white" stroke-width="2"/>',
                    f'<text x="{x+cell_w/2}" y="{y+cell_h/2+12}" text-anchor="middle" fill="{text_color}" font-family="Arial,sans-serif" font-size="34">{100*value:.0f}%</text>',
                ]
            )
    for col, task in enumerate(TASKS):
        x = left + col * cell_w + cell_w / 2
        svg_parts.append(
            f'<text x="{x}" y="{top+len(GROUPS)*cell_h+62}" text-anchor="middle" font-family="Arial,sans-serif" font-size="34">{escape(task)}</text>'
        )
    svg_parts.append(
        f'<text x="{left+len(TASKS)*cell_w/2}" y="{height-75}" text-anchor="middle" font-family="Arial,sans-serif" font-size="34">LIBERO Object task</text>'
    )
    svg_parts.append("</svg>")
    (out / "figure3b_fixed_layer_heatmap.svg").write_text("\n".join(svg_parts) + "\n", encoding="utf-8")


def build_outputs(attempts: list[dict], out: Path, device_tag: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    registry = []
    task_rows = []
    episode_rows = []
    for record in attempts:
        checkpoint = (
            f"outputs/FlowMLP-ablation-routing_only-seed{record['seed']}-{device_tag}-"
            f"fixed{record['fixed_layer_index']}--10000_chkpt"
        )
        registry.append(
            {
                **record,
                "group": GROUPS.get(
                    record["fixed_layer_index"], f"audit fixed index {record['fixed_layer_index']}"
                ),
                "checkpoint": checkpoint,
            }
        )
        for task_index, task in enumerate(TASKS):
            start = task_index * TRIALS_PER_TASK
            stop = start + TRIALS_PER_TASK
            task_outcomes = record["outcomes"][start:stop]
            task_episodes = len(task_outcomes)
            task_successes = int(sum(task_outcomes))
            rate = task_successes / task_episodes if task_episodes else math.nan
            logged_rate = (
                record["task_rates"][task_index]
                if task_index < len(record["task_rates"])
                else math.nan
            )
            task_rows.append(
                {
                    "fixed_layer_index": record["fixed_layer_index"],
                    "group": GROUPS.get(
                        record["fixed_layer_index"], f"audit fixed index {record['fixed_layer_index']}"
                    ),
                    "seed": record["seed"],
                    "task": task,
                    "attempt": record["attempt"],
                    "canonical": record["canonical"],
                    "formal_eligible": record["formal_eligible"],
                    "audit_only": record["audit_only"],
                    "episodes": task_episodes,
                    "successes": task_successes,
                    "task_success_rate": rate,
                    "logged_task_success_rate": logged_rate,
                    "config_episodes": record["episodes"],
                    "config_successes": record["successes"],
                    "status": record["status"],
                    "raw_log": record["eval_log"],
                }
            )
        for index, success in enumerate(record["outcomes"]):
            task_index = min(index // TRIALS_PER_TASK, len(TASKS) - 1)
            episode_rows.append(
                {
                    "fixed_layer_index": record["fixed_layer_index"],
                    "seed": record["seed"],
                    "task": TASKS[task_index],
                    "attempt": record["attempt"],
                    "episode_index": index % TRIALS_PER_TASK,
                    "success": success,
                    "canonical": record["canonical"],
                    "formal_eligible": record["formal_eligible"],
                    "audit_only": record["audit_only"],
                    "raw_log": record["eval_log"],
                }
            )

    # Formal rows are deduplicated by fixed_layer, seed, task through canonical attempts only.
    formal = [
        row
        for row in task_rows
        if row["canonical"] and row["formal_eligible"] and row["fixed_layer_index"] in GROUPS
    ]
    keys = [(r["fixed_layer_index"], r["seed"], r["task"]) for r in formal]
    if len(keys) != len(set(keys)):
        raise RuntimeError("canonical aggregation key is not unique")

    registry_fields = [
        "fixed_layer_index", "group", "seed", "attempt", "device_tag", "canonical", "formal_eligible",
        "audit_only", "complete", "status", "episodes", "successes", "overall_success_rate", "checkpoint",
        "train_log", "eval_log",
    ]
    write_csv(out / "run_attempt_registry.csv", registry, registry_fields)
    write_csv(out / "task_layer_attempts.csv", task_rows, list(task_rows[0]) if task_rows else ["fixed_layer_index"])
    write_csv(out / "task_layer_runs.csv", formal, list(formal[0]) if formal else list(task_rows[0]))
    audit_task_rows = [row for row in task_rows if row["canonical"] and row["audit_only"]]
    write_csv(
        out / "audit_task_layer_runs.csv",
        audit_task_rows,
        list(task_rows[0]) if task_rows else ["fixed_layer_index"],
    )
    formal_episode_rows = [
        row for row in episode_rows if row["canonical"] and row["formal_eligible"]
    ]
    audit_episode_rows = [row for row in episode_rows if row["canonical"] and row["audit_only"]]
    write_csv(
        out / "episode_results.csv",
        formal_episode_rows,
        list(episode_rows[0]) if episode_rows else ["fixed_layer_index", "seed", "task", "attempt"],
    )
    write_csv(
        out / "episode_attempts.csv",
        episode_rows,
        list(episode_rows[0]) if episode_rows else ["fixed_layer_index", "seed", "task", "attempt"],
    )
    write_csv(
        out / "audit_episode_results.csv",
        audit_episode_rows,
        list(episode_rows[0]) if episode_rows else ["fixed_layer_index", "seed", "task", "attempt"],
    )

    matrix = []
    for fixed, group in GROUPS.items():
        for task in TASKS:
            values = [r["task_success_rate"] for r in formal if r["fixed_layer_index"] == fixed and r["task"] == task]
            mean = float(np.mean(values)) if values else math.nan
            seed_sd = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            matrix.append(
                {
                    "fixed_layer_index": fixed,
                    "group": group,
                    "task": task,
                    "n_runs": len(values),
                    "mean_success_rate": mean,
                    "seed_sd": seed_sd,
                    "seed_min": float(np.min(values)) if values else math.nan,
                    "seed_max": float(np.max(values)) if values else math.nan,
                }
            )
    write_csv(out / "task_layer_matrix.csv", matrix, list(matrix[0]))

    fixed_depth_summary = []
    formal_configs = [
        row for row in registry if row["canonical"] and row["formal_eligible"]
    ]
    for fixed, label in GROUPS.items():
        configs = [row for row in formal_configs if row["fixed_layer_index"] == fixed]
        rates = [float(row["overall_success_rate"]) for row in configs]
        fixed_depth_summary.append(
            {
                "fixed_layer_index": fixed,
                "label": label,
                "n_seeds": len(rates),
                "episodes": int(sum(row["episodes"] for row in configs)),
                "successes": int(sum(row["successes"] for row in configs)),
                "mean_success_rate": float(np.mean(rates)) if rates else math.nan,
                "seed_sd": float(np.std(rates, ddof=1)) if len(rates) > 1 else math.nan,
                "seed_min": float(np.min(rates)) if rates else math.nan,
                "seed_max": float(np.max(rates)) if rates else math.nan,
            }
        )
    write_csv(out / "fixed_depth_summary.csv", fixed_depth_summary, list(fixed_depth_summary[0]))

    heat = np.asarray([[r["mean_success_rate"] for r in matrix if r["fixed_layer_index"] == fixed] for fixed in GROUPS])
    if plt is None:
        _render_heatmap_fallback(heat, out)
    else:
        fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
        image = ax.imshow(heat, vmin=0.0, vmax=1.0, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(TASKS)), TASKS)
        ax.set_yticks(range(len(GROUPS)), list(GROUPS.values()))
        ax.set_xlabel("LIBERO Object task")
        ax.set_ylabel("independently trained hidden-state index")
        ax.set_title("Figure 3(b) - representative fixed-index comparison")
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                if np.isfinite(heat[i, j]):
                    text_color = "white" if heat[i, j] >= 0.72 else "black"
                    ax.text(
                        j,
                        i,
                        f"{100 * heat[i, j]:.0f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=10,
                    )
        fig.colorbar(image, ax=ax, label="success rate")
        fig.savefig(out / "figure3b_fixed_layer_heatmap.png", dpi=300)
        fig.savefig(out / "figure3b_fixed_layer_heatmap.svg")
        plt.close(fig)

    canonical_configs = {(r["fixed_layer_index"], r["seed"]) for r in formal}
    missing = [
        {"fixed_layer_index": fixed, "seed": seed}
        for fixed in GROUPS
        for seed in SEEDS_EXPECTED
        if (fixed, seed) not in canonical_configs
    ]
    manifest = {
        "experiment": "E04",
        "protocol": f"{device_tag}-only; independent fixed-layer training; minimum {MIN_EPISODES} evaluation trajectories",
        "device_tag": device_tag,
        "fixed_groups": GROUPS,
        "final_fixed_layer_index": 24,
        "legacy_nonfinal_index_retained_for_audit": 16,
        "tasks": TASKS,
        "seeds_expected": SEEDS_EXPECTED,
        "attempt_count": len(attempts),
        "canonical_formal_config_count": len(canonical_configs),
        "expected_formal_config_count": len(GROUPS) * len(SEEDS_EXPECTED),
        "formal_episode_count": len(formal_episode_rows),
        "canonical_audit_config_count": len(
            {(r["fixed_layer_index"], r["seed"]) for r in registry if r["canonical"] and r["audit_only"]}
        ),
        "audit_episode_count": len(audit_episode_rows),
        "missing_formal_configs": missing,
        "deduplication_key": ["fixed_layer_index", "seed", "task"],
        "canonical_rule": "latest attempt with at least 40 parsed evaluation trajectories",
        "outputs": [
            "run_attempt_registry.csv", "task_layer_attempts.csv", "task_layer_runs.csv", "episode_results.csv",
            "episode_attempts.csv", "audit_task_layer_runs.csv", "audit_episode_results.csv",
            "task_layer_matrix.csv", "fixed_depth_summary.csv", "figure3b_fixed_layer_heatmap.png",
            "figure3b_fixed_layer_heatmap.svg",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checksums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (out / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device_tag", default="a100")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempts = discover_attempts(args.log_dir, args.device_tag)
    if not attempts:
        raise SystemExit("No E04 logs found")
    manifest = build_outputs(attempts, args.output_dir, args.device_tag)
    print(json.dumps({"attempts": len(attempts), "formal_configs": manifest["canonical_formal_config_count"], "missing": manifest["missing_formal_configs"]}))


if __name__ == "__main__":
    main()

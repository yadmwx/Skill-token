"""Build the auditable E07 comparison tables from canonical evaluator logs only."""
from __future__ import annotations

import argparse, csv, hashlib, json, re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

VARIANTS = {"no_skill": "R0", "continuous_context": "R1", "routing_only": "R2"}
TASKS = ["task3", "task4", "task7", "task9"]
TASK_TEXT = {
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket": "task3",
    "pick_up_the_ketchup_and_place_it_in_the_basket": "task4",
    "pick_up_the_milk_and_place_it_in_the_basket": "task7",
    "pick_up_the_orange_juice_and_place_it_in_the_basket": "task9",
}

def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    logs = args.raw_root / "train_logs"
    episodes, registry = [], []
    for key, label in VARIANTS.items():
        for seed in (7, 8, 9):
            candidates = sorted(logs.glob(f"flowmlp_ablation_{key}_seed{seed}_v48*.eval.log"))
            if not candidates: raise SystemExit(f"missing canonical eval log: {key} seed{seed}")
            log = candidates[-1]
            text = log.read_text(errors="replace")
            hits = re.findall(r"episode=(\d+)--success=(True|False)--task=([^\s]+)\.mp4", text)
            if len(hits) != 40: raise SystemExit(f"{log}: expected 40 episode records, got {len(hits)}")
            train = Path(str(log).replace(".eval.log", ".train.log"))
            p = re.search(r"# trainable params in action_head:\s*(\d+)", train.read_text(errors="replace")) if train.exists() else None
            registry.append({"variant": label, "seed": seed, "attempt": "legacy_seed8" if seed == 8 and key != "continuous_context" else "e07_canonical", "canonical": True, "eval_log": str(log), "train_log": str(train), "episode_count": 40, "action_head_trainable_params": p.group(1) if p else ""})
            for ep, success, task_text in hits:
                task = TASK_TEXT.get(task_text)
                if task is None:
                    # The evaluator truncates long rollout filenames at 100 chars.
                    task = next((v for k, v in TASK_TEXT.items() if k.startswith(task_text)), None)
                if task is None: raise SystemExit(f"unknown task text {task_text}")
                episodes.append({"variant": label, "seed": seed, "task": task, "episode": int(ep), "success": int(success == "True"), "attempt": registry[-1]["attempt"], "eval_log": str(log)})
    write_csv(args.out / "per_episode_results.csv", episodes)
    write_csv(args.out / "run_attempt_registry.csv", registry)
    by = defaultdict(list)
    for r in episodes: by[(r["variant"], r["seed"], r["task"])].append(r["success"])
    task_rows = []
    for (v,s,t), xs in sorted(by.items()): task_rows.append({"variant":v,"seed":s,"task":t,"episodes":len(xs),"successes":sum(xs),"success_rate":sum(xs)/len(xs)})
    write_csv(args.out / "task_variant_matrix.csv", task_rows)
    paired = []
    for task in TASKS:
        for seed in (7, 8, 9):
            r0 = sum(by[("R0", seed, task)]) / len(by[("R0", seed, task)])
            r1 = sum(by[("R1", seed, task)]) / len(by[("R1", seed, task)])
            r2 = sum(by[("R2", seed, task)]) / len(by[("R2", seed, task)])
            paired.append({"task": task, "seed": seed, "R0": r0, "R1": r1, "R2": r2, "R2_minus_R0": r2-r0, "R2_minus_R1": r2-r1})
    write_csv(args.out / "paired_task_differences.csv", paired)
    totals = defaultdict(list)
    for (v,s,_), xs in by.items(): totals[(v,s)].extend(xs)
    summary=[]
    for v in ("R0","R1","R2"):
        rates=[sum(totals[(v,s)])/len(totals[(v,s)]) for s in (7,8,9)]
        summary.append({"variant":v,"seeds":"7,8,9","total_episodes":sum(len(totals[(v,s)]) for s in (7,8,9)),"total_successes":sum(sum(totals[(v,s)]) for s in (7,8,9)),"mean_success_rate":mean(rates),"std_success_rate":stdev(rates),"seed7":rates[0],"seed8":rates[1],"seed9":rates[2]})
    rmap={r["variant"]:r for r in summary}
    for r in summary:
        r["R2_minus_R0"] = rmap["R2"]["mean_success_rate"]-rmap["R0"]["mean_success_rate"] if r["variant"]=="R2" else ""
        r["R2_minus_R1"] = rmap["R2"]["mean_success_rate"]-rmap["R1"]["mean_success_rate"] if r["variant"]=="R2" else ""
    write_csv(args.out / "controlled_context_ablation.csv", summary)
    params=[]
    for v in ("R0","R1","R2"):
        vals=sorted({x["action_head_trainable_params"] for x in registry if x["variant"]==v and x["action_head_trainable_params"]})
        params.append({"variant":v,"action_head_trainable_params": ";".join(vals),"routing_only_direct_action_conditioning":"False" if v in ("R1","R2") else "not_applicable"})
    write_csv(args.out / "parameter_count.csv", params)
    manifest={"protocol":"E07_context_control","variants":{"R0":"m_l only router; proprio retained by shared decoder","R1":"m_l,g,u continuous context router","R2":"m_l,g,u discrete skill router; direct action conditioning disabled"},"formal_episodes":len(episodes),"formal_configs":9,"tasks":TASKS,"seeds":[7,8,9],"canonical_only":True}
    (args.out / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    checks=[]
    for p in sorted(args.out.glob("*.csv")) + [args.out/"manifest.json"]:
        checks.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (args.out/"checksums.sha256").write_text("\n".join(checks)+"\n",encoding="utf-8")

if __name__ == "__main__": main()

#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
PY=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python
cd "$REPO_DIR"
mkdir -p experiment_results/skill_depth/E00 experiment_results/skill_depth/E01

# Never compete with an existing formal CALVIN evaluation.
while pgrep -f '[e]valuate_calvin.py' >/dev/null; do
  sleep 30
done

CKPT=outputs/FlowMLP-ablation-routing_only-seed9-a100--10000_chkpt
COMMON=(
  --pretrained_checkpoint "$CKPT" --action_head_type FlowMLP
  --task_suite_name libero_object --task_ids 3 --num_trials_per_task 2
  --use_depth_interface False --depth_interface_mode none
  --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1
  --flowmlp_use_latent_skill_token True --flowmlp_skill_use_layer_routing True
  --flowmlp_skill_use_direct_conditioning False --flowmlp_num_skill_tokens 16
  --flowmlp_skill_token_dim 128 --flowmlp_skill_temperature 1.0
  --flowmlp_skill_entropy_weight 0.0 --flowmlp_num_inference_steps 5
  --flowmlp_num_inference_samples 8 --flowmlp_supervised_anchor_weight 0.0
  --flowmlp_anchor_blend 0.0 --flowmlp_detach_flow_conditioning False
  --use_proprio True --num_images_in_input 2 --use_film False --use_minivlm True
  --use_pro_version True --use_wandb False --center_crop True --seed 9
)

# E00: same seed/initial states, diagnostic on versus off.
"$PY" experiments/robot/libero/run_libero_eval.py "${COMMON[@]}" \
  --run_id_note E00_trace --routing_trace_enabled True \
  --routing_trace_path experiment_results/skill_depth/E00/trace_enabled.jsonl \
  --routing_action_capture_path experiment_results/skill_depth/E00/actions_enabled.jsonl \
  > experiment_results/skill_depth/E00/eval_enabled.log 2>&1

"$PY" experiments/robot/libero/run_libero_eval.py "${COMMON[@]}" \
  --run_id_note E00_no_trace --routing_trace_enabled False \
  --routing_action_capture_path experiment_results/skill_depth/E00/actions_disabled.jsonl \
  > experiment_results/skill_depth/E00/eval_disabled.log 2>&1

"$PY" - <<'PY'
import json
import numpy as np
from pathlib import Path

def read(path):
    return [json.loads(line)["actions"] for line in Path(path).read_text().splitlines() if line.strip()]

a = read("experiment_results/skill_depth/E00/actions_enabled.jsonl")
b = read("experiment_results/skill_depth/E00/actions_disabled.jsonl")
same = len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
result = {"enabled_queries": len(a), "disabled_queries": len(b), "actions_equal": bool(same)}
Path("experiment_results/skill_depth/E00/equality_check.json").write_text(json.dumps(result, indent=2) + "\n")
print(result)
if not same:
    raise SystemExit("E00 failed: diagnostic switch changed captured actions")
PY

# E01: ten-trial pilot; trace contains every query and keeps failures.
COMMON10=("${COMMON[@]}")
for ((i = 0; i < ${#COMMON10[@]}; i++)); do
  if [[ "${COMMON10[$i]}" == "--num_trials_per_task" ]]; then
    COMMON10[$((i + 1))]=10
  fi
done
"$PY" experiments/robot/libero/run_libero_eval.py "${COMMON10[@]}" \
  --run_id_note E01_pilot --routing_trace_enabled True \
  --routing_trace_path experiment_results/skill_depth/E01/trace.jsonl \
  > experiment_results/skill_depth/E01/eval.log 2>&1

date --iso-8601=seconds > experiment_results/skill_depth/E01/complete.marker

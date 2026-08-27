#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
PY=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python
cd "$REPO_DIR"
mkdir -p experiment_results/skill_depth/Figure5_pilot_expanded
OUT_DIR=${OUT_DIR:-experiment_results/skill_depth/Figure5_pilot_expanded}
CKPT=${CKPT:-outputs/FlowMLP-ablation-routing_only-seed9-a100--10000_chkpt}
SEED=${SEED:-9}
TASK_IDS=${TASK_IDS:-3,4,7,9}
TRIALS=${TRIALS:-10}
RUN_NOTE=${RUN_NOTE:-Figure5_pilot_expanded}
mkdir -p "$OUT_DIR"
"$PY" experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CKPT" --action_head_type FlowMLP \
  --task_suite_name libero_object --task_ids "$TASK_IDS" --num_trials_per_task "$TRIALS" \
  --use_depth_interface False --depth_interface_mode none --use_adaptive_bridge True \
  --bridge_mode adaptive --fixed_layer_index -1 --flowmlp_use_latent_skill_token True \
  --flowmlp_skill_use_layer_routing True --flowmlp_skill_use_direct_conditioning False \
  --flowmlp_num_skill_tokens 16 --flowmlp_skill_token_dim 128 --flowmlp_skill_temperature 1.0 \
  --flowmlp_skill_entropy_weight 0.0 --flowmlp_num_inference_steps 5 --flowmlp_num_inference_samples 8 \
  --flowmlp_supervised_anchor_weight 0.0 --flowmlp_anchor_blend 0.0 \
  --flowmlp_detach_flow_conditioning False --use_proprio True --num_images_in_input 2 \
  --use_film False --use_minivlm True --use_pro_version True --use_wandb False \
  --center_crop True --seed "$SEED" --run_id_note "$RUN_NOTE" \
  --routing_trace_enabled True \
  --routing_trace_path "$OUT_DIR/trace.jsonl" \
  > "$OUT_DIR/eval.log" 2>&1
date --iso-8601=seconds > "$OUT_DIR/complete.marker"

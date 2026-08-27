#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/e07_a100_src
ROOT=/root/autodl-tmp/e07_a100_src
VLM=/root/autodl-tmp/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b
DATA=/root/autodl-tmp/e07_a100_data
CFG=/root/autodl-tmp/e07_a100_hf/e07_base_ckpt
run_eval() {
  local seed="$1"; local rid="$2"; local out="$3"
  env LIBERO_CONFIG_PATH=/root/.libero_a100 PYTHONPATH=.:/root/autodl-tmp/e07_a100_src/LIBERO PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /root/miniconda3/bin/python experiments/robot/libero/run_libero_eval.py \
   --pretrained_checkpoint "$out/outputs/$rid" --action_head_type FlowMLP --task_suite_name libero_object --task_ids 3,4,7,9 \
   --num_trials_per_task 10 --use_depth_interface False --depth_interface_mode none --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
   --flowmlp_use_latent_skill_token True --flowmlp_use_continuous_context False --flowmlp_skill_use_layer_routing True \
   --flowmlp_skill_use_direct_conditioning False --flowmlp_continuous_context_use_direct_conditioning False \
   --flowmlp_num_skill_tokens 16 --flowmlp_skill_token_dim 128 --flowmlp_skill_temperature 1.0 --flowmlp_skill_entropy_weight 0.0 \
   --flowmlp_num_inference_steps 5 --flowmlp_num_inference_samples 8 --flowmlp_anchor_blend 0.0 --flowmlp_detach_flow_conditioning False \
   --use_proprio True --num_images_in_input 2 --use_film False --use_minivlm True --use_pro_version True --use_wandb False \
   --center_crop True --seed "$seed" > "$ROOT/$out/eval.log" 2>&1 || true
}
run_train() {
  local seed="$1"; local rid="$2"; local out="$3"
  mkdir -p "$ROOT/$out"
  nohup /root/miniconda3/bin/torchrun --standalone --nnodes 1 --nproc_per_node 1 vla-scripts/finetune.py \
   --config_file_path "$CFG" --vlm_path "$VLM" --data_root_dir "$DATA" --dataset_name libero_object_no_noops \
   --run_root_dir "$ROOT/$out/outputs" --train_from vlm_base --vlm_training freeze_lora --action_head_type FlowMLP \
   --use_lora True --lora_rank 64 --lora_dropout 0.0 --freeze_vlm True --merge_lora_during_training False \
   --use_minivlm True --use_pro_version True --use_proprio True --num_images_in_input 2 --use_film False \
   --learning_rate 0.0002 --use_constant_lr True --lr_warmup_steps 200 --max_steps 10000 --save_freq 5000 \
   --save_latest_checkpoint_only True --image_aug True --flowmlp_use_latent_skill_token True --flowmlp_use_continuous_context False \
   --flowmlp_skill_use_layer_routing True --flowmlp_skill_use_direct_conditioning False --flowmlp_continuous_context_use_direct_conditioning False \
   --flowmlp_num_skill_tokens 16 --flowmlp_skill_token_dim 128 --flowmlp_skill_temperature 1.0 --flowmlp_skill_entropy_weight 0.0 \
   --flowmlp_num_inference_steps 5 --flowmlp_num_inference_samples 8 --flowmlp_supervised_anchor_weight 0.0 \
   --flowmlp_anchor_blend 0.0 --flowmlp_detach_flow_conditioning False --use_adaptive_bridge True --bridge_mode adaptive \
   --fixed_layer_index -1 --ddp_find_unused_params True --seed "$seed" --run_id_note "$rid" --run_id_override "$rid" \
   --batch_size 8 --grad_accumulation_steps 2 > "$ROOT/$out/train.log" 2>&1 &
  echo $! > "$ROOT/$out/train.pid"
}
# Wait for currently running canonical seed7 training to publish provenance.
S7=clean_figures_seed7_retry2; R7=FigureClean-FlowMLP-routing_only-seed7-a100-10000-retry2
while [ ! -f "$ROOT/$S7/outputs/$R7/training_provenance.json" ]; do sleep 60; done
# The existing seed7 watcher is replaced by this one; run exactly one evaluation.
if ! grep -q 'Total episodes: *40' "$ROOT/$S7/eval.log" 2>/dev/null; then
  run_eval 7 "$R7" "$S7"
fi
# Do not start the next training unless seed7 evaluation produced all 40 episodes.
if ! grep -q 'Total episodes: *40' "$ROOT/$S7/eval.log" 2>/dev/null; then
  echo 'seed7 evaluation incomplete; stopping supervisor' >&2; exit 2
fi
S8=clean_figures_seed8_retry2; R8=FigureClean-FlowMLP-routing_only-seed8-a100-10000-retry2
if [ ! -f "$ROOT/$S8/outputs/$R8/training_provenance.json" ]; then
  run_train 8 "$R8" "$S8"
fi
while [ ! -f "$ROOT/$S8/outputs/$R8/training_provenance.json" ]; do sleep 60; done
if ! grep -q 'Total episodes: *40' "$ROOT/$S8/eval.log" 2>/dev/null; then
  run_eval 8 "$R8" "$S8"
fi
if ! grep -q 'Total episodes: *40' "$ROOT/$S8/eval.log" 2>/dev/null; then
  echo 'seed8 evaluation incomplete' >&2; exit 3
fi
echo 'E07 seed7 and seed8 training/evaluation complete'

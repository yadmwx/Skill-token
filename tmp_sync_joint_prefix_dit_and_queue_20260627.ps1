param(
    [string]$Remote = "xiaguanxiao@100.106.143.20",
    [string]$RemoteRepo = "/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation"
)

$ErrorActionPreference = "Stop"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

$files = @(
    "prismatic/models/ditx_blocks.py",
    "prismatic/models/ditx_vla_adapter.py",
    "prismatic/models/action_heads.py",
    "prismatic/models/flow_matching_head.py",
    "vla-scripts/finetune.py",
    "eval.sh",
    "experiments/robot/openvla_utils.py",
    "experiments/robot/libero/run_libero_eval.py",
    "scripts/tmp_smoke_dit_condition_injection_20260627.py",
    "scripts/tmp_probe_offline_action_error_20260627.py",
    "scripts/tmp_queue_dit12_mlp_anchorinit_prompt_taskonly_20260627.sh"
)

foreach ($file in $files) {
    Invoke-NativeChecked -FilePath scp -Arguments @("-o", "ConnectTimeout=20", $file, "${Remote}:$RemoteRepo/$file")
}

$remoteCmd = @'
set -euo pipefail
cd '$RemoteRepo'
PY=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
$PY -m py_compile \
  prismatic/models/ditx_blocks.py \
  prismatic/models/ditx_vla_adapter.py \
  prismatic/models/action_heads.py \
  prismatic/models/flow_matching_head.py \
  vla-scripts/finetune.py \
  experiments/robot/openvla_utils.py \
  experiments/robot/libero/run_libero_eval.py \
  scripts/tmp_smoke_dit_condition_injection_20260627.py \
  scripts/tmp_probe_offline_action_error_20260627.py
$PY scripts/tmp_smoke_dit_condition_injection_20260627.py
chmod +x scripts/tmp_queue_dit12_mlp_anchorinit_prompt_taskonly_20260627.sh
cat > scripts/tmp_queue_dit_recovery_joint_prefix_20260627.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export TOKENIZERS_PARALLELISM=false

QUEUE_LOG=train_logs/queue_dit_recovery_joint_prefix_20260627.log
OLD_RUN_ID=configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-mlp-anchorinit-prompt-taskonly-1000from5000-20260627
EVAL_GPU_MEM_LIMIT="${EVAL_GPU_MEM_LIMIT:-65000}"
TRAIN_GPU_MEM_LIMIT="${TRAIN_GPU_MEM_LIMIT:-13000}"

wait_for_gpu() {
  local limit="$1"
  local label="$2"
  echo "[recovery] waiting for GPU label=${label} limit=${limit} $(date '+%F %T')" >> "$QUEUE_LOG"
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    active=$(ps -eo args= | awk '/[f]inetune.py|[r]un_libero_eval.py/ {n++} END {print n + 0}')
    echo "[recovery] $(date '+%F %T') label=${label} gpu_used=${used} util=${util} active=${active}" >> "$QUEUE_LOG"
    if [ "$active" -eq 0 ] && [ "$used" -lt "$limit" ] && [ "$util" -lt 20 ]; then
      break
    fi
    sleep 60
  done
}

eval_gripper_fixed_5500() {
  local ckpt="outputs/${OLD_RUN_ID}--5500_chkpt"
  [ -d "$ckpt" ] || { echo "[recovery] missing old 5500 checkpoint: $ckpt" >> "$QUEUE_LOG"; return 0; }
  for seed in 0 3 7; do
    local log="train_logs/dit12_gripper_fixed_existing5500_seed${seed}_eval3_20260627.log"
    echo "[recovery] gripper_fixed_existing5500 seed=${seed} $(date '+%F %T')" >> "$QUEUE_LOG"
    ./eval.sh "$ckpt" \
      --task_suite libero_object --task_ids 0 --num_trials 3 --action_head_type DIT \
      --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples 1 \
      --dit_supervised_anchor_weight 1.0 --dit_anchor_blend 0.0 --dit_disable_inference_anchor False --dit_pure_inference False \
      --dit_anchor_gripper_weight 1.0 --dit_anchor_gripper_bce_weight 0.0 \
      --dit_gripper_head_weight 1.0 --dit_gripper_head_override True \
      --dit_condition_mode task_only --dit_condition_injection_mode cross_attn \
      --dit_include_prompt_tokens True --dit_task_token_mode vision_prompt \
      --dit_clip_normalized_actions False --debug_dit_group_action_tokens_to_chunk False \
      --dit_zero_init_adaln False --dit_zero_init_output False \
      --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
      --seed "$seed" > "$log" 2>&1
    cat "$log" >> "$QUEUE_LOG"
    successes=$(grep -E "Total successes:" "$log" | tail -1 | sed -E 's/.*Total successes:[[:space:]]*([0-9]+).*/\1/' || true)
    echo "[recovery] gripper_fixed_existing5500 seed=${seed} successes=${successes:-NA}/3" >> "$QUEUE_LOG"
  done
}

echo "[recovery] start $(date '+%F %T')" >> "$QUEUE_LOG"
wait_for_gpu "$EVAL_GPU_MEM_LIMIT" "eval"
eval_gripper_fixed_5500
wait_for_gpu "$TRAIN_GPU_MEM_LIMIT" "train"
env DIT_CONDITION_INJECTION_MODE=joint_prefix DIT_TASK_TOKEN_MODE=vision_prompt DIT_CONDITION_MODE=task_only RUN_OFFLINE_PROBE=True GPU_IDLE_MEM_LIMIT="$TRAIN_GPU_MEM_LIMIT" DEBUG_DIT_GROUP_ACTION_TOKENS_TO_CHUNK=False DIT_CLIP_NORMALIZED_ACTIONS=False bash scripts/tmp_queue_dit12_mlp_anchorinit_prompt_taskonly_20260627.sh
echo "[recovery] all done $(date '+%F %T')" >> "$QUEUE_LOG"
SH
chmod +x scripts/tmp_queue_dit_recovery_joint_prefix_20260627.sh
nohup bash scripts/tmp_queue_dit_recovery_joint_prefix_20260627.sh > train_logs/nohup_dit_recovery_joint_prefix_20260627.out 2>&1 &
echo "[queue] recovery_joint_prefix PID=$!"
'@

$remoteCmd = $remoteCmd.Replace('$RemoteRepo', $RemoteRepo)
Invoke-NativeChecked -FilePath ssh -Arguments @("-o", "ConnectTimeout=20", $Remote, $remoteCmd)

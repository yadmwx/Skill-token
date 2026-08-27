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

$localFile = "prismatic/models/flow_matching_head.py"
$remoteFile = "$RemoteRepo/prismatic/models/flow_matching_head.py"

Invoke-NativeChecked scp -o ConnectTimeout=20 $localFile "${Remote}:$remoteFile"

$remoteCmd = @'
set -euo pipefail
cd '$RemoteRepo'
/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python -m py_compile prismatic/models/flow_matching_head.py
grep -n 'torch.tanh(gripper_logits)' prismatic/models/flow_matching_head.py
cat > scripts/tmp_eval_dit_gripper_scale_fix_20260627.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export TOKENIZERS_PARALLELISM=false

RUN_ID=configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-mlp-anchorinit-prompt-taskonly-1000from5000-20260627
QUEUE_LOG=train_logs/queue_dit12_gripper_scale_fix_eval_20260627.log

wait_for_gpu() {
  echo "[gripfix] waiting for GPU $(date '+%F %T')" >> "$QUEUE_LOG"
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    active=$(ps -eo args= | awk '/[f]inetune.py|[r]un_libero_eval.py/ {n++} END {print n + 0}')
    echo "[gripfix] $(date '+%F %T') gpu_used=${used} util=${util} active=${active}" >> "$QUEUE_LOG"
    if [ "$active" -eq 0 ] && [ "$used" -lt 9000 ] && [ "$util" -lt 20 ]; then
      break
    fi
    sleep 60
  done
}

eval_one() {
  local step="$1"
  local seed="$2"
  local ckpt="outputs/${RUN_ID}--${step}_chkpt"
  local log="train_logs/dit12_gripper_scale_fix_step${step}_seed${seed}_eval3_20260627.log"
  [ -d "$ckpt" ] || { echo "[gripfix] missing $ckpt" >> "$QUEUE_LOG"; return 0; }
  echo "[gripfix] eval step=${step} seed=${seed} $(date '+%F %T')" >> "$QUEUE_LOG"
  ./eval.sh "$ckpt" \
    --task_suite libero_object --task_ids 0 --num_trials 3 --action_head_type DIT \
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 1.0 --dit_anchor_blend 0.0 --dit_disable_inference_anchor False --dit_pure_inference False \
    --dit_anchor_gripper_weight 1.0 --dit_anchor_gripper_bce_weight 0.0 \
    --dit_gripper_head_weight 1.0 --dit_gripper_head_override True \
    --dit_condition_mode task_only --dit_include_prompt_tokens True --dit_task_token_mode vision_prompt \
    --dit_clip_normalized_actions False --debug_dit_group_action_tokens_to_chunk False \
    --dit_zero_init_adaln False --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
    --seed "$seed" > "$log" 2>&1
  cat "$log" >> "$QUEUE_LOG"
  successes=$(grep -E "Total successes:" "$log" | tail -1 | sed -E 's/.*Total successes:[[:space:]]*([0-9]+).*/\1/' || true)
  echo "[gripfix] step=${step} seed=${seed} successes=${successes:-NA}/3" >> "$QUEUE_LOG"
}

wait_for_gpu
for step in 5500 6000; do
  for seed in 0 3 7; do
    eval_one "$step" "$seed"
  done
done
echo "[gripfix] all done $(date '+%F %T')" >> "$QUEUE_LOG"
SH
chmod +x scripts/tmp_eval_dit_gripper_scale_fix_20260627.sh
echo '[sync] gripper scale fix synced and eval script prepared'
'@

$remoteCmd = $remoteCmd.Replace('$RemoteRepo', $RemoteRepo)

Invoke-NativeChecked ssh -o ConnectTimeout=20 $Remote $remoteCmd

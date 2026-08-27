param(
    [string]$Remote = "xiaguanxiao@100.106.143.20",
    [string]$RemoteRepo = "/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation"
)

$ErrorActionPreference = "Stop"

$files = @(
    "prismatic/models/ditx_blocks.py",
    "prismatic/models/ditx_vla_adapter.py",
    "prismatic/models/action_heads.py",
    "prismatic/models/flow_matching_head.py",
    "prismatic/extern/hf/modeling_prismatic.py",
    "vla-scripts/finetune.py",
    "experiments/robot/openvla_utils.py",
    "experiments/robot/libero/run_libero_eval.py",
    "eval.sh",
    "scripts/tmp_probe_offline_action_error_20260627.py",
    "scripts/tmp_probe_dit_task_token_modes_20260627.sh",
    "scripts/tmp_queue_dit12_mlp_anchorinit_prompt_taskonly_20260627.sh",
    "scripts/tmp_run_dit_recovery_probe_after_sync_20260627.sh"
)

$syncDir = "$RemoteRepo/__tmp_sync_dit_maskaware_20260627"
ssh -o ConnectTimeout=20 $Remote "mkdir -p '$syncDir/prismatic/models' '$syncDir/prismatic/extern/hf' '$syncDir/vla-scripts' '$syncDir/experiments/robot/libero' '$syncDir/experiments/robot' '$syncDir/scripts'"

foreach ($file in $files) {
    scp $file "${Remote}:$syncDir/$file"
}

$remoteCmd = @"
set -euo pipefail
cd '$RemoteRepo'
cp '$syncDir/prismatic/models/action_heads.py' prismatic/models/action_heads.py
cp '$syncDir/prismatic/models/ditx_blocks.py' prismatic/models/ditx_blocks.py
cp '$syncDir/prismatic/models/ditx_vla_adapter.py' prismatic/models/ditx_vla_adapter.py
cp '$syncDir/prismatic/models/flow_matching_head.py' prismatic/models/flow_matching_head.py
cp '$syncDir/prismatic/extern/hf/modeling_prismatic.py' prismatic/extern/hf/modeling_prismatic.py
cp '$syncDir/vla-scripts/finetune.py' vla-scripts/finetune.py
cp '$syncDir/experiments/robot/openvla_utils.py' experiments/robot/openvla_utils.py
cp '$syncDir/experiments/robot/libero/run_libero_eval.py' experiments/robot/libero/run_libero_eval.py
cp '$syncDir/eval.sh' eval.sh
cp '$syncDir/scripts/tmp_probe_offline_action_error_20260627.py' scripts/tmp_probe_offline_action_error_20260627.py
cp '$syncDir/scripts/tmp_probe_dit_task_token_modes_20260627.sh' scripts/tmp_probe_dit_task_token_modes_20260627.sh
cp '$syncDir/scripts/tmp_queue_dit12_mlp_anchorinit_prompt_taskonly_20260627.sh' scripts/tmp_queue_dit12_mlp_anchorinit_prompt_taskonly_20260627.sh
cp '$syncDir/scripts/tmp_run_dit_recovery_probe_after_sync_20260627.sh' scripts/tmp_run_dit_recovery_probe_after_sync_20260627.sh
chmod +x eval.sh scripts/tmp_probe_dit_task_token_modes_20260627.sh scripts/tmp_queue_dit12_mlp_anchorinit_prompt_taskonly_20260627.sh scripts/tmp_run_dit_recovery_probe_after_sync_20260627.sh
/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python -m py_compile prismatic/models/ditx_blocks.py prismatic/models/ditx_vla_adapter.py prismatic/models/action_heads.py prismatic/models/flow_matching_head.py prismatic/extern/hf/modeling_prismatic.py vla-scripts/finetune.py experiments/robot/openvla_utils.py experiments/robot/libero/run_libero_eval.py scripts/tmp_probe_offline_action_error_20260627.py
bash -n scripts/tmp_probe_dit_task_token_modes_20260627.sh
bash -n scripts/tmp_queue_dit12_mlp_anchorinit_prompt_taskonly_20260627.sh
bash -n scripts/tmp_run_dit_recovery_probe_after_sync_20260627.sh
bash -n eval.sh
echo '[sync] DIT mask-aware fixes synced and verified'
"@

ssh -o ConnectTimeout=20 $Remote $remoteCmd

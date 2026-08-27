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

$remoteCmd = @'
set -euo pipefail
cd '$RemoteRepo'
echo "=== TIME ==="
date
echo "=== PROCESSES ==="
pgrep -af '[f]inetune.py' || true
pgrep -af '[r]un_libero_eval.py' || true
pgrep -af 'tmp_queue_dit_recovery_joint_prefix_20260627.sh' || true
pgrep -af 'tmp_queue_dit12_mlp_anchorinit_prompt_taskonly_20260627.sh' || true
echo "=== CHECKPOINTS ==="
ls -d outputs/*DIT12-mlp-anchorinit*joint_prefix*--*_chkpt 2>/dev/null || true
ls -d outputs/*DIT12-mlp-anchorinit*1000from5000*--5500_chkpt 2>/dev/null || true
echo "=== RECOVERY QUEUE ==="
tail -260 train_logs/queue_dit_recovery_joint_prefix_20260627.log 2>/dev/null || true
echo "=== JOINT PREFIX QUEUE ==="
tail -260 train_logs/queue_dit12_mlp_anchorinit_vision_prompt_task_only_joint_prefix_20260627.log 2>/dev/null || true
echo "=== GRIPPER FIX SUMMARY ==="
grep -R "gripper_fixed_existing5500.*successes=" train_logs/queue_dit_recovery_joint_prefix_20260627.log 2>/dev/null || true
echo "=== JOINT PREFIX ONLINE SUMMARY ==="
grep -R "mlp_anchorinit step=.*successes=" train_logs/queue_dit12_mlp_anchorinit_vision_prompt_task_only_joint_prefix_20260627.log 2>/dev/null || true
echo "=== JOINT PREFIX OFFLINE SUMMARY ==="
grep -R -E 'teacher_flow_recon|teacher_flow_velocity|MAE|split_info|condition_injection' train_logs/dit12_mlp_anchorinit_vision_prompt_task_only_joint_prefix_step*_offline_probe_20260627.log 2>/dev/null || true
'@

$remoteCmd = $remoteCmd.Replace('$RemoteRepo', $RemoteRepo)
Invoke-NativeChecked -FilePath ssh -Arguments @("-o", "ConnectTimeout=20", $Remote, $remoteCmd)

param(
    [string]$Mode = "state"
)

$ErrorActionPreference = "Stop"

$repo = "/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation"

function Invoke-RemotePython {
    param(
        [string]$Code
    )

    $escaped = $Code.Replace('"', '\"')
    ssh xiaguanxiao@100.106.143.20 "bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla-flow && cd $repo && python -c \"$escaped\"'"
}

switch ($Mode) {
    "state" {
        $code = @'
import torch

paths = [
    "outputs/VLA-Adapter-DIT12-anchor-bce-lr5e5-cont-20260620--2000_chkpt/training_state--2000_checkpoint.pt",
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-residual-anchor-object-1000-20260620-r2--2000_chkpt/training_state--2000_checkpoint.pt",
]
interesting = [
    "action_head_type",
    "dit_num_blocks",
    "dit_num_inference_steps",
    "dit_num_inference_samples",
    "dit_supervised_anchor_weight",
    "dit_anchor_blend",
    "dit_anchor_gripper_weight",
    "dit_anchor_gripper_bce_weight",
    "dit_detach_flow_conditioning",
    "use_adaptive_bridge",
    "bridge_mode",
    "fixed_layer_index",
    "flow_ratio",
]

for path in paths:
    print(f"=== {path} ===")
    obj = torch.load(path, map_location="cpu")
    print("top-level type:", type(obj).__name__)
    if not isinstance(obj, dict):
        print(obj)
        print()
        continue
    print("keys:", sorted(obj.keys()))
    cfg = None
    for key in ("cfg", "config", "args", "run_config"):
        if key in obj:
            cfg = obj[key]
            print("config_container:", key, type(cfg).__name__)
            break
    if cfg is None and "state" in obj and isinstance(obj["state"], dict):
        state = obj["state"]
        for key in ("cfg", "config", "args", "run_config"):
            if key in state:
                cfg = state[key]
                print("config_container:", f"state.{key}", type(cfg).__name__)
                break
    if cfg is None:
        print("no config-like object found")
        print()
        continue
    for name in interesting:
        if isinstance(cfg, dict):
            value = cfg.get(name, "<missing>")
        else:
            value = getattr(cfg, name, "<missing>")
        print(f"{name}={value}")
    print()
'@
        Invoke-RemotePython -Code $code
    }
    "files" {
        ssh xiaguanxiao@100.106.143.20 "bash -lc 'cd $repo && find outputs/VLA-Adapter-DIT12-anchor-bce-lr5e5-cont-20260620--2000_chkpt -maxdepth 1 -type f | sort && echo ===== && find outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-residual-anchor-object-1000-20260620-r2--2000_chkpt -maxdepth 1 -type f | sort'"
    }
    "logs" {
        ssh xiaguanxiao@100.106.143.20 "bash -lc 'cd $repo && tail -n 120 experiments/logs/EVAL-libero_object-openvla-2026_06_21-14_30_39.txt && echo ===== && tail -n 120 experiments/logs/EVAL-libero_object-openvla-2026_06_21-14_37_04.txt'"
    }
    default {
        throw "Unknown mode: $Mode"
    }
}

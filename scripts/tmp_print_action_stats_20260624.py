import json
from pathlib import Path


PATHS = [
    Path(
        "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
        "--image_aug--VLA-Adapter-DIT12-fullaction-pure-gripfix-nozeroinit-object-1000-20260623--1000_chkpt"
        "/dataset_statistics.json"
    ),
    Path(
        "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
        "--image_aug--VLA-Adapter-DIT12-fullaction-pure-actionhead-only1200-20260624--1200_chkpt"
        "/dataset_statistics.json"
    ),
    Path(
        "data/libero/libero_object_no_noops/1.0.0/"
        "dataset_statistics_3d3d2846db15a1d4ae01e96021e1b696a3912ee5714ec63b6ebfae7a110ff6df.json"
    ),
]


def main():
    for path in PATHS:
        print(f"PATH {path}")
        if not path.exists():
            print("MISSING")
            print()
            continue
        data = json.loads(path.read_text())
        stats = data.get("libero_object_no_noops", data)
        action = stats.get("action", {})
        for key in ["mask", "min", "max", "q01", "q99", "mean", "std"]:
            if key in action:
                print(f"{key} {action[key]}")
        print()


if __name__ == "__main__":
    main()

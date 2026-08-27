#!/usr/bin/env bash
# Materialize the official CALVIN-D evaluation environment without downloading
# the 165-GB task_D_D archive.  CALVIN evaluation only consumes
# validation/.hydra/merged_config.yaml; it is byte-identical to the config
# shipped in the official calvin_debug_dataset archive.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation}
SOURCE="$REPO_DIR/calvin/dataset/calvin_debug_dataset/validation/.hydra/merged_config.yaml"
TARGET="$REPO_DIR/calvin/dataset/task_D_D/validation/.hydra/merged_config.yaml"
OFFICIAL_SHA256=75c1c15067e4bd543d1b69a654769836f6cbbf55c4e44ae30da255c45609523f

test -f "$SOURCE" || {
  echo "Missing CALVIN debug dataset config: $SOURCE" >&2
  exit 1
}

actual=$(sha256sum "$SOURCE" | awk '{print $1}')
test "$actual" = "$OFFICIAL_SHA256" || {
  echo "Unexpected CALVIN-D config checksum: $actual" >&2
  exit 1
}

mkdir -p "$(dirname "$TARGET")"
cp "$SOURCE" "$TARGET"
echo "Prepared official CALVIN-D evaluation environment: $TARGET"
